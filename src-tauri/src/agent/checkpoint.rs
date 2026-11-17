//! Persistent workspace checkpoints for Agent Mode.
//!
//! A checkpoint is a commit in a private bare Git repository under the app data
//! directory. It deliberately does not touch the workspace's own `.git`
//! directory, index, branches or stash: snapshots use a temporary index and
//! the shadow repository's own refs.
//!
//! Each run writes two commits (`pre` and `post`) plus a small JSON record. The
//! record and repository objects survive app restarts. Undo compares every
//! changed path with the post-run blob and only restores paths that still match
//! what the run left behind, so edits made after the run are never overwritten.
//!
//! Git-ignored paths are intentionally excluded from checkpoints (the shadow
//! repository uses the workspace's normal ignore rules plus a small list of
//! heavy build directories). Checkpoints therefore cover source/workspace file
//! changes, not build output or other ignored files.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::ffi::OsString;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

const FORMAT_VERSION: u32 = 1;
const MAX_RUNS: usize = 20;
const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_SNAPSHOT_BYTES: u64 = 1_500 * 1024 * 1024;
const MAX_SNAPSHOT_FILES: usize = 50_000;
const STALE_RUNNING_MS: u64 = 24 * 60 * 60 * 1000;
const MAX_ATTEMPTS: usize = 20;

const GIT_EXCLUDES: &str = "\
.git/
.tools/
node_modules/
target/
dist/
build/
.next/
.cache/
.venv/
venv/
__pycache__/
";

/// Git stores only the executable bit, so disabling filters and text
/// conversion here is what makes the snapshot byte-for-byte restorable. This
/// attributes file belongs to the private shadow repository, not the workspace.
const GIT_ATTRIBUTES: &str = "* -filter -text\n";

#[derive(Debug, Clone, Copy)]
struct Limits {
    max_runs: usize,
    max_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_runs: MAX_RUNS,
            max_bytes: MAX_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MetadataStatus {
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttributedPath {
    path: PathBuf,
    /// Git mode of the version the run left behind; 0 when it deleted the path.
    mode: u32,
    /// Blob id of the version the run left behind; empty for a deletion.
    #[serde(default)]
    oid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunMetadata {
    version: u32,
    run_id: String,
    checkpoint_id: String,
    workspace: PathBuf,
    pre: String,
    #[serde(default)]
    post: Option<String>,
    created_at: u64,
    #[serde(default)]
    changed_files: u32,
    status: MetadataStatus,
    /// Paths the agent itself changed during the run, with the exact content
    /// it left behind. `None` marks checkpoints written before attribution
    /// existed; those fall back to comparing against the final snapshot.
    #[serde(default)]
    attribution: Option<Vec<AttributedPath>>,
    /// Per-file undo state, including the reason each attempt produced. Older
    /// checkpoints leave this empty and seed it from `attribution` on first use.
    #[serde(default)]
    undo_files: Vec<FileUndoEntry>,
    /// Selective restores recorded by older checkpoints. Migrated into
    /// `undo_files` the next time the checkpoint is read.
    #[serde(default)]
    restored_paths: Vec<PathBuf>,
    /// Filled when a metadata file is read; never serialized.
    #[serde(skip, default)]
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileUndoState {
    /// Not attempted yet, or waiting after a retryable skip or failure.
    Pending,
    Restored,
    Skipped,
    Failed,
    /// The user accepted their own edit, so the pre-image must not be applied.
    Kept,
}

impl FileUndoState {
    fn unresolved(self) -> bool {
        matches!(self, Self::Pending | Self::Skipped | Self::Failed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoAction {
    Restore,
    Keep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoAttempt {
    pub action: UndoAction,
    pub state: FileUndoState,
    /// Full explanation for the attempt, kept so a retry does not erase why an
    /// earlier restore was skipped.
    pub reason: Option<String>,
    pub at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileUndoEntry {
    pub path: String,
    /// `created`, `modified` or `deleted`, for the run-change summary. Older
    /// checkpoints default to `modified` and are corrected from the diff.
    #[serde(default = "default_change_kind")]
    pub kind: String,
    pub state: FileUndoState,
    pub reason: Option<String>,
    pub attempts: u32,
    pub history: Vec<UndoAttempt>,
}

impl FileUndoEntry {
    pub(crate) fn pending(path: String, kind: String) -> Self {
        Self {
            path,
            kind,
            state: FileUndoState::Pending,
            reason: None,
            attempts: 0,
            history: Vec::new(),
        }
    }

    fn set_kind(&mut self, kind: &str) {
        self.kind = kind.to_string();
    }

    pub(crate) fn record(&mut self, action: UndoAction, state: FileUndoState, reason: Option<String>, at: u64) {
        self.state = state;
        self.reason = reason.clone();
        self.attempts = self.attempts.saturating_add(1);
        self.history.push(UndoAttempt {
            action,
            state,
            reason,
            at,
        });
        if self.history.len() > MAX_ATTEMPTS {
            let extra = self.history.len() - MAX_ATTEMPTS;
            self.history.drain(0..extra);
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct UndoOutcome {
    pub restored: u32,
    pub deleted: u32,
    /// Every file the run changed, with its current state and history.
    pub files: Vec<FileUndoEntry>,
    /// Files that still need a restore or a keep decision.
    pub pending: u32,
    /// True once nothing is pending; the checkpoint is removed at that point.
    pub complete: bool,
    pub incomplete: bool,
}

#[derive(Debug, Clone)]
struct DiffEntry {
    path: PathBuf,
    old_mode: u32,
    new_mode: u32,
    old_oid: String,
    new_oid: String,
}

pub struct CheckpointStore {
    root: PathBuf,
    repo: PathBuf,
    runs: PathBuf,
    limits: Limits,
    lock: Mutex<()>,
}

impl CheckpointStore {
    pub fn new(root: PathBuf) -> Self {
        Self {
            repo: root.join("repo.git"),
            runs: root.join("runs"),
            root,
            limits: Limits::default(),
            lock: Mutex::new(()),
        }
    }

    #[cfg(test)]
    fn with_limits(root: PathBuf, max_runs: usize, max_bytes: u64) -> Self {
        let mut store = Self::new(root);
        store.limits = Limits {
            max_runs,
            max_bytes,
        };
        store
    }

    /// Snapshot the workspace before a run. The returned guard must be finished
    /// once the run has stopped changing the workspace.
    pub fn begin(
        self: &Arc<Self>,
        run_id: &str,
        workspace: &Path,
    ) -> Result<RunCheckpoint, String> {
        let workspace = fs::canonicalize(workspace)
            .map_err(|error| format!("Could not resolve the workspace: {error}"))?;
        if !workspace.is_dir() {
            return Err(format!(
                "The workspace is not a directory: {}",
                workspace.display()
            ));
        }
        if workspace == Path::new("/") {
            return Err("Refusing to checkpoint the filesystem root.".into());
        }
        let store_root = fs::canonicalize(&self.root).unwrap_or_else(|_| self.root.clone());
        let (bytes, files) = workspace_usage(&workspace, Some(&store_root))?;
        if bytes > MAX_SNAPSHOT_BYTES {
            return Err(format!(
                "The workspace is too large to checkpoint safely ({} MiB; limit {} MiB).",
                bytes / (1024 * 1024),
                MAX_SNAPSHOT_BYTES / (1024 * 1024)
            ));
        }
        if files > MAX_SNAPSHOT_FILES {
            return Err(format!(
                "The workspace has too many files to checkpoint safely ({}; limit {}).",
                files, MAX_SNAPSHOT_FILES
            ));
        }

        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        self.ensure_repo()?;
        self.exclude_store_from_workspace(&workspace)?;
        // A previous crash can leave running/failed records behind.
        let _ = self.cleanup_locked(None);

        let (pre_tree, pre) = self.snapshot_locked(&workspace, "pre", None)?;
        let checkpoint_id = Uuid::new_v4().simple().to_string();
        let created_at = now_ms();
        let metadata = RunMetadata {
            version: FORMAT_VERSION,
            run_id: run_id.to_string(),
            checkpoint_id: checkpoint_id.clone(),
            workspace: workspace.clone(),
            pre: pre.clone(),
            post: None,
            created_at,
            changed_files: 0,
            status: MetadataStatus::Running,
            attribution: None,
            undo_files: Vec::new(),
            restored_paths: Vec::new(),
            path: PathBuf::new(),
        };

        self.set_ref(&checkpoint_id, "pre", &pre)?;
        let metadata_path = match self.write_new_metadata(&metadata) {
            Ok(path) => path,
            Err(error) => {
                let _ = self.delete_refs(&checkpoint_id);
                return Err(error);
            }
        };

        let recorder = Arc::new(RunRecorder {
            store: self.clone(),
            workspace: workspace.clone(),
            checkpoint_id: checkpoint_id.clone(),
            state: Mutex::new(RecorderState {
                last_tree: Some(pre_tree),
                last_commit: None,
                attribution: BTreeMap::new(),
            }),
        });

        Ok(RunCheckpoint {
            store: self.clone(),
            workspace,
            checkpoint_id,
            pre,
            metadata_path,
            recorder,
        })
    }

    /// Restore a completed run by its persisted run id. This works from a new
    /// process because the metadata and Git objects live under app data.
    pub fn undo(&self, run_id: &str) -> Result<UndoOutcome, String> {
        self.undo_selected(run_id, None)
    }

    /// Restore a single changed file. Files that were already restored or kept
    /// are left alone, so a retry only touches what still needs it.
    pub fn undo_path(&self, run_id: &str, path: &str) -> Result<UndoOutcome, String> {
        self.undo_selected(run_id, Some(path))
    }

    /// Accept the user's own version of a file without changing it, resolving
    /// that file's pending undo. The checkpoint is removed once the last file
    /// is resolved this way.
    pub fn keep_path(&self, run_id: &str, path: &str) -> Result<UndoOutcome, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let mut metadata = self.complete_locked(run_id)?;
        let entries = self.attributed_entries_locked(&metadata)?;
        ensure_undo_files(&mut metadata, &entries);
        let index = metadata
            .undo_files
            .iter()
            .position(|file| file.path == path)
            .ok_or_else(|| format!("{path} is not part of this run's undo set."))?;
        match metadata.undo_files[index].state {
            FileUndoState::Restored => {
                return Err(format!("{path} was already restored by undo."));
            }
            FileUndoState::Kept => {
                let outcome = outcome_from_files(metadata.undo_files.clone());
                self.persist_outcome_locked(metadata, &outcome)?;
                return Ok(outcome);
            }
            _ => {}
        }
        let at = now_ms();
        metadata.undo_files[index].record(
            UndoAction::Keep,
            FileUndoState::Kept,
            Some("kept your version; the file was left unchanged".to_string()),
            at,
        );
        let outcome = outcome_from_files(metadata.undo_files.clone());
        self.persist_outcome_locked(metadata, &outcome)?;
        Ok(outcome)
    }

    fn undo_selected(&self, run_id: &str, only: Option<&str>) -> Result<UndoOutcome, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let mut metadata = self.complete_locked(run_id)?;
        let outcome = self.undo_metadata_locked(&mut metadata, only)?;
        self.persist_outcome_locked(metadata, &outcome)?;
        Ok(outcome)
    }

    /// Persist the latest per-file state, or remove the checkpoint once every
    /// file is either restored or kept.
    fn persist_outcome_locked(
        &self,
        mut metadata: RunMetadata,
        outcome: &UndoOutcome,
    ) -> Result<(), String> {
        if outcome.complete {
            self.delete_entry_locked(&metadata);
            let _ = self.cleanup_locked(None);
            return Ok(());
        }
        metadata.undo_files = outcome.files.clone();
        metadata.restored_paths = outcome
            .files
            .iter()
            .filter(|file| file.state == FileUndoState::Restored)
            .map(|file| PathBuf::from(&file.path))
            .collect();
        self.write_metadata_at(&metadata.path, &metadata)?;
        let _ = self.cleanup_locked(Some(&metadata.checkpoint_id));
        Ok(())
    }

    pub fn available(&self, run_ids: &[String]) -> Vec<String> {
        let Ok(_guard) = self.lock.lock() else {
            return Vec::new();
        };
        let Ok(metadatas) = self.read_metadata_locked() else {
            return Vec::new();
        };
        let wanted: HashSet<&str> = run_ids.iter().map(String::as_str).collect();
        let mut seen = HashSet::new();
        metadatas
            .iter()
            .filter(|metadata| {
                metadata.status == MetadataStatus::Complete
                    && metadata_has_pending(metadata)
                    && metadata.post.is_some()
                    && metadata.workspace.is_dir()
                    && wanted.contains(metadata.run_id.as_str())
            })
            .filter_map(|metadata| {
                if seen.insert(metadata.run_id.clone()) {
                    Some(metadata.run_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn has_complete(&self, run_id: &str) -> bool {
        let Ok(_guard) = self.lock.lock() else {
            return false;
        };
        self.find_complete_locked(run_id)
            .ok()
            .flatten()
            .map(|metadata| metadata_has_pending(&metadata))
            .unwrap_or(false)
    }

    /// True when a completed checkpoint exists for this run, even if every
    /// file has already been restored. This keeps the in-memory journal from
    /// making a fully restored run look undoable again.
    pub fn has_record(&self, run_id: &str) -> bool {
        let Ok(_guard) = self.lock.lock() else {
            return false;
        };
        self.find_complete_locked(run_id)
            .ok()
            .flatten()
            .map(|metadata| metadata.changed_files > 0)
            .unwrap_or(false)
    }

    /// Current per-file undo state without changing anything. Used to show the
    /// file list as soon as a run finishes and after a restart.
    pub fn state(&self, run_id: &str) -> Result<UndoOutcome, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let mut metadata = self.complete_locked(run_id)?;
        let entries = self.attributed_entries_locked(&metadata)?;
        ensure_undo_files(&mut metadata, &entries);
        Ok(outcome_from_files(metadata.undo_files))
    }

    /// Workspace a completed checkpoint belongs to, used to keep an undo from
    /// racing a new run in the same directory.
    pub fn workspace_of(&self, run_id: &str) -> Option<PathBuf> {
        let _guard = self.lock.lock().ok()?;
        self.find_complete_locked(run_id)
            .ok()
            .flatten()
            .map(|metadata| metadata.workspace)
    }

    fn undo_metadata_locked(
        &self,
        metadata: &mut RunMetadata,
        only: Option<&str>,
    ) -> Result<UndoOutcome, String> {
        let workspace = fs::canonicalize(&metadata.workspace)
            .map_err(|error| format!("The checkpoint workspace is unavailable: {error}"))?;
        if !workspace.is_dir() {
            return Err(format!(
                "The checkpoint workspace is not a directory: {}",
                workspace.display()
            ));
        }
        let entries = self.attributed_entries_locked(metadata)?;
        ensure_undo_files(metadata, &entries);
        if let Some(only) = only {
            if !metadata.undo_files.iter().any(|file| file.path == only) {
                return Err(format!("{only} is not part of this run's undo set."));
            }
        }

        let mut outcome = UndoOutcome::default();
        for (entry, expected) in &entries {
            let label = path_label(&entry.path);
            if only.map(|only| only != label).unwrap_or(false) {
                continue;
            }
            let Some(index) = metadata
                .undo_files
                .iter()
                .position(|file| file.path == label)
            else {
                continue;
            };
            // A file that is already restored or kept stays resolved. This is
            // what makes a per-file retry target only the files that still
            // need a decision.
            if !metadata.undo_files[index].state.unresolved() {
                continue;
            }
            let at = now_ms();
            match self.restore_entry(&workspace, entry, expected.as_ref()) {
                Ok(RestoreKind::Restored) => {
                    metadata.undo_files[index].record(
                        UndoAction::Restore,
                        FileUndoState::Restored,
                        None,
                        at,
                    );
                    outcome.restored += 1;
                }
                Ok(RestoreKind::Deleted) => {
                    metadata.undo_files[index].record(
                        UndoAction::Restore,
                        FileUndoState::Restored,
                        None,
                        at,
                    );
                    outcome.deleted += 1;
                }
                Ok(RestoreKind::Skipped(reason)) => {
                    metadata.undo_files[index].record(
                        UndoAction::Restore,
                        FileUndoState::Skipped,
                        Some(reason),
                        at,
                    );
                }
                Err(error) => {
                    outcome.incomplete = true;
                    metadata.undo_files[index].record(
                        UndoAction::Restore,
                        FileUndoState::Failed,
                        Some(error),
                        at,
                    );
                }
            }
        }

        outcome.files = metadata.undo_files.clone();
        outcome.pending = outcome
            .files
            .iter()
            .filter(|file| file.state.unresolved())
            .count() as u32;
        outcome.complete = outcome.pending == 0;
        Ok(outcome)
    }

    /// Changed paths the run itself touched, paired with the content it left
    /// behind. Anything else in the pre/post diff was changed by another
    /// process and is never offered to undo.
    fn attributed_entries_locked(
        &self,
        metadata: &RunMetadata,
    ) -> Result<Vec<(DiffEntry, Option<AttributedPath>)>, String> {
        let post = metadata
            .post
            .as_deref()
            .ok_or_else(|| "This checkpoint has no post-run snapshot.".to_string())?;
        let entries = self.diff_locked(&metadata.pre, post)?;
        let attributed: Option<HashMap<PathBuf, AttributedPath>> = metadata.attribution.as_ref().map(|paths| {
            paths
                .iter()
                .map(|path| (path.path.clone(), path.clone()))
                .collect()
        });
        Ok(entries
            .into_iter()
            .filter_map(|entry| match &attributed {
                Some(paths) => paths
                    .get(&entry.path)
                    .cloned()
                    .map(|expected| (entry, Some(expected))),
                None => Some((entry, None)),
            })
            .collect())
    }

    fn restore_entry(
        &self,
        workspace: &Path,
        entry: &DiffEntry,
        expected: Option<&AttributedPath>,
    ) -> Result<RestoreKind, String> {
        if !safe_relative(&entry.path) {
            return Err("Git returned an unsafe path.".into());
        }
        let path = workspace.join(&entry.path);
        if !path.starts_with(workspace) {
            return Err("The path escaped the workspace.".into());
        }
        // The final component is checked by symlink_metadata below, but a
        // parent directory could have been swapped for a symlink pointing
        // outside the workspace after the run. Never write through one.
        if !parent_inside_workspace(workspace, &path) {
            return Ok(RestoreKind::Skipped(
                "its directory moved outside the workspace".to_string(),
            ));
        }

        // Compare with the version the agent left behind, not with the final
        // snapshot: a later edit must win even if the run had touched the file.
        let (expected_mode, expected_blob) = match expected {
            Some(expected) => {
                let blob = if expected.mode == 0 {
                    None
                } else {
                    Some(self.cat_file(&expected.oid)?)
                };
                (expected.mode, blob)
            }
            None => {
                let blob = if entry.new_mode == 0 {
                    None
                } else {
                    Some(self.cat_file(&entry.new_oid)?)
                };
                (entry.new_mode, blob)
            }
        };
        if !self.matches_post(&path, expected_mode, expected_blob.as_deref())? {
            return Ok(RestoreKind::Skipped(
                "changed since the run ended".to_string(),
            ));
        }

        if entry.old_mode == 0 {
            remove_current(&path)?;
            return Ok(RestoreKind::Deleted);
        }

        let pre_blob = self.cat_file(&entry.old_oid)?;
        if entry.new_mode != 0 {
            remove_current(&path)?;
        }

        match entry.old_mode {
            0o120000 => restore_symlink(&path, &pre_blob)?,
            0o100644 | 0o100755 => restore_regular(&path, &pre_blob, entry.old_mode)?,
            0o160000 => {
                return Ok(RestoreKind::Skipped(
                    "submodule checkpoints are not restorable".to_string(),
                ))
            }
            mode => {
                return Err(format!("unsupported Git mode {mode:o} in the checkpoint"));
            }
        }
        Ok(RestoreKind::Restored)
    }

    fn matches_post(
        &self,
        path: &Path,
        post_mode: u32,
        post_blob: Option<&[u8]>,
    ) -> Result<bool, String> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(format!("could not inspect the current path: {error}"));
            }
        };

        if post_mode == 0 {
            return Ok(metadata.is_none());
        }
        let Some(metadata) = metadata else {
            return Ok(false);
        };
        let Some(post_blob) = post_blob else {
            return Ok(false);
        };

        match post_mode {
            0o120000 => {
                if !metadata.file_type().is_symlink() {
                    return Ok(false);
                }
                let target =
                    fs::read_link(path).map_err(|error| format!("could not read the symlink: {error}"))?;
                Ok(os_bytes(target.as_os_str()) == post_blob)
            }
            0o100644 | 0o100755 => {
                if !metadata.is_file() {
                    return Ok(false);
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let executable = metadata.permissions().mode() & 0o111 != 0;
                    if executable != (post_mode == 0o100755) {
                        return Ok(false);
                    }
                }
                let current = fs::read(path)
                    .map_err(|error| format!("could not read the current file: {error}"))?;
                Ok(current == post_blob)
            }
            0o160000 => Ok(false),
            _ => Ok(false),
        }
    }

    fn cat_file(&self, oid: &str) -> Result<Vec<u8>, String> {
        self.git_bytes(&["cat-file", "blob", oid], None, None)
    }

    fn diff_locked(&self, pre: &str, post: &str) -> Result<Vec<DiffEntry>, String> {
        let output = self.git_bytes(
            &[
                "diff",
                "--raw",
                "-z",
                "--no-renames",
                "--no-color",
                "--no-ext-diff",
                pre,
                post,
            ],
            None,
            None,
        )?;
        let mut entries = Vec::new();
        let mut chunks = output.split(|byte| *byte == 0);
        while let Some(chunk) = chunks.next() {
            if chunk.is_empty() {
                continue;
            }
            let header = std::str::from_utf8(chunk)
                .map_err(|_| "Git returned a non-UTF-8 diff header.".to_string())?;
            let header = header
                .strip_prefix(':')
                .ok_or_else(|| "Git returned an unexpected raw diff.".to_string())?;
            let fields: Vec<&str> = header.split(' ').collect();
            if fields.len() < 5 {
                return Err("Git returned an incomplete raw diff.".to_string());
            }
            let path = chunks
                .next()
                .ok_or_else(|| "Git returned a diff entry without a path.".to_string())?;
            let path = git_path(path);
            entries.push(DiffEntry {
                path,
                old_mode: parse_mode(fields[0])?,
                new_mode: parse_mode(fields[1])?,
                old_oid: fields[2].to_string(),
                new_oid: fields[3].to_string(),
            });
        }
        Ok(entries)
    }

    fn tree_locked(&self, workspace: &Path) -> Result<String, String> {
        let index = self.root.join(format!("index-{}.tmp", Uuid::new_v4().simple()));
        let add = self.git(&["add", "-A"], Some(workspace), Some(&index));
        if let Err(error) = add {
            let _ = fs::remove_file(&index);
            return Err(error);
        }
        let tree = match self.git(&["write-tree"], None, Some(&index)) {
            Ok(tree) => tree.trim().to_string(),
            Err(error) => {
                let _ = fs::remove_file(&index);
                return Err(error);
            }
        };
        let _ = fs::remove_file(&index);
        Ok(tree)
    }

    fn snapshot_locked(
        &self,
        workspace: &Path,
        label: &str,
        parent: Option<&str>,
    ) -> Result<(String, String), String> {
        let tree = self.tree_locked(workspace)?;

        let mut args = vec!["commit-tree", tree.as_str()];
        if let Some(parent) = parent {
            args.push("-p");
            args.push(parent);
        }
        args.push("-m");
        args.push(label);
        let commit = self.git(&args, None, None)?.trim().to_string();
        Ok((tree, commit))
    }

    /// Snapshot the workspace into a tree without creating a checkpoint commit.
    /// The recorder uses this to bracket mutating tool calls.
    fn tree_for(&self, workspace: &Path) -> Result<String, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        self.tree_locked(workspace)
    }

    fn diff_trees(&self, before: &str, after: &str) -> Result<Vec<DiffEntry>, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        self.diff_locked(before, after)
    }

    /// Keep a step tree reachable from a ref so its blobs survive cleanup and
    /// `gc` until this run is undone or evicted.
    fn protect_step(
        &self,
        checkpoint_id: &str,
        tree: &str,
        parent: Option<&str>,
    ) -> Result<String, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let mut args = vec!["commit-tree", tree];
        if let Some(parent) = parent {
            args.push("-p");
            args.push(parent);
        }
        args.push("-m");
        args.push("step");
        let commit = self.git(&args, None, None)?.trim().to_string();
        self.git(
            &[
                "update-ref",
                &format!("refs/janas/{checkpoint_id}/steps"),
                &commit,
            ],
            None,
            None,
        )?;
        Ok(commit)
    }

    fn set_ref(&self, checkpoint_id: &str, phase: &str, commit: &str) -> Result<(), String> {
        self.git(
            &[
                "update-ref",
                &format!("refs/janas/{checkpoint_id}/{phase}"),
                commit,
            ],
            None,
            None,
        )
        .map(|_| ())
    }

    fn delete_refs(&self, checkpoint_id: &str) -> Result<(), String> {
        for phase in ["pre", "post", "steps"] {
            self.git(
                &[
                    "update-ref",
                    "-d",
                    &format!("refs/janas/{checkpoint_id}/{phase}"),
                ],
                None,
                None,
            )?;
        }
        Ok(())
    }

    fn delete_entry_locked(&self, metadata: &RunMetadata) {
        let _ = self.delete_refs(&metadata.checkpoint_id);
        if !metadata.path.as_os_str().is_empty() {
            let _ = fs::remove_file(&metadata.path);
        }
    }

    fn ensure_repo(&self) -> Result<(), String> {
        if self.repo.join("HEAD").is_file() {
            self.ensure_repo_config()?;
            return Ok(());
        }
        fs::create_dir_all(&self.root)
            .map_err(|error| format!("Could not create the checkpoint directory: {error}"))?;
        let output = Command::new("git")
            .args(["init", "--bare", "--quiet"])
            .arg(&self.repo)
            .output()
            .map_err(|error| {
                format!(
                    "Could not run Git to create a workspace checkpoint. Install Git or make it available on PATH: {error}"
                )
            })?;
        if !output.status.success() {
            return Err(format!(
                "Git could not initialise the checkpoint repository: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        self.ensure_repo_config()
    }

    fn ensure_repo_config(&self) -> Result<(), String> {
        for (key, value) in [
            ("user.name", "Janas Agent"),
            ("user.email", "agent@janas.local"),
            ("core.autocrlf", "false"),
            ("core.safecrlf", "false"),
            ("gc.auto", "0"),
        ] {
            self.git(&["config", key, value], None, None)?;
        }
        #[cfg(unix)]
        {
            self.git(&["config", "core.fileMode", "true"], None, None)?;
            self.git(&["config", "core.symlinks", "true"], None, None)?;
        }

        let info = self.repo.join("info");
        fs::create_dir_all(&info)
            .map_err(|error| format!("Could not create the checkpoint Git info directory: {error}"))?;
        fs::write(info.join("attributes"), GIT_ATTRIBUTES)
            .map_err(|error| format!("Could not configure checkpoint attributes: {error}"))?;

        // Preserve the workspace-specific exclusion added by begin(), because
        // the app data directory can itself be inside a wide workspace (for
        // example when the user selects their home directory).
        let exclude_path = info.join("exclude");
        let mut excludes = fs::read_to_string(&exclude_path).unwrap_or_default();
        for line in GIT_EXCLUDES.lines() {
            if line.is_empty() || excludes.lines().any(|existing| existing == line) {
                continue;
            }
            if !excludes.ends_with('\n') && !excludes.is_empty() {
                excludes.push('\n');
            }
            excludes.push_str(line);
            excludes.push('\n');
        }
        fs::write(&exclude_path, excludes)
            .map_err(|error| format!("Could not configure checkpoint excludes: {error}"))?;
        Ok(())
    }

    fn exclude_store_from_workspace(&self, workspace: &Path) -> Result<(), String> {
        let store = fs::canonicalize(&self.root).unwrap_or_else(|_| self.root.clone());
        let Ok(relative) = store.strip_prefix(workspace) else {
            return Ok(());
        };
        if relative.as_os_str().is_empty() {
            return Ok(());
        }
        let line = format!("/{}/", relative.to_string_lossy().replace('\\', "/"));
        let path = self.repo.join("info/exclude");
        let mut excludes = fs::read_to_string(&path).unwrap_or_default();
        if excludes.lines().any(|existing| existing == line) {
            return Ok(());
        }
        if !excludes.ends_with('\n') && !excludes.is_empty() {
            excludes.push('\n');
        }
        excludes.push_str(&line);
        excludes.push('\n');
        fs::write(&path, excludes)
            .map_err(|error| format!("Could not exclude the checkpoint store from the workspace: {error}"))
    }

    fn write_new_metadata(&self, metadata: &RunMetadata) -> Result<PathBuf, String> {
        fs::create_dir_all(&self.runs)
            .map_err(|error| format!("Could not create the checkpoint journal: {error}"))?;
        let path = self.runs.join(format!(
            "{}-{}.json",
            metadata.created_at, metadata.checkpoint_id
        ));
        self.write_metadata_at(&path, metadata)?;
        Ok(path)
    }

    fn write_metadata_at(&self, path: &Path, metadata: &RunMetadata) -> Result<(), String> {
        let temp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(metadata)
            .map_err(|error| format!("Could not encode checkpoint metadata: {error}"))?;
        fs::write(&temp, bytes)
            .map_err(|error| format!("Could not write checkpoint metadata: {error}"))?;
        if path.exists() {
            let _ = fs::remove_file(path);
        }
        fs::rename(&temp, path)
            .map_err(|error| format!("Could not commit checkpoint metadata: {error}"))
    }

    fn read_metadata_path(&self, path: &Path) -> Result<RunMetadata, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("Could not read checkpoint metadata: {error}"))?;
        let mut metadata: RunMetadata = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Could not parse checkpoint metadata: {error}"))?;
        metadata.path = path.to_path_buf();
        Ok(metadata)
    }

    fn read_metadata_locked(&self) -> Result<Vec<RunMetadata>, String> {
        let entries = match fs::read_dir(&self.runs) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(format!("Could not read the checkpoint journal: {error}"));
            }
        };
        let mut metadata = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            match self.read_metadata_path(&path) {
                Ok(value) => metadata.push(value),
                Err(_) => {
                    // A torn or old record cannot be used; remove it. Any refs
                    // it owned are swept by cleanup_locked.
                    let _ = fs::remove_file(path);
                }
            }
        }
        Ok(metadata)
    }

    fn find_complete_locked(&self, run_id: &str) -> Result<Option<RunMetadata>, String> {
        let mut found: Option<RunMetadata> = None;
        for metadata in self.read_metadata_locked()? {
            if metadata.run_id != run_id || metadata.status != MetadataStatus::Complete {
                continue;
            }
            if metadata.post.is_none() {
                continue;
            }
            if found
                .as_ref()
                .map(|current| metadata.created_at > current.created_at)
                .unwrap_or(true)
            {
                found = Some(metadata);
            }
        }
        Ok(found)
    }

    fn complete_locked(&self, run_id: &str) -> Result<RunMetadata, String> {
        self.find_complete_locked(run_id)?.ok_or_else(|| {
            "No undo checkpoint is available for that run. It may have been removed by cleanup."
                .to_string()
        })
    }

    fn cleanup_locked(&self, keep: Option<&str>) -> Result<(), String> {
        let mut metadata = self.read_metadata_locked()?;
        let now = now_ms();
        let mut removed = false;

        metadata.retain(|entry| {
            let stale = entry.status == MetadataStatus::Failed
                || (entry.status == MetadataStatus::Running
                    && now.saturating_sub(entry.created_at) > STALE_RUNNING_MS);
            if stale {
                self.delete_entry_locked(entry);
                removed = true;
                false
            } else {
                true
            }
        });

        // Keep the newest records. The list is small (the cap is 20), so a
        // linear selection is simpler and safer than sorting by filename.
        while metadata.len() > self.limits.max_runs.max(1) {
            let Some(index) = oldest_removable(&metadata, keep) else {
                break;
            };
            let entry = metadata.remove(index);
            self.delete_entry_locked(&entry);
            removed = true;
        }

        // Git objects from evicted runs are only garbage after their refs are
        // gone. This is a best-effort disk bound; a single oversized run is
        // still kept so that it remains undoable.
        if self.repo_size_bytes() > self.limits.max_bytes {
            while self.repo_size_bytes() > self.limits.max_bytes {
                let Some(index) = oldest_removable(&metadata, keep) else {
                    break;
                };
                let entry = metadata.remove(index);
                self.delete_entry_locked(&entry);
                removed = true;
            }
        }

        if self.sweep_refs_locked(&metadata)? {
            removed = true;
        }
        if removed {
            let _ = self.git(&["gc", "--prune=now", "--quiet"], None, None);
        }
        Ok(())
    }

    fn sweep_refs_locked(&self, metadata: &[RunMetadata]) -> Result<bool, String> {
        let output = self.git(
            &["for-each-ref", "--format=%(refname)", "refs/janas/"],
            None,
            None,
        )?;
        let known: HashSet<&str> = metadata
            .iter()
            .map(|entry| entry.checkpoint_id.as_str())
            .collect();
        let mut removed = false;
        for reference in output.lines() {
            let Some(rest) = reference.strip_prefix("refs/janas/") else {
                continue;
            };
            let Some((checkpoint_id, _)) = rest.split_once('/') else {
                continue;
            };
            if known.contains(checkpoint_id) {
                continue;
            }
            let _ = self.git(&["update-ref", "-d", reference], None, None);
            removed = true;
        }
        Ok(removed)
    }

    fn repo_size_bytes(&self) -> u64 {
        let Ok(output) = self.git(&["count-objects", "-v"], None, None) else {
            return 0;
        };
        let mut kib = 0u64;
        for line in output.lines() {
            let value = line
                .strip_prefix("size: ")
                .or_else(|| line.strip_prefix("size-pack: "));
            if let Some(value) = value {
                kib = kib.saturating_add(value.trim().parse().unwrap_or(0));
            }
        }
        kib.saturating_mul(1024)
    }

    fn git(
        &self,
        args: &[&str],
        work_tree: Option<&Path>,
        index: Option<&Path>,
    ) -> Result<String, String> {
        let output = self.git_output(args, work_tree, index)?;
        Ok(String::from_utf8_lossy(&output).to_string())
    }

    fn git_bytes(
        &self,
        args: &[&str],
        work_tree: Option<&Path>,
        index: Option<&Path>,
    ) -> Result<Vec<u8>, String> {
        self.git_output(args, work_tree, index)
    }

    fn git_output(
        &self,
        args: &[&str],
        work_tree: Option<&Path>,
        index: Option<&Path>,
    ) -> Result<Vec<u8>, String> {
        let mut command = Command::new("git");
        command.arg("--no-optional-locks").arg("--git-dir").arg(&self.repo);
        if let Some(work_tree) = work_tree {
            command.arg("--work-tree").arg(work_tree);
        }
        command.args(args);
        command.env("GIT_TERMINAL_PROMPT", "0");
        if let Some(index) = index {
            command.env("GIT_INDEX_FILE", index);
        }
        let output = command.output().map_err(|error| {
            format!("Could not run Git. Install Git or make it available on PATH: {error}")
        })?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr);
            let detail = detail.trim();
            return Err(if detail.is_empty() {
                format!("Git {} failed.", args.join(" "))
            } else {
                format!("Git {} failed: {detail}", args.join(" "))
            });
        }
        Ok(output.stdout)
    }
}

pub struct RunCheckpoint {
    store: Arc<CheckpointStore>,
    workspace: PathBuf,
    checkpoint_id: String,
    pre: String,
    metadata_path: PathBuf,
    recorder: Arc<RunRecorder>,
}

impl RunCheckpoint {
    /// Attribution recorder to bracket each mutating tool call with.
    pub fn recorder(&self) -> Arc<RunRecorder> {
        self.recorder.clone()
    }

    /// Mark the run complete and return the number of paths changed during it.
    /// The pre snapshot remains available if this fails, but it is not offered
    /// as an undo target because the workspace could have changed afterwards.
    pub fn finish(self) -> Result<u32, String> {
        let store = self.store;
        let _guard = store
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        store.ensure_repo()?;

        let attributed = self.recorder.snapshot();
        let result = (|| {
            let (_, post) = store.snapshot_locked(&self.workspace, "post", Some(&self.pre))?;
            let entries = store.diff_locked(&self.pre, &post)?;
            // Keep only attributed paths that still differ from the pre-run
            // state. External edits that happened during the run stay in the
            // snapshot but are never offered to undo.
            let tracked: HashMap<&Path, &AttributedPath> = attributed
                .iter()
                .map(|entry| (entry.path.as_path(), entry))
                .collect();
            let kept: Vec<AttributedPath> = entries
                .iter()
                .filter_map(|entry| tracked.get(entry.path.as_path()).map(|path| (*path).clone()))
                .collect();
            let changed_files = kept.len() as u32;
            store.set_ref(&self.checkpoint_id, "post", &post)?;

            let mut metadata = store.read_metadata_path(&self.metadata_path)?;
            metadata.post = Some(post);
            metadata.changed_files = changed_files;
            let kinds: HashMap<PathBuf, String> = entries
                .iter()
                .map(|entry| (entry.path.clone(), change_kind(entry).to_string()))
                .collect();
            metadata.undo_files = kept
                .iter()
                .map(|path| {
                    FileUndoEntry::pending(
                        path_label(&path.path),
                        kinds
                            .get(&path.path)
                            .cloned()
                            .unwrap_or_else(default_change_kind),
                    )
                })
                .collect();
            metadata.attribution = Some(kept);
            metadata.status = MetadataStatus::Complete;
            store.write_metadata_at(&self.metadata_path, &metadata)?;
            Ok(changed_files)
        })();

        match result {
            Ok(changed_files) => {
                let _ = store.cleanup_locked(Some(&self.checkpoint_id));
                Ok(changed_files)
            }
            Err(error) => {
                if let Ok(mut metadata) = store.read_metadata_path(&self.metadata_path) {
                    metadata.status = MetadataStatus::Failed;
                    let _ = store.write_metadata_at(&self.metadata_path, &metadata);
                }
                Err(error)
            }
        }
    }
}

enum RestoreKind {
    Restored,
    Deleted,
    Skipped(String),
}

#[derive(Debug, Default)]
struct RecorderState {
    last_tree: Option<String>,
    last_commit: Option<String>,
    attribution: BTreeMap<PathBuf, AttributedPath>,
}

/// Records which paths the agent's own tools changed while a run is active.
///
/// Undo must not revert files the user (or another process) edited during the
/// run. A single pre/post diff cannot tell those apart, so each mutating tool
/// call is bracketed: a fresh baseline is taken just before the tool runs (so
/// edits made while the model was thinking are absorbed, not attributed), and
/// the diff afterwards is recorded along with the content the tool left behind.
///
/// Trees recorded for each step are kept reachable from a per-run ref so their
/// blobs survive cleanup and garbage collection until the run is undone.
pub struct RunRecorder {
    store: Arc<CheckpointStore>,
    workspace: PathBuf,
    checkpoint_id: String,
    state: Mutex<RecorderState>,
}

impl RunRecorder {
    /// Absorb everything that happened since the previous step, so only the
    /// tool call that follows is attributed.
    pub fn baseline(&self) -> Result<(), String> {
        let tree = match self.store.tree_for(&self.workspace) {
            Ok(tree) => tree,
            Err(error) => {
                // Without a baseline the next diff would include unrelated
                // edits. Drop back to recording nothing rather than guess.
                if let Ok(mut state) = self.state.lock() {
                    state.last_tree = None;
                }
                return Err(error);
            }
        };
        if let Ok(mut state) = self.state.lock() {
            state.last_tree = Some(tree);
        }
        Ok(())
    }

    /// Record the trusted paths changed by the tool call since the last
    /// baseline. A path that the tool did not name is left out even when it
    /// changed in the same window, so a concurrent user edit can never become
    /// undoable as an agent change.
    pub fn commit_step(&self, trusted_paths: &[PathBuf]) -> Result<(), String> {
        let tree = self.store.tree_for(&self.workspace)?;
        let (previous, parent, tracked) = {
            let state = self
                .state
                .lock()
                .map_err(|_| "The checkpoint recorder is locked.".to_string())?;
            (
                state.last_tree.clone(),
                state.last_commit.clone(),
                state.attribution.clone(),
            )
        };
        let mut attribution = tracked;
        let mut last_commit = parent;
        if let Some(previous) = previous.filter(|previous| previous != &tree) {
            let trusted: HashSet<&Path> = trusted_paths.iter().map(PathBuf::as_path).collect();
            for entry in self.store.diff_trees(&previous, &tree)? {
                if !trusted.contains(entry.path.as_path()) {
                    continue;
                }
                attribution.insert(
                    entry.path.clone(),
                    AttributedPath {
                        path: entry.path.clone(),
                        mode: entry.new_mode,
                        oid: if entry.new_mode == 0 {
                            String::new()
                        } else {
                            entry.new_oid.clone()
                        },
                    },
                );
            }
            last_commit = Some(
                self.store
                    .protect_step(&self.checkpoint_id, &tree, last_commit.as_deref())?,
            );
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| "The checkpoint recorder is locked.".to_string())?;
        state.attribution = attribution;
        state.last_commit = last_commit;
        state.last_tree = Some(tree);
        Ok(())
    }

    fn snapshot(&self) -> Vec<AttributedPath> {
        self.state
            .lock()
            .map(|state| state.attribution.values().cloned().collect())
            .unwrap_or_default()
    }
}

fn oldest_removable(metadata: &[RunMetadata], keep: Option<&str>) -> Option<usize> {
    metadata
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.status != MetadataStatus::Running)
        .filter(|(_, entry)| Some(entry.checkpoint_id.as_str()) != keep)
        .min_by_key(|(_, entry)| entry.created_at)
        .map(|(index, _)| index)
}

/// Stable label for a checkpoint path. Git reports `/` separators on every
/// platform, so normalising them would corrupt a legal `\` in a Unix filename
/// and could make two distinct paths share a label.
fn path_label(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Older checkpoints have no per-file records. Seed them as pending so the
/// first undo (or keep) can still be targeted by path.
fn ensure_undo_files(metadata: &mut RunMetadata, entries: &[(DiffEntry, Option<AttributedPath>)]) {
    for (entry, _) in entries {
        let path = path_label(&entry.path);
        let kind = change_kind(entry);
        if let Some(file) = metadata.undo_files.iter_mut().find(|file| file.path == path) {
            // The diff is authoritative; this also backfills checkpoints saved
            // before change kinds were recorded.
            file.set_kind(kind);
        } else {
            let mut file = FileUndoEntry::pending(path, kind.to_string());
            if metadata.restored_paths.contains(&entry.path) {
                file.state = FileUndoState::Restored;
                file.attempts = 1;
                file.history.push(UndoAttempt {
                    action: UndoAction::Restore,
                    state: FileUndoState::Restored,
                    reason: Some("restored by an earlier selective undo".into()),
                    at: now_ms(),
                });
            }
            metadata.undo_files.push(file);
        }
    }
    metadata
        .undo_files
        .sort_by(|left, right| left.path.cmp(&right.path));
}

fn default_change_kind() -> String {
    "modified".to_string()
}

fn change_kind(entry: &DiffEntry) -> &'static str {
    if entry.old_mode == 0 {
        "created"
    } else if entry.new_mode == 0 {
        "deleted"
    } else {
        "modified"
    }
}

/// A checkpoint is offered for undo while at least one attributed file still
/// needs a decision. Checkpoints written before per-file state existed fall
/// back to their change count.
fn metadata_has_pending(metadata: &RunMetadata) -> bool {
    if metadata.changed_files == 0 {
        return false;
    }
    if metadata.undo_files.is_empty() {
        return metadata.changed_files > metadata.restored_paths.len() as u32;
    }
    metadata
        .undo_files
        .iter()
        .any(|file| file.state.unresolved())
}

fn outcome_from_files(files: Vec<FileUndoEntry>) -> UndoOutcome {
    let pending = files.iter().filter(|file| file.state.unresolved()).count() as u32;
    UndoOutcome {
        files,
        pending,
        complete: pending == 0,
        ..UndoOutcome::default()
    }
}

fn workspace_usage(root: &Path, exclude: Option<&Path>) -> Result<(u64, usize), String> {
    fn heavy(name: &str) -> bool {
        matches!(
            name,
            ".git"
                | ".tools"
                | "node_modules"
                | "target"
                | "dist"
                | "build"
                | ".next"
                | ".cache"
                | ".venv"
                | "venv"
                | "__pycache__"
        )
    }

    fn visit(
        directory: &Path,
        exclude: Option<&Path>,
        bytes: &mut u64,
        files: &mut usize,
    ) -> Result<(), String> {
        let reader = match fs::read_dir(directory) {
            Ok(reader) => reader,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "Could not inspect {} for a checkpoint: {error}",
                    directory.display()
                ))
            }
        };
        for entry in reader.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                if !heavy(name.as_ref()) && exclude != Some(path.as_path()) {
                    visit(&path, exclude, bytes, files)?;
                }
                continue;
            }
            *files += 1;
            if file_type.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    *bytes = bytes.saturating_add(metadata.len());
                }
            }
        }
        Ok(())
    }

    let mut bytes = 0u64;
    let mut files = 0usize;
    visit(root, exclude, &mut bytes, &mut files)?;
    Ok((bytes, files))
}

fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// True when the nearest existing ancestor of `path` is inside `workspace`.
/// Missing directories are allowed: restoring a file the run deleted may need
/// to recreate them, and the nearest existing ancestor still bounds the write.
fn parent_inside_workspace(workspace: &Path, path: &Path) -> bool {
    let mut ancestor = path.parent();
    while let Some(directory) = ancestor {
        match fs::symlink_metadata(directory) {
            // Follow a symlink only when it resolves inside the workspace. A
            // dangling link has no safe target, so it is refused as well.
            Ok(metadata) if metadata.is_dir() || metadata.file_type().is_symlink() => {
                return fs::canonicalize(directory)
                    .map(|canonical| canonical.starts_with(workspace))
                    .unwrap_or(false);
            }
            Ok(_) => return false,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = directory.parent();
            }
            Err(_) => return false,
        }
    }
    false
}

fn parse_mode(value: &str) -> Result<u32, String> {
    u32::from_str_radix(value, 8)
        .map_err(|error| format!("Git returned an invalid file mode: {error}"))
}

fn remove_current(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            fs::remove_dir(path).map_err(|error| format!("could not remove the directory: {error}"))
        }
        Ok(_) => fs::remove_file(path).map_err(|error| format!("could not remove the file: {error}")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("could not inspect the path: {error}")),
    }
}

fn restore_regular(path: &Path, bytes: &[u8], mode: u32) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "could not resolve the parent directory".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create the parent directory: {error}"))?;
    let temp = parent.join(format!(".janas-undo-{}.tmp", Uuid::new_v4().simple()));
    fs::write(&temp, bytes).map_err(|error| format!("could not write the restored file: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&temp)
            .map_err(|error| format!("could not inspect the restored file: {error}"))?
            .permissions();
        permissions.set_mode(if mode == 0o100755 { 0o755 } else { 0o644 });
        fs::set_permissions(&temp, permissions)
            .map_err(|error| format!("could not set restored file permissions: {error}"))?;
    }
    // On Unix, rename atomically replaces the verified regular file. Avoid a
    // remove-then-rename window that could lose the only copy on failure.
    #[cfg(unix)]
    if path.exists() || fs::symlink_metadata(path).is_ok() {
        return fs::rename(&temp, path).map_err(|error| {
            let _ = fs::remove_file(&temp);
            format!("could not restore the file: {error}")
        });
    }
    if path.exists() || fs::symlink_metadata(path).is_ok() {
        remove_current(path)?;
    }
    fs::rename(&temp, path).map_err(|error| {
        let _ = fs::remove_file(&temp);
        format!("could not restore the file: {error}")
    })
}

#[cfg(unix)]
fn restore_symlink(path: &Path, target: &[u8]) -> Result<(), String> {
    use std::os::unix::ffi::OsStringExt;
    let parent = path
        .parent()
        .ok_or_else(|| "could not resolve the parent directory".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create the parent directory: {error}"))?;
    let temp = parent.join(format!(".janas-undo-{}.tmp", Uuid::new_v4().simple()));
    std::os::unix::fs::symlink(OsString::from_vec(target.to_vec()), &temp)
        .map_err(|error| format!("could not create the restored symlink: {error}"))?;
    fs::rename(&temp, path).map_err(|error| {
        let _ = fs::remove_file(&temp);
        format!("could not restore the symlink: {error}")
    })
}

#[cfg(not(unix))]
fn restore_symlink(_path: &Path, _target: &[u8]) -> Result<(), String> {
    Err("symlink restore is not supported on this platform".to_string())
}

#[cfg(unix)]
fn os_bytes(value: &std::ffi::OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(not(unix))]
fn os_bytes(value: &std::ffi::OsStr) -> Vec<u8> {
    value.to_string_lossy().as_bytes().to_vec()
}

#[cfg(unix)]
fn git_path(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;
    PathBuf::from(OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn git_path(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "janas-checkpoint-{label}-{}",
            Uuid::new_v4()
        ));
        fs::create_dir_all(root.join("ws")).expect("workspace");
        root
    }

    #[test]
    fn checkpoint_undo_survives_a_new_store_and_skips_later_edits() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the checkpoint integration test.");
            return;
        }
        let root = temp_root("restart");
        let workspace = root.join("ws");
        fs::write(workspace.join("keep.txt"), "before\n").expect("keep");
        fs::write(workspace.join("gone.txt"), "gone\n").expect("gone");

        let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let checkpoint = store.begin("run-1", &workspace).expect("begin");
        let recorder = checkpoint.recorder();
        recorder.baseline().expect("baseline");
        fs::write(workspace.join("keep.txt"), "after\n").expect("modify");
        fs::remove_file(workspace.join("gone.txt")).expect("delete");
        fs::write(workspace.join("new.txt"), "created\n").expect("create");
        #[cfg(unix)]
        {
            // This stands in for an agent run_command/MCP mutation: the
            // checkpoint sees the result even though no file tool wrote it.
            let status = Command::new("sh")
                .args(["-c", "printf 'shell\\n' > shell.txt"])
                .current_dir(&workspace)
                .status()
                .expect("shell");
            assert!(status.success());
        }
        recorder
            .commit_step(&[
                PathBuf::from("keep.txt"),
                PathBuf::from("gone.txt"),
                PathBuf::from("new.txt"),
            ])
            .expect("step");
        assert_eq!(
            checkpoint.finish().expect("finish"),
            3,
            "shell-created paths have no trustworthy provenance"
        );

        // A different store instance is what a restarted process sees.
        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        assert!(reopened.has_complete("run-1"));
        assert_eq!(
            reopened.available(&["run-1".to_string(), "missing".to_string()]),
            vec!["run-1".to_string()]
        );

        // A post-run user edit must win over the checkpoint.
        fs::write(workspace.join("keep.txt"), "user edit\n").expect("user edit");
        let outcome = reopened.undo("run-1").expect("undo");
        assert!(!outcome.incomplete);
        assert_eq!(outcome.restored, 1, "the deleted file is restored");
        assert_eq!(outcome.deleted, 1, "only the created file is removed");
        assert_eq!(
            fs::read_to_string(workspace.join("keep.txt")).expect("keep"),
            "user edit\n"
        );
        assert_eq!(
            fs::read_to_string(workspace.join("gone.txt")).expect("gone"),
            "gone\n"
        );
        assert!(!workspace.join("new.txt").exists());
        #[cfg(unix)]
        assert!(
            workspace.join("shell.txt").exists(),
            "an un-attributed shell edit must survive undo"
        );
        let kept = outcome
            .files
            .iter()
            .find(|file| file.path == "keep.txt")
            .expect("keep.txt entry");
        assert_eq!(kept.state, FileUndoState::Skipped);
        assert!(kept
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("changed since the run"));
        assert_eq!(outcome.pending, 1);
        assert!(!outcome.complete);
        // A skipped file is still pending, so its checkpoint must survive for
        // a retry or a keep decision.
        assert!(reopened.has_complete("run-1"));

        // Accepting the external edit resolves the last pending file without
        // changing it, and the checkpoint is cleared.
        let kept_outcome = reopened.keep_path("run-1", "keep.txt").expect("keep");
        assert_eq!(kept_outcome.pending, 0);
        assert!(kept_outcome.complete);
        assert_eq!(
            fs::read_to_string(workspace.join("keep.txt")).expect("keep"),
            "user edit\n"
        );
        assert!(!reopened.has_complete("run-1"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cleanup_keeps_only_the_newest_runs() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the checkpoint cleanup test.");
            return;
        }
        let root = temp_root("cleanup");
        let workspace = root.join("ws");
        let store = Arc::new(CheckpointStore::with_limits(
            root.join("checkpoints"),
            2,
            u64::MAX,
        ));
        for index in 0..3 {
            let run_id = format!("run-{index}");
            let checkpoint = store.begin(&run_id, &workspace).expect("begin");
            let recorder = checkpoint.recorder();
            recorder.baseline().expect("baseline");
            fs::write(
                workspace.join(format!("file-{index}.txt")),
                format!("run {index}\n"),
            )
            .expect("write");
            recorder
                .commit_step(&[PathBuf::from(format!("file-{index}.txt"))])
                .expect("step");
            checkpoint.finish().expect("finish");
            thread::sleep(Duration::from_millis(3));
        }
        let available = store.available(&[
            "run-0".to_string(),
            "run-1".to_string(),
            "run-2".to_string(),
        ]);
        assert_eq!(available.len(), 2);
        assert!(!available.contains(&"run-0".to_string()));
        assert!(available.contains(&"run-1".to_string()));
        assert!(available.contains(&"run-2".to_string()));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn undo_leaves_edits_made_outside_the_run_untouched() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the checkpoint attribution test.");
            return;
        }
        let root = temp_root("external");
        let workspace = root.join("ws");
        fs::write(workspace.join("shared.txt"), "before\n").expect("shared");
        fs::write(workspace.join("notes.txt"), "notes\n").expect("notes");

        let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let checkpoint = store.begin("run-external", &workspace).expect("begin");
        let recorder = checkpoint.recorder();

        // The agent changes notes.txt through a tool call.
        recorder.baseline().expect("baseline");
        fs::write(workspace.join("notes.txt"), "agent\n").expect("agent write");
        recorder
            .commit_step(&[PathBuf::from("notes.txt")])
            .expect("step");

        // Something else edits a different file and creates one while the run
        // is still active. Neither went through a tool, so undo must not touch
        // them even though they are part of the final snapshot.
        fs::write(workspace.join("shared.txt"), "user\n").expect("user edit");
        fs::write(workspace.join("scratch.txt"), "scratch\n").expect("user create");

        assert_eq!(
            checkpoint.finish().expect("finish"),
            1,
            "only the agent-touched file counts as a run change"
        );

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let outcome = reopened.undo("run-external").expect("undo");
        assert_eq!(outcome.restored, 1);
        assert!(!outcome.incomplete);
        assert!(outcome.complete);
        assert_eq!(
            fs::read_to_string(workspace.join("notes.txt")).expect("notes"),
            "notes\n"
        );
        assert_eq!(
            fs::read_to_string(workspace.join("shared.txt")).expect("shared"),
            "user\n"
        );
        assert_eq!(
            fs::read_to_string(workspace.join("scratch.txt")).expect("scratch"),
            "scratch\n"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn undo_skips_a_file_edited_after_the_agent_touched_it() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the checkpoint attribution test.");
            return;
        }
        let root = temp_root("after-touch");
        let workspace = root.join("ws");
        fs::write(workspace.join("shared.txt"), "before\n").expect("shared");

        let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let checkpoint = store.begin("run-after-touch", &workspace).expect("begin");
        let recorder = checkpoint.recorder();
        recorder.baseline().expect("baseline");
        fs::write(workspace.join("shared.txt"), "agent\n").expect("agent write");
        recorder
            .commit_step(&[PathBuf::from("shared.txt")])
            .expect("step");
        // The user edits the same file after the tool returned.
        fs::write(workspace.join("shared.txt"), "user\n").expect("user edit");

        assert_eq!(checkpoint.finish().expect("finish"), 1);

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let outcome = reopened.undo("run-after-touch").expect("undo");
        assert_eq!(outcome.restored, 0);
        let skipped = outcome
            .files
            .iter()
            .find(|file| file.path == "shared.txt")
            .expect("shared entry");
        assert_eq!(skipped.state, FileUndoState::Skipped);
        assert!(skipped
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("changed since the run"));
        assert_eq!(
            fs::read_to_string(workspace.join("shared.txt")).expect("shared"),
            "user\n"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn undo_refuses_a_directory_replaced_by_an_outside_symlink() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the parent symlink test.");
            return;
        }
        #[cfg(not(unix))]
        return;

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let root = temp_root("parent-symlink");
            let workspace = root.join("ws");
            let outside = root.join("outside");
            fs::create_dir_all(workspace.join("sub")).expect("sub");
            fs::create_dir_all(&outside).expect("outside");
            fs::write(workspace.join("sub/note.txt"), "before\n").expect("note");

            let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
            let checkpoint = store.begin("run-parent-symlink", &workspace).expect("begin");
            let recorder = checkpoint.recorder();
            recorder.baseline().expect("baseline");
            fs::write(workspace.join("sub/note.txt"), "after\n").expect("agent");
            recorder
                .commit_step(&[PathBuf::from("sub/note.txt")])
                .expect("step");
            assert_eq!(checkpoint.finish().expect("finish"), 1);

            // Replace the directory with a link to a copy outside the workspace.
            // The copy has the same content the run left behind, so only the
            // parent check can prevent an outside write.
            fs::rename(workspace.join("sub"), outside.join("sub")).expect("move");
            symlink(outside.join("sub"), workspace.join("sub")).expect("link");

            let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
            let outcome = reopened.undo("run-parent-symlink").expect("undo");
            assert_eq!(outcome.restored, 0);
            assert_eq!(
                fs::read_to_string(outside.join("sub/note.txt")).expect("outside"),
                "after\n",
                "undo must never write through an outside symlink"
            );
            let skipped = outcome
                .files
                .iter()
                .find(|file| file.path == "sub/note.txt")
                .expect("note entry");
            assert_eq!(skipped.state, FileUndoState::Skipped);
            assert!(skipped
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("outside the workspace"));

            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn per_file_undo_tracks_attempts_and_keeps_external_edits() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the per-file undo test.");
            return;
        }
        let root = temp_root("per-file");
        let workspace = root.join("ws");
        fs::write(workspace.join("alpha.txt"), "before alpha\n").expect("alpha");
        fs::write(workspace.join("beta.txt"), "before beta\n").expect("beta");

        let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let checkpoint = store.begin("run-per-file", &workspace).expect("begin");
        let recorder = checkpoint.recorder();
        recorder.baseline().expect("baseline");
        fs::write(workspace.join("alpha.txt"), "after alpha\n").expect("alpha");
        fs::write(workspace.join("beta.txt"), "after beta\n").expect("beta");
        recorder
            .commit_step(&[PathBuf::from("alpha.txt"), PathBuf::from("beta.txt")])
            .expect("step");
        assert_eq!(checkpoint.finish().expect("finish"), 2);

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));

        let initial = reopened.state("run-per-file").expect("state");
        assert_eq!(initial.pending, 2);
        assert!(initial
            .files
            .iter()
            .all(|file| file.state == FileUndoState::Pending));
        assert!(initial
            .files
            .iter()
            .all(|file| file.kind == "modified"));

        // Restoring one file leaves the other pending and keeps the checkpoint.
        let alpha = reopened.undo_path("run-per-file", "alpha.txt").expect("alpha");
        assert_eq!(alpha.restored, 1);
        assert_eq!(alpha.pending, 1);
        assert!(!alpha.complete);
        assert_eq!(
            fs::read_to_string(workspace.join("alpha.txt")).expect("alpha"),
            "before alpha\n"
        );
        let beta = alpha
            .files
            .iter()
            .find(|file| file.path == "beta.txt")
            .expect("beta entry");
        assert_eq!(beta.state, FileUndoState::Pending);
        assert!(reopened.has_complete("run-per-file"));

        // The partial state is on disk, so a restarted app still knows which
        // file was restored and which one is pending.
        let restarted = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let persisted = restarted.state("run-per-file").expect("state");
        assert_eq!(persisted.pending, 1);
        assert_eq!(
            persisted
                .files
                .iter()
                .find(|file| file.path == "alpha.txt")
                .expect("alpha entry")
                .state,
            FileUndoState::Restored
        );

        // A user edit after the run is skipped, and the retry history is kept.
        fs::write(workspace.join("beta.txt"), "my beta\n").expect("user edit");
        let skipped = reopened
            .undo_path("run-per-file", "beta.txt")
            .expect("beta undo");
        assert_eq!(skipped.restored, 0);
        assert_eq!(skipped.pending, 1);
        assert!(!skipped.complete);
        let beta = skipped
            .files
            .iter()
            .find(|file| file.path == "beta.txt")
            .expect("beta entry");
        assert_eq!(beta.state, FileUndoState::Skipped);
        assert_eq!(beta.attempts, 1);
        assert!(beta
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("changed since the run"));
        assert_eq!(
            fs::read_to_string(workspace.join("beta.txt")).expect("beta"),
            "my beta\n"
        );

        // Keeping the user's version resolves the last file without writing it.
        let kept = reopened.keep_path("run-per-file", "beta.txt").expect("keep");
        assert_eq!(kept.pending, 0);
        assert!(kept.complete);
        let beta = kept
            .files
            .iter()
            .find(|file| file.path == "beta.txt")
            .expect("beta entry");
        assert_eq!(beta.state, FileUndoState::Kept);
        assert_eq!(beta.attempts, 2);
        assert_eq!(beta.history.len(), 2, "the skip stays visible after a retry");
        assert_eq!(beta.history[0].state, FileUndoState::Skipped);
        assert_eq!(beta.history[1].action, UndoAction::Keep);
        assert_eq!(
            fs::read_to_string(workspace.join("beta.txt")).expect("beta"),
            "my beta\n"
        );
        assert!(!reopened.has_complete("run-per-file"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn last_per_file_restore_completes_the_run() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the per-file completion test.");
            return;
        }
        let root = temp_root("per-file-complete");
        let workspace = root.join("ws");
        fs::write(workspace.join("one.txt"), "before one\n").expect("one");
        fs::write(workspace.join("two.txt"), "before two\n").expect("two");

        let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let checkpoint = store.begin("run-complete", &workspace).expect("begin");
        let recorder = checkpoint.recorder();
        recorder.baseline().expect("baseline");
        fs::write(workspace.join("one.txt"), "after one\n").expect("one");
        fs::write(workspace.join("two.txt"), "after two\n").expect("two");
        recorder
            .commit_step(&[PathBuf::from("one.txt"), PathBuf::from("two.txt")])
            .expect("step");
        assert_eq!(checkpoint.finish().expect("finish"), 2);

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let first = reopened.undo_path("run-complete", "one.txt").expect("one");
        assert!(!first.complete);
        assert_eq!(first.pending, 1);
        let last = reopened.undo_path("run-complete", "two.txt").expect("two");
        assert!(last.complete, "the last restore finishes the run");
        assert_eq!(last.pending, 0);
        assert!(last
            .files
            .iter()
            .all(|file| file.state == FileUndoState::Restored));
        assert!(!reopened.has_complete("run-complete"));

        let _ = fs::remove_dir_all(root);
    }
}
