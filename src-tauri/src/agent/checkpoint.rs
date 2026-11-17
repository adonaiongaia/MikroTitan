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
    /// Paths that have already been restored by a selective undo. The
    /// attribution list stays intact so the run's change summary remains
    /// available after a partial restore.
    #[serde(default)]
    restored_paths: Vec<PathBuf>,
    /// Filled when a metadata file is read; never serialized.
    #[serde(skip, default)]
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunFileChange {
    /// Workspace-relative path, using forward slashes for display.
    pub path: String,
    /// `created`, `modified` or `deleted`.
    pub kind: String,
    pub restored: bool,
}

#[derive(Debug, Clone, Default)]
pub struct UndoOutcome {
    pub restored: u32,
    pub deleted: u32,
    pub skipped: Vec<String>,
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
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let mut metadata = self
            .find_complete_locked(run_id)?
            .ok_or_else(|| {
                "No undo checkpoint is available for that run. It may have been removed by cleanup."
                    .to_string()
            })?;
        let outcome = self.undo_metadata_locked(&mut metadata)?;
        if outcome.skipped.is_empty() && !outcome.incomplete {
            self.delete_entry_locked(&metadata);
            let _ = self.cleanup_locked(None);
        } else {
            // Keep the un-restored files available for a retry after the user
            // resolves the external edit or permission problem.
            self.write_metadata_at(&metadata.path, &metadata)?;
        }
        Ok(outcome)
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
                    && metadata.changed_files > metadata.restored_paths.len() as u32
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
            .map(|metadata| {
                metadata.changed_files > metadata.restored_paths.len() as u32
            })
            .unwrap_or(false)
    }

    /// True when a completed checkpoint exists with attributed changes, even
    /// if every file has already been restored. This keeps the in-memory
    /// journal from making a fully restored run look undoable again, while a
    /// checkpoint with no attributed changes still lets the journal fallback
    /// cover built-in file edits.
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

    /// Every path the run changed, including paths already restored by a
    /// selective undo. Restore protection lives in `undo_paths`.
    pub fn changes(&self, run_id: &str) -> Result<Vec<RunFileChange>, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let metadata = self
            .find_complete_locked(run_id)?
            .ok_or_else(|| "No change summary is available for that run.".to_string())?;
        self.changes_locked(&metadata)
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

    fn undo_metadata_locked(&self, metadata: &mut RunMetadata) -> Result<UndoOutcome, String> {
        let post = metadata
            .post
            .as_deref()
            .ok_or_else(|| "This checkpoint has no post-run snapshot.".to_string())?;
        let workspace = fs::canonicalize(&metadata.workspace)
            .map_err(|error| format!("The checkpoint workspace is unavailable: {error}"))?;
        if !workspace.is_dir() {
            return Err(format!(
                "The checkpoint workspace is not a directory: {}",
                workspace.display()
            ));
        }
        let entries = self.diff_locked(&metadata.pre, post)?;
        let attributed: Option<HashMap<PathBuf, AttributedPath>> = metadata.attribution.as_ref().map(|paths| {
            paths
                .iter()
                .map(|path| (path.path.clone(), path.clone()))
                .collect()
        });
        let completed: HashSet<PathBuf> = metadata.restored_paths.iter().cloned().collect();
        let mut outcome = UndoOutcome::default();
        for entry in entries {
            if completed.contains(&entry.path) {
                continue;
            }
            // With attribution, anything the run did not touch is left alone: it
            // was changed by something else while the run was active.
            let expected = match &attributed {
                Some(paths) => match paths.get(&entry.path) {
                    Some(path) => Some(path),
                    None => continue,
                },
                None => None,
            };
            match self.restore_entry(&workspace, &entry, expected) {
                Ok(RestoreKind::Restored) => {
                    outcome.restored += 1;
                    metadata.restored_paths.push(entry.path.clone());
                }
                Ok(RestoreKind::Deleted) => {
                    outcome.deleted += 1;
                    metadata.restored_paths.push(entry.path.clone());
                }
                Ok(RestoreKind::Skipped(reason)) => {
                    outcome
                        .skipped
                        .push(format!("{}: {reason}", relative_label(&entry.path)));
                }
                Err(error) => {
                    outcome.incomplete = true;
                    outcome
                        .skipped
                        .push(format!("{}: {error}", relative_label(&entry.path)));
                }
            }
        }
        Ok(outcome)
    }

    /// Restore only the requested paths, leaving the checkpoint available for
    /// the remaining files. Paths already restored are ignored.
    pub fn undo_paths(
        &self,
        run_id: &str,
        paths: &[PathBuf],
    ) -> Result<UndoOutcome, String> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "The checkpoint store is locked.".to_string())?;
        let mut metadata = self
            .find_complete_locked(run_id)?
            .ok_or_else(|| {
                "No undo checkpoint is available for that run. It may have been removed by cleanup."
                    .to_string()
            })?;
        let requested: HashSet<PathBuf> = paths.iter().cloned().collect();
        let outcome = self.undo_selected_locked(&mut metadata, &requested)?;
        if outcome.restored > 0 || outcome.deleted > 0 {
            self.write_metadata_at(&metadata.path, &metadata)?;
        }
        Ok(outcome)
    }

    fn undo_selected_locked(
        &self,
        metadata: &mut RunMetadata,
        requested: &HashSet<PathBuf>,
    ) -> Result<UndoOutcome, String> {
        let Some(post) = metadata.post.as_deref() else {
            return Err("This checkpoint has no post-run snapshot.".into());
        };
        let workspace = fs::canonicalize(&metadata.workspace)
            .map_err(|error| format!("The checkpoint workspace is unavailable: {error}"))?;
        if !workspace.is_dir() {
            return Err(format!(
                "The checkpoint workspace is not a directory: {}",
                workspace.display()
            ));
        }
        let entries = self.diff_locked(&metadata.pre, post)?;
        let attributed: Option<HashMap<PathBuf, AttributedPath>> = metadata.attribution.as_ref().map(|paths| {
            paths
                .iter()
                .map(|path| (path.path.clone(), path.clone()))
                .collect()
        });
        let restored: HashSet<PathBuf> = metadata.restored_paths.iter().cloned().collect();
        let mut outcome = UndoOutcome::default();
        for entry in entries {
            if !requested.contains(&entry.path) {
                continue;
            }
            let expected = match &attributed {
                Some(paths) => match paths.get(&entry.path) {
                    Some(path) => Some(path),
                    None => continue,
                },
                None => None,
            };
            if restored.contains(&entry.path) {
                continue;
            }
            match self.restore_entry(&workspace, &entry, expected) {
                Ok(RestoreKind::Restored) => {
                    outcome.restored += 1;
                    metadata.restored_paths.push(entry.path.clone());
                }
                Ok(RestoreKind::Deleted) => {
                    outcome.deleted += 1;
                    metadata.restored_paths.push(entry.path.clone());
                }
                Ok(RestoreKind::Skipped(reason)) => {
                    outcome
                        .skipped
                        .push(format!("{}: {reason}", relative_label(&entry.path)));
                }
                Err(error) => {
                    outcome.incomplete = true;
                    outcome
                        .skipped
                        .push(format!("{}: {error}", relative_label(&entry.path)));
                }
            }
        }
        Ok(outcome)
    }

    fn changes_locked(&self, metadata: &RunMetadata) -> Result<Vec<RunFileChange>, String> {
        let Some(post) = metadata.post.as_deref() else {
            return Ok(Vec::new());
        };
        let entries = self.diff_locked(&metadata.pre, post)?;
        let attributed: Option<HashSet<PathBuf>> = metadata.attribution.as_ref().map(|paths| {
            paths.iter().map(|path| path.path.clone()).collect()
        });
        let restored: HashSet<PathBuf> = metadata.restored_paths.iter().cloned().collect();
        let mut changes: Vec<RunFileChange> = entries
            .into_iter()
            .filter(|entry| {
                attributed
                    .as_ref()
                    .map(|paths| paths.contains(&entry.path))
                    .unwrap_or(true)
            })
            .map(|entry| RunFileChange {
                path: relative_label(&entry.path),
                kind: change_kind(&entry).to_string(),
                restored: restored.contains(&entry.path),
            })
            .collect();
        changes.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(changes)
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
        if !parent_inside_workspace(workspace, &path) {
            return Err("The path's parent directory escaped the workspace.".into());
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
    pub fn finish(self) -> Result<Vec<RunFileChange>, String> {
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
            let kept_paths: HashSet<PathBuf> =
                kept.iter().map(|path| path.path.clone()).collect();
            let mut changes: Vec<RunFileChange> = entries
                .iter()
                .filter(|entry| kept_paths.contains(&entry.path))
                .map(|entry| RunFileChange {
                    path: relative_label(&entry.path),
                    kind: change_kind(entry).to_string(),
                    restored: false,
                })
                .collect();
            changes.sort_by(|left, right| left.path.cmp(&right.path));
            store.set_ref(&self.checkpoint_id, "post", &post)?;

            let mut metadata = store.read_metadata_path(&self.metadata_path)?;
            metadata.post = Some(post);
            metadata.changed_files = changed_files;
            metadata.attribution = Some(kept);
            metadata.status = MetadataStatus::Complete;
            store.write_metadata_at(&self.metadata_path, &metadata)?;
            Ok(changes)
        })();

        match result {
            Ok(changes) => {
                let _ = store.cleanup_locked(Some(&self.checkpoint_id));
                Ok(changes)
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

    /// Record the paths changed by the tool call since the last baseline.
    pub fn commit_step(&self) -> Result<(), String> {
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
            for entry in self.store.diff_trees(&previous, &tree)? {
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

fn relative_label(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
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

/// Resolve the nearest existing parent. A symlinked directory must not turn a
/// restore into a write outside the workspace, while a genuinely missing
/// parent is allowed (restore_regular creates it).
fn parent_inside_workspace(workspace: &Path, path: &Path) -> bool {
    let Some(mut ancestor) = path.parent() else {
        return false;
    };
    loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => {
                return fs::canonicalize(ancestor)
                    .map(|resolved| resolved.starts_with(workspace))
                    .unwrap_or(false);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = match ancestor.parent() {
                    Some(parent) => parent,
                    None => return false,
                };
            }
            Err(_) => return false,
        }
    }
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
    #[cfg(unix)]
    fn restore_parent_must_stay_inside_the_workspace() {
        let root = temp_root("parent-guard");
        let workspace = fs::canonicalize(root.join("ws")).expect("workspace");
        let outside = root.join("outside");
        fs::create_dir_all(&outside).expect("outside");
        std::os::unix::fs::symlink(&outside, workspace.join("link")).expect("symlink");

        assert!(!parent_inside_workspace(
            &workspace,
            &workspace.join("link/file.txt")
        ));
        assert!(parent_inside_workspace(
            &workspace,
            &workspace.join("new/file.txt")
        ));
        let _ = fs::remove_dir_all(root);
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
        recorder.commit_step().expect("step");
        assert_eq!(
            checkpoint.finish().expect("finish").len(),
            if cfg!(unix) { 4 } else { 3 }
        );

        // A different store instance is what a restarted process sees.
        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        assert!(reopened.has_record("run-1"));
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
        assert_eq!(
            outcome.deleted,
            if cfg!(unix) { 2 } else { 1 },
            "the created files are removed"
        );
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
        assert!(!workspace.join("shell.txt").exists());
        assert!(outcome.skipped.iter().any(|line| line.contains("keep.txt")));
        assert!(
            reopened.has_complete("run-1"),
            "the externally edited file must remain undoable"
        );
        fs::write(workspace.join("keep.txt"), "after\n").expect("resolve edit");
        let retry = reopened.undo("run-1").expect("retry undo");
        assert_eq!(retry.restored, 1);
        assert_eq!(
            fs::read_to_string(workspace.join("keep.txt")).expect("keep"),
            "before\n"
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
            recorder.commit_step().expect("step");
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
        recorder.commit_step().expect("step");

        // Something else edits a different file and creates one while the run
        // is still active. Neither went through a tool, so undo must not touch
        // them even though they are part of the final snapshot.
        fs::write(workspace.join("shared.txt"), "user\n").expect("user edit");
        fs::write(workspace.join("scratch.txt"), "scratch\n").expect("user create");

        assert_eq!(
            checkpoint.finish().expect("finish").len(),
            1,
            "only the agent-touched file counts as a run change"
        );

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let outcome = reopened.undo("run-external").expect("undo");
        assert_eq!(outcome.restored, 1);
        assert!(!outcome.incomplete);
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
        recorder.commit_step().expect("step");
        // The user edits the same file after the tool returned.
        fs::write(workspace.join("shared.txt"), "user\n").expect("user edit");

        assert_eq!(checkpoint.finish().expect("finish").len(), 1);

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let outcome = reopened.undo("run-after-touch").expect("undo");
        assert_eq!(outcome.restored, 0);
        assert!(outcome
            .skipped
            .iter()
            .any(|line| line.contains("shared.txt")));
        assert_eq!(
            fs::read_to_string(workspace.join("shared.txt")).expect("shared"),
            "user\n"
        );
        assert!(
            reopened.has_complete("run-after-touch"),
            "a skipped file must stay available for a retry"
        );
        fs::write(workspace.join("shared.txt"), "agent\n").expect("resolve edit");
        let retry = reopened.undo("run-after-touch").expect("retry undo");
        assert_eq!(retry.restored, 1);
        assert_eq!(
            fs::read_to_string(workspace.join("shared.txt")).expect("shared"),
            "before\n"
        );
        assert!(!reopened.has_complete("run-after-touch"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn selective_undo_restores_one_file_and_keeps_external_edits_safe() {
        if !git_available() {
            eprintln!("Git is unavailable; skipping the selective checkpoint test.");
            return;
        }
        let root = temp_root("selective");
        let workspace = root.join("ws");
        fs::write(workspace.join("one.txt"), "before one\n").expect("one");
        fs::write(workspace.join("two.txt"), "before two\n").expect("two");

        let store = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        let checkpoint = store.begin("run-selective", &workspace).expect("begin");
        let recorder = checkpoint.recorder();
        recorder.baseline().expect("baseline");
        fs::write(workspace.join("one.txt"), "agent one\n").expect("agent one");
        fs::write(workspace.join("two.txt"), "agent two\n").expect("agent two");
        recorder.commit_step().expect("step");
        let changes = checkpoint.finish().expect("finish");
        assert_eq!(changes.len(), 2);
        assert!(changes.iter().all(|change| change.kind == "modified"));

        let reopened = Arc::new(CheckpointStore::new(root.join("checkpoints")));
        // A post-run edit wins over a selective restore for that file.
        fs::write(workspace.join("one.txt"), "user one\n").expect("user one");
        let skipped = reopened
            .undo_paths("run-selective", &[PathBuf::from("one.txt")])
            .expect("selective skip");
        assert_eq!(skipped.restored, 0);
        assert!(skipped.skipped.iter().any(|line| line.contains("one.txt")));
        assert_eq!(
            fs::read_to_string(workspace.join("one.txt")).expect("one"),
            "user one\n"
        );

        // Restoring a different file works and leaves the run available.
        let restored = reopened
            .undo_paths("run-selective", &[PathBuf::from("two.txt")])
            .expect("selective restore");
        assert_eq!(restored.restored, 1);
        assert_eq!(
            fs::read_to_string(workspace.join("two.txt")).expect("two"),
            "before two\n"
        );
        let summary = reopened.changes("run-selective").expect("summary");
        assert!(summary
            .iter()
            .any(|change| change.path == "two.txt" && change.restored));
        assert!(summary
            .iter()
            .any(|change| change.path == "one.txt" && !change.restored));
        assert!(reopened.available(&["run-selective".to_string()]).len() == 1);

        let all = reopened.undo("run-selective").expect("undo remaining");
        assert_eq!(all.restored, 0);
        assert!(all.skipped.iter().any(|line| line.contains("one.txt")));
        assert_eq!(
            fs::read_to_string(workspace.join("one.txt")).expect("one"),
            "user one\n"
        );
        assert!(
            reopened.has_complete("run-selective"),
            "the skipped file must stay available for a retry"
        );
        fs::write(workspace.join("one.txt"), "agent one\n").expect("resolve one");
        let retry = reopened.undo("run-selective").expect("retry undo");
        assert_eq!(retry.restored, 1);
        assert_eq!(
            fs::read_to_string(workspace.join("one.txt")).expect("one"),
            "before one\n"
        );
        assert!(!reopened.has_complete("run-selective"));
        assert!(!reopened.has_record("run-selective"));

        let _ = fs::remove_dir_all(root);
    }
}
