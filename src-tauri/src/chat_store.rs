//! Disk-backed chat history.
//!
//! The webview used to keep every conversation in one `localStorage` blob.
//! That meant a synchronous rewrite of the whole history on every streamed
//! token, a 60-chat cap and a silent quota failure once a few images were
//! attached. Sessions now live as one JSON file each under the app data
//! directory, and inline `data:` images are extracted to
//! `assets/<session-id>/` so the session file only carries a small reference.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{AppError, AppResult};
use crate::util::now_ms;

/// A single session file should never come close to this; the cap keeps a
/// corrupt or hand-edited file from exhausting memory.
const MAX_SESSION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ASSET_BYTES: u64 = 32 * 1024 * 1024;
const CURRENT_SCHEMA_VERSION: u32 = 2;

fn default_schema_version() -> u32 {
    1
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoredSession {
    pub id: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub updated_at: u64,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub messages: Vec<Value>,
    /// Inactive branches kept alive by a message edit or regenerate. The
    /// active path stays in `messages`; this is the rest of the tree.
    #[serde(default)]
    pub archived_messages: Vec<Value>,
    /// Context items the user pinned, kept as the frontend's JSON objects so a
    /// new item kind needs no Rust change. Absent in schema-v2 files written
    /// before pins existed, which is the same as an empty list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pinned_context: Vec<Value>,
    /// Items dropped from the prompt but kept visible in the transcript.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evicted_context_ids: Vec<String>,
    /// UI-owned fields such as draft text, attached project and per-chat
    /// workspace/tool scope. Flattening them here means the backend preserves
    /// future session fields without requiring a Rust change for each one.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatBackupInfo {
    pub id: String,
    pub title: String,
    pub updated_at: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatStoreStats {
    pub sessions: usize,
    pub session_bytes: u64,
    pub asset_bytes: u64,
    pub backups: usize,
    pub backup_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatImportSummary {
    pub imported: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

pub struct ChatStore {
    root: PathBuf,
}

impl ChatStore {
    pub fn load(root: PathBuf) -> Self {
        let _ = fs::create_dir_all(&root);
        let _ = fs::create_dir_all(root.join("assets"));
        let _ = fs::create_dir_all(root.join("backups"));
        Self { root }
    }

    pub fn list(&self) -> Vec<StoredSession> {
        let Ok(entries) = fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut sessions = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_string();
            match self.read_session(&path) {
                // Session files are always `<id>.json`. This also keeps a
                // user-exported copy dropped into the store folder from
                // appearing as a second, read-only chat.
                Ok(session) if session.id == stem => sessions.push(session),
                Ok(_) | Err(_) => {
                    // One unreadable chat must not hide the rest of the shelf.
                }
            }
        }
        sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        sessions
    }

    pub fn save(&self, mut session: StoredSession) -> AppResult<StoredSession> {
        validate_id(&session.id)?;
        if session.schema_version > CURRENT_SCHEMA_VERSION {
            return Err(AppError::message(format!(
                "That chat was written by a newer version of Studio for Janas (schema {}).",
                session.schema_version
            )));
        }
        session.schema_version = CURRENT_SCHEMA_VERSION;
        if session.title.trim().is_empty() {
            session.title = "New chat".into();
        }
        if session.created_at == 0 {
            session.created_at = now_ms();
        }
        if session.updated_at == 0 {
            session.updated_at = now_ms();
        }
        let path = self.session_path(&session.id)?;
        // A debounced save and a stream-flush save can overlap. Never let an
        // older snapshot overwrite a newer one when they finish out of order.
        if let Ok(existing) = self.read_session(&path) {
            if existing.updated_at > session.updated_at {
                return Ok(session);
            }
        }
        // Keep the previous on-disk version before replacing it. A failed
        // copy is best-effort: it must not block the actual save.
        if path.exists() {
            let _ = self.backup_existing(&path, &session.id);
        }
        self.extract_images(&mut session)?;

        let bytes = serde_json::to_vec_pretty(&session)
            .map_err(|err| AppError::message(format!("Could not encode that chat: {err}")))?;
        if bytes.len() as u64 > MAX_SESSION_BYTES {
            return Err(AppError::message(
                "That chat is too large to save. Remove some image attachments and try again.",
            ));
        }
        write_atomic(&path, &bytes)?;
        // Prune only after the new session file is on disk; a failed write
        // must never remove images the old file still references. Cleanup is
        // best-effort: a permission problem must not report the save failed.
        let _ = self.prune_assets(&session);
        Ok(session)
    }

    pub fn delete(&self, id: &str) -> AppResult<bool> {
        validate_id(id)?;
        let path = self.session_path(id)?;
        let removed = match fs::remove_file(&path) {
            Ok(()) => true,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
            Err(err) => return Err(err.into()),
        };
        let assets = self.root.join("assets").join(id);
        if assets.exists() {
            let _ = fs::remove_dir_all(&assets);
        }
        let _ = fs::remove_file(self.backup_path(id)?);
        Ok(removed)
    }

    /// Resolve an `assets/<session>/<file>` reference to a data URL. Only the
    /// exact two-component shape is accepted, so no path can escape the asset
    /// directory.
    pub fn asset_data_url(&self, asset: &str) -> AppResult<String> {
        let (session_id, file_name) = asset
            .split_once('/')
            .ok_or_else(|| AppError::message("That image reference is not valid."))?;
        if file_name.contains('/') || file_name.contains('\\') {
            return Err(AppError::message("That image reference is not valid."));
        }
        validate_id(session_id)?;
        validate_asset_name(file_name)?;

        let path = self.root.join("assets").join(session_id).join(file_name);
        let metadata = fs::metadata(&path)
            .map_err(|_| AppError::message("That image is no longer on disk."))?;
        if !metadata.is_file() {
            return Err(AppError::message("That image is no longer on disk."));
        }
        if metadata.len() > MAX_ASSET_BYTES {
            return Err(AppError::message("That image is too large to open."));
        }
        let bytes = fs::read(&path)?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let mime = match file_name
            .rsplit_once('.')
            .map(|(_, extension)| extension.to_ascii_lowercase())
            .as_deref()
        {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            Some("gif") => "image/gif",
            _ => "application/octet-stream",
        };
        Ok(format!("data:{mime};base64,{encoded}"))
    }

    /// Write a portable JSON copy with every asset inlined as a data URL.
    pub fn export(&self, id: &str, path: &str) -> AppResult<()> {
        validate_id(id)?;
        let target = Path::new(path);
        if path.trim().is_empty() {
            return Err(AppError::message("Choose where to save that chat."));
        }
        if target.is_dir() {
            return Err(AppError::message("That export path is a folder."));
        }
        let mut session = self.read_session(&self.session_path(id)?)?;
        self.inline_images(&mut session)?;
        let bytes = serde_json::to_vec_pretty(&session)
            .map_err(|err| AppError::message(format!("Could not encode that chat: {err}")))?;
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(target, bytes)?;
        Ok(())
    }

    /// Import a chat exported by this app. The new copy always gets a fresh id
    /// so importing twice cannot overwrite the original.
    pub fn import(&self, path: &str) -> AppResult<StoredSession> {
        let source = Path::new(path);
        let metadata = fs::metadata(source)
            .map_err(|err| AppError::message(format!("Could not read that chat file: {err}")))?;
        if !metadata.is_file() {
            return Err(AppError::message("That chat import path is not a file."));
        }
        if metadata.len() > MAX_SESSION_BYTES {
            return Err(AppError::message("That chat file is too large to import."));
        }
        let raw = fs::read_to_string(source)?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|err| AppError::message(format!("That is not a chat export: {err}")))?;
        self.import_value(value)
    }

    /// Write every chat, with images inlined, into one portable backup file.
    /// Returns the number of sessions written.
    pub fn export_all(&self, path: &str) -> AppResult<usize> {
        if path.trim().is_empty() {
            return Err(AppError::message("Choose where to save the backup."));
        }
        let target = Path::new(path);
        if target.is_dir() {
            return Err(AppError::message("That backup path is a folder."));
        }
        let mut sessions = self.list();
        let mut values = Vec::with_capacity(sessions.len());
        for session in &mut sessions {
            self.inline_images(session)?;
            values.push(
                serde_json::to_value(session)
                    .map_err(|err| AppError::message(format!("Could not encode that chat: {err}")))?,
            );
        }
        let payload = json!({
            "schema_version": CURRENT_SCHEMA_VERSION,
            "exported_at": now_ms(),
            "sessions": values,
        });
        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|err| AppError::message(format!("Could not encode the backup: {err}")))?;
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        write_atomic(target, &bytes)?;
        Ok(sessions.len())
    }

    /// Import a backup produced by `export_all`. Every session gets a fresh
    /// id, so restoring twice cannot overwrite the current shelf.
    pub fn import_all(&self, path: &str) -> AppResult<ChatImportSummary> {
        let source = Path::new(path);
        let metadata = fs::metadata(source)
            .map_err(|err| AppError::message(format!("Could not read that backup: {err}")))?;
        if !metadata.is_file() {
            return Err(AppError::message("That backup path is not a file."));
        }
        let raw = fs::read_to_string(source)?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|err| AppError::message(format!("That is not a chat backup: {err}")))?;
        let sessions = value
            .get("sessions")
            .and_then(Value::as_array)
            .cloned()
            .or_else(|| value.as_array().cloned())
            .ok_or_else(|| AppError::message("That backup has no sessions."))?;
        let mut summary = ChatImportSummary {
            imported: 0,
            failed: 0,
            errors: Vec::new(),
        };
        for session in sessions {
            match self.import_value(session) {
                Ok(_) => summary.imported += 1,
                Err(err) => {
                    summary.failed += 1;
                    if summary.errors.len() < 8 {
                        summary.errors.push(err.to_string());
                    }
                }
            }
        }
        Ok(summary)
    }

    /// Copy the prefix ending at `message_id` into a fresh session. This is the
    /// explicit "branch from here" action; normal edits keep their old suffix
    /// in `archived_messages` instead of creating a second file.
    pub fn branch_session(&self, id: &str, message_id: &str) -> AppResult<StoredSession> {
        validate_id(id)?;
        let mut source = self.read_session(&self.session_path(id)?)?;
        let index = source
            .messages
            .iter()
            .position(|message| message.get("id").and_then(Value::as_str) == Some(message_id))
            .ok_or_else(|| AppError::message("That message is no longer in this chat."))?;
        source.messages.truncate(index + 1);
        source.archived_messages.clear();
        source.extra.remove("draft");
        // A branch inherits the source chat's pins and evictions. Message
        // entries whose turn is not in the branch prefix would point at a
        // message the branch does not have, so those are dropped.
        let surviving: HashSet<String> = source
            .messages
            .iter()
            .filter_map(|message| message.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        source.pinned_context.retain(|item| {
            item.get("id")
                .and_then(Value::as_str)
                .map(|id| match id.strip_prefix("message:") {
                    Some(message_id) => surviving.contains(message_id),
                    None => true,
                })
                .unwrap_or(false)
        });
        source.evicted_context_ids.retain(|id| {
            match id.strip_prefix("message:") {
                Some(message_id) => surviving.contains(message_id),
                None => true,
            }
        });
        let new_id = uuid::Uuid::new_v4().simple().to_string();
        let old_id = source.id.clone();
        source.id = new_id.clone();
        source.title = format!("Branch of {}", source.title)
            .chars()
            .take(80)
            .collect();
        source.created_at = now_ms();
        source.updated_at = source.created_at;
        source.pinned = false;
        source.schema_version = CURRENT_SCHEMA_VERSION;
        self.copy_assets(&old_id, &new_id)?;
        for message in &mut source.messages {
            let Some(images) = message.get_mut("images").and_then(Value::as_array_mut) else {
                continue;
            };
            for image in images {
                let Some(object) = image.as_object_mut() else {
                    continue;
                };
                let Some(asset) = object.get("asset").and_then(Value::as_str) else {
                    continue;
                };
                if let Some((_, file_name)) = asset.split_once('/') {
                    object.insert("asset".into(), json!(format!("{new_id}/{file_name}")));
                }
            }
        }
        self.save(source)
    }

    pub fn backup_list(&self) -> Vec<ChatBackupInfo> {
        let directory = self.root.join("backups");
        let Ok(entries) = fs::read_dir(directory) else {
            return Vec::new();
        };
        let mut backups = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default()
                .to_string();
            let Ok(session) = self.read_session(&path) else {
                continue;
            };
            if session.id != id {
                continue;
            }
            backups.push(ChatBackupInfo {
                id,
                title: session.title,
                updated_at: session.updated_at,
                bytes: fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0),
            });
        }
        backups.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        backups
    }

    pub fn restore_backup(&self, id: &str) -> AppResult<StoredSession> {
        validate_id(id)?;
        let mut session = self.read_session(&self.backup_path(id)?)?;
        session.id = id.to_string();
        session.updated_at = now_ms();
        session.schema_version = CURRENT_SCHEMA_VERSION;
        self.save(session)
    }

    /// Remove asset directories that no session references. A crash between
    /// writing an image and saving its session leaves these behind.
    pub fn gc_assets(&self) -> AppResult<u64> {
        let mut live = HashSet::new();
        for session in self.list() {
            live.insert(session.id);
        }
        let directory = self.root.join("assets");
        let Ok(entries) = fs::read_dir(directory) else {
            return Ok(0);
        };
        let mut freed = 0u64;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !path.is_dir() || live.contains(&name) {
                continue;
            }
            freed += directory_bytes(&path);
            let _ = fs::remove_dir_all(&path);
        }
        Ok(freed)
    }

    pub fn stats(&self) -> ChatStoreStats {
        let sessions = self.list();
        let session_bytes = sessions
            .iter()
            .filter_map(|session| self.session_path(&session.id).ok())
            .filter_map(|path| fs::metadata(path).ok())
            .map(|meta| meta.len())
            .sum();
        let asset_bytes = directory_bytes(&self.root.join("assets"));
        let backups = self.backup_list();
        ChatStoreStats {
            sessions: sessions.len(),
            session_bytes,
            asset_bytes,
            backups: backups.len(),
            backup_bytes: backups.iter().map(|backup| backup.bytes).sum(),
        }
    }

    fn session_path(&self, id: &str) -> AppResult<PathBuf> {
        validate_id(id)?;
        Ok(self.root.join(format!("{id}.json")))
    }

    fn read_session(&self, path: &Path) -> AppResult<StoredSession> {
        let metadata = fs::metadata(path)?;
        if metadata.len() > MAX_SESSION_BYTES {
            return Err(AppError::message("That chat file is too large to open."));
        }
        let raw = fs::read_to_string(path)?;
        let mut session: StoredSession = serde_json::from_str(&raw)
            .map_err(|err| AppError::message(format!("Could not read that chat: {err}")))?;
        if session.schema_version > CURRENT_SCHEMA_VERSION {
            return Err(AppError::message(format!(
                "That chat uses a newer schema ({}).",
                session.schema_version
            )));
        }
        session.schema_version = CURRENT_SCHEMA_VERSION;
        validate_id(&session.id)?;
        Ok(session)
    }

    fn import_value(&self, value: Value) -> AppResult<StoredSession> {
        let has_messages = value.get("messages").and_then(Value::as_array).is_some();
        if !has_messages {
            return Err(AppError::message("That chat export has no messages."));
        }
        let mut session: StoredSession = serde_json::from_value(value)
            .map_err(|err| AppError::message(format!("Could not read that chat export: {err}")))?;
        let now = now_ms();
        session.id = uuid::Uuid::new_v4().simple().to_string();
        session.schema_version = CURRENT_SCHEMA_VERSION;
        session.created_at = if session.created_at == 0 { now } else { session.created_at };
        session.updated_at = now;
        if session.title.trim().is_empty() {
            session.title = "Imported chat".into();
        }
        self.save(session)
    }

    fn backup_path(&self, id: &str) -> AppResult<PathBuf> {
        validate_id(id)?;
        Ok(self.root.join("backups").join(format!("{id}.json")))
    }

    fn backup_existing(&self, path: &Path, id: &str) -> AppResult<()> {
        let backup = self.backup_path(id)?;
        if let Some(parent) = backup.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(path, &backup)?;
        Ok(())
    }

    fn copy_assets(&self, from: &str, to: &str) -> AppResult<()> {
        let source = self.root.join("assets").join(from);
        if !source.is_dir() {
            return Ok(());
        }
        let target = self.root.join("assets").join(to);
        fs::create_dir_all(&target)?;
        for entry in fs::read_dir(&source)? {
            let entry = entry?;
            let path = entry.path();
            if !entry.file_type().map(|kind| kind.is_file()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name();
            fs::copy(&path, target.join(name))?;
        }
        Ok(())
    }

    fn extract_images(&self, session: &mut StoredSession) -> AppResult<()> {
        let id = session.id.clone();
        extract_images_in(&self.root, &id, &mut session.messages)?;
        extract_images_in(&self.root, &id, &mut session.archived_messages)?;
        Ok(())
    }

    /// Remove images a chat no longer references. Clearing a chat or deleting
    /// a message should not leave the screenshot behind in app data forever.
    fn prune_assets(&self, session: &StoredSession) -> AppResult<()> {
        let directory = self.root.join("assets").join(&session.id);
        if !directory.is_dir() {
            return Ok(());
        }
        let mut keep = HashSet::new();
        collect_assets(&session.id, &session.messages, &mut keep);
        collect_assets(&session.id, &session.archived_messages, &mut keep);
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !keep.contains(&name) {
                let _ = fs::remove_file(entry.path());
            }
        }
        Ok(())
    }

    fn inline_images(&self, session: &mut StoredSession) -> AppResult<()> {
        inline_images_in(self, &mut session.messages);
        inline_images_in(self, &mut session.archived_messages);
        Ok(())
    }
}

fn extract_images_in(root: &Path, session_id: &str, messages: &mut [Value]) -> AppResult<()> {
    for message in messages {
        let Some(images) = message.get_mut("images").and_then(Value::as_array_mut) else {
            continue;
        };
        for image in images {
            let Some(object) = image.as_object_mut() else {
                continue;
            };
            let Some(data_url) = object.get("data_url").and_then(Value::as_str) else {
                continue;
            };
            let Some((header, payload)) = data_url.split_once(";base64,") else {
                continue;
            };
            if !header.starts_with("data:image/") {
                continue;
            }
            let mime = header.trim_start_matches("data:");
            let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(payload) else {
                // Leave a malformed data URL untouched rather than losing it.
                continue;
            };
            if bytes.len() as u64 > MAX_ASSET_BYTES {
                return Err(AppError::message(
                    "An attached image is too large to save. Remove it and try again.",
                ));
            }
            let extension = match mime {
                "image/png" => "png",
                "image/jpeg" => "jpg",
                "image/webp" => "webp",
                "image/gif" => "gif",
                _ => "bin",
            };
            // Content-addressed names make repeated saves of the same
            // chat reuse one file instead of leaking a copy per turn.
            let file_name = format!("{:016x}.{}", content_hash(&bytes), extension);
            let directory = root.join("assets").join(session_id);
            fs::create_dir_all(&directory)?;
            fs::write(directory.join(&file_name), bytes)?;
            object.insert("asset".into(), json!(format!("{session_id}/{file_name}")));
            object.remove("data_url");
        }
    }
    Ok(())
}

fn collect_assets(session_id: &str, messages: &[Value], keep: &mut HashSet<String>) {
    for message in messages {
        let Some(images) = message.get("images").and_then(Value::as_array) else {
            continue;
        };
        for image in images {
            let Some(asset) = image.get("asset").and_then(Value::as_str) else {
                continue;
            };
            if let Some((owner, file_name)) = asset.split_once('/') {
                if owner == session_id {
                    keep.insert(file_name.to_string());
                }
            }
        }
    }
}

fn inline_images_in(store: &ChatStore, messages: &mut [Value]) {
    for message in messages {
        let Some(images) = message.get_mut("images").and_then(Value::as_array_mut) else {
            continue;
        };
        for image in images {
            let Some(object) = image.as_object_mut() else {
                continue;
            };
            let Some(asset) = object.get("asset").and_then(Value::as_str) else {
                continue;
            };
            // A missing image should not block exporting the text of a
            // conversation. Leave the reference so the UI can say so.
            if let Ok(data_url) = store.asset_data_url(asset) {
                object.insert("data_url".into(), json!(data_url));
                object.remove("asset");
            }
        }
    }
}

fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() {
            total = total.saturating_add(directory_bytes(&path));
        } else if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    total
}

fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Fold the length in so a truncated prefix cannot collide with a shorter
    // image by construction.
    hash ^ (bytes.len() as u64)
}

fn validate_id(id: &str) -> AppResult<()> {
    let valid = !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_');
    if valid {
        Ok(())
    } else {
        Err(AppError::message("That chat id is not valid."))
    }
}

fn validate_asset_name(name: &str) -> AppResult<()> {
    let valid = !name.is_empty()
        && name.len() <= 160
        && !name.starts_with('.')
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.');
    if valid {
        Ok(())
    } else {
        Err(AppError::message("That image reference is not valid."))
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> AppResult<()> {
    // A per-write temp name means a manual flush cannot race a debounced save
    // of the same session and rename a half-written file into place.
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("session"),
        uuid::Uuid::new_v4().simple()
    ));
    fs::write(&temporary, bytes)?;
    #[cfg(windows)]
    {
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = fs::remove_file(&temporary);
            Err(err.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> ChatStore {
        let root = std::env::temp_dir().join(format!(
            "janas-chat-store-{}-{}",
            name,
            uuid::Uuid::new_v4().simple()
        ));
        ChatStore::load(root)
    }

    fn session_with_image(id: &str) -> StoredSession {
        StoredSession {
            id: id.into(),
            title: "Image chat".into(),
            created_at: 1,
            updated_at: 2,
            pinned: false,
            messages: vec![json!({
                "id": "m1",
                "role": "user",
                "content": "what is this",
                "reasoning": "",
                "created_at": 1,
                "images": [{
                    "name": "shot.png",
                    "data_url": "data:image/png;base64,aGVsbG8=",
                    "tokens": 3
                }]
            })],
            ..Default::default()
        }
    }

    #[test]
    fn empty_mcp_scope_survives_save_export_import_and_branch() {
        let store = store("mcp-scope");
        let mut session = session_with_image("scoped");
        session.extra.insert("mcp_server_ids".into(), json!([]));
        session.extra.insert(
            "mcp_server_names".into(),
            json!({ "server-1": "Files" }),
        );
        store.save(session).expect("save");

        let loaded = store
            .list()
            .into_iter()
            .find(|candidate| candidate.id == "scoped")
            .expect("saved session");
        assert_eq!(loaded.extra.get("mcp_server_ids"), Some(&json!([])));
        assert_eq!(
            loaded.extra.get("mcp_server_names"),
            Some(&json!({ "server-1": "Files" }))
        );

        let export = store.root.join("scoped-export.json");
        store
            .export("scoped", export.to_str().expect("path"))
            .expect("export");
        let imported = store.import(export.to_str().expect("path")).expect("import");
        assert_eq!(imported.extra.get("mcp_server_ids"), Some(&json!([])));
        assert_eq!(
            imported.extra.get("mcp_server_names"),
            Some(&json!({ "server-1": "Files" }))
        );

        let branch = store
            .branch_session("scoped", "m1")
            .expect("branch");
        assert_eq!(branch.extra.get("mcp_server_ids"), Some(&json!([])));
        assert_eq!(
            branch.extra.get("mcp_server_names"),
            Some(&json!({ "server-1": "Files" }))
        );
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn repeated_saves_reuse_one_asset_file() {
        let store = store("dedupe");
        store
            .save(session_with_image("same"))
            .expect("first save");
        store
            .save(session_with_image("same"))
            .expect("second save");
        let assets = store.root.join("assets").join("same");
        let count = fs::read_dir(&assets)
            .expect("assets dir")
            .filter_map(Result::ok)
            .count();
        assert_eq!(count, 1, "the same image should not be copied twice");
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn saves_extracts_and_lists_sessions() {
        let store = store("save");
        let saved = store
            .save(session_with_image("abc"))
            .expect("save session");
        assert_eq!(saved.id, "abc");
        let image = &saved.messages[0]["images"][0];
        assert!(image.get("data_url").is_none());
        let asset = image["asset"].as_str().expect("asset reference");
        assert!(asset.starts_with("abc/"));
        let data_url = store.asset_data_url(asset).expect("load asset");
        assert_eq!(data_url, "data:image/png;base64,aGVsbG8=");

        let sessions = store.list();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "abc");
        assert_eq!(sessions[0].messages.len(), 1);
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn saving_without_images_prunes_old_assets() {
        let store = store("prune");
        store
            .save(session_with_image("prune"))
            .expect("first save");
        let mut cleared = session_with_image("prune");
        cleared.messages[0]["images"] = json!([]);
        store.save(cleared).expect("save cleared chat");
        let assets = store.root.join("assets").join("prune");
        assert_eq!(fs::read_dir(assets).expect("assets dir").count(), 0);
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn export_inlines_and_import_renames() {
        let store = store("export");
        store
            .save(session_with_image("abc"))
            .expect("save session");
        let export_path = store.root.join("export.json");
        store
            .export("abc", export_path.to_str().expect("path"))
            .expect("export");
        let exported: Value =
            serde_json::from_str(&fs::read_to_string(&export_path).expect("read export"))
                .expect("parse export");
        assert_eq!(
            exported["messages"][0]["images"][0]["data_url"],
            "data:image/png;base64,aGVsbG8="
        );

        let imported = store
            .import(export_path.to_str().expect("path"))
            .expect("import");
        assert_ne!(imported.id, "abc");
        assert_eq!(imported.title, "Image chat");
        let asset = imported.messages[0]["images"][0]["asset"]
            .as_str()
            .expect("imported asset");
        assert!(store.asset_data_url(asset).is_ok());
        assert_eq!(store.list().len(), 2);
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn rejects_asset_traversal_and_invalid_ids() {
        let store = store("safe");
        assert!(store.asset_data_url("../../etc/passwd").is_err());
        assert!(store.asset_data_url("abc/../../etc/passwd").is_err());
        assert!(store.asset_data_url("abc/../passwd").is_err());
        assert!(store.save(StoredSession {
            id: "../escape".into(),
            title: "bad".into(),
            created_at: 0,
            updated_at: 0,
            pinned: false,
            messages: Vec::new(),
            ..Default::default()
        }).is_err());
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn delete_removes_the_session_and_assets() {
        let store = store("delete");
        let saved = store
            .save(session_with_image("gone"))
            .expect("save session");
        let asset = saved.messages[0]["images"][0]["asset"]
            .as_str()
            .expect("asset")
            .to_string();
        assert!(store.delete("gone").expect("delete"));
        assert!(store.asset_data_url(&asset).is_err());
        assert!(store.list().is_empty());
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn schema_version_is_upgraded_on_save() {
        let store = store("schema");
        let mut session = session_with_image("versioned");
        session.schema_version = 1;
        let saved = store.save(session).expect("save");
        assert_eq!(saved.schema_version, CURRENT_SCHEMA_VERSION);
        let loaded = store.list();
        assert_eq!(loaded[0].schema_version, CURRENT_SCHEMA_VERSION);
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn branch_keeps_prefix_and_copies_asset() {
        let store = store("branch");
        let mut session = session_with_image("root");
        session.messages.push(json!({
            "id": "m2",
            "role": "assistant",
            "content": "later answer",
            "reasoning": "",
            "created_at": 2
        }));
        let saved = store.save(session).expect("save");
        let asset = saved.messages[0]["images"][0]["asset"]
            .as_str()
            .expect("asset")
            .to_string();
        let branch = store
            .branch_session("root", "m1")
            .expect("branch from first message");
        assert_ne!(branch.id, "root");
        assert_eq!(branch.messages.len(), 1);
        assert!(branch.title.starts_with("Branch of"));
        let copied = branch.messages[0]["images"][0]["asset"]
            .as_str()
            .expect("copied asset");
        assert!(copied.starts_with(&format!("{}/", branch.id)));
        assert!(store.asset_data_url(copied).is_ok());
        assert!(store.asset_data_url(&asset).is_ok());
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn legacy_session_files_without_pins_still_load() {
        let store = store("legacy-pins");
        let path = store.root.join("legacy.json");
        fs::write(
            &path,
            r#"{"id":"legacy","schema_version":2,"title":"Legacy","created_at":1,"updated_at":2,"messages":[{"id":"m1","role":"user","content":"hi"}]}"#,
        )
        .expect("write legacy session");
        let sessions = store.list();
        assert_eq!(sessions.len(), 1);
        assert!(sessions[0].pinned_context.is_empty());
        assert!(sessions[0].evicted_context_ids.is_empty());
        let saved = store.save(sessions.into_iter().next().expect("session"))
            .expect("resave legacy session");
        assert!(saved.pinned_context.is_empty());
        assert_eq!(saved.schema_version, CURRENT_SCHEMA_VERSION);
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn pinned_context_round_trips_and_branches_inherit_it() {
        let store = store("pins");
        let mut session = session_with_image("pinned-chat");
        session.messages.push(json!({
            "id": "m2",
            "role": "assistant",
            "content": "answer",
            "reasoning": "",
            "created_at": 2
        }));
        session.pinned_context = vec![
            json!({"id":"file:src/main.rs","kind":"file","label":"src/main.rs","tokens":42,"added_at":1}),
            json!({"id":"message:m2","kind":"message","label":"answer","tokens":9,"added_at":1}),
        ];
        session.evicted_context_ids = vec!["file:src/old.rs".into(), "message:m2".into()];
        store.save(session).expect("save pinned chat");

        let loaded = store
            .list()
            .into_iter()
            .find(|candidate| candidate.id == "pinned-chat")
            .expect("saved pinned chat");
        assert_eq!(loaded.pinned_context.len(), 2);
        assert_eq!(loaded.evicted_context_ids.len(), 2);

        let branch = store
            .branch_session("pinned-chat", "m1")
            .expect("branch from first message");
        // m2 is outside the branch prefix, so its pin and eviction are gone.
        assert_eq!(branch.pinned_context.len(), 1);
        assert_eq!(branch.pinned_context[0]["id"], "file:src/main.rs");
        assert_eq!(
            branch.evicted_context_ids,
            vec!["file:src/old.rs".to_string()]
        );
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn export_all_and_import_all_round_trip() {
        let store = store("all");
        store
            .save(session_with_image("one"))
            .expect("first");
        store
            .save(session_with_image("two"))
            .expect("second");
        let path = store.root.join("backup.json");
        assert_eq!(store.export_all(path.to_str().expect("path")).expect("export"), 2);
        let summary = store
            .import_all(path.to_str().expect("path"))
            .expect("import");
        assert_eq!(summary.imported, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(store.list().len(), 4);
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn save_keeps_one_previous_backup_and_restores_it() {
        let store = store("backup");
        let mut first = session_with_image("chat");
        first.title = "First title".into();
        store.save(first).expect("first save");
        let mut second = session_with_image("chat");
        second.title = "Second title".into();
        second.updated_at = 3;
        store.save(second).expect("second save");
        let backups = store.backup_list();
        assert_eq!(backups.len(), 1);
        assert_eq!(backups[0].title, "First title");
        let restored = store.restore_backup("chat").expect("restore");
        assert_eq!(restored.title, "First title");
        let _ = fs::remove_dir_all(&store.root);
    }

    #[test]
    fn gc_removes_only_orphaned_asset_directories() {
        let store = store("gc");
        store.save(session_with_image("live")).expect("save");
        let orphan = store.root.join("assets").join("orphan");
        fs::create_dir_all(&orphan).expect("orphan dir");
        fs::write(orphan.join("a.png"), b"orphan").expect("orphan file");
        let freed = store.gc_assets().expect("gc");
        assert!(freed > 0);
        assert!(!orphan.exists());
        assert!(store.root.join("assets").join("live").exists());
        let _ = fs::remove_dir_all(&store.root);
    }
}
