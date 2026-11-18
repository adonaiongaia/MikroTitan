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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSession {
    pub id: String,
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
}

pub struct ChatStore {
    root: PathBuf,
}

impl ChatStore {
    pub fn load(root: PathBuf) -> Self {
        let _ = fs::create_dir_all(&root);
        let _ = fs::create_dir_all(root.join("assets"));
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
        let messages = value
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| AppError::message("That chat export has no messages."))?;
        let now = now_ms();
        let title = value
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| "Imported chat".into());
        let session = StoredSession {
            id: uuid::Uuid::new_v4().simple().to_string(),
            title,
            created_at: value
                .get("created_at")
                .and_then(Value::as_u64)
                .unwrap_or(now),
            updated_at: now,
            pinned: value
                .get("pinned")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            messages,
        };
        self.save(session)
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
        let session: StoredSession = serde_json::from_str(&raw)
            .map_err(|err| AppError::message(format!("Could not read that chat: {err}")))?;
        validate_id(&session.id)?;
        Ok(session)
    }

    fn extract_images(&self, session: &mut StoredSession) -> AppResult<()> {
        for message in &mut session.messages {
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
                let directory = self.root.join("assets").join(&session.id);
                fs::create_dir_all(&directory)?;
                fs::write(directory.join(&file_name), bytes)?;
                object.insert(
                    "asset".into(),
                    json!(format!("{}/{}", session.id, file_name)),
                );
                object.remove("data_url");
            }
        }
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
        for message in &session.messages {
            let Some(images) = message.get("images").and_then(Value::as_array) else {
                continue;
            };
            for image in images {
                let Some(asset) = image.get("asset").and_then(Value::as_str) else {
                    continue;
                };
                if let Some((session_id, file_name)) = asset.split_once('/') {
                    if session_id == session.id {
                        keep.insert(file_name.to_string());
                    }
                }
            }
        }
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
        for message in &mut session.messages {
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
                if let Ok(data_url) = self.asset_data_url(asset) {
                    object.insert("data_url".into(), json!(data_url));
                    object.remove("asset");
                }
            }
        }
        Ok(())
    }
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
        }
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
}
