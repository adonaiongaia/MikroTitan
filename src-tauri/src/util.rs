use std::{
    fs, io,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

/// Write a temporary credential or rule file with owner-only permissions from
/// the moment it is created. A stale temporary file is replaced so an old,
/// broader mode cannot leak the new contents.
///
/// On Unix the file is created with mode 0600. On other platforms it is a new
/// file that inherits the parent directory's ACL, so that directory must remain
/// private to the user.
pub fn write_private_temp(path: &Path, contents: &[u8]) -> io::Result<()> {
    use std::io::Write;

    let open = || {
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(path)
    };

    let mut file = match open() {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            fs::remove_file(path)?;
            open()?
        }
        Err(err) => return Err(err),
    };

    if let Err(err) = file.write_all(contents).and_then(|_| file.flush()) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(err);
    }
    Ok(())
}

/// Milliseconds since the Unix epoch, used for UI timestamps and duration math.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

/// Keep process output bounded before it crosses the IPC boundary.
pub fn truncate_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_string();
    }
    let mut clipped: String = line.chars().take(max_chars).collect();
    clipped.push('…');
    clipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn private_temp_is_owner_only_from_creation() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "janas-private-temp-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        write_private_temp(&path, b"secret").expect("write private temp");
        let mode = fs::metadata(&path)
            .expect("temp metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "temporary secrets must be owner-only");

        fs::write(&path, b"stale").expect("write stale temp");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))
            .expect("widen stale temp");
        write_private_temp(&path, b"replacement").expect("replace private temp");
        let mode = fs::metadata(&path)
            .expect("replacement metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "a stale temp file must be replaced owner-only");
        assert_eq!(fs::read(&path).expect("read replacement"), b"replacement");
        let _ = fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn private_temp_replaces_a_symlink_without_following_it() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let root = std::env::temp_dir().join(format!(
            "janas-private-temp-symlink-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("create temp directory");
        let victim = root.join("victim");
        let path = root.join("settings.json.tmp");
        fs::write(&victim, b"keep me").expect("write victim");
        fs::set_permissions(&victim, fs::Permissions::from_mode(0o644))
            .expect("widen victim");
        symlink(&victim, &path).expect("create symlink");

        write_private_temp(&path, b"secret").expect("replace symlink");

        assert_eq!(fs::read(&victim).expect("read victim"), b"keep me");
        assert_eq!(fs::read(&path).expect("read replacement"), b"secret");
        assert!(
            !fs::symlink_metadata(&path)
                .expect("temp metadata")
                .file_type()
                .is_symlink(),
            "the temporary path must become a regular file"
        );
        let mode = fs::metadata(&path)
            .expect("temp permissions")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "replacement must be owner-only");
        let _ = fs::remove_dir_all(root);
    }
}
