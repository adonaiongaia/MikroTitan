//! Keep Tauri diagnostics on stderr, but never print an IPC invoke key.
//!
//! Tauri formats a key mismatch as
//! `__TAURI_INVOKE_KEY__ expected <key> but received <key>`. A subscriber is
//! installed so other Tauri warnings and errors remain visible; this writer
//! replaces only lines carrying that marker with a generic diagnostic.

use std::io::{self, Write};

use tracing_subscriber::fmt::MakeWriter;

const KEY_MARKER: &[u8] = b"__TAURI_INVOKE_KEY__";
const REDACTED_LINE: &[u8] =
    b"__TAURI_INVOKE_KEY__ mismatch (expected and received values redacted)\n";

/// Install the redacting subscriber.
///
/// If another global subscriber already exists, redaction cannot be
/// guaranteed. The caller must fail closed instead of starting Tauri without
/// this writer.
pub fn init() -> Result<(), String> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(RedactingStderr)
        .try_init()
        .map_err(|error| format!("Could not install the redacting IPC logger: {error}"))
}

struct RedactingStderr;

impl<'a> MakeWriter<'a> for RedactingStderr {
    type Writer = RedactingWriter;

    fn make_writer(&'a self) -> Self::Writer {
        RedactingWriter::default()
    }
}

#[derive(Default)]
struct RedactingWriter {
    pending: Vec<u8>,
}

impl Write for RedactingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=end).collect();
            write_line(&line)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            write_line(&line)?;
        }
        let mut stderr = io::stderr();
        stderr.flush()
    }
}

impl Drop for RedactingWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

fn redact_line(line: &[u8]) -> &[u8] {
    if line
        .windows(KEY_MARKER.len())
        .any(|window| window == KEY_MARKER)
    {
        REDACTED_LINE
    } else {
        line
    }
}

fn write_line(line: &[u8]) -> io::Result<()> {
    let mut stderr = io::stderr();
    stderr.write_all(redact_line(line))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_line_is_redacted_without_exposing_values() {
        let line = b"ERROR tauri: __TAURI_INVOKE_KEY__ expected expected-secret but received fake-secret\n";
        let output = redact_line(line);
        assert_eq!(output, REDACTED_LINE);
        assert!(!output
            .windows(b"expected-secret".len())
            .any(|window| window == b"expected-secret"));
        assert!(!output
            .windows(b"fake-secret".len())
            .any(|window| window == b"fake-secret"));
    }

    #[test]
    fn ordinary_line_is_forwarded_unchanged() {
        let line = b"WARN tauri: useful diagnostic\n";
        assert_eq!(redact_line(line), line);
    }
}
