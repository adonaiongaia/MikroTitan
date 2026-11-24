//! Native Linux microphone capture for voice input.
//!
//! WebKitGTK's `getUserMedia` gives the webview the first enumerated capture
//! device rather than the Pulse/PipeWire default. That is wrong on machines
//! with an unused line-in and a USB microphone: the recording is silent even
//! though the OS has the USB microphone selected. This module records through
//! `parec` (or `pw-record` when PulseAudio utilities are absent), which follows
//! the same default source the rest of the desktop uses.
//!
//! The recorder deliberately produces canonical 16 kHz mono 16-bit WAV. It
//! captures raw PCM in a background thread, reports an RMS level for the
//! composer meter, and builds the WAV header only when recording stops.

use std::{
    env,
    ffi::OsStr,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Serialize;
use tauri::State;

use crate::error::{AppError, AppResult};

const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const BYTES_PER_SAMPLE: usize = 2;
const MAX_RECORDING_SECONDS: usize = 30 * 60;
const MAX_PCM_BYTES: usize = SAMPLE_RATE as usize
    * CHANNELS as usize
    * BYTES_PER_SAMPLE
    * MAX_RECORDING_SECONDS;

pub struct NativeAudioCapture {
    session: Mutex<Option<CaptureSession>>,
    level: Arc<AtomicU32>,
}

struct CaptureSession {
    backend: CaptureBackend,
    child: Child,
    buffer: Arc<Mutex<Vec<u8>>>,
    reader: Option<JoinHandle<()>>,
}

#[derive(Clone, Copy, Debug)]
enum CaptureBackend {
    Parec,
    PwRecord,
}

impl CaptureBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Parec => "parec",
            Self::PwRecord => "pw-record",
        }
    }

    fn command(self, path: &Path) -> Command {
        let mut command = Command::new(path);
        match self {
            Self::Parec => {
                command
                    .arg("--device=@DEFAULT_SOURCE@")
                    .arg("--format=s16le")
                    .arg(format!("--rate={SAMPLE_RATE}"))
                    .arg(format!("--channels={CHANNELS}"))
                    .arg("--latency-msec=50");
            }
            Self::PwRecord => {
                // `-` means raw PCM on stdout. pw-record's file mode would
                // write WAV, which is unnecessary because this module builds
                // the canonical WAV header itself.
                command
                    .arg("--rate")
                    .arg(SAMPLE_RATE.to_string())
                    .arg("--channels")
                    .arg(CHANNELS.to_string())
                    .arg("--format")
                    .arg("s16")
                    .arg("-");
            }
        }
        // A shell-level PULSE_SOURCE would override the desktop default,
        // recreating the exact problem this recorder exists to avoid.
        command
            .env_remove("PULSE_SOURCE")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }
}

impl CaptureSession {
    /// Stop the recorder, drain its output, and return the captured PCM plus
    /// diagnostics. This is shared by stop, cancel and Drop so every exit path
    /// closes the child and joins its reader.
    fn terminate(mut self) -> (Vec<u8>, Option<ExitStatus>, String) {
        let _ = self.child.kill();
        let status = self.child.wait().ok();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        let pcm = self
            .buffer
            .lock()
            .map(|buffer| buffer.clone())
            .unwrap_or_default();
        let mut stderr = String::new();
        if let Some(mut pipe) = self.child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr);
        }
        (pcm, status, stderr)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeRecording {
    pub audio_base64: String,
    pub mime: String,
    pub duration_ms: u64,
}

impl NativeAudioCapture {
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
            level: Arc::new(AtomicU32::new(0)),
        }
    }

    fn start(&self) -> AppResult<()> {
        let mut slot = self
            .session
            .lock()
            .map_err(|_| AppError::message("The microphone recorder is unavailable."))?;
        if slot.is_some() {
            return Err(AppError::message(
                "A native microphone recording is already running.",
            ));
        }

        let (backend, executable) = find_backend().ok_or_else(|| {
            AppError::message(
                "Native microphone capture needs `parec` or `pw-record` from PulseAudio or PipeWire. Install pulseaudio-utils or pipewire-bin and retry.",
            )
        })?;
        let mut child = backend
            .command(&executable)
            .spawn()
            .map_err(|err| {
                AppError::message(format!(
                    "Could not start native microphone capture with {} ({}): {err}",
                    backend.label(),
                    executable.display()
                ))
            })?;

        // Both tools can exit before the UI has finished switching to
        // recording (for example when the default source disappeared). Catch
        // that here and include the diagnostic instead of returning an empty
        // recording.
        std::thread::sleep(Duration::from_millis(80));
        if let Ok(Some(status)) = child.try_wait() {
            let mut stderr = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                let _ = pipe.read_to_string(&mut stderr);
            }
            return Err(AppError::message(format!(
                "The system default microphone could not be opened ({} exited with {status}).{}",
                backend.label(),
                if stderr.trim().is_empty() {
                    String::new()
                } else {
                    format!(" {}", stderr.trim())
                }
            )));
        }

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(AppError::message(
                    "Native microphone capture produced no audio stream.",
                ));
            }
        };
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let level = self.level.clone();
        level.store(0, Ordering::SeqCst);
        let reader_buffer = buffer.clone();
        let reader = thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stdout);
            let mut chunk = [0u8; 6_400];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => {
                        let data = &chunk[..count];
                        if let Ok(mut samples) = reader_buffer.lock() {
                            if samples.len() < MAX_PCM_BYTES {
                                let room = MAX_PCM_BYTES - samples.len();
                                samples.extend_from_slice(&data[..count.min(room)]);
                            }
                        }

                        // RMS over complete 16-bit samples in this read. An
                        // odd split sample is ignored for the meter; the raw
                        // buffer keeps every byte for the WAV payload.
                        let mut sum = 0.0f64;
                        let mut count_samples = 0usize;
                        for pair in data.chunks_exact(BYTES_PER_SAMPLE) {
                            let sample =
                                i16::from_le_bytes([pair[0], pair[1]]) as f64 / 32768.0;
                            sum += sample * sample;
                            count_samples += 1;
                        }
                        if count_samples > 0 {
                            let rms = (sum / count_samples as f64).sqrt();
                            let normalized = (rms * 8.0).clamp(0.0, 1.0);
                            level.store((normalized * 1000.0).round() as u32, Ordering::SeqCst);
                        }
                    }
                }
            }
        });

        *slot = Some(CaptureSession {
            backend,
            child,
            buffer,
            reader: Some(reader),
        });
        Ok(())
    }

    fn stop(&self) -> AppResult<NativeRecording> {
        let session = self
            .session
            .lock()
            .map_err(|_| AppError::message("The microphone recorder is unavailable."))?
            .take()
            .ok_or_else(|| AppError::message("No native microphone recording is running."))?;
        self.level.store(0, Ordering::SeqCst);

        let backend = session.backend;
        let (pcm, status, stderr) = session.terminate();
        if pcm.len() < BYTES_PER_SAMPLE {
            return Err(AppError::message(format!(
                "No audio was captured from the system default microphone.{}{}",
                status
                    .map(|value| format!(" {} exited with {value}.", backend.label()))
                    .unwrap_or_default(),
                if stderr.trim().is_empty() {
                    String::new()
                } else {
                    format!(" {}", stderr.trim())
                }
            )));
        }
        let pcm = &pcm[..pcm.len() & !1];
        let wav = pcm16_wav(pcm);
        let duration_ms = (pcm.len() as u64 * 1000) / (SAMPLE_RATE as u64 * BYTES_PER_SAMPLE as u64);
        Ok(NativeRecording {
            audio_base64: STANDARD.encode(wav),
            mime: "audio/wav".into(),
            duration_ms,
        })
    }

    fn cancel(&self) -> bool {
        let Ok(mut slot) = self.session.lock() else {
            return false;
        };
        self.level.store(0, Ordering::SeqCst);
        let Some(session) = slot.take() else {
            return false;
        };
        let _ = session.terminate();
        true
    }

    fn level(&self) -> f32 {
        self.level.load(Ordering::SeqCst) as f32 / 1000.0
    }
}

impl Drop for NativeAudioCapture {
    fn drop(&mut self) {
        // Closing the app while recording must not leave a capture child
        // holding the microphone open.
        if let Ok(mut slot) = self.session.lock() {
            if let Some(session) = slot.take() {
                let _ = session.terminate();
            }
        }
    }
}

fn find_backend() -> Option<(CaptureBackend, PathBuf)> {
    find_backend_in(&env::var_os("PATH")?)
}

fn find_backend_in(path: &OsStr) -> Option<(CaptureBackend, PathBuf)> {
    for backend in [CaptureBackend::Parec, CaptureBackend::PwRecord] {
        if let Some(executable) = find_executable_in(backend.label(), path) {
            return Some((backend, executable));
        }
    }
    None
}

fn find_executable_in(name: &str, path: &OsStr) -> Option<PathBuf> {
    env::split_paths(path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

fn pcm16_wav(pcm: &[u8]) -> Vec<u8> {
    let data_len = pcm.len().min(u32::MAX as usize - 44);
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * CHANNELS as u32 * BYTES_PER_SAMPLE as u32).to_le_bytes());
    out.extend_from_slice(&(CHANNELS * BYTES_PER_SAMPLE as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    out.extend_from_slice(&pcm[..data_len]);
    out
}

#[tauri::command]
pub fn start_native_capture(capture: State<'_, Arc<NativeAudioCapture>>) -> AppResult<()> {
    capture.start()
}

#[tauri::command]
pub fn stop_native_capture(
    capture: State<'_, Arc<NativeAudioCapture>>,
) -> AppResult<NativeRecording> {
    capture.stop()
}

#[tauri::command]
pub fn cancel_native_capture(capture: State<'_, Arc<NativeAudioCapture>>) -> bool {
    capture.cancel()
}

#[tauri::command]
pub fn native_capture_level(capture: State<'_, Arc<NativeAudioCapture>>) -> f32 {
    capture.level()
}

#[cfg(test)]
mod tests {
    use super::{find_backend_in, pcm16_wav, CaptureBackend};
    use std::{
        ffi::OsString,
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn pcm16_wav_has_the_canonical_header() {
        let pcm = [0i16, 1, -1, 32767, -32768]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let wav = pcm16_wav(&pcm);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + pcm.len());
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(
            u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize,
            pcm.len()
        );
    }

    #[test]
    fn backend_prefers_parec_and_falls_back_when_it_is_missing() {
        let root = std::env::temp_dir().join(format!(
            "janas-native-audio-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let parec = root.join("parec");
        let pw_record = root.join("pw-record");
        fs::write(&parec, b"stub").unwrap();
        fs::write(&pw_record, b"stub").unwrap();
        let path = OsString::from(root.as_os_str());

        let (backend, executable) = find_backend_in(&path).expect("parec should be found");
        assert!(matches!(backend, CaptureBackend::Parec));
        assert_eq!(executable, parec);

        fs::remove_file(&parec).unwrap();
        let (backend, executable) = find_backend_in(&path).expect("pw-record should be found");
        assert!(matches!(backend, CaptureBackend::PwRecord));
        assert_eq!(executable, pw_record);

        fs::remove_file(&pw_record).unwrap();
        assert!(find_backend_in(&path).is_none());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn pw_record_is_the_fallback_and_requests_raw_pcm() {
        let command = CaptureBackend::PwRecord.command(std::path::Path::new("/usr/bin/pw-record"));
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            ["--rate", "16000", "--channels", "1", "--format", "s16", "-"]
        );
    }
}
