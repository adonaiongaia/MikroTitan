//! 100% local speech-to-text with `whisper.cpp`.
//!
//! Two independently managed pieces live in the application's local data
//! directory, never inside the read-only AppImage mount:
//!
//! ```text
//! <app_local_data_dir>/
//!   bin/
//!     whisper/                 exactly one complete runtime
//!       whisper-cli            managed binary (downloaded on demand)
//!       libwhisper*/libggml*   its libraries and CPU backends
//!     whisper.bak/             previous complete runtime, kept on update
//!     .staging/<uuid>/         in-flight archive extraction
//!   whisper-models/
//!     ggml-base.en.bin         curated GGML models
//! ```
//!
//! The CLI and every one of its libraries live together in `whisper/`.
//! ggml discovers CPU backends next to the executable, so a shared `bin/`
//! layout would also discover llama.cpp's incompatible `libggml-vulkan.so`
//! and abort inside ggml. Keeping the complete runtime in its own directory
//! (and prepending that directory to `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` or
//! `PATH` only for this process) isolates the two engines.
//!
//! Audio handling is intentionally codec-independent. The webview records
//! Float32 PCM with the Web Audio API and encodes a canonical 16 kHz mono
//! 16-bit WAV before calling [`transcribe_audio`]. The Rust side also parses
//! WAV itself (PCM 8/16/24/32-bit and IEEE float), downmixes, resamples with a
//! monotonic linear filter and writes a canonical WAV. MP3, OGG and FLAC files
//! are handed to the release build's own decoders, which whisper.cpp ships
//! without FFmpeg.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::{io::AsyncWriteExt, sync::Mutex};
use uuid::Uuid;

use crate::{
    error::{AppError, AppResult},
    engines::{
        binary::{self, validate_url},
        managed_bin_dir,
    },
    settings::{AppSettings, SettingsStore},
    util::{truncate_line, write_private_temp},
};

pub const PROGRESS_EVENT: &str = "whisper://progress";
const MODEL_REPO: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=20";
const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_BINARY_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_AUDIO_BYTES: usize = 200 * 1024 * 1024;
const MAX_DECODED_SAMPLES: usize = 16_000 * 60 * 30;
const WHISPER_RATE: u32 = 16_000;
const MODEL_DIR_NAME: &str = "whisper-models";
const RUNTIME_DIR_NAME: &str = "whisper";
const BACKUP_DIR_NAME: &str = "whisper.bak";

// ---------------------------------------------------------------------------
// Curated model catalogue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct WhisperModelSpec {
    pub id: &'static str,
    pub file_name: &'static str,
    pub label: &'static str,
    pub size_bytes: u64,
    pub ram_mb: u32,
    pub multilingual: bool,
    pub note: &'static str,
    pub sha256: &'static str,
}

/// Sizes and SHA-256 digests are the LFS pointers published by
/// `ggerganov/whisper.cpp` on the Hugging Face Hub. They are pinned so a model
/// install is verified even if the Hub API is unavailable.
pub const WHISPER_MODELS: &[WhisperModelSpec] = &[
    WhisperModelSpec {
        id: "tiny.en",
        file_name: "ggml-tiny.en.bin",
        label: "Tiny English",
        size_bytes: 77_704_715,
        ram_mb: 390,
        multilingual: false,
        note: "Fastest; best for short English dictation on low-power machines.",
        sha256: "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
    },
    WhisperModelSpec {
        id: "base.en",
        file_name: "ggml-base.en.bin",
        label: "Base English",
        size_bytes: 147_964_211,
        ram_mb: 520,
        multilingual: false,
        note: "The balanced default for English notes and chat prompts.",
        sha256: "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
    },
    WhisperModelSpec {
        id: "small",
        file_name: "ggml-small.bin",
        label: "Small multilingual",
        size_bytes: 487_601_967,
        ram_mb: 1_500,
        multilingual: true,
        note: "Noticeably better accuracy in any supported language.",
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    },
    WhisperModelSpec {
        id: "large-v3-turbo-q5_0",
        file_name: "ggml-large-v3-turbo-q5_0.bin",
        label: "Large v3 Turbo (Q5_0)",
        size_bytes: 574_041_195,
        ram_mb: 2_600,
        multilingual: true,
        note: "Best multilingual accuracy; the Q5_0 build keeps it under 600 MB.",
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    },
];

fn model_spec(model_id: &str) -> AppResult<&'static WhisperModelSpec> {
    let needle = model_id.trim().to_ascii_lowercase();
    WHISPER_MODELS.iter().find(|spec| spec.id == needle).ok_or_else(|| {
        AppError::message(format!(
            "Unknown Whisper model \"{model_id}\". Choose one of: {}.",
            WHISPER_MODELS
                .iter()
                .map(|spec| spec.id)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })
}

// ---------------------------------------------------------------------------
// Public payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct WhisperModelStatus {
    pub id: String,
    pub file_name: String,
    pub label: String,
    pub size_bytes: u64,
    pub ram_mb: u32,
    pub multilingual: bool,
    pub note: String,
    pub installed: bool,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhisperStatus {
    pub installed: bool,
    pub binary_path: String,
    pub library_path: String,
    pub version: Option<String>,
    pub models_dir: String,
    pub models: Vec<WhisperModelStatus>,
    pub selected_model: String,
    pub active_model: Option<String>,
    pub language: String,
    pub voice_enabled: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhisperInstallResult {
    pub installed_path: String,
    pub version: Option<String>,
    pub release_tag: String,
    pub asset_name: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhisperModelDownload {
    pub model_id: String,
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhisperProgress {
    pub phase: String,
    /// Model id for model downloads; `None` for binary installs.
    pub model: Option<String>,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub percent: Option<f64>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    /// Wall-clock time spent decoding audio and running whisper-cli.
    pub duration_ms: u64,
    pub model_used: String,
    /// Approximate duration of the normalised audio, when known.
    pub audio_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct GhRelease {
    tag_name: String,
    draft: bool,
    assets: Vec<GhAsset>,
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Serializes installs and downloads. `cancel` is swapped for every model
/// download so the Cancel button can interrupt the exact transfer it started.
pub struct WhisperManager {
    gate: Mutex<()>,
    cancel: Mutex<Option<Arc<AtomicBool>>>,
}

impl WhisperManager {
    pub fn new() -> Self {
        Self {
            gate: Mutex::new(()),
            cancel: Mutex::new(None),
        }
    }

    async fn begin_cancellable(&self) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        let mut slot = self.cancel.lock().await;
        *slot = Some(flag.clone());
        flag
    }

    async fn clear_cancellable(&self) {
        *self.cancel.lock().await = None;
    }

    async fn request_cancel(&self) -> bool {
        let slot = self.cancel.lock().await;
        match slot.as_ref() {
            Some(flag) => {
                flag.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }
}

impl Default for WhisperManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Paths and commands
// ---------------------------------------------------------------------------

fn whisper_binary_name() -> &'static str {
    if cfg!(windows) {
        "whisper-cli.exe"
    } else {
        "whisper-cli"
    }
}

fn whisper_paths(app: &AppHandle) -> AppResult<(PathBuf, PathBuf, PathBuf)> {
    let bin = managed_bin_dir(app)?;
    let runtime = bin.join(RUNTIME_DIR_NAME);
    migrate_legacy_install(&bin, &runtime);
    let binary = runtime.join(whisper_binary_name());
    let libs = runtime;
    let models = app
        .path()
        .app_local_data_dir()
        .map_err(|err| {
            AppError::message(format!(
                "Could not resolve the app's local data directory for Whisper models: {err}"
            ))
        })?
        .join(MODEL_DIR_NAME);
    Ok((binary, libs, models))
}

/// Move a runtime installed by an older build out of the shared `bin/` folder.
/// The old layout put the CLI next to llama.cpp's libraries, which lets ggml
/// load the wrong Vulkan backend. Migration is best effort and atomic: a
/// complete runtime is staged and renamed into place, leaving the old files
/// untouched if anything fails.
fn migrate_legacy_install(bin: &Path, runtime: &Path) {
    let binary_name = whisper_binary_name();
    let legacy_binary = bin.join(binary_name);
    let legacy_libs = bin.join("whisper-libs");
    if runtime.join(binary_name).is_file()
        || !legacy_binary.is_file()
        || !legacy_libs.is_dir()
        || runtime.exists()
    {
        return;
    }

    let staging = bin.join(format!(".whisper-migrate-{}", Uuid::new_v4()));
    let result = (|| -> AppResult<()> {
        fs::create_dir_all(&staging).map_err(|err| {
            crate::engines::io_error("Could not stage the Whisper runtime", &staging, err)
        })?;
        let staged_binary = staging.join(binary_name);
        fs::copy(&legacy_binary, &staged_binary).map_err(|err| {
            crate::engines::io_error("Could not migrate whisper-cli", &legacy_binary, err)
        })?;
        binary::set_executable(&staged_binary)?;

        let entries = fs::read_dir(&legacy_libs).map_err(|err| {
            crate::engines::io_error("Could not read the old Whisper libraries", &legacy_libs, err)
        })?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name() else {
                continue;
            };
            let destination = staging.join(name);
            fs::copy(&path, &destination).map_err(|err| {
                crate::engines::io_error("Could not migrate a Whisper library", &path, err)
            })?;
            binary::set_executable(&destination)?;
        }

        fs::rename(&staging, runtime).map_err(|err| {
            crate::engines::io_error("Could not activate the Whisper runtime", runtime, err)
        })?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            let _ = fs::remove_file(&legacy_binary);
            let _ = fs::remove_dir_all(&legacy_libs);
        }
        Err(err) => {
            eprintln!("Studio for Janas: could not migrate the existing Whisper install: {err}");
            let _ = fs::remove_dir_all(&staging);
        }
    }
}

fn ensure_models_dir(app: &AppHandle) -> AppResult<PathBuf> {
    let (_, _, models) = whisper_paths(app)?;
    fs::create_dir_all(&models)
        .map_err(|err| crate::engines::io_error("Could not create the Whisper models folder", &models, err))?;
    Ok(models)
}

#[tauri::command]
pub async fn get_whisper_status(
    app: AppHandle,
    store: State<'_, Arc<SettingsStore>>,
) -> AppResult<WhisperStatus> {
    let settings = store.get();
    build_status(&app, &settings).await
}

async fn build_status(app: &AppHandle, settings: &AppSettings) -> AppResult<WhisperStatus> {
    let (binary, libs, models_dir) = whisper_paths(app)?;
    let installed = binary.is_file();
    let version = if installed {
        probe_whisper_version(&binary, &libs).await.ok().flatten()
    } else {
        None
    };

    let models = WHISPER_MODELS
        .iter()
        .map(|spec| {
            let path = models_dir.join(spec.file_name);
            let installed = path
                .metadata()
                .map(|meta| meta.is_file() && meta.len() > 0)
                .unwrap_or(false);
            WhisperModelStatus {
                id: spec.id.into(),
                file_name: spec.file_name.into(),
                label: spec.label.into(),
                size_bytes: spec.size_bytes,
                ram_mb: spec.ram_mb,
                multilingual: spec.multilingual,
                note: spec.note.into(),
                installed,
                path: path.to_string_lossy().to_string(),
            }
        })
        .collect::<Vec<_>>();

    let selected = settings.whisper_model.trim().to_ascii_lowercase();
    let active = models
        .iter()
        .find(|model| model.id == selected && model.installed)
        .or_else(|| models.iter().find(|model| model.installed))
        .map(|model| model.id.clone());

    let message = if installed {
        match active.as_deref() {
            Some(model) => format!("whisper-cli is ready with {model}."),
            None => "whisper-cli is installed. Download a speech model to start transcribing.".into(),
        }
    } else if models.iter().any(|model| model.installed) {
        "A speech model is present. Install whisper-cli to start transcribing.".into()
    } else {
        "Whisper is not installed yet. Install whisper-cli and a model in Settings → Voice.".into()
    };

    Ok(WhisperStatus {
        installed,
        binary_path: binary.to_string_lossy().to_string(),
        library_path: libs.to_string_lossy().to_string(),
        version,
        models_dir: models_dir.to_string_lossy().to_string(),
        models,
        selected_model: selected,
        active_model: active,
        language: settings.whisper_language.clone(),
        voice_enabled: settings.voice_input_enabled,
        message,
    })
}

#[tauri::command]
pub async fn install_whisper_binary(
    app: AppHandle,
    manager: State<'_, Arc<WhisperManager>>,
    backend: Option<String>,
) -> AppResult<WhisperInstallResult> {
    let manager = manager.inner().clone();
    install_binary(app, manager, backend).await
}

#[tauri::command]
pub async fn download_whisper_model(
    app: AppHandle,
    manager: State<'_, Arc<WhisperManager>>,
    store: State<'_, Arc<SettingsStore>>,
    model_id: String,
) -> AppResult<WhisperModelDownload> {
    let manager = manager.inner().clone();
    let settings = store.get();
    download_model(app, manager, settings, model_id).await
}

#[tauri::command]
pub async fn cancel_whisper_download(
    manager: State<'_, Arc<WhisperManager>>,
) -> AppResult<bool> {
    Ok(manager.inner().request_cancel().await)
}

// ---------------------------------------------------------------------------
// Binary provisioning
// ---------------------------------------------------------------------------

fn nvidia_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT.get_or_init(|| {
        if cfg!(target_os = "linux") && Path::new("/dev/nvidia0").exists() {
            return true;
        }
        let mut command = std::process::Command::new("nvidia-smi");
        command
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        command
            .output()
            .map(|output| output.status.success() && !output.stdout.is_empty())
            .unwrap_or(false)
    })
}

fn preferred_backend(requested: Option<&str>) -> String {
    let requested = requested.unwrap_or("auto").trim().to_ascii_lowercase();
    match requested.as_str() {
        "cpu" | "cuda" | "vulkan" | "metal" => requested,
        "auto" | "" => {
            if nvidia_present() && cfg!(windows) {
                "cuda".into()
            } else {
                "cpu".into()
            }
        }
        other => format!("cpu ({other} is not a supported Whisper backend on this platform)"),
    }
}

/// Choose the release asset that matches this OS, architecture and backend.
/// The newest release (`v1.9.6` at the time of writing) can have no assets at
/// all, so callers walk the release list until one yields a usable archive.
fn select_asset<'a>(
    assets: &'a [GhAsset],
    os: &str,
    arch: &str,
    backend: &str,
) -> Option<&'a GhAsset> {
    let os = os.to_ascii_lowercase();
    let arch = arch.to_ascii_lowercase();
    let mut wanted = Vec::new();
    match backend {
        "cuda" => wanted.push("cuda"),
        "vulkan" => wanted.push("vulkan"),
        _ => {}
    }
    wanted.push("cpu");

    for want in wanted {
        for asset in assets {
            let name = asset.name.to_ascii_lowercase();
            if !name.contains("whisper-bin") {
                continue;
            }
            if name.contains("xcframework")
                || name.contains("source")
                || name.contains("talk-llama")
                || name.contains("parakeet")
            {
                continue;
            }
            let os_matches = match os.as_str() {
                "linux" => name.contains("ubuntu") || name.contains("linux"),
                "windows" => {
                    name.contains("win") || name.ends_with("-x64.zip") || name.contains("-x64.")
                }
                "macos" => name.contains("macos") || name.contains("osx") || name.contains("darwin"),
                _ => false,
            };
            if !os_matches {
                continue;
            }
            let arch_matches = match arch.as_str() {
                "x86_64" => {
                    (name.contains("x64") || name.contains("amd64")) && !name.contains("arm64")
                }
                "aarch64" => name.contains("arm64") || name.contains("aarch64"),
                "x86" => name.contains("win32") || name.contains("x86") || name.contains("i686"),
                _ => false,
            };
            if !arch_matches {
                continue;
            }
            let is_cuda = name.contains("cuda") || name.contains("cublas");
            let is_vulkan = name.contains("vulkan");
            let is_opencl = name.contains("opencl");
            let backend_matches = match want {
                "cuda" => is_cuda,
                "vulkan" => is_vulkan,
                "cpu" => !is_cuda && !is_vulkan && !is_opencl,
                _ => false,
            };
            if backend_matches {
                return Some(asset);
            }
        }
    }
    None
}

async fn resolve_release(
    backend: &str,
) -> AppResult<(GhRelease, GhAsset)> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent("Studio-for-Janas/0.1")
        .build()
        .map_err(|err| AppError::message(format!("Could not build the release client: {err}")))?;
    let response = client
        .get(RELEASES_API)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|err| {
            AppError::message(format!(
                "Could not query whisper.cpp releases: {err}. Check your network connection and retry."
            ))
        })?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(AppError::message(format!(
            "GitHub returned HTTP {status} while resolving whisper.cpp releases: {}",
            truncate_line(&body, 240)
        )));
    }
    let releases: Vec<GhRelease> = response
        .json()
        .await
        .map_err(|err| AppError::message(format!("Unexpected whisper.cpp release data: {err}")))?;

    let mut available = Vec::new();
    for release in releases {
        // whisper.cpp attaches its platform archives to CI build tags, which
        // GitHub marks as prereleases; the newest stable tag often has no
        // assets at all. Any usable non-draft release is a valid install.
        if release.draft {
            continue;
        }
        for asset in &release.assets {
            if asset.name.to_ascii_lowercase().contains("whisper-bin") {
                available.push(asset.name.clone());
            }
        }
        if let Some(asset) =
            select_asset(&release.assets, std::env::consts::OS, std::env::consts::ARCH, backend)
        {
            let asset = asset.clone();
            return Ok((release, asset));
        }
    }

    let hint = if available.is_empty() {
        "The release list exposed no whisper-cli archives.".to_string()
    } else {
        format!(
            "Available archives were: {}.",
            available.into_iter().take(8).collect::<Vec<_>>().join(", ")
        )
    };
    Err(AppError::message(format!(
        "No whisper.cpp release archive matches {} / {} with a {backend} backend. {hint} Upstream does not publish a macOS whisper-cli; install whisper.cpp with your package manager and place whisper-cli in the app's managed bin folder.",
        std::env::consts::OS,
        std::env::consts::ARCH
    )))
}

async fn install_binary(
    app: AppHandle,
    manager: Arc<WhisperManager>,
    backend: Option<String>,
) -> AppResult<WhisperInstallResult> {
    let _gate = manager.gate.lock().await;
    let bin = managed_bin_dir(&app)?;
    let staging_root = bin.join(".staging");
    fs::create_dir_all(&staging_root)
        .map_err(|err| crate::engines::io_error("Could not create the staging folder", &staging_root, err))?;
    let work = staging_root.join(Uuid::new_v4().to_string());
    fs::create_dir_all(&work)
        .map_err(|err| crate::engines::io_error("Could not create an install workspace", &work, err))?;

    let backend = preferred_backend(backend.as_deref());
    emit_progress(
        &app,
        None,
        "preparing",
        0,
        None,
        "Resolving the latest whisper.cpp release…".into(),
    );

    let result = install_binary_inner(&app, &work, &backend).await;
    if let Err(err) = &result {
        emit_progress(
            &app,
            None,
            "error",
            0,
            None,
            truncate_line(&err.to_string(), 400),
        );
    }
    let _ = fs::remove_dir_all(&work);
    result
}

async fn install_binary_inner(
    app: &AppHandle,
    work: &Path,
    backend: &str,
) -> AppResult<WhisperInstallResult> {
    let (release, asset) = resolve_release(backend).await?;
    if asset.size > MAX_BINARY_BYTES {
        return Err(AppError::message(format!(
            "The {} archive is {} bytes, larger than the {} byte safety limit.",
            asset.name, asset.size, MAX_BINARY_BYTES
        )));
    }
    let url = validate_url(&asset.browser_download_url)?;
    let archive = work.join("whisper-archive");
    emit_progress(
        app,
        None,
        "downloading",
        0,
        Some(asset.size),
        format!(
            "Downloading {} ({})…",
            asset.name,
            format_bytes(asset.size)
        ),
    );
    download_with_progress(app, None, url, &archive, MAX_BINARY_BYTES, None).await?;

    emit_progress(
        app,
        None,
        "extracting",
        0,
        None,
        "Verifying the archive and extracting whisper-cli…".into(),
    );
    let payload = work.join("payload");
    fs::create_dir_all(&payload)
        .map_err(|err| crate::engines::io_error("Could not create the extraction folder", &payload, err))?;
    let archive_for_extract = archive.clone();
    let payload_for_extract = payload.clone();
    let binary_name = whisper_binary_name().to_string();
    let staged = tokio::task::spawn_blocking(move || {
        binary::extract_archive(&archive_for_extract, &payload_for_extract, &binary_name)
    })
    .await
    .map_err(|err| AppError::message(format!("The extraction task failed: {err}")))??;

    let (target, libs, _) = whisper_paths(app)?;
    emit_progress(
        app,
        None,
        "installing",
        0,
        None,
        "Installing whisper-cli and its shared libraries…".into(),
    );
    let prepared = tokio::task::spawn_blocking(move || {
        prepare_binary_install(&target, &libs, &staged.binary, &staged.companions)
    })
    .await
    .map_err(|err| AppError::message(format!("The install task failed: {err}")))??;

    emit_progress(
        app,
        None,
        "verifying",
        0,
        None,
        "Checking that whisper-cli starts with its shared libraries…".into(),
    );
    let verification = probe_whisper_version(&prepared.target, &prepared.libs).await;
    match verification {
        Ok(version) => {
            emit_progress(
                app,
                None,
                "complete",
                0,
                None,
                format!(
                    "whisper-cli installed{}.",
                    version
                        .as_ref()
                        .map(|value| format!(" ({value})"))
                        .unwrap_or_default()
                ),
            );
            Ok(WhisperInstallResult {
                installed_path: prepared.target.to_string_lossy().to_string(),
                version,
                release_tag: release.tag_name.clone(),
                asset_name: asset.name.clone(),
                message: format!(
                    "whisper-cli was installed from {} ({} backend). The previous runtime is kept as whisper.bak.",
                    release.tag_name, backend
                ),
            })
        }
        Err(err) => {
            let runtime = prepared.libs.clone();
            let backup = prepared.backup.clone();
            let restore = tokio::task::spawn_blocking(move || {
                restore_runtime(&runtime, backup.as_deref())
            })
            .await;
            let note = match restore {
                Ok(Ok(())) => "The previous binary was restored.".to_string(),
                Ok(Err(restore_err)) => format!(
                    "The automatic restore also failed: {restore_err}. Reinstall from Settings → Voice."
                ),
                Err(join_err) => format!("The restore task failed: {join_err}."),
            };
            Err(AppError::message(format!(
                "whisper-cli failed its startup check, so the update was rolled back. {note} Original error: {err}"
            )))
        }
    }
}

struct PreparedBinary {
    target: PathBuf,
    libs: PathBuf,
    backup: Option<PathBuf>,
}

fn prepare_binary_install(
    target: &Path,
    libs: &Path,
    staged_binary: &Path,
    companions: &[PathBuf],
) -> AppResult<PreparedBinary> {
    let runtime = target
        .parent()
        .ok_or_else(|| AppError::message("The managed binary path has no parent directory."))?;
    if runtime != libs {
        return Err(AppError::message(
            "Whisper's binary and libraries must share one runtime directory.",
        ));
    }
    let bin = runtime
        .parent()
        .ok_or_else(|| AppError::message("The Whisper runtime path has no parent directory."))?;
    let backup = bin.join(BACKUP_DIR_NAME);
    let temp = runtime.join(format!(".whisper-cli.{}.new", Uuid::new_v4()));
    let had_runtime = runtime.is_dir();

    if backup.exists() {
        if backup.is_dir() {
            fs::remove_dir_all(&backup).map_err(|err| {
                crate::engines::io_error("Could not remove the old Whisper backup", &backup, err)
            })?;
        } else {
            fs::remove_file(&backup).map_err(|err| {
                crate::engines::io_error("Could not remove the old Whisper backup", &backup, err)
            })?;
        }
    }

    if had_runtime {
        fs::rename(runtime, &backup).map_err(|err| {
            crate::engines::io_error("Could not back up the Whisper runtime", runtime, err)
        })?;
    }

    let outcome = (|| -> AppResult<()> {
        fs::create_dir_all(runtime).map_err(|err| {
            crate::engines::io_error("Could not create the Whisper runtime folder", runtime, err)
        })?;
        for companion in companions {
            let Some(name) = companion.file_name() else {
                continue;
            };
            let destination = runtime.join(name);
            fs::copy(companion, &destination).map_err(|err| {
                crate::engines::io_error("Could not install a Whisper library", &destination, err)
            })?;
            binary::set_executable(&destination)?;
        }

        fs::copy(staged_binary, &temp).map_err(|err| {
            crate::engines::io_error("Could not stage whisper-cli", &temp, err)
        })?;
        binary::set_executable(&temp)?;
        fs::rename(&temp, target).map_err(|err| {
            crate::engines::io_error("Could not atomically install whisper-cli", target, err)
        })?;
        Ok(())
    })();

    match outcome {
        Ok(()) => {
            Ok(PreparedBinary {
                target: target.to_path_buf(),
                libs: libs.to_path_buf(),
                backup: had_runtime.then_some(backup),
            })
        }
        Err(err) => {
            let _ = fs::remove_file(&temp);
            let _ = fs::remove_dir_all(runtime);
            if had_runtime && backup.is_dir() {
                let _ = fs::rename(&backup, runtime);
            }
            Err(err)
        }
    }
}

fn restore_runtime(runtime: &Path, backup: Option<&Path>) -> AppResult<()> {
    if runtime.exists() {
        fs::remove_dir_all(runtime).map_err(|err| {
            crate::engines::io_error("Could not remove the failed Whisper runtime", runtime, err)
        })?;
    }
    if let Some(backup) = backup {
        if backup.is_dir() {
            fs::rename(backup, runtime).map_err(|err| {
                crate::engines::io_error("Could not restore the previous Whisper runtime", runtime, err)
            })?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Model downloads
// ---------------------------------------------------------------------------

async fn download_model(
    app: AppHandle,
    manager: Arc<WhisperManager>,
    settings: AppSettings,
    model_id: String,
) -> AppResult<WhisperModelDownload> {
    let spec = model_spec(&model_id)?;
    let _gate = manager.gate.lock().await;
    let models_dir = ensure_models_dir(&app)?;
    let final_path = models_dir.join(spec.file_name);
    let cancel = manager.begin_cancellable().await;

    let result = download_model_inner(
        &app,
        &settings,
        spec,
        &models_dir,
        &final_path,
        cancel,
    )
    .await;
    manager.clear_cancellable().await;
    result
}

async fn download_model_inner(
    app: &AppHandle,
    settings: &AppSettings,
    spec: &WhisperModelSpec,
    models_dir: &Path,
    final_path: &Path,
    cancel: Arc<AtomicBool>,
) -> AppResult<WhisperModelDownload> {
    if final_path.is_file() {
        let metadata = final_path
            .metadata()
            .map_err(|err| crate::engines::io_error("Could not inspect the Whisper model", final_path, err))?;
        if metadata.len() == spec.size_bytes {
            let path = final_path.to_path_buf();
            let probe = tokio::task::spawn_blocking(move || sha256_file(&path))
                .await
                .map_err(|err| AppError::message(format!("The checksum task failed: {err}")))??;
            if probe.eq_ignore_ascii_case(spec.sha256) {
                emit_progress(
                    app,
                    Some(spec.id),
                    "complete",
                    spec.size_bytes,
                    Some(spec.size_bytes),
                    format!("{} is already installed and verified.", spec.label),
                );
                return Ok(WhisperModelDownload {
                    model_id: spec.id.into(),
                    path: final_path.to_string_lossy().to_string(),
                    size_bytes: spec.size_bytes,
                    sha256: probe,
                    message: format!("{} is already installed.", spec.label),
                });
            }
        }
        // A corrupt or stale partial file is replaced instead of trusted.
        fs::remove_file(final_path).map_err(|err| {
            crate::engines::io_error("Could not remove the damaged Whisper model", final_path, err)
        })?;
    }

    let part = models_dir.join(format!(".{}.part", spec.file_name));
    let _ = fs::remove_file(&part);
    let url = validate_url(&format!("{MODEL_REPO}/{}", spec.file_name))?;
    emit_progress(
        app,
        Some(spec.id),
        "downloading",
        0,
        Some(spec.size_bytes),
        format!(
            "Downloading {} ({})…",
            spec.label,
            format_bytes(spec.size_bytes)
        ),
    );
    download_with_progress(
        app,
        Some(spec.id),
        url,
        &part,
        MAX_MODEL_BYTES,
        Some((&cancel, spec.size_bytes, spec.file_name)),
    )
    .await?;

    emit_progress(
        app,
        Some(spec.id),
        "verifying",
        0,
        Some(spec.size_bytes),
        "Verifying the model with SHA-256…".into(),
    );
    let part_for_hash = part.clone();
    let digest = tokio::task::spawn_blocking(move || sha256_file(&part_for_hash))
        .await
        .map_err(|err| AppError::message(format!("The checksum task failed: {err}")))??;
    if !digest.eq_ignore_ascii_case(spec.sha256) {
        let _ = fs::remove_file(&part);
        return Err(AppError::message(format!(
            "{} failed SHA-256 verification (expected {}, received {}). The download was deleted; retry it from Settings → Voice.",
            spec.label, spec.sha256, digest
        )));
    }

    if cancel.load(Ordering::SeqCst) {
        let _ = fs::remove_file(&part);
        return Err(AppError::message("The model download was cancelled."));
    }

    fs::rename(&part, final_path).map_err(|err| {
        crate::engines::io_error("Could not move the verified model into place", final_path, err)
    })?;
    emit_progress(
        app,
        Some(spec.id),
        "complete",
        spec.size_bytes,
        Some(spec.size_bytes),
        format!("{} installed and verified.", spec.label),
    );

    // Keep the selected model coherent with what is actually installed.
    if settings.whisper_model.trim().is_empty() {
        let _ = app.emit(
            "whisper://model-ready",
            serde_json::json!({ "model": spec.id }),
        );
    }

    Ok(WhisperModelDownload {
        model_id: spec.id.into(),
        path: final_path.to_string_lossy().to_string(),
        size_bytes: spec.size_bytes,
        sha256: digest,
        message: format!("{} downloaded and verified with SHA-256.", spec.label),
    })
}

async fn download_with_progress(
    app: &AppHandle,
    model: Option<&str>,
    url: reqwest::Url,
    destination: &Path,
    max_bytes: u64,
    cancel: Option<(&Arc<AtomicBool>, u64, &str)>,
) -> AppResult<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(30 * 60))
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent("Studio-for-Janas/0.1")
        .build()
        .map_err(|err| AppError::message(format!("Could not build the HTTP client: {err}")))?;
    let response = client.get(url).send().await.map_err(|err| {
        AppError::message(format!(
            "Could not start the download: {err}. Check your network and proxy settings."
        ))
    })?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(AppError::message(format!(
            "The download server returned HTTP {status}: {}",
            truncate_line(&body, 300)
        )));
    }

    let total = response.content_length().or_else(|| cancel.map(|(_, size, _)| size));
    if let Some(total) = total {
        if total > max_bytes {
            return Err(AppError::message(format!(
                "The download is {total} bytes, larger than the {} byte safety limit.",
                max_bytes
            )));
        }
    }

    let mut file = tokio::fs::File::create(destination)
        .await
        .map_err(|err| crate::engines::io_error("Could not create the download file", destination, err))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = tokio::fs::set_permissions(
            destination,
            fs::Permissions::from_mode(0o600),
        )
        .await;
    }
    let mut stream = response.bytes_stream();
    let mut downloaded = 0u64;
    let mut last_emit = Instant::now();
    let started = Instant::now();

    while let Some(chunk) = futures_util::StreamExt::next(&mut stream).await {
        if let Some((flag, _, _)) = cancel {
            if flag.load(Ordering::SeqCst) {
                drop(file);
                let _ = tokio::fs::remove_file(destination).await;
                emit_progress(
                    app,
                    model,
                    "cancelled",
                    downloaded,
                    total,
                    format!("Cancelled after {}.", format_bytes(downloaded)),
                );
                return Err(AppError::message("The download was cancelled."));
            }
        }
        let chunk = chunk.map_err(|err| {
            AppError::message(format!("The download was interrupted after {downloaded} bytes: {err}"))
        })?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded > max_bytes {
            drop(file);
            let _ = tokio::fs::remove_file(destination).await;
            return Err(AppError::message(format!(
                "The download exceeded the {} byte safety limit.",
                max_bytes
            )));
        }
        file.write_all(&chunk)
            .await
            .map_err(|err| crate::engines::io_error("Could not write the download file", destination, err))?;

        if last_emit.elapsed() >= Duration::from_millis(200) {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            let rate = downloaded as f64 / elapsed;
            let prefix = cancel
                .map(|(_, _, name)| name)
                .unwrap_or("whisper.cpp archive");
            emit_progress(
                app,
                model,
                "downloading",
                downloaded,
                total,
                format!(
                    "Downloading {prefix} … {} / {} · {}/s",
                    format_bytes(downloaded),
                    total.map(format_bytes).unwrap_or_else(|| "?".into()),
                    format_bytes(rate as u64)
                ),
            );
            last_emit = Instant::now();
        }
    }

    file.flush()
        .await
        .map_err(|err| crate::engines::io_error("Could not flush the download", destination, err))?;
    drop(file);

    if let Some(total) = total {
        if total != downloaded {
            let _ = fs::remove_file(destination);
            return Err(AppError::message(format!(
                "The download ended early: received {downloaded} of {total} bytes. Retry it from Settings → Voice."
            )));
        }
    }
    emit_progress(
        app,
        model,
        "downloaded",
        downloaded,
        total.or(Some(downloaded)),
        format!("Downloaded {}.", format_bytes(downloaded)),
    );
    Ok(())
}

fn sha256_file(path: &Path) -> AppResult<String> {
    let mut file = fs::File::open(path)
        .map_err(|err| crate::engines::io_error("Could not open the downloaded file", path, err))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| crate::engines::io_error("Could not read the downloaded file", path, err))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn emit_progress(
    app: &AppHandle,
    model: Option<&str>,
    phase: &str,
    downloaded: u64,
    total: Option<u64>,
    message: String,
) {
    let percent = total
        .filter(|total| *total > 0)
        .map(|total| (downloaded as f64 / total as f64 * 100.0).clamp(0.0, 100.0));
    let _ = app.emit(
        PROGRESS_EVENT,
        WhisperProgress {
            phase: phase.into(),
            model: model.map(str::to_string),
            downloaded,
            total,
            percent,
            message,
        },
    );
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

/// RAII guard: the temporary recording leaves the system even when decoding,
/// verification or whisper-cli returns an error.
struct TempAudio {
    path: PathBuf,
}

impl TempAudio {
    fn new(extension: &str) -> AppResult<Self> {
        let path = std::env::temp_dir().join(format!(
            "janas-voice-{}.{}",
            Uuid::new_v4(),
            extension
        ));
        Ok(Self { path })
    }

    fn write(&self, bytes: &[u8]) -> AppResult<()> {
        write_private_temp(&self.path, bytes).map_err(|err| {
            AppError::message(format!(
                "Could not write the temporary recording {}: {err}",
                self.path.display()
            ))
        })
    }
}

impl Drop for TempAudio {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub async fn transcribe_audio_payload(
    app: &AppHandle,
    settings: &AppSettings,
    audio_base64: &str,
    mime: Option<&str>,
    file_name: Option<&str>,
    model_override: Option<&str>,
    language_override: Option<&str>,
    strip_non_speech: Option<bool>,
) -> AppResult<TranscriptionResult> {
    let bytes = STANDARD
        .decode(audio_base64.trim())
        .map_err(|err| AppError::message(format!("The recording could not be decoded: {err}")))?;
    if bytes.is_empty() {
        return Err(AppError::message("The recording is empty."));
    }
    if bytes.len() > MAX_AUDIO_BYTES {
        return Err(AppError::message(format!(
            "The recording is {} MB; the limit is {} MB.",
            bytes.len() / (1024 * 1024),
            MAX_AUDIO_BYTES / (1024 * 1024)
        )));
    }

    transcribe_bytes(
        app,
        settings,
        bytes,
        mime,
        file_name,
        model_override,
        language_override,
        strip_non_speech.unwrap_or(true),
    )
    .await
}

pub async fn transcribe_audio_path(
    app: &AppHandle,
    settings: &AppSettings,
    path: &str,
    model_override: Option<&str>,
    language_override: Option<&str>,
) -> AppResult<TranscriptionResult> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::message("Choose an audio file to transcribe."));
    }
    let file_path = PathBuf::from(trimmed);
    if !file_path.is_file() {
        return Err(AppError::message(format!(
            "The audio file {} does not exist.",
            file_path.display()
        )));
    }
    let metadata = file_path
        .metadata()
        .map_err(|err| crate::engines::io_error("Could not inspect the audio file", &file_path, err))?;
    if metadata.len() > MAX_AUDIO_BYTES as u64 {
        return Err(AppError::message(format!(
            "That audio file is {} MB; the limit is {} MB.",
            metadata.len() / (1024 * 1024),
            MAX_AUDIO_BYTES / (1024 * 1024)
        )));
    }
    let extension = file_path
        .extension()
        .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if !matches!(extension.as_str(), "wav" | "mp3" | "ogg" | "oga" | "flac") {
        return Err(AppError::message(
            "Whisper accepts .wav, .mp3, .ogg and .flac files. Convert other formats before attaching them.",
        ));
    }
    let bytes = tokio::fs::read(&file_path)
        .await
        .map_err(|err| crate::engines::io_error("Could not read the audio file", &file_path, err))?;
    let file_name = file_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string());
    transcribe_bytes(
        app,
        settings,
        bytes,
        None,
        file_name.as_deref(),
        model_override,
        language_override,
        true,
    )
    .await
}

async fn transcribe_bytes(
    app: &AppHandle,
    settings: &AppSettings,
    bytes: Vec<u8>,
    mime: Option<&str>,
    file_name: Option<&str>,
    model_override: Option<&str>,
    language_override: Option<&str>,
    strip_non_speech: bool,
) -> AppResult<TranscriptionResult> {
    let started = Instant::now();
    let (binary, libs, models_dir) = whisper_paths(app)?;
    if !binary.is_file() {
        return Err(AppError::message(
            "Whisper is not installed. Open Settings → Voice and click Install whisper-cli, then download a model.",
        ));
    }

    let explicit_model = model_override
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let configured = explicit_model.unwrap_or_else(|| settings.whisper_model.trim());
    let installed_spec = |spec: &WhisperModelSpec| models_dir.join(spec.file_name).is_file();
    let spec = if explicit_model.is_some() {
        let spec = model_spec(configured)?;
        if !installed_spec(spec) {
            return Err(AppError::message(format!(
                "The requested Whisper model {} is not installed. Download it in Settings → Voice.",
                spec.id
            )));
        }
        spec
    } else if !configured.is_empty() && model_spec(configured).is_ok_and(installed_spec) {
        model_spec(configured)?
    } else {
        // The saved selection may name a model the user never finished
        // downloading. The status panel already reports the first installed
        // model as active, so transcription follows the same rule.
        WHISPER_MODELS
            .iter()
            .find(|spec| installed_spec(spec))
            .ok_or_else(|| {
                if configured.is_empty() {
                    AppError::message(
                        "No Whisper model is installed. Download one in Settings → Voice.",
                    )
                } else {
                    AppError::message(format!(
                        "The selected Whisper model {configured} is not installed, and no other model is available. Download one in Settings → Voice."
                    ))
                }
            })?
    };
    let model_path = models_dir.join(spec.file_name);
    if !model_path.is_file() {
        return Err(AppError::message(format!(
            "The selected Whisper model {} is not installed. Download it in Settings → Voice.",
            spec.id
        )));
    }

    let language = normalise_language(
        language_override
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| settings.whisper_language.trim()),
    )?;

    let format = AudioFormat::detect(&bytes, mime, file_name);
    let (temp, audio_ms) = prepare_audio_input(&bytes, format)?;

    let threads = std::thread::available_parallelism()
        .map(|value| value.get().clamp(1, 8))
        .unwrap_or(4);
    // Compressed inputs (MP3/OGG/FLAC) have no decoded duration here, so give
    // them the maximum budget rather than killing a long recording early.
    // Decoded WAV scales with its real duration up to 30 minutes.
    let timeout_secs = if audio_ms == 0 {
        1_800
    } else {
        ((audio_ms as f64 / 1000.0 * 3.0) + 120.0).clamp(180.0, 1_800.0) as u64
    };
    let timeout = Duration::from_secs(timeout_secs);

    let mut command = tokio::process::Command::new(&binary);
    command
        .arg("-m")
        .arg(&model_path)
        .arg("-f")
        .arg(&temp.path)
        .arg("-l")
        .arg(&language)
        .arg("-t")
        .arg(threads.to_string())
        .arg("-nt")
        .arg("-np")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    apply_whisper_environment(&mut command, &libs);

    let child = command
        .spawn()
        .map_err(|err| whisper_spawn_error(&binary, &libs, err))?;
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| {
            AppError::message(format!(
                "Whisper did not finish within {} seconds. Try a smaller model or a shorter recording.",
                timeout.as_secs()
            ))
        })?
        .map_err(|err| {
            AppError::message(format!("Could not run whisper-cli: {err}"))
        })?;

    if !output.status.success() {
        let code = output
            .status
            .code()
            .map(|code| format!("code {code}"))
            .unwrap_or_else(|| "a signal".into());
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = truncate_line(stderr.trim(), 500);
        return Err(AppError::message(format!(
            "whisper-cli failed with {code}. {}",
            if detail.is_empty() {
                "It produced no diagnostics."
            } else {
                &detail
            }
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let text = clean_transcript(&stdout, strip_non_speech);
    if text.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = truncate_line(stderr.trim(), 300);
        return Err(AppError::message(if detail.is_empty() {
            "Whisper returned no speech. Try again closer to the microphone.".into()
        } else {
            format!("Whisper returned no speech. {detail}")
        }));
    }

    Ok(TranscriptionResult {
        text,
        duration_ms: started.elapsed().as_millis() as u64,
        model_used: spec.id.into(),
        audio_ms,
    })
}

#[cfg(target_os = "linux")]
fn apply_whisper_environment(command: &mut tokio::process::Command, libs: &Path) {
    let mut paths = vec![libs.to_path_buf()];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        command.env("LD_LIBRARY_PATH", joined);
    }
}

#[cfg(target_os = "macos")]
fn apply_whisper_environment(command: &mut tokio::process::Command, libs: &Path) {
    let mut paths = vec![libs.to_path_buf()];
    if let Some(existing) = std::env::var_os("DYLD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        command.env("DYLD_LIBRARY_PATH", joined);
    }
}

#[cfg(windows)]
fn apply_whisper_environment(command: &mut tokio::process::Command, libs: &Path) {
    let mut paths = vec![libs.to_path_buf()];
    if let Some(existing) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        command.env("PATH", joined);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn apply_whisper_environment(_command: &mut tokio::process::Command, _libs: &Path) {}

fn whisper_spawn_error(binary: &Path, libs: &Path, err: std::io::Error) -> AppError {
    match err.kind() {
        std::io::ErrorKind::NotFound => AppError::message(format!(
            "{} could not start because a shared library is missing. Reinstall whisper-cli from Settings → Voice so libwhisper*/libggml* are placed in {}.",
            binary.display(),
            libs.display()
        )),
        std::io::ErrorKind::PermissionDenied => AppError::message(format!(
            "{} is not executable. Reinstall it from Settings → Voice.",
            binary.display()
        )),
        _ if err.raw_os_error() == Some(8) => AppError::message(format!(
            "{} was built for a different CPU architecture. Reinstall whisper-cli from Settings → Voice.",
            binary.display()
        )),
        _ => AppError::message(format!(
            "Could not execute {}: {err}. Reinstall whisper-cli from Settings → Voice.",
            binary.display()
        )),
    }
}

/// Probe whisper-cli with the same library environment used for transcription.
/// `engines::run_cli_probe` cannot set that environment on Windows, where the
/// managed DLL folder has to be added to PATH explicitly.
async fn probe_whisper_version(binary: &Path, libs: &Path) -> AppResult<Option<String>> {
    let mut command = tokio::process::Command::new(binary);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    apply_whisper_environment(&mut command, libs);
    let output = tokio::time::timeout(Duration::from_secs(8), command.output())
        .await
        .map_err(|_| {
            AppError::message(format!(
                "{} --version did not finish within 8 seconds.",
                binary.display()
            ))
        })?
        .map_err(|err| whisper_spawn_error(binary, libs, err))?;

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        if !text.trim().is_empty() {
            text.push('\n');
        }
        text.push_str(&stderr);
    }
    let version = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .ok_or_else(|| {
            AppError::message(format!(
                "{} reported no version output.",
                binary.display()
            ))
        })?
        .to_string();
    if !output.status.success() && !text.to_ascii_lowercase().contains("version") {
        return Err(AppError::message(format!(
            "{} did not report a version: {}",
            binary.display(),
            truncate_line(text.trim(), 300)
        )));
    }
    Ok(Some(version))
}

fn normalise_language(raw: &str) -> AppResult<String> {
    let value = raw.trim().to_ascii_lowercase();
    if value.is_empty() || value == "auto" {
        return Ok("auto".into());
    }
    if value.len() <= 10
        && value
            .chars()
            .all(|character| character.is_ascii_lowercase() || character == '-')
    {
        Ok(value)
    } else {
        Err(AppError::message(format!(
            "Unknown Whisper language \"{raw}\". Use \"auto\" or a two-letter language code such as \"en\"."
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioFormat {
    Wav,
    Pcm16,
    Mp3,
    Ogg,
    Flac,
    WebM,
    Unsupported,
}

impl AudioFormat {
    fn detect(bytes: &[u8], mime: Option<&str>, file_name: Option<&str>) -> Self {
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
            return Self::Wav;
        }
        if bytes.len() >= 4 && &bytes[0..4] == b"OggS" {
            return Self::Ogg;
        }
        if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
            return Self::Flac;
        }
        if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
            return Self::Mp3;
        }
        if bytes.len() >= 2 && bytes[0] == 0xff && (bytes[1] & 0xe0) == 0xe0 {
            return Self::Mp3;
        }
        if bytes.len() >= 4 && bytes[0..4] == [0x1a, 0x45, 0xdf, 0xa3] {
            return Self::WebM;
        }

        let extension = file_name
            .and_then(|name| Path::new(name).extension())
            .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let mime = mime.unwrap_or_default().to_ascii_lowercase();
        match extension.as_str() {
            "wav" => return Self::Wav,
            "mp3" => return Self::Mp3,
            "ogg" | "oga" => return Self::Ogg,
            "flac" => return Self::Flac,
            "webm" => return Self::WebM,
            "pcm" | "raw" => return Self::Pcm16,
            _ => {}
        }
        match mime.as_str() {
            "audio/wav" | "audio/wave" | "audio/x-wav" => Self::Wav,
            "audio/mpeg" | "audio/mp3" => Self::Mp3,
            "audio/ogg" | "application/ogg" => Self::Ogg,
            "audio/flac" | "audio/x-flac" => Self::Flac,
            "audio/webm" | "video/webm" => Self::WebM,
            "audio/pcm" | "audio/raw" | "audio/l16" => Self::Pcm16,
            _ if mime.is_empty() || mime == "application/octet-stream" => {
                if bytes.len() >= 2 && bytes.len() % 2 == 0 {
                    Self::Pcm16
                } else {
                    Self::Unsupported
                }
            }
            _ => Self::Unsupported,
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Pcm16 => "pcm",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
            Self::Flac => "flac",
            Self::WebM => "webm",
            Self::Unsupported => "bin",
        }
    }
}

fn prepare_audio_input(bytes: &[u8], format: AudioFormat) -> AppResult<(TempAudio, u64)> {
    match format {
        AudioFormat::Wav => {
            let (samples, sample_rate, _channels) = decode_wav(bytes)?;
            let mono = resample_mono_16k(&samples, sample_rate)?;
            let wav = encode_wav_pcm16(&mono);
            let temp = TempAudio::new("wav")?;
            temp.write(&wav)?;
            Ok((temp, duration_ms(mono.len())))
        }
        AudioFormat::Pcm16 => {
            if bytes.len() < 2 || bytes.len() % 2 != 0 {
                return Err(AppError::message(
                    "The raw PCM recording has an odd byte length; it must be 16-bit little-endian mono samples.",
                ));
            }
            if bytes.len() / 2 > MAX_DECODED_SAMPLES {
                return Err(AppError::message(
                    "The raw PCM recording is longer than 30 minutes.",
                ));
            }
            let samples: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
                .collect();
            let wav = encode_wav_pcm16(&samples);
            let temp = TempAudio::new("wav")?;
            temp.write(&wav)?;
            Ok((temp, duration_ms(samples.len())))
        }
        AudioFormat::Mp3 | AudioFormat::Ogg | AudioFormat::Flac => {
            // whisper.cpp's release builds decode these with their bundled
            // miniaudio path, so no Rust codec dependency is required.
            let temp = TempAudio::new(format.extension())?;
            temp.write(bytes)?;
            Ok((temp, 0))
        }
        AudioFormat::WebM => Err(AppError::message(
            "WebM/Opus audio is not decoded by whisper.cpp. The built-in microphone recorder sends 16 kHz mono WAV directly; re-record with the mic button or attach a .wav, .mp3, .ogg or .flac file.",
        )),
        AudioFormat::Unsupported => Err(AppError::message(
            "The audio format is not supported. Use 16 kHz WAV, raw 16-bit PCM, MP3, OGG or FLAC.",
        )),
    }
}

fn duration_ms(samples: usize) -> u64 {
    (samples as u64).saturating_mul(1000) / WHISPER_RATE as u64
}

/// Decode PCM WAV into mono float samples. Supports the formats a browser or
/// desktop recorder can emit: 8/16/24/32-bit integer PCM and 32/64-bit float.
pub(crate) fn decode_wav(bytes: &[u8]) -> AppResult<(Vec<f32>, u32, u16)> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AppError::message(
            "The WAV recording is missing its RIFF/WAVE header.",
        ));
    }

    let mut cursor = 12usize;
    let mut format: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        cursor += 8;
        let available = bytes.len().saturating_sub(cursor);
        let size = size.min(available);
        let chunk = &bytes[cursor..cursor + size];
        match id {
            b"fmt " => {
                if chunk.len() < 16 {
                    return Err(AppError::message("The WAV format chunk is truncated."));
                }
                let code = u16::from_le_bytes([chunk[0], chunk[1]]);
                let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
                let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let bits = u16::from_le_bytes([chunk[14], chunk[15]]);
                let mut effective = code;
                if code == 0xfffe && chunk.len() >= 26 {
                    effective = u16::from_le_bytes([chunk[24], chunk[25]]);
                }
                format = Some((effective, channels, sample_rate, bits));
            }
            b"data" => {
                data = Some(chunk);
            }
            _ => {}
        }
        cursor += size + (size & 1);
    }

    let (code, channels, sample_rate, bits) =
        format.ok_or_else(|| AppError::message("The WAV file has no format chunk."))?;
    let data = data.ok_or_else(|| AppError::message("The WAV file has no audio data chunk."))?;
    if channels == 0 {
        return Err(AppError::message("The WAV file declares zero channels."));
    }
    if sample_rate == 0 {
        return Err(AppError::message("The WAV file declares a zero sample rate."));
    }
    if channels > 8 {
        return Err(AppError::message(
            "The WAV file has too many channels (the limit is 8).",
        ));
    }

    let bytes_per_sample = match (code, bits) {
        (1, 8) => 1,
        (1, 16) => 2,
        (1, 24) => 3,
        (1, 32) => 4,
        (3, 32) => 4,
        (3, 64) => 8,
        _ => {
            return Err(AppError::message(format!(
                "Unsupported WAV encoding (format {code}, {bits} bits). Record with 16-bit PCM or attach an MP3, OGG or FLAC file."
            )))
        }
    };
    let frame_bytes = bytes_per_sample * channels as usize;
    if frame_bytes == 0 {
        return Err(AppError::message("The WAV file has an invalid frame size."));
    }
    let frames = data.len() / frame_bytes;
    if frames == 0 {
        return Err(AppError::message("The WAV file contains no samples."));
    }
    if frames > MAX_DECODED_SAMPLES * channels as usize {
        return Err(AppError::message("The recording is longer than 30 minutes."));
    }

    let mut mono = Vec::with_capacity(frames);
    for frame in data.chunks_exact(frame_bytes) {
        let mut sum = 0.0f32;
        for channel in 0..channels as usize {
            let offset = channel * bytes_per_sample;
            let sample = match (code, bits) {
                (1, 8) => (frame[offset] as f32 - 128.0) / 128.0,
                (1, 16) => {
                    i16::from_le_bytes([frame[offset], frame[offset + 1]]) as f32 / 32768.0
                }
                (1, 24) => {
                    let value = i32::from_le_bytes([
                        0,
                        frame[offset],
                        frame[offset + 1],
                        frame[offset + 2],
                    ]) >> 8;
                    value as f32 / 8_388_608.0
                }
                (1, 32) => {
                    i32::from_le_bytes([
                        frame[offset],
                        frame[offset + 1],
                        frame[offset + 2],
                        frame[offset + 3],
                    ]) as f32
                        / 2_147_483_648.0
                }
                (3, 32) => f32::from_le_bytes([
                    frame[offset],
                    frame[offset + 1],
                    frame[offset + 2],
                    frame[offset + 3],
                ]),
                (3, 64) => f64::from_le_bytes([
                    frame[offset],
                    frame[offset + 1],
                    frame[offset + 2],
                    frame[offset + 3],
                    frame[offset + 4],
                    frame[offset + 5],
                    frame[offset + 6],
                    frame[offset + 7],
                ]) as f32,
                _ => 0.0,
            };
            sum += sample;
        }
        mono.push((sum / channels as f32).clamp(-1.0, 1.0));
    }
    Ok((mono, sample_rate, channels))
}

/// Linear resampling is monotonic and artifact-free enough for 16 kHz speech;
/// it avoids another native dependency in the AppImage.
pub(crate) fn resample_mono_16k(samples: &[f32], sample_rate: u32) -> AppResult<Vec<f32>> {
    if sample_rate == 0 {
        return Err(AppError::message("The audio has a zero sample rate."));
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if sample_rate == WHISPER_RATE {
        if samples.len() > MAX_DECODED_SAMPLES {
            return Err(AppError::message("The recording is longer than 30 minutes."));
        }
        return Ok(samples.to_vec());
    }
    let output_len = (samples.len() as u128 * WHISPER_RATE as u128 / sample_rate as u128) as usize;
    if output_len == 0 {
        return Err(AppError::message("The recording is too short to transcribe."));
    }
    if output_len > MAX_DECODED_SAMPLES {
        return Err(AppError::message("The recording is longer than 30 minutes."));
    }

    let step = sample_rate as f64 / WHISPER_RATE as f64;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let position = index as f64 * step;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (position - left as f64) as f32;
        let value = samples[left] * (1.0 - fraction) + samples[right] * fraction;
        output.push(value.clamp(-1.0, 1.0));
    }
    Ok(output)
}

pub(crate) fn encode_wav_pcm16(samples: &[f32]) -> Vec<u8> {
    let data_len = samples.len().saturating_mul(2);
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&WHISPER_RATE.to_le_bytes());
    out.extend_from_slice(&(WHISPER_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }
    out
}

/// Whisper emits bracketed or parenthesised sound annotations for silence,
/// music and noise. They are useful in a transcript but not in a chat draft.
pub(crate) fn strip_non_speech_annotations(text: &str) -> String {
    const TOKENS: &[&str] = &[
        "blank_audio",
        "blank audio",
        "silence",
        "music",
        "applause",
        "laughter",
        "laughs",
        "noise",
        "inaudible",
        "sound",
        "sounds",
        "no speech",
        "background noise",
        "wind",
        "click",
        "clicking",
    ];
    let mut output = String::with_capacity(text.len());
    let mut buffer = String::new();
    let mut closer: Option<char> = None;
    for character in text.chars() {
        if let Some(expected) = closer {
            buffer.push(character);
            if character == expected {
                let inner = buffer
                    .trim_matches(|value| value == '[' || value == ']' || value == '(' || value == ')')
                    .trim()
                    .to_ascii_lowercase();
                let is_token = TOKENS.iter().any(|token| inner == *token);
                if !is_token {
                    output.push_str(&buffer);
                }
                buffer.clear();
                closer = None;
            } else if buffer.len() > 64 {
                // Not an annotation after all; flush so text is never trapped.
                output.push_str(&buffer);
                buffer.clear();
                closer = None;
            }
            continue;
        }
        match character {
            '[' => {
                closer = Some(']');
                buffer.push(character);
            }
            '(' => {
                closer = Some(')');
                buffer.push(character);
            }
            '♪' | '♫' => {}
            _ => output.push(character),
        }
    }
    if !buffer.is_empty() {
        output.push_str(&buffer);
    }
    output
}

pub(crate) fn clean_transcript(raw: &str, strip_non_speech: bool) -> String {
    let cleaned = if strip_non_speech {
        strip_non_speech_annotations(raw)
    } else {
        raw.to_string()
    };
    cleaned
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn asset(name: &str) -> GhAsset {
        GhAsset {
            name: name.into(),
            browser_download_url: format!("https://example.invalid/{name}"),
            size: 1,
        }
    }

    #[test]
    fn catalogue_ids_are_unique_and_checksums_are_hex() {
        let mut seen = HashSet::new();
        for spec in WHISPER_MODELS {
            assert!(seen.insert(spec.id), "duplicate model id {}", spec.id);
            assert_eq!(spec.sha256.len(), 64, "{} needs a SHA-256", spec.id);
            assert!(
                spec.sha256.chars().all(|value| value.is_ascii_hexdigit()),
                "{} has a non-hex digest",
                spec.id
            );
            assert!(spec.size_bytes > 0 && spec.ram_mb > 0);
        }
        assert!(model_spec("BASE.EN").is_ok());
        assert!(model_spec("../evil").is_err());
    }

    #[test]
    fn linux_assets_match_arch_and_fall_back_to_cpu() {
        let assets = vec![
            asset("whisper-bin-ubuntu-x64.tar.gz"),
            asset("whisper-bin-ubuntu-arm64.tar.gz"),
            asset("whisper-bin-win-cuda-12.4.0-x64.zip"),
            asset("whisper-blas-bin-x64.zip"),
        ];
        let chosen = select_asset(&assets, "linux", "x86_64", "cpu").expect("linux x64 asset");
        assert_eq!(chosen.name, "whisper-bin-ubuntu-x64.tar.gz");
        let chosen =
            select_asset(&assets, "linux", "aarch64", "cpu").expect("linux arm64 asset");
        assert_eq!(chosen.name, "whisper-bin-ubuntu-arm64.tar.gz");
        // A host without a matching CUDA archive must still get the CPU build.
        let chosen = select_asset(&assets, "linux", "x86_64", "cuda").expect("cpu fallback");
        assert_eq!(chosen.name, "whisper-bin-ubuntu-x64.tar.gz");
        let chosen = select_asset(&assets, "windows", "x86_64", "cuda").expect("windows cuda");
        assert_eq!(chosen.name, "whisper-bin-win-cuda-12.4.0-x64.zip");
    }

    #[test]
    fn windows_cpu_assets_do_not_pick_cuda_or_blas() {
        let assets = vec![
            asset("whisper-bin-win-cuda-12.4.0-x64.zip"),
            asset("whisper-bin-x64.zip"),
            asset("whisper-blas-bin-x64.zip"),
        ];
        let chosen = select_asset(&assets, "windows", "x86_64", "cpu").expect("cpu asset");
        assert_eq!(chosen.name, "whisper-bin-x64.zip");
        assert!(select_asset(&assets, "macos", "aarch64", "metal").is_none());
    }

    #[test]
    fn wav_round_trip_downmixes_and_resamples() {
        // 0.5 s of a 440 Hz tone at 48 kHz, stereo float32.
        let rate = 48_000u32;
        let frames = rate as usize / 2;
        let mut wav = Vec::new();
        let data_len = frames * 2 * 4;
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * 8).to_le_bytes());
        wav.extend_from_slice(&8u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        for frame in 0..frames {
            let value = (2.0 * std::f32::consts::PI * 440.0 * frame as f32 / rate as f32).sin();
            wav.extend_from_slice(&value.to_le_bytes());
            wav.extend_from_slice(&value.to_le_bytes());
        }

        let (mono, decoded_rate, channels) = decode_wav(&wav).expect("decode float wav");
        assert_eq!(decoded_rate, 48_000);
        assert_eq!(channels, 2);
        assert_eq!(mono.len(), frames);
        let resampled = resample_mono_16k(&mono, decoded_rate).expect("resample");
        assert_eq!(resampled.len(), 8_000);
        assert!(resampled.iter().all(|value| value.is_finite()));

        let encoded = encode_wav_pcm16(&resampled);
        assert_eq!(encoded.len(), 44 + resampled.len() * 2);
        let (again, rate, channels) = decode_wav(&encoded).expect("re-decode");
        assert_eq!(rate, 16_000);
        assert_eq!(channels, 1);
        assert_eq!(again.len(), resampled.len());
    }

    #[test]
    fn raw_pcm_is_wrapped_as_16k_mono_wav() {
        let mut pcm = Vec::new();
        for frame in 0..1_600i16 {
            pcm.extend_from_slice(&(frame * 8).to_le_bytes());
        }
        let (temp, audio_ms) =
            prepare_audio_input(&pcm, AudioFormat::Pcm16).expect("wrap raw pcm");
        assert_eq!(audio_ms, 100);
        let bytes = fs::read(&temp.path).expect("read temporary wav");
        let (samples, rate, channels) = decode_wav(&bytes).expect("decode wrapped wav");
        assert_eq!(rate, 16_000);
        assert_eq!(channels, 1);
        assert_eq!(samples.len(), 1_600);
    }

    #[test]
    fn transcript_cleanup_drops_annotations_and_extra_whitespace() {
        let raw = "\n [BLANK_AUDIO]  And so my fellow Americans...\n [Music]\n ask not (silence) what your country can do for you.";
        let cleaned = clean_transcript(raw, true);
        assert_eq!(
            cleaned,
            "And so my fellow Americans... ask not what your country can do for you."
        );
        let kept = clean_transcript("[Music] hello", false);
        assert_eq!(kept, "[Music] hello");
        assert_eq!(clean_transcript("[BLANK_AUDIO]", true), "");
    }

    #[test]
    fn language_validation_accepts_codes_and_auto() {
        assert_eq!(normalise_language(" EN ").expect("en"), "en");
        assert_eq!(normalise_language("auto").expect("auto"), "auto");
        assert_eq!(normalise_language("").expect("empty"), "auto");
        assert!(normalise_language("en_US;rm -rf /").is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_install_keeps_a_working_backup_and_verifies_the_cli() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!("janas-whisper-install-{}", Uuid::new_v4()));
        let bin = root.join("bin");
        let runtime = bin.join(RUNTIME_DIR_NAME);
        let libs = runtime.clone();
        let target = runtime.join(whisper_binary_name());
        fs::create_dir_all(&bin).expect("create bin");

        let first = root.join("first");
        fs::write(&first, "#!/bin/sh\nprintf 'whisper.cpp version: first\\n'\n")
            .expect("write first binary");
        fs::set_permissions(&first, fs::Permissions::from_mode(0o755)).unwrap();
        let first_lib = root.join("libwhisper.so.1");
        fs::write(&first_lib, b"first library").unwrap();
        let prepared = prepare_binary_install(&target, &libs, &first, &[first_lib])
            .expect("install first binary");
        assert_eq!(fs::read_to_string(&prepared.target).unwrap(), fs::read_to_string(&first).unwrap());
        assert!(prepared.backup.is_none());
        assert_eq!(fs::read(libs.join("libwhisper.so.1")).unwrap(), b"first library");
        let probe = probe_whisper_version(&prepared.target, &libs)
            .await
            .expect("probe first");
        assert_eq!(probe.as_deref(), Some("whisper.cpp version: first"));

        let second = root.join("second");
        fs::write(&second, "#!/bin/sh\nprintf 'whisper.cpp version: second\\n'\n")
            .expect("write second binary");
        fs::set_permissions(&second, fs::Permissions::from_mode(0o755)).unwrap();
        let second_lib = root.join("libwhisper.so.1");
        fs::write(&second_lib, b"second library").unwrap();
        let prepared = prepare_binary_install(&target, &libs, &second, &[second_lib])
            .expect("install second binary");
        let backup = prepared.backup.clone().expect("backup exists after an update");
        assert_eq!(
            fs::read_to_string(backup.join(whisper_binary_name())).unwrap(),
            fs::read_to_string(&first).unwrap()
        );
        assert_eq!(fs::read(backup.join("libwhisper.so.1")).unwrap(), b"first library");
        assert_eq!(fs::read(libs.join("libwhisper.so.1")).unwrap(), b"second library");

        restore_runtime(&prepared.libs, Some(&backup))
            .expect("restore previous install");
        assert_eq!(fs::read_to_string(&target).unwrap(), fs::read_to_string(&first).unwrap());
        assert_eq!(fs::read(libs.join("libwhisper.so.1")).unwrap(), b"first library");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sha256_file_matches_a_known_digest() {
        let path = std::env::temp_dir().join(format!("janas-whisper-sha-{}", Uuid::new_v4()));
        fs::write(&path, b"abc").expect("write digest fixture");
        assert_eq!(
            sha256_file(&path).expect("hash fixture"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn audio_format_detection_prefers_magic_bytes() {
        assert_eq!(
            AudioFormat::detect(b"RIFF\0\0\0\0WAVE", Some("audio/webm"), None),
            AudioFormat::Wav
        );
        assert_eq!(
            AudioFormat::detect(b"OggS\0", Some("audio/ogg"), None),
            AudioFormat::Ogg
        );
        assert_eq!(
            AudioFormat::detect(b"\x1a\x45\xdf\xa3rest", Some("audio/webm"), None),
            AudioFormat::WebM
        );
        assert_eq!(
            AudioFormat::detect(&[0, 1, 2, 3], Some("audio/mpeg"), None),
            AudioFormat::Mp3
        );
        assert_eq!(
            AudioFormat::detect(&[0, 1], None, Some("take.pcm")),
            AudioFormat::Pcm16
        );
        assert_eq!(
            AudioFormat::detect(&[0, 1], None, Some("take.flac")),
            AudioFormat::Flac
        );
    }
}
