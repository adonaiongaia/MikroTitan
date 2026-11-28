use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{OnceLock, RwLock},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};

/// Everything the studio needs to find and launch the Janas tools. Stored as
/// JSON in the platform config directory and editable from the Settings view.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// A full path, or a name looked up on PATH (e.g. "janas-server").
    pub engine_path: String,
    /// "janas-convert" for the studio's converter contract, or "gguf2jns" which
    /// upstream Janas ships. The launcher adapts the arguments automatically.
    pub converter_path: String,
    pub host: String,
    pub port: u16,
    /// Where downloaded and converted models are kept.
    pub models_dir: String,
    /// The .jns file handed to janas-server on launch.
    pub model_path: String,
    /// Additional janas-server arguments, parsed with shell-word rules.
    pub extra_args: String,
    /// Optional `--api-key` for the local server. Never leaves this machine.
    pub api_key: String,
    /// Optional Hugging Face access token for gated or private repositories.
    /// It is only sent to huggingface.co, never logged, and stored locally.
    pub hf_token: String,
    /// `--ctx`: context window in tokens.
    pub ctx_size: u32,
    /// `--cache`: GiB reserved for streamed experts. 0 means automatic.
    pub cache_gib: f32,
    /// `--mtp` / `--no-mtp`: "auto", "off" or "file".
    pub mtp_mode: String,
    /// Optional MTP block file when `mtp_mode` is "file".
    pub mtp_path: String,
    /// `--draft`: a small model of the same family used for speculation.
    pub draft_path: String,
    /// `--prompt-on`: auto, cpu, gpu, npu, npu+gpu, or a comma-separated list.
    pub prompt_on: String,
    /// `--reply-on`: where reply tokens are computed.
    pub reply_on: String,
    /// `--drafts-on`: where speculative passes are checked.
    pub drafts_on: String,
    /// `--think`: auto, on or off.
    pub think: String,
    /// `--npu-int8`: eight-bit activations for NPU prompt reading.
    pub npu_int8: bool,
    /// `--mode`: auto, eco or max.
    pub engine_mode: String,
    /// `JANAS_ATTN`: auto, exact (float) or fast (16-bit query).
    pub janas_attention: String,
    /// `JANAS_KV`: auto, 8 or 16 bits for both keys and values.
    pub janas_kv_cache: String,
    /// A full path to a .gguf model for llama-server.
    pub llama_model_path: String,
    /// Optional `--mmproj` GGUF. llama.cpp needs this vision projector to
    /// accept image content parts in chat requests.
    pub llama_mmproj_path: String,
    /// llama.cpp only: `--spec-type draft-mtp` uses the multi-token-prediction
    /// heads that ship with the selected model. "auto" leaves the build's own
    /// default (no speculation), "off" passes `--spec-type none`.
    pub llama_mtp_mode: String,
    /// llama.cpp only: MTP heads that live in a separate GGUF. It is handed to
    /// llama.cpp as the speculative draft model alongside `draft-mtp`.
    pub llama_mtp_path: String,
    /// llama.cpp only: `--spec-draft-model`, a smaller same-family GGUF used
    /// for speculative decoding.
    pub llama_draft_path: String,
    /// `--draft-max`: how many tokens the draft model proposes per step.
    pub llama_draft_max: u32,
    /// `--draft-min`: the shortest draft run accepted for a speculative step.
    pub llama_draft_min: u32,
    /// `--draft-p-min`: the minimum draft probability llama.cpp keeps.
    pub llama_draft_p_min: f32,
    /// llama.cpp only: where the draft model runs. "auto", "cpu"
    /// (`--spec-draft-device none`) or "gpu" (`--spec-draft-ngl all`).
    pub llama_drafts_on: String,
    /// llama-server's loopback port. Kept separate from Janas's port so the
    /// two engines can be switched without editing the other configuration.
    pub llama_port: u16,
    /// `--n-gpu-layers`: 0 keeps everything on the CPU. Values are capped to
    /// the selected GGUF's block count when its metadata can be read; 999 is
    /// the fallback that asks llama.cpp to offload every layer.
    pub llama_gpu_layers: i32,
    /// `--ctx-size` for llama-server.
    pub llama_ctx_size: u32,
    /// `--flash-attn`: auto, on or off.
    pub llama_flash_attn: String,
    /// `--cache-type-k`: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0 or q5_1.
    pub llama_kv_cache_k: String,
    /// `--cache-type-v`: the same set, with f16 as the safe default.
    pub llama_kv_cache_v: String,
    /// llama.cpp only: run MoE cache misses on host experts through the
    /// llama_moe_cache pipeline instead of uploading every miss to the GPU.
    /// janas-server never reads this setting.
    pub llama_direct_cpu_moe: bool,
    /// llama.cpp only: maximum CPU threads for direct MoE compute. Zero keeps
    /// llama.cpp's own default. janas-server never reads this setting.
    pub llama_cpu_threads: u32,
    /// Additional llama-server flags, parsed with shell-word rules.
    pub llama_extra_args: String,
    /// Which llama.cpp release automatic setup installs. "baseline" pins the
    /// tested `KNOWN_GOOD_LLAMA_TAG` (or a `compat.json` recommendation),
    /// "latest" tracks upstream releases.
    pub llama_update_channel: String,
    /// Not persisted: the GPU rung the supervisor selected for this launch.
    /// "auto" is the requested full-GPU launch, "host-kv" retries with
    /// `-nkvo`, "cpu" retries with `--device none`.
    #[serde(skip)]
    pub llama_launch_step: String,
    /// Janas: request/keep the SSD streaming path when the build advertises it.
    pub ssd_streaming: bool,
    /// Janas: optional SSD cache directory, when the build advertises it.
    pub ssd_cache_dir: String,
    /// Server-side default Top-K, mirrored from the sampling panel.
    pub top_k: u32,
    /// Saved (main, projector, draft) bundles. The library can activate one
    /// in a click and the engine loads every component it names.
    pub model_bundles: Vec<crate::companions::ModelBundle>,
    /// The bundle the user last activated.
    pub active_bundle_id: String,
    /// Curated whisper.cpp model id used for voice input ("base.en", ...).
    pub whisper_model: String,
    /// whisper-cli `-l` value: "auto" or a language code such as "en".
    pub whisper_language: String,
    /// Show the microphone in the chat composer.
    pub voice_input_enabled: bool,
    /// "toggle" (click to start/stop) or "hold" (push-to-talk).
    pub voice_activation: String,
    /// Send a finished transcript immediately instead of inserting it.
    pub voice_auto_send: bool,
    /// Drop Whisper annotations such as [BLANK_AUDIO] or (music).
    pub voice_strip_non_speech: bool,
    /// Expose the loopback OpenAI-compatible gateway to external clients.
    pub gateway_enabled: bool,
    /// The gateway's loopback port. Kept separate from both engine ports so a
    /// client configuration never follows the engine across a switch.
    pub gateway_port: u16,
    /// Let an incoming `model` name switch the engine and model.
    pub gateway_auto_switch_models: bool,
    /// Add Memory Bank facts to external gateway requests.
    pub gateway_inject_memory: bool,
    /// Document Vault collections injected into external gateway requests.
    pub gateway_default_rag_collections: Vec<String>,
    /// Per-launch environment overrides. This is deliberately not serialized:
    /// the benchmark runner puts `JANAS_ATTN=float` and similar target flags on
    /// a cloned AppSettings so a measurement never rewrites saved settings.
    #[serde(skip)]
    pub launch_env: Vec<(String, String)>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            engine_path: "janas-server".into(),
            converter_path: "janas-convert".into(),
            host: "127.0.0.1".into(),
            port: 8080,
            models_dir: String::new(),
            model_path: String::new(),
            extra_args: String::new(),
            api_key: String::new(),
            hf_token: String::new(),
            ctx_size: 16_384,
            cache_gib: 0.0,
            mtp_mode: "auto".into(),
            mtp_path: String::new(),
            draft_path: String::new(),
            prompt_on: "auto".into(),
            reply_on: "auto".into(),
            drafts_on: "auto".into(),
            think: "auto".into(),
            npu_int8: false,
            engine_mode: "auto".into(),
            janas_attention: "auto".into(),
            janas_kv_cache: "auto".into(),
            llama_model_path: String::new(),
            llama_mmproj_path: String::new(),
            llama_mtp_mode: "auto".into(),
            llama_mtp_path: String::new(),
            llama_draft_path: String::new(),
            llama_draft_max: 8,
            llama_draft_min: 0,
            llama_draft_p_min: 0.75,
            llama_drafts_on: "auto".into(),
            llama_port: 8081,
            llama_gpu_layers: 0,
            llama_ctx_size: 16_384,
            llama_flash_attn: "auto".into(),
            llama_kv_cache_k: "f16".into(),
            llama_kv_cache_v: "f16".into(),
            llama_direct_cpu_moe: false,
            llama_cpu_threads: 0,
            llama_extra_args: String::new(),
            llama_update_channel: "baseline".into(),
            llama_launch_step: "auto".into(),
            ssd_streaming: false,
            ssd_cache_dir: String::new(),
            top_k: 40,
            model_bundles: Vec::new(),
            active_bundle_id: String::new(),
            whisper_model: "base.en".into(),
            whisper_language: "auto".into(),
            voice_input_enabled: false,
            voice_activation: "toggle".into(),
            voice_auto_send: false,
            voice_strip_non_speech: true,
            gateway_enabled: false,
            gateway_port: DEFAULT_GATEWAY_PORT,
            gateway_auto_switch_models: true,
            gateway_inject_memory: false,
            gateway_default_rag_collections: Vec::new(),
            launch_env: Vec::new(),
        }
    }
}

impl AppSettings {
    pub fn connect_host(&self) -> &str {
        match self.host.trim() {
            "" | "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
            other => other,
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.connect_host(), self.port)
    }

    pub fn model_display_name(&self) -> Option<String> {
        Path::new(self.model_path.trim())
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .filter(|stem| !stem.is_empty())
    }

    pub fn llama_model_display_name(&self) -> Option<String> {
        Path::new(self.llama_model_path.trim())
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .filter(|stem| !stem.is_empty())
    }

    /// Keep deserialized files from older builds in a range the adapters can
    /// safely turn into command-line arguments.
    pub fn normalize(&mut self) {
        self.port = self.port.clamp(1, 65_535);
        self.llama_port = self.llama_port.clamp(1, 65_535);
        self.llama_gpu_layers = self.llama_gpu_layers.clamp(0, 999);
        self.llama_ctx_size = self.llama_ctx_size.clamp(512, 1_048_576);
        self.janas_attention = normalize_janas_attention(&self.janas_attention);
        self.janas_kv_cache = normalize_janas_kv_cache(&self.janas_kv_cache);
        self.mtp_mode = normalize_janas_mtp_mode(&self.mtp_mode);
        self.llama_mtp_mode = normalize_llama_mtp_mode(&self.llama_mtp_mode);
        self.llama_drafts_on = normalize_draft_placement(&self.llama_drafts_on);
        self.mtp_path = self.mtp_path.trim().to_string();
        self.draft_path = self.draft_path.trim().to_string();
        self.llama_mtp_path = self.llama_mtp_path.trim().to_string();
        self.llama_draft_path = self.llama_draft_path.trim().to_string();
        self.llama_draft_max = self.llama_draft_max.clamp(1, 64);
        self.llama_draft_min = self.llama_draft_min.min(self.llama_draft_max);
        if !self.llama_draft_p_min.is_finite() {
            self.llama_draft_p_min = 0.75;
        }
        self.llama_draft_p_min = self.llama_draft_p_min.clamp(0.0, 1.0);
        self.active_bundle_id = self.active_bundle_id.trim().to_string();
        self.llama_flash_attn = normalize_flash_attn(&self.llama_flash_attn);
        self.llama_kv_cache_k = normalize_kv_cache(&self.llama_kv_cache_k);
        self.llama_kv_cache_v = normalize_kv_cache(&self.llama_kv_cache_v);
        self.llama_cpu_threads = self.llama_cpu_threads.min(256);
        self.llama_model_path = self.llama_model_path.trim().to_string();
        self.llama_mmproj_path = self.llama_mmproj_path.trim().to_string();
        for bundle in &mut self.model_bundles {
            bundle.main_model_path = bundle.main_model_path.trim().to_string();
            bundle.mmproj_path = bundle.mmproj_path.trim().to_string();
            bundle.draft_model_path = bundle.draft_model_path.trim().to_string();
            bundle.draft_max_tokens = bundle.draft_max_tokens.clamp(1, 64);
            bundle.draft_min_tokens = bundle.draft_min_tokens.min(bundle.draft_max_tokens);
            if !bundle.draft_p_min.is_finite() {
                bundle.draft_p_min = 0.75;
            }
            bundle.draft_p_min = bundle.draft_p_min.clamp(0.0, 1.0);
        }
        // A deleted model must not leave a dangling active bundle pointer.
        if !self.active_bundle_id.is_empty()
            && !self
                .model_bundles
                .iter()
                .any(|bundle| bundle.id == self.active_bundle_id)
        {
            self.active_bundle_id.clear();
        }
        self.llama_extra_args = self.llama_extra_args.trim().to_string();
        self.llama_update_channel = normalize_llama_update_channel(&self.llama_update_channel);
        self.llama_launch_step = normalize_llama_launch_step(&self.llama_launch_step);
        self.ssd_cache_dir = self.ssd_cache_dir.trim().to_string();
        self.hf_token = self.hf_token.trim().to_string();
        // Tokens never contain whitespace. Strip line breaks and spaces from a
        // pasted value so the HTTP header builder cannot reject it.
        self.hf_token
            .retain(|character| !character.is_whitespace() && !character.is_control());
        self.top_k = self.top_k.clamp(1, 100);
        self.whisper_model = self.whisper_model.trim().to_ascii_lowercase();
        if self.whisper_model.is_empty() {
            self.whisper_model = "base.en".into();
        }
        self.whisper_language = self.whisper_language.trim().to_ascii_lowercase();
        if self.whisper_language.is_empty() {
            self.whisper_language = "auto".into();
        }
        self.voice_activation = match self.voice_activation.trim().to_ascii_lowercase().as_str() {
            "hold" => "hold".into(),
            _ => "toggle".into(),
        };
        if self.gateway_port == 0 {
            self.gateway_port = DEFAULT_GATEWAY_PORT;
        }
        self.gateway_port = self.gateway_port.clamp(1, 65_535);
        let mut collections: Vec<String> = Vec::new();
        for id in self.gateway_default_rag_collections.drain(..) {
            let id = id.trim();
            if !id.is_empty() && !collections.iter().any(|existing| existing == id) {
                collections.push(id.to_string());
            }
            if collections.len() >= 64 {
                break;
            }
        }
        self.gateway_default_rag_collections = collections;
    }

    /// Align the offload setting with the layers the selected model actually
    /// contains. This is deliberately best-effort: an unreadable model must
    /// never prevent unrelated settings from saving.
    pub fn clamp_gpu_layers_to_model(&mut self) {
        if self.llama_gpu_layers <= 0 {
            return;
        }
        if let Some(layers) = crate::gguf::layer_count(&self.llama_model_path) {
            self.llama_gpu_layers = self
                .llama_gpu_layers
                .min(layers.min(i32::MAX as u32) as i32);
        }
    }
}

/// The default loopback port for the external OpenAI-compatible gateway. It is
/// intentionally not 8080/8081, so it never collides with either engine.
pub const DEFAULT_GATEWAY_PORT: u16 = 11_435;

pub const KV_CACHE_TYPES: [&str; 9] = [
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

pub fn normalize_flash_attn(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "on" => "on".into(),
        "off" => "off".into(),
        _ => "auto".into(),
    }
}

/// Janas keeps its own three-way `--mtp` switch: auto, `--no-mtp`, or an
/// explicit MTP block file.
pub fn normalize_janas_mtp_mode(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" | "no" => "off".into(),
        "file" => "file".into(),
        _ => "auto".into(),
    }
}

/// llama.cpp enables MTP through `--spec-type draft-mtp`. A stored Janas-style
/// "file" value still means "use MTP" here, because llama.cpp takes the MTP
/// heads as its speculative draft model.
pub fn normalize_llama_mtp_mode(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" | "no" => "off".into(),
        "mtp" | "file" | "on" | "draft-mtp" => "mtp".into(),
        _ => "auto".into(),
    }
}

/// Which upstream release the automatic setup installs. Anything unrecognized
/// (including a settings file written before this field existed) stays on the
/// tested baseline instead of silently jumping to a daily build.
pub fn normalize_llama_update_channel(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "latest" | "upstream" | "nightly" => "latest".into(),
        _ => "baseline".into(),
    }
}

/// The GPU fallback rung is chosen by the supervisor, never by the user, so
/// anything unfamiliar falls back to the requested full-GPU launch.
pub fn normalize_llama_launch_step(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "host-kv" => "host-kv".into(),
        "cpu" => "cpu".into(),
        _ => "auto".into(),
    }
}

/// llama.cpp only moves a whole draft model, so its placement is a narrower
/// choice than Janas's `--drafts-on` device list.
pub fn normalize_draft_placement(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => "cpu".into(),
        "gpu" => "gpu".into(),
        _ => "auto".into(),
    }
}

pub fn normalize_kv_cache(value: &str) -> String {
    let lower = value.trim().to_ascii_lowercase();
    if KV_CACHE_TYPES.contains(&lower.as_str()) {
        lower
    } else {
        "f16".into()
    }
}

pub fn normalize_janas_attention(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "exact" | "float" => "exact".into(),
        "fast" | "int16" => "fast".into(),
        _ => "auto".into(),
    }
}

pub fn normalize_janas_kv_cache(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "8" | "8-bit" | "8bit" => "8".into(),
        "16" | "16-bit" | "16bit" => "16".into(),
        _ => "auto".into(),
    }
}

/// The bundle identifier the app shipped with before the MikroTitan rename.
/// Tauri derives the default config, data and cache folders from it.
pub const LEGACY_IDENTIFIER: &str = "dev.janas.studio";

/// The folder the previous identifier used beside `current`.
pub fn legacy_sibling(current: &Path) -> Option<PathBuf> {
    current
        .parent()
        .map(|parent| parent.join(LEGACY_IDENTIFIER))
}

/// Move the contents of the pre-rename profile into the renamed folder.
///
/// Entries the new profile already contains always win, so a second launch can
/// never overwrite a live profile. The old folder is removed only when every
/// entry moved, which keeps the migration non-destructive.
pub fn migrate_legacy_dir(current: &Path) -> bool {
    let Some(legacy) = legacy_sibling(current) else {
        return false;
    };
    if legacy == current || !legacy.is_dir() || fs::create_dir_all(current).is_err() {
        return false;
    }
    let Ok(entries) = fs::read_dir(&legacy) else {
        return false;
    };
    let mut moved = false;
    for entry in entries.flatten() {
        let target = current.join(entry.file_name());
        if target.exists() {
            // The webview may have already created an empty folder here.
            let reclaimable = target.is_dir()
                && fs::read_dir(&target)
                    .map(|mut entries| entries.next().is_none())
                    .unwrap_or(false)
                && fs::remove_dir(&target).is_ok();
            if !reclaimable {
                continue;
            }
        }
        if fs::rename(entry.path(), &target).is_ok() {
            moved = true;
        }
    }
    let _ = fs::remove_dir(&legacy);
    moved
}

fn remap_path(value: &mut String, from: &Path, to: &Path) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return;
    }
    let source = Path::new(trimmed);
    let Ok(rest) = source.strip_prefix(from) else {
        return;
    };
    // A path that still exists was intentionally left in the old profile (the
    // new one already had that entry), so keep pointing at the real files.
    if source.exists() {
        return;
    }
    *value = to.join(rest).to_string_lossy().into_owned();
}

impl AppSettings {
    /// Rewrite the absolute paths that pointed into the pre-rename data
    /// folder. Paths the user chose elsewhere are left untouched.
    pub fn remap_data_paths(&mut self, legacy_data_dir: &Path, data_dir: &Path) {
        if legacy_data_dir == data_dir {
            return;
        }
        for value in [
            &mut self.models_dir,
            &mut self.model_path,
            &mut self.llama_model_path,
            &mut self.llama_mmproj_path,
            &mut self.llama_mtp_path,
            &mut self.llama_draft_path,
            &mut self.mtp_path,
            &mut self.draft_path,
            &mut self.ssd_cache_dir,
        ] {
            remap_path(value, legacy_data_dir, data_dir);
        }
        for bundle in &mut self.model_bundles {
            for value in [
                &mut bundle.main_model_path,
                &mut bundle.mmproj_path,
                &mut bundle.draft_model_path,
            ] {
                remap_path(value, legacy_data_dir, data_dir);
            }
        }
    }
}

pub struct SettingsStore {
    path: PathBuf,
    settings: RwLock<AppSettings>,
}

impl SettingsStore {
    /// `legacy_data_dir` is the folder the previous bundle identifier used.
    /// Paths that profile saved are rewritten to `data_dir`.
    pub fn load(
        config_dir: PathBuf,
        data_dir: PathBuf,
        legacy_data_dir: Option<PathBuf>,
    ) -> AppResult<Self> {
        fs::create_dir_all(&config_dir)?;
        let default_models = data_dir.join("models");
        fs::create_dir_all(&default_models)?;

        let path = config_dir.join("settings.json");
        let mut defaults = AppSettings::default();
        defaults.models_dir = default_models.to_string_lossy().to_string();

        let mut settings = if path.exists() {
            match fs::read_to_string(&path)
                .ok()
                .and_then(|raw| serde_json::from_str::<AppSettings>(&raw).ok())
            {
                Some(stored) => stored,
                None => defaults,
            }
        } else {
            defaults
        };

        if let Some(legacy) = legacy_data_dir.as_deref() {
            settings.remap_data_paths(legacy, &data_dir);
        }

        if settings.models_dir.trim().is_empty() {
            settings.models_dir = default_models.to_string_lossy().to_string();
        }
        settings.normalize();
        settings.clamp_gpu_layers_to_model();
        if settings.host.trim().is_empty() {
            settings.host = "127.0.0.1".into();
        }
        let _ = fs::create_dir_all(&settings.models_dir);

        // A first run is friendlier when an already-converted model is picked
        // up automatically instead of asking the user to type a path.
        if settings.model_path.trim().is_empty() {
            if let Some(first) = scan_models(&settings.models_dir, None, 3).into_iter().next() {
                settings.model_path = first.path;
            }
        }

        let store = Self {
            path,
            settings: RwLock::new(settings),
        };
        let initial = store.get();
        let _ = store.persist(&initial);
        Ok(store)
    }

    pub fn get(&self) -> AppSettings {
        self.settings
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    pub fn save(&self, next: AppSettings) -> AppResult<AppSettings> {
        let mut clean = next;
        clean.normalize();
        clean.clamp_gpu_layers_to_model();
        clean.host = if clean.host.trim().is_empty() {
            "127.0.0.1".into()
        } else {
            clean.host.trim().to_string()
        };
        if clean.models_dir.trim().is_empty() {
            return Err(AppError::message("The models folder cannot be empty."));
        }
        fs::create_dir_all(clean.models_dir.trim())?;
        clean.models_dir = clean.models_dir.trim().to_string();

        {
            let mut guard = self
                .settings
                .write()
                .map_err(|_| AppError::message("Settings are locked by another operation."))?;
            self.persist(&clean)?;
            *guard = clean.clone();
        }
        Ok(clean)
    }

    fn persist(&self, settings: &AppSettings) -> AppResult<()> {
        let json = serde_json::to_string_pretty(settings)
            .map_err(|err| AppError::message(format!("Could not encode settings: {err}")))?;
        let tmp = self.path.with_extension("json.tmp");
        // The local API key and Hugging Face token live here, so the temporary
        // file is created owner-only before it is renamed into place.
        crate::util::write_private_temp(&tmp, json.as_bytes()).map_err(|err| {
            AppError::message(format!("Could not write the settings file: {err}"))
        })?;
        if fs::rename(&tmp, &self.path).is_err() {
            // Windows cannot rename over an existing file; write the existing
            // path directly and clean up the temporary file.
            let result = fs::write(&self.path, &json);
            let _ = fs::remove_file(&tmp);
            result.map_err(|err| {
                AppError::message(format!("Could not replace the settings file: {err}"))
            })?;
        }
        restrict_owner_only(&self.path);
        Ok(())
    }
}

/// Keep the settings file and its temporary copy readable only by the owner
/// where POSIX modes are available.
fn restrict_owner_only(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

pub fn expand_home(input: &str) -> PathBuf {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("~/") {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        return Path::new(&home).join(rest);
    }
    PathBuf::from(trimmed)
}

/// The app's managed `bin` directory, set once at startup. Tools downloaded
/// from the current Janas release are found here, so bare names work without
/// the user having to link them or add the folder to PATH.
static MANAGED_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn set_managed_dir(path: PathBuf) {
    let _ = MANAGED_DIR.set(path);
}

fn managed_binary(name: &str) -> Option<PathBuf> {
    let dir = MANAGED_DIR.get()?;
    let candidate = dir.join(name);
    candidate.is_file().then_some(candidate)
}

/// Resolve a binary specified either as a path or as a bare name on PATH.
pub fn resolve_binary(input: &str) -> Option<PathBuf> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = expand_home(trimmed);
    if candidate.components().count() > 1 || candidate.is_absolute() {
        return candidate.is_file().then_some(candidate);
    }

    let path_var = std::env::var_os("PATH")?;
    let mut names = vec![trimmed.to_string()];
    if cfg!(windows) && !trimmed.to_ascii_lowercase().ends_with(".exe") {
        names.push(format!("{trimmed}.exe"));
    }
    for dir in std::env::split_paths(&path_var) {
        for name in &names {
            let full = dir.join(name);
            if full.is_file() {
                return Some(full);
            }
        }
    }
    for name in &names {
        if let Some(full) = managed_binary(name) {
            return Some(full);
        }
    }
    None
}

#[derive(Debug, Clone, Serialize)]
pub struct BinaryProbe {
    pub ok: bool,
    pub resolved_path: Option<String>,
    pub version: Option<String>,
    pub message: String,
}

/// Run `<binary> --version` with a short deadline, then report what was found.
/// A non-zero exit is not fatal: many local tools only implement `--help`.
pub fn probe_binary(input: &str) -> BinaryProbe {
    let Some(path) = resolve_binary(input) else {
        return BinaryProbe {
            ok: false,
            resolved_path: None,
            version: None,
            message: "Not found on PATH or at this location.".into(),
        };
    };

    let mut command = Command::new(&path);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }

    let resolved = path.to_string_lossy().to_string();
    let Ok(mut child) = command.spawn() else {
        return BinaryProbe {
            ok: true,
            resolved_path: Some(resolved.clone()),
            version: None,
            message: "Binary found, but it could not be executed.".into(),
        };
    };

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(None) => {
                let _ = child.kill();
                break;
            }
            Err(_) => break,
        }
    }

    let output = child.wait_with_output().ok();
    let version = output
        .as_ref()
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let text = if stdout.trim().is_empty() {
                stderr.to_string()
            } else {
                stdout.to_string()
            };
            text.lines()
                .map(str::trim)
                .find(|line| !line.is_empty())
                .map(|line| line.chars().take(120).collect::<String>())
        });

    BinaryProbe {
        ok: true,
        resolved_path: Some(resolved),
        version,
        message: "Binary found and executable.".into(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalModel {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
}

/// Find `.jns` files under the models folder (and an explicitly selected file).
pub fn scan_models(dir: &str, extra_file: Option<&str>, max_depth: usize) -> Vec<LocalModel> {
    fn walk(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<LocalModel>) {
        if depth > max_depth || out.len() >= 200 {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, depth + 1, max_depth, out);
                continue;
            }
            if path
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("jns"))
                .unwrap_or(false)
            {
                let size_bytes = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model.jns".into());
                out.push(LocalModel {
                    name,
                    path: path.to_string_lossy().to_string(),
                    size_bytes,
                });
            }
        }
    }

    let mut found = Vec::new();
    if !dir.trim().is_empty() {
        walk(&expand_home(dir), 0, max_depth, &mut found);
    }
    if let Some(extra) = extra_file.filter(|value| !value.trim().is_empty()) {
        let path = expand_home(extra);
        if path.is_file()
            && path
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("jns"))
                .unwrap_or(false)
            && !found.iter().any(|model| Path::new(&model.path) == path)
        {
            let size_bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            found.push(LocalModel {
                name: path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model.jns".into()),
                path: path.to_string_lossy().to_string(),
                size_bytes,
            });
        }
    }

    found.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hugging_face_token_is_trimmed_and_single_line() {
        let mut settings = AppSettings::default();
        settings.hf_token = " hf_abc\n def\t ".into();
        settings.normalize();
        assert_eq!(settings.hf_token, "hf_abcdef");
    }

    #[cfg(unix)]
    #[test]
    fn settings_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "janas-settings-perms-{}-{}",
            std::process::id(),
            crate::util::now_ms()
        ));
        let config = root.join("config");
        let store = SettingsStore::load(config.clone(), root.join("data"), None)
            .expect("settings store should load");
        store.save(store.get()).expect("settings should save");

        let path = config.join("settings.json");
        let mode = fs::metadata(&path)
            .expect("settings file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "settings may contain secrets");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_save_keeps_the_previous_settings() {
        let root = std::env::temp_dir().join(format!(
            "janas-settings-failure-{}-{}",
            std::process::id(),
            crate::util::now_ms()
        ));
        fs::create_dir_all(&root).expect("create settings test root");
        let settings_path = root.join("settings.json");
        fs::create_dir(&settings_path).expect("make settings path a directory");
        let store = SettingsStore {
            path: settings_path,
            settings: RwLock::new(AppSettings::default()),
        };
        let next = AppSettings {
            models_dir: root.join("models").to_string_lossy().to_string(),
            ..AppSettings::default()
        };

        assert!(store.save(next).is_err(), "a failed write must be reported");
        assert!(
            store.get().models_dir.is_empty(),
            "in-memory settings must not change when persistence fails"
        );
        let _ = fs::remove_dir_all(&root);
    }
}

