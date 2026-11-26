//! Side-by-side engine & quantization benchmark runner.
//!
//! A suite runs two to four [`BenchmarkTarget`]s through the one
//! [`EngineSupervisor`] slot, in order, with an optional warm-up pass. Every
//! target switch goes through `switch_engine`'s supervisor path and waits for
//! `/health` before the workload is sent, so the two engines can never overlap
//! and a measurement never starts while a model is still loading.
//!
//! Measurement uses the same streaming chat contract as the chat proxy
//! (`stream: true`, `stream_options.include_usage`) and samples
//! `/proc/<pid>/io` and `/proc/<pid>/statm` every 200 ms during the measured
//! generation (and the discarded warm-up). SSD figures are generation-only:
//! model loading happens before sampling, and page-cache reads count as read
//! calls but not as block-device bytes.
//!
//! Finished suites are written atomically to
//! `app_local_data_dir()/benchmarks/<id>.json`, one file per suite.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex as StdMutex, RwLock,
    },
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Runtime, State};
use tokio::{sync::Mutex, task::JoinHandle};

use crate::{
    engines::{pid_alive, process_counters, EngineKind, EngineSupervisor},
    error::{AppError, AppResult},
    settings::{expand_home, AppSettings, SettingsStore},
    system::{hardware_summary, HardwareSummary},
    util::{now_ms, truncate_line},
};

pub const BENCHMARK_PROGRESS_EVENT: &str = "benchmark://progress";
pub const BENCHMARK_FINISHED_EVENT: &str = "benchmark://finished";

const SAMPLE_INTERVAL_MS: u64 = 200;
const HEALTH_TIMEOUT_SECONDS: u64 = 300;
const STREAM_INACTIVITY_SECONDS: u64 = 300;
const MAX_PROMPT_CHARS: usize = 200_000;
const MAX_TOKENS_LIMIT: u32 = 8_192;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTarget {
    pub id: String,
    pub label: String,
    pub engine: EngineKind,
    /// A `.jns` file for Janas, a `.gguf` file for llama.cpp.
    pub model_path: String,
    pub ctx_size: u32,
    /// llama.cpp `--n-gpu-layers`. Ignored by Janas.
    pub gpu_layers: i32,
    /// Shell-style extra arguments plus `KEY=value` environment overrides,
    /// for example `--cpu-moe` or `JANAS_ATTN=int16 JANAS_KV=8`.
    pub extra_flags: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkWorkload {
    /// A built-in preset id, or `"custom"`.
    pub preset: String,
    pub label: String,
    pub prompt: String,
    pub seed: u64,
    /// Always 0.0 for a comparable benchmark; the runner normalizes it.
    pub temperature: f32,
    pub max_tokens: u32,
    /// Send `response_format: {"type": "json_object"}` for the JSON preset.
    #[serde(default)]
    pub structured: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkPreset {
    pub id: String,
    pub label: String,
    pub description: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub structured: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkStatus {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPhase {
    SwitchingEngine,
    Warmup,
    Measuring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunResult {
    pub target_id: String,
    pub target_label: String,
    pub engine: EngineKind,
    pub ttft_ms: u64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub total_duration_ms: u64,
    pub tokens_generated: u64,
    /// Peak resident set size of the engine process during the measured pass.
    pub peak_rss_bytes: u64,
    /// Block-device bytes read during the measured pass. Model loading is
    /// outside this window.
    pub ssd_bytes_read: u64,
    /// Peak block-device read throughput during the measured pass.
    pub peak_ssd_mbps: f64,
    /// Peak read syscalls/s from `/proc/<pid>/io` (`syscr`). This includes
    /// page-cache hits, so it is not device IOPS.
    pub peak_iops: f64,
    #[serde(default)]
    pub adapter_warnings: Vec<String>,
    pub status: BenchmarkStatus,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub engine_version: Option<String>,
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub id: String,
    pub created_at_ms: u64,
    pub workload: BenchmarkWorkload,
    pub warmup: bool,
    pub targets: Vec<BenchmarkTarget>,
    pub results: Vec<BenchmarkRunResult>,
    pub hardware: HardwareSummary,
    pub status: BenchmarkStatus,
    #[serde(default)]
    pub error: Option<String>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkProgress {
    pub suite_id: String,
    pub target_index: usize,
    pub target_count: usize,
    pub target_id: String,
    pub target_label: String,
    pub engine: EngineKind,
    pub phase: BenchmarkPhase,
    pub tokens_generated: u64,
    pub live_tps: f64,
    pub elapsed_ms: u64,
    pub rss_bytes: u64,
    pub read_bytes_per_sec: f64,
    pub read_ops_per_sec: f64,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Workload presets
// ---------------------------------------------------------------------------

const LONG_CONTEXT_PASSAGE: &str = "A storage engine reads a page from a Solid State Drive only when the page cache does not already hold it. \
Each miss has a measurable cost: the device queues the command, address translation resolves the logical block, the controller \
schedules the NAND dies and the host copy copies the returned sectors into ordinary memory. Streaming inference changes the shape \
of that traffic because experts are read on demand rather than mapped up front. The working set of a mixture-of-experts decoder \
can exceed system memory by a wide margin, so the order in which tokens arrive matters as much as the total number of bytes. \
Prefill touches a large set of prompt tokens at once and can therefore read many experts in a burst, while decode reads a much \
smaller slice per step and is usually limited by latency and queue depth. Measuring both phases separately explains why one \
configuration can look fast on short prompts and slow on long documents. ";

/// Build the built-in workload catalog. The long-context prompt is materialized
/// deterministically so every run of a suite sends exactly the same bytes.
pub fn workload_presets() -> Vec<BenchmarkPreset> {
    let mut long_context = String::from(
        "The following technical reference is provided as context. Read all of it and answer the question at the end. \
Do not summarize the reference.\n\n",
    );
    for section in 0..22 {
        long_context.push_str(&format!(
            "Section {section}. Cache and device behaviour:\n{LONG_CONTEXT_PASSAGE}\n\n"
        ));
    }
    long_context.push_str(
        "Question: using only the reference above, explain in two sentences why a cold expert-streaming run can read more \
bytes from the SSD than a warm run. Answer concisely.",
    );

    vec![
        BenchmarkPreset {
            id: "short-reasoning".into(),
            label: "Short Reasoning".into(),
            description: "A small logic puzzle that measures time to first token and steady decode.".into(),
            prompt: "Three machines can produce 36 widgets in 6 hours. If two more identical machines join, how long will it \
take all five machines to produce 120 widgets? Work through the rates step by step, then give the final answer on its own \
line."
                .into(),
            max_tokens: 256,
            structured: false,
        },
        BenchmarkPreset {
            id: "long-context-prefill".into(),
            label: "Long Context Prefill (~4k tokens)".into(),
            description: "A roughly 4k-token reference followed by one short question. Stresses prefill and SSD reads.".into(),
            prompt: long_context,
            max_tokens: 128,
            structured: false,
        },
        BenchmarkPreset {
            id: "code-generation".into(),
            label: "Code Generation".into(),
            description: "A prompt that asks for a complete Rust function with tests, rewarding sustained decode speed.".into(),
            prompt: "Write a complete, compilable Rust 2021 function `merge_intervals(intervals: Vec<(u64, u64)>) -> Vec<(u64, u64)>` \
that merges overlapping and touching half-open intervals, sorts its input first, and handles empty input. Then write a \
`#[cfg(test)] mod tests` covering empty input, already merged input, touching intervals, nested intervals, and an unsorted \
input. Output only Rust code in one code block."
                .into(),
            max_tokens: 512,
            structured: false,
        },
        BenchmarkPreset {
            id: "json-structured".into(),
            label: "JSON Structured Output".into(),
            description: "Forces an OpenAI-style JSON response with a fixed shape and no surrounding prose.".into(),
            prompt: "Return one JSON object describing a local inference benchmark. It must have exactly these keys: \
\"engine\" (string), \"quantization\" (string), \"tokens_per_second\" (number), \"peak_rss_gib\" (number), \
\"ssd_cache\" (boolean), and \"notes\" (array of exactly two short strings). Use plausible sample values. Do not wrap the \
object in prose or markdown fences."
                .into(),
            max_tokens: 256,
            structured: true,
        },
    ]
}

#[cfg(test)]
fn preset_workload(id: &str) -> Option<BenchmarkWorkload> {
    workload_presets()
        .into_iter()
        .find(|preset| preset.id == id)
        .map(|preset| BenchmarkWorkload {
            preset: preset.id,
            label: preset.label,
            prompt: preset.prompt,
            seed: 42,
            temperature: 0.0,
            max_tokens: preset.max_tokens,
            structured: preset.structured,
        })
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

fn valid_suite_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-' || character == '_')
}

struct BenchmarkStore {
    dir: PathBuf,
}

impl BenchmarkStore {
    fn new(dir: PathBuf) -> AppResult<Self> {
        fs::create_dir_all(&dir).map_err(|err| {
            AppError::message(format!(
                "Could not create the benchmark folder {}: {err}",
                dir.display()
            ))
        })?;
        Ok(Self { dir })
    }

    fn suite_path(&self, id: &str) -> AppResult<PathBuf> {
        if !valid_suite_id(id) {
            return Err(AppError::message(
                "That benchmark suite id is not valid. Reload the history and try again.",
            ));
        }
        Ok(self.dir.join(format!("{id}.json")))
    }

    fn load_all(&self) -> Vec<BenchmarkSuite> {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut suites: Vec<BenchmarkSuite> = entries
            .flatten()
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|extension| extension.eq_ignore_ascii_case("json"))
                    .unwrap_or(false)
            })
            .filter_map(|entry| {
                let path = entry.path();
                let raw = fs::read(&path).ok()?;
                match serde_json::from_slice::<BenchmarkSuite>(&raw) {
                    Ok(suite) if valid_suite_id(&suite.id) => Some(suite),
                    Ok(_) | Err(_) => {
                        eprintln!(
                            "benchmark: ignoring unreadable suite {}",
                            path.display()
                        );
                        None
                    }
                }
            })
            .collect();
        suites.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
        suites
    }

    /// Write to a sibling temporary file, then rename. A reader never sees a
    /// half-written JSON document, even if the app is killed mid-save.
    fn save(&self, suite: &BenchmarkSuite) -> AppResult<()> {
        let path = self.suite_path(&suite.id)?;
        let tmp = self.dir.join(format!("{}.json.tmp", suite.id));
        let bytes = serde_json::to_vec_pretty(suite)
            .map_err(|err| AppError::message(format!("Could not encode the benchmark suite: {err}")))?;
        fs::write(&tmp, bytes).map_err(|err| {
            AppError::message(format!(
                "Could not write the benchmark suite to {}: {err}",
                tmp.display()
            ))
        })?;
        match fs::rename(&tmp, &path) {
            Ok(()) => Ok(()),
            Err(first_err) => {
                // Windows cannot replace an existing file with rename. The id
                // is unique in practice, but a retry after a crash may leave
                // the previous copy behind.
                if path.exists() {
                    fs::remove_file(&path).map_err(|err| {
                        AppError::message(format!(
                            "Could not replace the benchmark suite {}: {err}",
                            path.display()
                        ))
                    })?;
                    fs::rename(&tmp, &path).map_err(|err| {
                        AppError::message(format!(
                            "Could not finish saving the benchmark suite {}: {err}",
                            path.display()
                        ))
                    })
                } else {
                    let _ = fs::remove_file(&tmp);
                    Err(AppError::message(format!(
                        "Could not save the benchmark suite {}: {first_err}",
                        path.display()
                    )))
                }
            }
        }
    }

    fn delete(&self, id: &str) -> AppResult<bool> {
        let path = self.suite_path(id)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(AppError::message(format!(
                "Could not delete the benchmark suite {}: {err}",
                path.display()
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Runner state
// ---------------------------------------------------------------------------

struct ActiveRun {
    suite_id: String,
    cancel: Arc<AtomicBool>,
}

pub struct BenchmarkRunner {
    store: BenchmarkStore,
    suites: RwLock<Vec<BenchmarkSuite>>,
    /// Only one suite may own the supervisor at a time. The lock is held only
    /// for the begin/finish bookkeeping, never for the run itself.
    active: Mutex<Option<ActiveRun>>,
}

impl BenchmarkRunner {
    pub fn load(dir: PathBuf) -> AppResult<Self> {
        let store = BenchmarkStore::new(dir)?;
        let suites = store.load_all();
        Ok(Self {
            store,
            suites: RwLock::new(suites),
            active: Mutex::new(None),
        })
    }

    fn list(&self) -> Vec<BenchmarkSuite> {
        self.suites
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    fn find(&self, id: &str) -> Option<BenchmarkSuite> {
        self.suites
            .read()
            .map(|guard| guard.iter().find(|suite| suite.id == id).cloned())
            .unwrap_or_else(|poisoned| {
                poisoned
                    .into_inner()
                    .iter()
                    .find(|suite| suite.id == id)
                    .cloned()
            })
    }

    async fn begin(&self, suite_id: &str) -> AppResult<Arc<AtomicBool>> {
        let mut guard = self.active.lock().await;
        if let Some(active) = guard.as_ref() {
            return Err(AppError::message(format!(
                "A benchmark suite ({}) is already running. Cancel it or wait for it to finish first.",
                active.suite_id
            )));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        *guard = Some(ActiveRun {
            suite_id: suite_id.to_string(),
            cancel: cancel.clone(),
        });
        Ok(cancel)
    }

    async fn cancel_active(&self) -> bool {
        let guard = self.active.lock().await;
        match guard.as_ref() {
            Some(active) => {
                active.cancel.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    /// Persist, publish in memory, and release the supervisor for the next
    /// suite. Returns the possibly amended suite (a save failure is attached as
    /// a user-visible suite error instead of losing the measurements).
    async fn finish(&self, mut suite: BenchmarkSuite) -> BenchmarkSuite {
        if let Err(err) = self.store.save(&suite) {
            eprintln!("benchmark: {err}");
            let note = format!("The suite finished but could not be saved: {err}");
            suite.error = Some(match suite.error.take() {
                Some(existing) => format!("{existing} {note}"),
                None => note,
            });
        }
        {
            let mut guard = self
                .suites
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.retain(|existing| existing.id != suite.id);
            guard.push(suite.clone());
            guard.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
        }
        let mut active = self.active.lock().await;
        if active
            .as_ref()
            .map(|run| run.suite_id == suite.id)
            .unwrap_or(false)
        {
            *active = None;
        }
        suite
    }
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn benchmark_workload_presets() -> Vec<BenchmarkPreset> {
    workload_presets()
}

#[tauri::command]
pub async fn start_benchmark_suite(
    app: AppHandle,
    runner: State<'_, Arc<BenchmarkRunner>>,
    supervisor: State<'_, Arc<EngineSupervisor>>,
    store: State<'_, Arc<SettingsStore>>,
    targets: Vec<BenchmarkTarget>,
    workload: BenchmarkWorkload,
    warmup: bool,
) -> AppResult<String> {
    if !(2..=4).contains(&targets.len()) {
        return Err(AppError::message(
            "Choose between two and four targets for a side-by-side benchmark.",
        ));
    }
    let mut ids = HashSet::new();
    for target in &targets {
        if !ids.insert(target.id.trim().to_string()) {
            return Err(AppError::message(format!(
                "Two benchmark targets share the id \"{}\". Give each target a unique id.",
                target.id
            )));
        }
        validate_target(target)?;
    }

    let prompt = workload.prompt.trim().to_string();
    if prompt.is_empty() {
        return Err(AppError::message(
            "The benchmark workload has no prompt. Choose a preset or type a prompt.",
        ));
    }
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(AppError::message(format!(
            "The benchmark prompt is longer than the {} character limit.",
            MAX_PROMPT_CHARS
        )));
    }
    if workload.max_tokens == 0 || workload.max_tokens > MAX_TOKENS_LIMIT {
        return Err(AppError::message(format!(
            "Max tokens must be between 1 and {MAX_TOKENS_LIMIT}."
        )));
    }

    let mut workload = workload;
    // Fixed sampling is a measurement requirement, not a preference.
    workload.temperature = 0.0;
    workload.prompt = prompt;
    if workload.label.trim().is_empty() {
        workload.label = "Custom prompt".into();
    }
    if workload.preset.trim().is_empty() {
        workload.preset = "custom".into();
    }

    let suite_id = uuid::Uuid::new_v4().to_string();
    let cancel = runner.inner().begin(&suite_id).await?;

    let runner = runner.inner().clone();
    let supervisor = supervisor.inner().clone();
    let base_settings = store.inner().get();
    let app_for_run = app.clone();
    let id_for_run = suite_id.clone();

    tauri::async_runtime::spawn(async move {
        execute_suite(
            app_for_run,
            runner,
            supervisor,
            base_settings,
            targets,
            workload,
            warmup,
            id_for_run,
            cancel,
        )
        .await;
    });

    Ok(suite_id)
}

#[tauri::command]
pub async fn cancel_benchmark(
    runner: State<'_, Arc<BenchmarkRunner>>,
) -> AppResult<bool> {
    Ok(runner.inner().cancel_active().await)
}

#[tauri::command]
pub fn list_benchmark_suites(
    runner: State<'_, Arc<BenchmarkRunner>>,
) -> AppResult<Vec<BenchmarkSuite>> {
    Ok(runner.inner().list())
}

#[tauri::command(rename_all = "snake_case")]
pub fn delete_benchmark_suite(
    runner: State<'_, Arc<BenchmarkRunner>>,
    id: String,
) -> AppResult<bool> {
    let removed = runner.inner().store.delete(&id)?;
    if removed {
        if let Ok(mut guard) = runner.inner().suites.write() {
            guard.retain(|suite| suite.id != id);
        }
    }
    Ok(removed)
}

#[tauri::command(rename_all = "snake_case")]
pub fn export_benchmark_markdown(
    runner: State<'_, Arc<BenchmarkRunner>>,
    suite_id: String,
) -> AppResult<String> {
    let suite = runner.inner().find(&suite_id).ok_or_else(|| {
        AppError::message(
            "That benchmark suite is not in the local history. Reload the list and try again.",
        )
    })?;
    Ok(render_markdown(&suite))
}

// ---------------------------------------------------------------------------
// Validation and per-target launch settings
// ---------------------------------------------------------------------------

fn validate_target(target: &BenchmarkTarget) -> AppResult<()> {
    if target.id.trim().is_empty() {
        return Err(AppError::message("Every benchmark target needs an id."));
    }
    let path = expand_home(target.model_path.trim());
    if target.model_path.trim().is_empty() {
        return Err(AppError::message(format!(
            "Target \"{}\" has no model selected.",
            target.label
        )));
    }
    if !path.is_file() {
        return Err(AppError::message(format!(
            "The model for target \"{}\" does not exist: {}",
            target.label,
            path.display()
        )));
    }
    let extension = path
        .extension()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match target.engine {
        EngineKind::Janas if extension != "jns" => {
            return Err(AppError::message(format!(
                "Target \"{}\" uses janas-server but {} is not a .jns file.",
                target.label,
                path.display()
            )))
        }
        EngineKind::Llama if extension != "gguf" => {
            return Err(AppError::message(format!(
                "Target \"{}\" uses llama.cpp but {} is not a .gguf file.",
                target.label,
                path.display()
            )))
        }
        _ => {}
    }
    if !(512..=1_048_576).contains(&target.ctx_size) {
        return Err(AppError::message(format!(
            "Target \"{}\" has a context window outside 512–1048576 tokens.",
            target.label
        )));
    }
    if !(0..=999).contains(&target.gpu_layers) {
        return Err(AppError::message(format!(
            "Target \"{}\" has GPU layers outside 0–999.",
            target.label
        )));
    }
    let _ = split_extra_flags(&target.extra_flags)?;
    Ok(())
}

fn env_name(value: &str) -> bool {
    let mut characters = value.chars();
    match characters.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Split target extra flags into environment overrides (`NAME=value`) and
/// command-line arguments. A leading dash is always an argument, so
/// `--cache-type-k=q8_0` keeps its meaning.
fn split_extra_flags(raw: &str) -> AppResult<(Vec<(String, String)>, Vec<String>)> {
    let tokens = shell_words::split(raw).map_err(|err| {
        AppError::message(format!(
            "Invalid extra flags: {err}. Use shell-style quoting, for example `--cpu-moe --threads 8`."
        ))
    })?;
    let mut environment = Vec::new();
    let mut arguments = Vec::new();
    for token in tokens {
        match token.split_once('=') {
            Some((name, value)) if env_name(name) && !name.starts_with('-') => {
                if value.contains('\0') {
                    return Err(AppError::message(
                        "An environment override in the extra flags contains a NUL byte.",
                    ));
                }
                environment.push((name.to_string(), value.to_string()));
            }
            _ => arguments.push(token),
        }
    }
    Ok((environment, arguments))
}

/// Produce the launch settings for one target without touching the saved
/// settings file. Global extra arguments are deliberately not inherited: a
/// side-by-side comparison must be defined by the target's own flags.
fn target_launch_settings(
    base: &AppSettings,
    target: &BenchmarkTarget,
) -> AppResult<(AppSettings, Vec<String>)> {
    let (environment, arguments) = split_extra_flags(&target.extra_flags)?;
    let mut settings = base.clone();
    let mut warnings = Vec::new();
    settings.launch_env = environment;

    match target.engine {
        EngineKind::Janas => {
            settings.model_path = expand_home(target.model_path.trim())
                .to_string_lossy()
                .to_string();
            settings.ctx_size = target.ctx_size.clamp(512, 1_048_576);
            settings.extra_args = shell_words::join(&arguments);
            // A draft model or MTP head chosen for a different model would
            // change this target's decode speed and can fail to load. Pass
            // `--mtp`/`--draft` inside extra_flags when a target needs them.
            settings.mtp_mode = "auto".into();
            settings.mtp_path = String::new();
            settings.draft_path = String::new();
            if !base.ssd_cache_dir.trim().is_empty() {
                warnings.push(
                    "The global Janas SSD cache directory is active. A cold run may read pages left by an earlier run, so clear the cache directory in Settings when a true cold measurement matters."
                        .into(),
                );
            }
            if target.gpu_layers != 0 {
                warnings.push(
                    "GPU layers apply only to llama.cpp targets; Janas ignores `gpu_layers` for this run."
                        .into(),
                );
            }
        }
        EngineKind::Llama => {
            settings.llama_model_path = expand_home(target.model_path.trim())
                .to_string_lossy()
                .to_string();
            settings.llama_ctx_size = target.ctx_size.clamp(512, 1_048_576);
            settings.llama_gpu_layers = target.gpu_layers.clamp(0, 999);
            settings.llama_extra_args = shell_words::join(&arguments);
            // A projector, draft model or MTP head from the previous target's
            // bundle would leak a different file into this measurement. The
            // benchmark is text-only and explicit: pass `--mmproj` or a draft
            // model inside extra_flags when a target really needs it.
            settings.llama_mmproj_path = String::new();
            settings.llama_draft_path = String::new();
            settings.llama_mtp_path = String::new();
            settings.llama_mtp_mode = "off".into();
            settings.llama_direct_cpu_moe = false;
        }
    }
    Ok((settings, warnings))
}

// ---------------------------------------------------------------------------
// Execution pipeline
// ---------------------------------------------------------------------------

async fn execute_suite<R: Runtime>(
    app: AppHandle<R>,
    runner: Arc<BenchmarkRunner>,
    supervisor: Arc<EngineSupervisor<R>>,
    base_settings: AppSettings,
    targets: Vec<BenchmarkTarget>,
    workload: BenchmarkWorkload,
    warmup: bool,
    suite_id: String,
    cancel: Arc<AtomicBool>,
) {
    let started = Instant::now();
    let created_at_ms = now_ms();
    let hardware = hardware_summary();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();
    let target_count = targets.len();
    let mut results = Vec::with_capacity(target_count);
    let mut status = BenchmarkStatus::Completed;
    let mut suite_error: Option<String> = None;

    for (index, target) in targets.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            status = BenchmarkStatus::Cancelled;
            break;
        }
        let result = run_target(
            &app,
            &client,
            &supervisor,
            &base_settings,
            target,
            &workload,
            warmup,
            index,
            target_count,
            &suite_id,
            &cancel,
        )
        .await;
        if result.status == BenchmarkStatus::Failed {
            status = BenchmarkStatus::Failed;
        }
        let cancelled = result.status == BenchmarkStatus::Cancelled;
        results.push(result);

        // Tear the slot down deliberately. `switch(..., Some(kind))` reuses a
        // live engine of the same kind, which would silently benchmark target
        // one's weights for target two.
        if let Err(err) = supervisor.switch(&app, &base_settings, None).await {
            suite_error = Some(format!(
                "The engine could not be stopped after target {}: {err}. The remaining targets were not started.",
                index + 1
            ));
            status = BenchmarkStatus::Failed;
            break;
        }
        if cancelled {
            status = BenchmarkStatus::Cancelled;
            break;
        }
    }

    // Final safety stop. A failure here is reported instead of hidden, because
    // a live engine process is exactly what this runner promises never to
    // leave behind.
    if let Err(err) = supervisor.switch(&app, &base_settings, None).await {
        let note = format!("The engine could not be stopped after the suite: {err}");
        suite_error = Some(match suite_error.take() {
            Some(existing) => format!("{existing} {note}"),
            None => note,
        });
        status = BenchmarkStatus::Failed;
    }

    let suite = BenchmarkSuite {
        id: suite_id,
        created_at_ms,
        workload,
        warmup,
        targets,
        results,
        hardware,
        status,
        error: suite_error,
        duration_ms: started.elapsed().as_millis() as u64,
    };
    let suite = runner.finish(suite).await;
    let _ = app.emit(BENCHMARK_FINISHED_EVENT, suite);
}

#[allow(clippy::too_many_arguments)]
async fn run_target<R: Runtime>(
    app: &AppHandle<R>,
    client: &reqwest::Client,
    supervisor: &Arc<EngineSupervisor<R>>,
    base_settings: &AppSettings,
    target: &BenchmarkTarget,
    workload: &BenchmarkWorkload,
    warmup: bool,
    index: usize,
    target_count: usize,
    suite_id: &str,
    cancel: &Arc<AtomicBool>,
) -> BenchmarkRunResult {
    let mut warnings = Vec::new();
    let mut result = BenchmarkRunResult {
        target_id: target.id.clone(),
        target_label: target.label.clone(),
        engine: target.engine,
        ttft_ms: 0,
        prefill_tps: 0.0,
        decode_tps: 0.0,
        total_duration_ms: 0,
        tokens_generated: 0,
        peak_rss_bytes: 0,
        ssd_bytes_read: 0,
        peak_ssd_mbps: 0.0,
        peak_iops: 0.0,
        adapter_warnings: Vec::new(),
        status: BenchmarkStatus::Completed,
        error: None,
        engine_version: None,
        adapter: None,
    };
    let progress = ProgressContext {
        app: app.clone(),
        suite_id: suite_id.to_string(),
        target_index: index,
        target_count,
        target_id: target.id.clone(),
        target_label: target.label.clone(),
        engine: target.engine,
    };
    progress.emit(
        BenchmarkPhase::SwitchingEngine,
        0,
        0.0,
        0,
        0,
        0.0,
        0.0,
        "Stopping the current engine, then launching this target.",
    );

    let (settings, mut target_warnings) = match target_launch_settings(base_settings, target) {
        Ok(value) => value,
        Err(err) => {
            return finish_result(result, vec![err.to_string()], BenchmarkStatus::Failed, Some(err.to_string()));
        }
    };
    warnings.append(&mut target_warnings);

    if cancel.load(Ordering::Relaxed) {
        return finish_result(
            result,
            warnings,
            BenchmarkStatus::Cancelled,
            Some("Cancelled by user.".into()),
        );
    }

    // The explicit stop above guarantees a fresh process per target even when
    // two targets select the same engine.
    if let Err(err) = supervisor.switch(app, base_settings, None).await {
        return finish_result(
            result,
            warnings,
            BenchmarkStatus::Failed,
            Some(format!(
                "Could not stop the previous engine before starting target \"{}\": {err}",
                target.label
            )),
        );
    }
    if cancel.load(Ordering::Relaxed) {
        return finish_result(
            result,
            warnings,
            BenchmarkStatus::Cancelled,
            Some("Cancelled by user.".into()),
        );
    }

    progress.emit(
        BenchmarkPhase::SwitchingEngine,
        0,
        0.0,
        0,
        0,
        0.0,
        0.0,
        format!(
            "Starting {} with {}.",
            target.engine.label(),
            truncate_line(&target.model_path, 160)
        ),
    );
    if let Err(err) = supervisor
        .switch(app, &settings, Some(target.engine))
        .await
    {
        return finish_result(
            result,
            warnings,
            BenchmarkStatus::Failed,
            Some(format!(
                "{} could not start target \"{}\": {err}",
                target.engine.label(),
                target.label
            )),
        );
    }

    let active = match wait_for_ready(client, supervisor, target.engine, cancel).await {
        Ok(active) => active,
        Err(BenchFailure::Cancelled) => {
            return finish_result(
                result,
                warnings,
                BenchmarkStatus::Cancelled,
                Some("Cancelled by user.".into()),
            )
        }
        Err(BenchFailure::Failed(message)) => {
            return finish_result(result, warnings, BenchmarkStatus::Failed, Some(message))
        }
    };
    result.engine_version = active.version.clone();
    result.adapter = Some(active.adapter.clone());
    if let Some(warning) = active
        .warning
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        warnings.push(warning.to_string());
    }
    let pid = active.pid;
    let url = format!("http://{}:{}/v1/chat/completions", active.host, active.port);

    if warmup {
        progress.emit(
            BenchmarkPhase::Warmup,
            0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            "Warm-up pass: filling the page cache; its measurements are discarded.",
        );
        // Warm the model/experts without priming llama.cpp's prefix KV cache
        // with the exact measured prompt. The measured TTFT stays honest while
        // the page cache is still warm.
        let mut warmup_workload = workload.clone();
        warmup_workload.prompt = format!(
            "Warm-up pass. Ignore the answer to this copy.\n\n{}",
            workload.prompt
        );
        let (outcome, _warmup_telemetry) = stream_with_telemetry(
            client,
            &url,
            &settings.api_key,
            target,
            &warmup_workload,
            pid,
            cancel,
            &progress,
            BenchmarkPhase::Warmup,
        )
        .await;
        match outcome {
            Ok(outcome) if outcome.cancelled => {
                return finish_result(
                    result,
                    warnings,
                    BenchmarkStatus::Cancelled,
                    Some("Cancelled during the warm-up pass.".into()),
                )
            }
            Ok(_) => {}
            Err(BenchFailure::Cancelled) => {
                return finish_result(
                    result,
                    warnings,
                    BenchmarkStatus::Cancelled,
                    Some("Cancelled during the warm-up pass.".into()),
                )
            }
            Err(BenchFailure::Failed(message)) => {
                return finish_result(
                    result,
                    warnings,
                    BenchmarkStatus::Failed,
                    Some(format!("The warm-up pass failed: {message}")),
                )
            }
        }
        if cancel.load(Ordering::Relaxed) {
            return finish_result(
                result,
                warnings,
                BenchmarkStatus::Cancelled,
                Some("Cancelled by user.".into()),
            );
        }
    }

    progress.emit(
        BenchmarkPhase::Measuring,
        0,
        0.0,
        0,
        0,
        0.0,
        0.0,
        "Measuring the identical workload.",
    );

    let (outcome, telemetry) = stream_with_telemetry(
        client,
        &url,
        &settings.api_key,
        target,
        workload,
        pid,
        cancel,
        &progress,
        BenchmarkPhase::Measuring,
    )
    .await;
    result.peak_rss_bytes = telemetry.peak_rss_bytes;
    result.ssd_bytes_read = telemetry.ssd_bytes_read;
    result.peak_ssd_mbps = round2(telemetry.peak_ssd_mbps);
    result.peak_iops = round2(telemetry.peak_iops);
    if !telemetry.io_available {
        warnings.push(
            "Process I/O counters were unavailable, so SSD bytes, SSD throughput and read-call rates were not measured. Their zero values are not results."
                .into(),
        );
    }

    match outcome {
        Ok(outcome) => {
            let report = outcome.report;
            warnings.extend(report.warnings.iter().cloned());
            result.ttft_ms = report.ttft_ms;
            result.prefill_tps = round2(report.prefill_tps);
            result.decode_tps = round2(report.decode_tps);
            result.total_duration_ms = report.total_duration_ms;
            result.tokens_generated = report.tokens_generated;
            if outcome.cancelled {
                finish_result(
                    result,
                    warnings,
                    BenchmarkStatus::Cancelled,
                    Some("Cancelled by user.".into()),
                )
            } else {
                finish_result(result, warnings, BenchmarkStatus::Completed, None)
            }
        }
        Err(BenchFailure::Cancelled) => finish_result(
            result,
            warnings,
            BenchmarkStatus::Cancelled,
            Some("Cancelled by user.".into()),
        ),
        Err(BenchFailure::Failed(message)) => {
            finish_result(result, warnings, BenchmarkStatus::Failed, Some(message))
        }
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn finish_result(
    mut result: BenchmarkRunResult,
    warnings: Vec<String>,
    status: BenchmarkStatus,
    error: Option<String>,
) -> BenchmarkRunResult {
    result.adapter_warnings = sanitize_warnings(warnings);
    result.status = status;
    result.error = error;
    result
}

fn sanitize_warnings(warnings: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for warning in warnings {
        let warning = truncate_line(warning.trim(), 900);
        if warning.is_empty() || !seen.insert(warning.clone()) {
            continue;
        }
        output.push(warning);
        if output.len() >= 12 {
            break;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Engine readiness
// ---------------------------------------------------------------------------

enum BenchFailure {
    Cancelled,
    Failed(String),
}

async fn wait_for_ready<R: Runtime>(
    client: &reqwest::Client,
    supervisor: &Arc<EngineSupervisor<R>>,
    kind: EngineKind,
    cancel: &AtomicBool,
) -> Result<crate::engines::ActiveEngineInfo, BenchFailure> {
    let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECONDS);
    let mut absent_since: Option<Instant> = None;

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(BenchFailure::Cancelled);
        }
        match supervisor.active_info().await {
            Some(info) if info.kind == kind => {
                absent_since = None;
                if !pid_alive(info.pid) {
                    return Err(BenchFailure::Failed(format!(
                        "{} exited before it answered /health. Check the engine log for a model or GPU error.",
                        kind.label()
                    )));
                }
                let url = format!("http://{}:{}/health", info.host, info.port);
                if let Ok(response) = client
                    .get(&url)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                {
                    if response.status().is_success() {
                        return Ok(info);
                    }
                }
            }
            Some(info) => {
                return Err(BenchFailure::Failed(format!(
                    "The supervisor started {} instead of {}. Stop the other engine and retry.",
                    info.kind.label(),
                    kind.label()
                )))
            }
            None => {
                // `switch` publishes the launch before the child binds its
                // port, so a short absence is normal. A lasting one is not.
                let since = *absent_since.get_or_insert_with(Instant::now);
                if since.elapsed() > Duration::from_secs(5) {
                    return Err(BenchFailure::Failed(format!(
                        "{} stopped before it answered /health. Check the engine log.",
                        kind.label()
                    )));
                }
            }
        }

        if Instant::now() >= deadline {
            return Err(BenchFailure::Failed(format!(
                "{} did not answer /health within {HEALTH_TIMEOUT_SECONDS} seconds. The model may be larger than the available memory or the binary may be missing a runtime library.",
                kind.label()
            )));
        }
        tokio::time::sleep(Duration::from_millis(350)).await;
    }
}

// ---------------------------------------------------------------------------
// Streaming measurement
// ---------------------------------------------------------------------------

#[derive(Default)]
struct LiveCounters {
    tokens: u64,
    tps: f64,
}

struct ProgressContext<R: Runtime> {
    app: AppHandle<R>,
    suite_id: String,
    target_index: usize,
    target_count: usize,
    target_id: String,
    target_label: String,
    engine: EngineKind,
}

impl<R: Runtime> Clone for ProgressContext<R> {
    fn clone(&self) -> Self {
        Self {
            app: self.app.clone(),
            suite_id: self.suite_id.clone(),
            target_index: self.target_index,
            target_count: self.target_count,
            target_id: self.target_id.clone(),
            target_label: self.target_label.clone(),
            engine: self.engine,
        }
    }
}

impl<R: Runtime> ProgressContext<R> {
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &self,
        phase: BenchmarkPhase,
        tokens_generated: u64,
        live_tps: f64,
        elapsed_ms: u64,
        rss_bytes: u64,
        read_bytes_per_sec: f64,
        read_ops_per_sec: f64,
        message: impl Into<String>,
    ) {
        let _ = self.app.emit(
            BENCHMARK_PROGRESS_EVENT,
            BenchmarkProgress {
                suite_id: self.suite_id.clone(),
                target_index: self.target_index,
                target_count: self.target_count,
                target_id: self.target_id.clone(),
                target_label: self.target_label.clone(),
                engine: self.engine,
                phase,
                tokens_generated,
                live_tps: round2(live_tps),
                elapsed_ms,
                rss_bytes,
                read_bytes_per_sec: round2(read_bytes_per_sec),
                read_ops_per_sec: round2(read_ops_per_sec),
                message: message.into(),
            },
        );
    }
}

struct StreamAccum {
    tokens: u64,
    ttft_ms: Option<u64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    predicted_tokens: Option<u64>,
    prompt_tps: Option<f64>,
    decode_tps: Option<f64>,
    malformed: u32,
    last_problem: Option<String>,
}

impl Default for StreamAccum {
    fn default() -> Self {
        Self {
            tokens: 0,
            ttft_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
            predicted_tokens: None,
            prompt_tps: None,
            decode_tps: None,
            malformed: 0,
            last_problem: None,
        }
    }
}

struct GenerationReport {
    ttft_ms: u64,
    prefill_tps: f64,
    decode_tps: f64,
    total_duration_ms: u64,
    tokens_generated: u64,
    warnings: Vec<String>,
}

struct StreamOutcome {
    report: GenerationReport,
    cancelled: bool,
}

fn token_budget_warning(tokens_generated: u64, max_tokens: u32) -> Option<String> {
    (tokens_generated > u64::from(max_tokens)).then(|| {
        format!(
            "The engine reported {tokens_generated} tokens for a {max_tokens}-token cap. The model may ignore `enable_thinking`, so this result is not token-bounded."
        )
    })
}

fn request_body(target: &BenchmarkTarget, workload: &BenchmarkWorkload) -> Value {
    let model = Path::new(target.model_path.trim())
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "benchmark".into());
    let mut body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": workload.prompt }],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": workload.seed,
        "max_tokens": workload.max_tokens,
        // Janas caps `max_tokens` on visible content only; a reasoning model
        // can emit hundreds of hidden tokens beyond it. Disabling thinking at
        // the template level makes the token budget and wall clock comparable
        // across engines. llama-server honours the same Jinja kwarg.
        "chat_template_kwargs": { "enable_thinking": false },
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if workload.structured {
        body["response_format"] = json!({ "type": "json_object" });
    }
    if target.engine == EngineKind::Llama {
        // llama-server returns exact prompt/predicted timings in the final
        // chunk. Janas does not need the field, so it never sees it.
        body["timings_per_token"] = json!(true);
    }
    body
}

async fn stream_completion(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    target: &BenchmarkTarget,
    workload: &BenchmarkWorkload,
    cancel: &AtomicBool,
    live: Option<Arc<Mutex<LiveCounters>>>,
) -> Result<StreamOutcome, BenchFailure> {
    let started = Instant::now();
    let mut builder = client.post(url).json(&request_body(target, workload));
    if !api_key.trim().is_empty() {
        builder = builder.bearer_auth(api_key.trim());
    }
    let response = builder.send().await.map_err(|err| {
        BenchFailure::Failed(format!(
            "Could not reach the local inference server: {err}"
        ))
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(BenchFailure::Failed(format!(
            "The engine returned HTTP {status}: {}",
            truncate_line(&body, 700)
        )));
    }

    let mut stream = response.bytes_stream();
    let mut sse = SseBuffer::default();
    let mut acc = StreamAccum::default();
    let mut done = false;
    let mut cancelled = false;
    let mut last_data = Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_millis(150));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    while !done {
        let chunk = tokio::select! {
            _ = ticker.tick() => {
                if cancel.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
                if last_data.elapsed() > Duration::from_secs(STREAM_INACTIVITY_SECONDS) {
                    return Err(BenchFailure::Failed(format!(
                        "The engine sent no data for {STREAM_INACTIVITY_SECONDS} seconds. The run was stopped so it could not hang the suite."
                    )));
                }
                continue;
            }
            next = stream.next() => match next {
                Some(Ok(chunk)) => chunk,
                Some(Err(err)) => {
                    return Err(BenchFailure::Failed(format!(
                        "The inference stream stopped: {err}"
                    )))
                }
                None => break,
            },
        };
        last_data = Instant::now();
        let before = acc.tokens;
        for payload in sse.push(&chunk) {
            if payload.trim().eq_ignore_ascii_case("[DONE]") {
                done = true;
                break;
            }
            if let Err(message) = apply_payload(&payload, &mut acc, started) {
                return Err(BenchFailure::Failed(message));
            }
        }
        if acc.tokens != before {
            update_live(&live, &acc, started).await;
        }
    }

    if !done && !cancelled {
        if let Some(payload) = sse.finish() {
            if payload.trim().eq_ignore_ascii_case("[DONE]") {
                // Nothing follows the terminal marker.
            } else if let Err(message) = apply_payload(&payload, &mut acc, started) {
                return Err(BenchFailure::Failed(message));
            }
            update_live(&live, &acc, started).await;
        }
    }

    let total_duration_ms = started.elapsed().as_millis() as u64;
    // Engine usage is authoritative when present. Stream chunks are only a
    // fallback for engines that omit usage; taking the maximum of the two
    // would let a multi-token delta fabricate a cap overrun.
    let tokens_generated = acc
        .completion_tokens
        .or(acc.predicted_tokens)
        .unwrap_or(acc.tokens);
    let prompt_tokens = acc.prompt_tokens.unwrap_or(0);
    let ttft_ms = acc.ttft_ms.unwrap_or(0);
    let mut warnings = Vec::new();

    if acc.malformed > 0 {
        warnings.push(format!(
            "Ignored {} malformed stream chunk(s).{}",
            acc.malformed,
            acc.last_problem
                .as_deref()
                .map(|problem| format!(" Last problem: {}", truncate_line(problem, 240)))
                .unwrap_or_default()
        ));
    }
    if prompt_tokens == 0 {
        warnings.push(
            "The engine did not report usage.prompt_tokens, so prompt-eval speed could not be computed. llama.cpp needs a current build; Janas must include usage in its stream."
                .into(),
        );
    }
    if acc.completion_tokens.is_none() && acc.predicted_tokens.is_none() {
        warnings.push(
            "The engine did not report completion token usage; token count and decode speed are counted from stream chunks."
                .into(),
        );
    }
    if !cancelled {
        if let Some(warning) = token_budget_warning(tokens_generated, workload.max_tokens) {
            warnings.push(warning);
        }
    }
    if !cancelled && tokens_generated == 0 {
        let detail = acc
            .last_problem
            .as_deref()
            .map(|problem| format!(" Last stream problem: {}", truncate_line(problem, 240)))
            .unwrap_or_default();
        return Err(BenchFailure::Failed(format!(
            "The engine finished without generating a token.{detail}"
        )));
    }

    let decode_seconds = (total_duration_ms.saturating_sub(ttft_ms)) as f64 / 1000.0;
    let decode_tps = acc.decode_tps.unwrap_or_else(|| {
        if tokens_generated > 0 && decode_seconds > 0.0 {
            tokens_generated as f64 / decode_seconds
        } else {
            0.0
        }
    });
    let prefill_tps = acc.prompt_tps.unwrap_or_else(|| {
        if prompt_tokens > 0 && ttft_ms > 0 {
            prompt_tokens as f64 / (ttft_ms as f64 / 1000.0)
        } else {
            0.0
        }
    });

    Ok(StreamOutcome {
        report: GenerationReport {
            ttft_ms,
            prefill_tps,
            decode_tps,
            total_duration_ms,
            tokens_generated,
            warnings,
        },
        cancelled,
    })
}

fn live_decode_tps(tokens: u64, ttft_ms: u64, elapsed_ms: u64) -> f64 {
    if tokens < 2 {
        return 0.0;
    }
    let decode_seconds = elapsed_ms.saturating_sub(ttft_ms) as f64 / 1000.0;
    if decode_seconds <= 0.0 {
        0.0
    } else {
        (tokens - 1) as f64 / decode_seconds
    }
}

async fn update_live(
    live: &Option<Arc<Mutex<LiveCounters>>>,
    acc: &StreamAccum,
    started: Instant,
) {
    if let Some(live) = live.as_ref() {
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let mut guard = live.lock().await;
        guard.tokens = acc.tokens;
        guard.tps = live_decode_tps(acc.tokens, acc.ttft_ms.unwrap_or(elapsed_ms), elapsed_ms);
    }
}

fn delta_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn json_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .or_else(|| value.get(key).and_then(Value::as_f64).map(|value| value.max(0.0) as u64))
}

fn error_text(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    if error.is_null() {
        return None;
    }
    if let Some(text) = error.as_str() {
        return Some(text.to_string());
    }
    error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| Some(error.to_string()))
}

/// Parse one SSE `data:` JSON payload. Malformed chunks are counted and
/// ignored; only a server-reported error aborts the stream.
fn apply_payload(payload: &str, acc: &mut StreamAccum, started: Instant) -> Result<(), String> {
    let value: Value = match serde_json::from_str(payload) {
        Ok(value) => value,
        Err(err) => {
            acc.malformed = acc.malformed.saturating_add(1);
            acc.last_problem = Some(format!("Malformed stream chunk: {err}"));
            return Ok(());
        }
    };

    if let Some(message) = error_text(&value) {
        return Err(format!("The engine reported: {message}"));
    }

    if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
        if let Some(prompt) = json_u64(usage, "prompt_tokens") {
            acc.prompt_tokens = Some(prompt);
        }
        if let Some(completion) = json_u64(usage, "completion_tokens") {
            acc.completion_tokens = Some(completion);
        }
        // Janas reports the exact prompt-eval and reply clocks in its own
        // usage block. Prefer them over the TCP-inclusive TTFT estimate.
        if let Some(janas) = usage.get("janas").filter(|janas| janas.is_object()) {
            if let Some(seconds) = json_f64(janas, "input_seconds").filter(|value| *value > 0.0)
            {
                if let Some(prompt) = acc.prompt_tokens.filter(|value| *value > 0) {
                    acc.prompt_tps = Some(prompt as f64 / seconds);
                }
            }
            if let Some(seconds) = json_f64(janas, "output_seconds").filter(|value| *value > 0.0)
            {
                if let Some(completion) = acc.completion_tokens.filter(|value| *value > 0) {
                    acc.decode_tps = Some(completion as f64 / seconds);
                }
            }
        }
    }

    if let Some(timings) = value.get("timings").filter(|timings| timings.is_object()) {
        // A zero is a missing timing, not a measured rate; do not let it erase
        // Janas's own clocks when both objects are present.
        if let Some(value) = json_f64(timings, "prompt_per_second").filter(|value| *value > 0.0) {
            acc.prompt_tps = Some(value);
        }
        if let Some(value) = json_f64(timings, "predicted_per_second").filter(|value| *value > 0.0) {
            acc.decode_tps = Some(value);
        }
        if let Some(value) = json_u64(timings, "prompt_n").filter(|value| *value > 0) {
            acc.prompt_tokens = Some(value);
        }
        if let Some(value) = json_u64(timings, "predicted_n").filter(|value| *value > 0) {
            acc.predicted_tokens = Some(value);
        }
    }

    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            if let Some(delta) = choice.get("delta") {
                let mut text = delta_text(delta.get("content"));
                if text.is_empty() {
                    text = delta_text(delta.get("reasoning_content"));
                }
                if text.is_empty() {
                    text = delta_text(delta.get("reasoning"));
                }
                if !text.is_empty() {
                    acc.tokens = acc.tokens.saturating_add(1);
                    if acc.ttft_ms.is_none() {
                        acc.ttft_ms = Some(started.elapsed().as_millis() as u64);
                    }
                }
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct SseBuffer {
    buffer: Vec<u8>,
}

impl SseBuffer {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        loop {
            let Some((index, width)) = event_boundary(&self.buffer) else {
                break;
            };
            let raw: Vec<u8> = self.buffer.drain(..index + width).collect();
            if let Some(payload) = event_data(&String::from_utf8_lossy(&raw)) {
                events.push(payload);
            }
        }
        events
    }

    fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let raw: Vec<u8> = std::mem::take(&mut self.buffer);
        event_data(&String::from_utf8_lossy(&raw))
    }
}

fn event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) => {
            if left <= right {
                Some((left, 2))
            } else {
                Some((right, 4))
            }
        }
        (Some(index), None) => Some((index, 2)),
        (None, Some(index)) => Some((index, 4)),
        (None, None) => None,
    }
}

fn event_data(raw: &str) -> Option<String> {
    let mut lines = Vec::new();
    for line in raw.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("data:") {
            lines.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// 200 ms process telemetry sampler
// ---------------------------------------------------------------------------

struct TelemetryAccum {
    pid: u32,
    io_available: bool,
    baseline_read_bytes: u64,
    last_at: Instant,
    last_read_bytes: u64,
    last_read_ops: u64,
    peak_rss_bytes: u64,
    peak_ssd_mbps: f64,
    peak_iops: f64,
    last_ssd_mbps: f64,
    last_iops: f64,
    ssd_bytes_read: u64,
}

struct TelemetrySummary {
    io_available: bool,
    peak_rss_bytes: u64,
    ssd_bytes_read: u64,
    peak_ssd_mbps: f64,
    peak_iops: f64,
}

fn io_telemetry_available(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string(format!("/proc/{pid}/io")).is_ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        false
    }
}

impl TelemetryAccum {
    fn new(pid: u32) -> Self {
        let (read_bytes, read_ops, rss_bytes) = process_counters(pid);
        Self {
            pid,
            io_available: io_telemetry_available(pid),
            baseline_read_bytes: read_bytes,
            last_at: Instant::now(),
            last_read_bytes: read_bytes,
            last_read_ops: read_ops,
            peak_rss_bytes: rss_bytes,
            peak_ssd_mbps: 0.0,
            peak_iops: 0.0,
            last_ssd_mbps: 0.0,
            last_iops: 0.0,
            ssd_bytes_read: 0,
        }
    }

    fn observe(&mut self) {
        let (read_bytes, read_ops, rss_bytes) = process_counters(self.pid);
        let now = Instant::now();
        let seconds = now.duration_since(self.last_at).as_secs_f64().max(0.001);
        self.last_ssd_mbps = 0.0;
        self.last_iops = 0.0;
        if read_bytes >= self.last_read_bytes {
            let mbps = (read_bytes - self.last_read_bytes) as f64 / seconds / (1024.0 * 1024.0);
            self.last_ssd_mbps = mbps;
            self.peak_ssd_mbps = self.peak_ssd_mbps.max(mbps);
        }
        if read_ops >= self.last_read_ops {
            let iops = (read_ops - self.last_read_ops) as f64 / seconds;
            self.last_iops = iops;
            self.peak_iops = self.peak_iops.max(iops);
        }
        self.peak_rss_bytes = self.peak_rss_bytes.max(rss_bytes);
        self.ssd_bytes_read = read_bytes.saturating_sub(self.baseline_read_bytes);
        self.last_at = now;
        self.last_read_bytes = read_bytes;
        self.last_read_ops = read_ops;
    }

    fn summary(&self) -> TelemetrySummary {
        TelemetrySummary {
            io_available: self.io_available,
            peak_rss_bytes: self.peak_rss_bytes,
            ssd_bytes_read: self.ssd_bytes_read,
            peak_ssd_mbps: self.peak_ssd_mbps,
            peak_iops: self.peak_iops,
        }
    }
}

fn lock_accum(accum: &Arc<StdMutex<TelemetryAccum>>) -> std::sync::MutexGuard<'_, TelemetryAccum> {
    accum
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

async fn stream_with_telemetry<R: Runtime>(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    target: &BenchmarkTarget,
    workload: &BenchmarkWorkload,
    pid: u32,
    cancel: &AtomicBool,
    progress: &ProgressContext<R>,
    phase: BenchmarkPhase,
) -> (Result<StreamOutcome, BenchFailure>, TelemetrySummary) {
    let live = Arc::new(Mutex::new(LiveCounters::default()));
    let accum = Arc::new(StdMutex::new(TelemetryAccum::new(pid)));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = spawn_sampler(
        progress.clone(),
        live.clone(),
        accum.clone(),
        stop.clone(),
        phase,
    );
    let outcome = stream_completion(
        client,
        url,
        api_key,
        target,
        workload,
        cancel,
        Some(live),
    )
    .await;

    stop.store(true, Ordering::Relaxed);
    sampler.abort();
    let _ = sampler.await;
    let telemetry = {
        let mut guard = lock_accum(&accum);
        // The final sample closes the gap between the last 200 ms tick and the
        // end of the stream, so the totals are not rounded down by a tick.
        guard.observe();
        guard.summary()
    };
    (outcome, telemetry)
}

fn spawn_sampler<R: Runtime>(
    progress: ProgressContext<R>,
    live: Arc<Mutex<LiveCounters>>,
    accum: Arc<StdMutex<TelemetryAccum>>,
    stop: Arc<AtomicBool>,
    phase: BenchmarkPhase,
) -> JoinHandle<()> {
    let started = Instant::now();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(SAMPLE_INTERVAL_MS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let (rss_bytes, read_bytes_per_sec, read_ops_per_sec) = {
                let mut guard = lock_accum(&accum);
                guard.observe();
                (
                    guard.peak_rss_bytes,
                    guard.last_ssd_mbps * 1024.0 * 1024.0,
                    guard.last_iops,
                )
            };
            let (tokens, tps) = {
                let guard = live.lock().await;
                (guard.tokens, guard.tps)
            };
            progress.emit(
                phase,
                tokens,
                tps,
                started.elapsed().as_millis() as u64,
                rss_bytes,
                read_bytes_per_sec,
                read_ops_per_sec,
                match phase {
                    BenchmarkPhase::Warmup => "Warming up the page cache",
                    _ => "Measuring",
                },
            );
        }
    })
}

// ---------------------------------------------------------------------------
// Real-engine verification harness
// ---------------------------------------------------------------------------

/// This ignored test runs the actual suite against the local `.jns` and `.gguf`
/// models with the managed engine binaries. It exists so engine switching and
/// SSE parsing can be exercised against real servers, not the fake harness:
///
/// ```text
/// cargo test --manifest-path src-tauri/Cargo.toml real_two_target_suite \
///   -- --ignored --nocapture --test-threads=1
/// ```
///
/// Override the automatic paths with `JANAS_BENCH_JNS`, `JANAS_BENCH_GGUF` and
/// `JANAS_BENCH_BIN` when the models or binaries live elsewhere.
#[cfg(all(test, target_os = "linux"))]
mod real_suite {
    use super::*;
    use crate::{
        engine::EngineManager,
        engines::set_test_bin_dir,
        settings::SettingsStore,
    };
    use tauri::{
        test::{mock_app, MockRuntime},
        Listener, Manager,
    };

    fn home() -> PathBuf {
        PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
    }

    fn free_port() -> u16 {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
            .expect("bind a free loopback port");
        listener.local_addr().expect("local addr").port()
    }

    fn env_path(key: &str) -> Option<PathBuf> {
        std::env::var_os(key)
            .map(PathBuf::from)
            .filter(|path| path.is_file())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "runs janas-server and llama-server against the local models"]
    async fn real_two_target_suite() {
        let data_dir = home().join(".local/share/dev.mikrotitan.studio");
        let config_dir = home().join(".config/dev.mikrotitan.studio");
        let store = SettingsStore::load(config_dir, data_dir.clone(), None)
            .expect("load the real Studio settings");
        let mut settings = store.get();

        let bin = env_path_dir("JANAS_BENCH_BIN").unwrap_or_else(|| data_dir.join("bin"));
        assert!(bin.join("janas-server").is_file(), "missing janas-server in {}", bin.display());
        assert!(bin.join("llama-server").is_file(), "missing llama-server in {}", bin.display());
        set_test_bin_dir(bin.clone());

        let jns = env_path("JANAS_BENCH_JNS")
            .or_else(|| {
                let path = PathBuf::from(settings.model_path.trim());
                path.is_file().then_some(path)
            })
            .expect("no .jns model: set JANAS_BENCH_JNS or pick one in Settings");
        let gguf = env_path("JANAS_BENCH_GGUF")
            .or_else(|| {
                let path = PathBuf::from(settings.llama_model_path.trim());
                path.is_file().then_some(path)
            })
            .expect("no .gguf model: set JANAS_BENCH_GGUF or pick one in Settings");

        settings.host = "127.0.0.1".into();
        settings.port = free_port();
        settings.llama_port = free_port();
        while settings.llama_port == settings.port {
            settings.llama_port = free_port();
        }
        settings.api_key.clear();
        settings.extra_args.clear();
        settings.llama_extra_args.clear();
        settings.mtp_mode = "auto".into();
        settings.mtp_path.clear();
        settings.draft_path.clear();
        settings.ssd_streaming = false;
        settings.ssd_cache_dir.clear();
        settings.llama_mmproj_path.clear();
        settings.llama_mtp_path.clear();
        settings.llama_mtp_mode = "off".into();
        settings.llama_draft_path.clear();
        settings.llama_direct_cpu_moe = false;
        settings.llama_cpu_threads = 8;
        settings.llama_gpu_layers = 0;
        settings.llama_ctx_size = 4_096;
        settings.ctx_size = 4_096;
        settings.llama_flash_attn = "auto".into();
        settings.llama_kv_cache_k = "f16".into();
        settings.llama_kv_cache_v = "f16".into();

        let targets = vec![
            BenchmarkTarget {
                id: "real-janas".into(),
                label: "Qwen3.8 4B (JNS)".into(),
                engine: EngineKind::Janas,
                model_path: jns.to_string_lossy().to_string(),
                ctx_size: 4_096,
                gpu_layers: 0,
                extra_flags: "JANAS_ATTN=int16 JANAS_KV=8".into(),
            },
            BenchmarkTarget {
                id: "real-llama".into(),
                label: "Spark 4B (GGUF)".into(),
                engine: EngineKind::Llama,
                model_path: gguf.to_string_lossy().to_string(),
                ctx_size: 4_096,
                gpu_layers: 0,
                extra_flags: "--threads 8".into(),
            },
        ];
        let mut workload = preset_workload("short-reasoning").expect("preset");
        workload.max_tokens = 32;
        let token_cap = workload.max_tokens;

        let app = mock_app();
        let handle = app.handle().clone();
        let phases = Arc::new(StdMutex::new(Vec::<Value>::new()));
        let phases_for_listener = phases.clone();
        let _progress_listener = app.listen("benchmark://progress", move |event| {
            if let Ok(value) = serde_json::from_str::<Value>(event.payload()) {
                phases_for_listener
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(value);
            }
        });

        let manager = Arc::new(EngineManager::new(settings.port));
        app.manage(manager.clone());
        let supervisor = Arc::new(EngineSupervisor::<MockRuntime>::new_mock(manager));
        app.manage(supervisor.clone());

        let root = std::env::temp_dir().join(format!(
            "janas-real-bench-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("create real benchmark root");
        let runner = Arc::new(
            BenchmarkRunner::load(root.join("benchmarks")).expect("load benchmark runner"),
        );
        let suite_id = uuid::Uuid::new_v4().to_string();
        let cancel = Arc::new(AtomicBool::new(false));

        execute_suite(
            handle.clone(),
            runner.clone(),
            supervisor.clone(),
            settings.clone(),
            targets,
            workload,
            true,
            suite_id.clone(),
            cancel,
        )
        .await;

        let suites = runner.list();
        let suite = suites
            .iter()
            .find(|suite| suite.id == suite_id)
            .expect("finished suite was persisted");
        println!("\n=== REAL SUITE ===\n{}\n", serde_json::to_string_pretty(suite).unwrap());

        assert_eq!(suite.status, BenchmarkStatus::Completed, "suite error: {:?}", suite.error);
        assert_eq!(suite.results.len(), 2, "both targets must produce a result");
        for result in &suite.results {
            assert_eq!(
                result.status,
                BenchmarkStatus::Completed,
                "{} failed: {:?} (warnings: {:?})",
                result.target_label,
                result.error,
                result.adapter_warnings
            );
            assert!(result.ttft_ms > 0, "{} reported no TTFT", result.target_label);
            assert!(
                result.tokens_generated > 0,
                "{} generated no tokens",
                result.target_label
            );
            assert!(
                result.tokens_generated <= u64::from(token_cap),
                "{} exceeded the {token_cap}-token cap with {} tokens",
                result.target_label,
                result.tokens_generated
            );
            assert!(
                result.total_duration_ms >= result.ttft_ms,
                "{} duration is shorter than TTFT",
                result.target_label
            );
        }

        let events = phases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let warmup: Vec<&Value> = events
            .iter()
            .filter(|event| event.get("phase").and_then(Value::as_str) == Some("warmup"))
            .collect();
        assert!(!warmup.is_empty(), "warm-up emitted no progress events");
        assert!(
            warmup.iter().any(|event| {
                event.get("rss_bytes").and_then(Value::as_u64).unwrap_or(0) > 0
            }),
            "warm-up progress had no live RSS telemetry: {:?}",
            warmup
        );
        assert!(
            warmup.iter().any(|event| {
                event.get("tokens_generated").and_then(Value::as_u64).unwrap_or(0) > 0
            }),
            "warm-up progress never reported live tokens: {:?}",
            warmup
        );
        assert!(
            warmup.iter().any(|event| {
                event.get("read_bytes_per_sec").and_then(Value::as_f64).is_some()
                    && event.get("read_ops_per_sec").and_then(Value::as_f64).is_some()
            }),
            "warm-up progress had no live SSD telemetry: {:?}",
            warmup
        );
        assert!(
            events.iter().any(|event| {
                event.get("phase").and_then(Value::as_str) == Some("measuring")
                    && event
                        .get("tokens_generated")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        > 0
            }),
            "measured progress never reported live tokens"
        );
        assert!(
            warmup.iter().any(|event| {
                event.get("live_tps").and_then(Value::as_f64).unwrap_or(0.0) > 0.0
            }),
            "warm-up progress never reported a decode-only live rate: {:?}",
            warmup
        );

        let _ = fs::remove_dir_all(&root);
    }

    fn env_path_dir(key: &str) -> Option<PathBuf> {
        std::env::var_os(key)
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
    }
}

// ---------------------------------------------------------------------------
// Markdown report
// ---------------------------------------------------------------------------

fn md_escape(value: &str) -> String {
    value
        .replace('|', "\\|")
        .replace(['\r', '\n'], " ")
        .trim()
        .to_string()
}

fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes == 0 {
        return "0 B".into();
    }
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn fmt_number(value: f64) -> String {
    if value >= 1000.0 {
        format!("{value:.0}")
    } else if value >= 100.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

fn fmt_ms(value: u64) -> String {
    if value >= 1000 {
        format!("{:.2} s", value as f64 / 1000.0)
    } else {
        format!("{value} ms")
    }
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn format_utc(ms: u64) -> String {
    let seconds = ms / 1000;
    let days = (seconds / 86_400) as i64;
    let remainder = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = remainder / 3600;
    let minute = (remainder % 3600) / 60;
    let second = remainder % 60;
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

fn value_for(result: &BenchmarkRunResult, metric: &str) -> Option<f64> {
    if result.status != BenchmarkStatus::Completed {
        return None;
    }
    let value = match metric {
        "ttft" => result.ttft_ms as f64,
        "prefill" => result.prefill_tps,
        "decode" => result.decode_tps,
        "duration" => result.total_duration_ms as f64,
        "rss" => result.peak_rss_bytes as f64,
        "ssd_bytes" => result.ssd_bytes_read as f64,
        "ssd_mbps" => result.peak_ssd_mbps,
        "iops" => result.peak_iops,
        _ => return None,
    };
    value.is_finite().then_some(value)
}

fn best_indices(values: &[Option<f64>], lower_is_better: bool) -> Vec<usize> {
    let mut best: Option<f64> = None;
    let mut winners = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let Some(value) = value else { continue };
        if !value.is_finite() {
            continue;
        }
        let improves = match best {
            None => true,
            Some(current) if lower_is_better => *value < current - 1e-9,
            Some(current) => *value > current + 1e-9,
        };
        if improves {
            best = Some(*value);
            winners.clear();
            winners.push(index);
        } else if best.is_some_and(|current| (*value - current).abs() <= 1e-9) {
            winners.push(index);
        }
    }
    winners
}

fn metric_cell(metric: &str, result: &BenchmarkRunResult) -> String {
    if result.status != BenchmarkStatus::Completed {
        return "—".into();
    }
    match metric {
        "ttft" => fmt_ms(result.ttft_ms),
        "prefill" => format!("{} tok/s", fmt_number(result.prefill_tps)),
        "decode" => format!("{} tok/s", fmt_number(result.decode_tps)),
        "duration" => fmt_ms(result.total_duration_ms),
        "rss" => fmt_bytes(result.peak_rss_bytes),
        "ssd_bytes" => fmt_bytes(result.ssd_bytes_read),
        "ssd_mbps" => format!("{} MB/s", fmt_number(result.peak_ssd_mbps)),
        "iops" => format!("{} read calls/s", fmt_number(result.peak_iops)),
        _ => "—".into(),
    }
}

/// Format a complete GitHub-ready report with hardware, workload, per-target
/// configuration, winner highlighting and adapter warnings.
pub fn render_markdown(suite: &BenchmarkSuite) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    let _ = writeln!(out, "# Side-by-Side Benchmark: {}", md_escape(&suite.workload.label));
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Generated {} · MikroTitan {} · suite `{}`",
        format_utc(suite.created_at_ms),
        env!("CARGO_PKG_VERSION"),
        suite.id
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "**Status:** {:?} · **Warm-up pass:** {}",
        suite.status,
        if suite.warmup {
            "one discarded pass first"
        } else {
            "none (measured generation starts after /health)"
        }
    );
    if let Some(error) = suite.error.as_deref().filter(|value| !value.trim().is_empty()) {
        let _ = writeln!(out);
        let _ = writeln!(out, "> {error}");
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "## Hardware");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- **CPU:** {} ({} threads)",
        md_escape(&suite.hardware.cpu),
        suite.hardware.cores
    );
    let _ = writeln!(
        out,
        "- **Memory:** {}",
        if suite.hardware.mem_total_bytes > 0 {
            fmt_bytes(suite.hardware.mem_total_bytes)
        } else {
            "unknown".into()
        }
    );
    let _ = writeln!(
        out,
        "- **GPU:** {}",
        if suite.hardware.gpu.present {
            if let (Some(used), Some(total)) = (
                suite.hardware.gpu.memory_used,
                suite.hardware.gpu.memory_total,
            ) {
                format!(
                    "{} ({} / {} VRAM)",
                    md_escape(&suite.hardware.gpu.name),
                    fmt_bytes(used),
                    fmt_bytes(total)
                )
            } else {
                md_escape(&suite.hardware.gpu.name)
            }
        } else {
            "none detected through sysfs".into()
        }
    );
    if suite.hardware.npu.present {
        let _ = writeln!(out, "- **NPU:** {}", md_escape(&suite.hardware.npu.name));
    }
    let _ = writeln!(
        out,
        "- **OS / architecture:** {} / {}",
        suite.hardware.os, suite.hardware.arch
    );

    let _ = writeln!(out);
    let _ = writeln!(out, "## Workload");
    let _ = writeln!(out);
    let _ = writeln!(out, "- **Preset:** `{}` — {}", suite.workload.preset, md_escape(&suite.workload.label));
    let _ = writeln!(out, "- **Seed:** {}", suite.workload.seed);
    let _ = writeln!(out, "- **Temperature:** 0.0");
    let _ = writeln!(out, "- **Max tokens:** {}", suite.workload.max_tokens);
    let _ = writeln!(out, "- **Structured JSON:** {}", if suite.workload.structured { "yes" } else { "no" });
    let _ = writeln!(out);
    let _ = writeln!(out, "<details><summary>Prompt preview</summary>");
    let _ = writeln!(out);
    let _ = writeln!(out, "```text");
    let _ = writeln!(out, "{}", truncate_line(suite.workload.prompt.trim(), 900));
    let _ = writeln!(out, "```");
    let _ = writeln!(out);
    let _ = writeln!(out, "</details>");

    let results = &suite.results;
    if results.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "_No target produced a result._");
        return out;
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "## Results");
    let _ = writeln!(out);
    let _ = write!(out, "| Metric |");
    for result in results {
        let _ = write!(out, " {} |", md_escape(&result.target_label));
    }
    let _ = writeln!(out);
    let _ = write!(out, "|---|");
    for _ in results {
        let _ = write!(out, "---:|");
    }
    let _ = writeln!(out);

    let metrics: [(&str, &str, bool); 8] = [
        ("ttft", "Time to first token", true),
        ("prefill", "Prompt eval", false),
        ("decode", "Generation", false),
        ("duration", "Total time", true),
        ("rss", "Peak process RSS (RAM)", true),
        ("ssd_bytes", "SSD bytes read (generation)", false),
        ("ssd_mbps", "Peak SSD read (generation)", false),
        ("iops", "Peak read calls/s", false),
    ];
    for (key, label, lower_is_better) in metrics {
        let values: Vec<Option<f64>> = results
            .iter()
            .map(|result| value_for(result, key))
            .collect();
        let winners = best_indices(&values, lower_is_better);
        let winner_label = if winners.len() > 1 { "tie" } else { "best" };
        let _ = write!(out, "| **{label}** |");
        for (index, result) in results.iter().enumerate() {
            let cell = metric_cell(key, result);
            if winners.contains(&index) {
                let _ = write!(out, " **{cell}** ({winner_label}) |");
            } else {
                let _ = write!(out, " {cell} |");
            }
        }
        let _ = writeln!(out);
    }
    let _ = write!(out, "| **Tokens generated** |");
    for result in results {
        if result.status == BenchmarkStatus::Completed {
            let _ = write!(out, " {} |", result.tokens_generated);
        } else {
            let _ = write!(out, " — |");
        }
    }
    let _ = writeln!(out);
    let _ = write!(out, "| **Status** |");
    for result in results {
        let _ = write!(out, " {:?} |", result.status);
    }
    let _ = writeln!(out);

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "_Lower is better for latency and RSS; higher is better for token rates and generation-only SSD/read activity, so the engine that streams the most from the drive wins those rows._"
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "_Thinking is disabled for every request, so `max_tokens` normally bounds the whole reply; an engine that ignores the template flag is flagged under warnings. SSD and read-call figures cover the measured generation only: model loading is excluded, and page-cache reads count as calls but not as SSD bytes, so 0 B with non-zero calls is normal._"
    );

    let _ = writeln!(out);
    let _ = writeln!(out, "## Target configuration");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Target | Engine | Backend | Version | Context | GPU layers | Extra flags |");
    let _ = writeln!(out, "|---|---|---|---|---:|---:|---|");
    for (index, target) in suite.targets.iter().enumerate() {
        let result = results.get(index);
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | `{}` |",
            md_escape(&target.label),
            target.engine.label(),
            result
                .and_then(|value| value.adapter.clone())
                .unwrap_or_else(|| "—".into()),
            result
                .and_then(|value| value.engine_version.clone())
                .map(|value| md_escape(&value))
                .unwrap_or_else(|| "—".into()),
            target.ctx_size,
            if target.engine == EngineKind::Llama {
                target.gpu_layers.to_string()
            } else {
                "n/a".into()
            },
            md_escape(&target.extra_flags)
        );
    }

    let warnings: Vec<(&str, &str)> = results
        .iter()
        .flat_map(|result| {
            result
                .adapter_warnings
                .iter()
                .map(move |warning| (result.target_label.as_str(), warning.as_str()))
        })
        .collect();
    let failures: Vec<(&str, &str)> = results
        .iter()
        .filter_map(|result| {
            result
                .error
                .as_deref()
                .map(|error| (result.target_label.as_str(), error))
        })
        .collect();
    if !failures.is_empty() || !warnings.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "## Warnings and failures");
        for (target, error) in failures {
            let _ = writeln!(out, "- **{}**: {}", md_escape(target), md_escape(error));
        }
        for (target, warning) in warnings {
            let _ = writeln!(out, "- **{}**: {}", md_escape(target), md_escape(warning));
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Report generated by MikroTitan — every prompt and byte stayed on this machine."
    );
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::EngineKind;
    use crate::system::DeviceMetrics;

    fn sample_suite(first_decode: f64, second_decode: f64) -> BenchmarkSuite {
        let workload = preset_workload("short-reasoning").expect("preset");
        let target = |id: &str, label: &str, engine: EngineKind| BenchmarkTarget {
            id: id.into(),
            label: label.into(),
            engine,
            model_path: format!("/tmp/{id}.{}", if engine == EngineKind::Janas { "jns" } else { "gguf" }),
            ctx_size: 8192,
            gpu_layers: 0,
            extra_flags: String::new(),
        };
        let result = |label: &str, engine: EngineKind, decode: f64| BenchmarkRunResult {
            target_id: label.to_lowercase(),
            target_label: label.into(),
            engine,
            ttft_ms: 120,
            prefill_tps: 400.0,
            decode_tps: decode,
            total_duration_ms: 2_000,
            tokens_generated: 128,
            peak_rss_bytes: 2 * 1024 * 1024 * 1024,
            ssd_bytes_read: 1024 * 1024 * 1024,
            peak_ssd_mbps: 900.0,
            peak_iops: 4_000.0,
            adapter_warnings: Vec::new(),
            status: BenchmarkStatus::Completed,
            error: None,
            engine_version: Some("test build".into()),
            adapter: Some("test".into()),
        };

        BenchmarkSuite {
            id: "suite-1".into(),
            created_at_ms: 1_700_000_000_000,
            workload,
            warmup: false,
            targets: vec![
                target("janas", "Janas", EngineKind::Janas),
                target("llama", "llama.cpp", EngineKind::Llama),
            ],
            results: vec![
                result("Janas", EngineKind::Janas, first_decode),
                result("llama.cpp", EngineKind::Llama, second_decode),
            ],
            hardware: HardwareSummary {
                os: "linux".into(),
                arch: "x86_64".into(),
                cpu: "Test CPU".into(),
                cores: 8,
                mem_total_bytes: 16 * 1024 * 1024 * 1024,
                gpu: DeviceMetrics::default(),
                npu: DeviceMetrics::default(),
            },
            status: BenchmarkStatus::Completed,
            error: None,
            duration_ms: 12_000,
        }
    }

    #[test]
    fn built_in_presets_are_present_and_unique() {
        let presets = workload_presets();
        let ids: Vec<&str> = presets.iter().map(|preset| preset.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "short-reasoning",
                "long-context-prefill",
                "code-generation",
                "json-structured"
            ]
        );
        let unique: HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len());
        for preset in &presets {
            assert!(!preset.prompt.trim().is_empty());
            assert!(preset.max_tokens > 0);
        }
        let json = presets
            .iter()
            .find(|preset| preset.id == "json-structured")
            .expect("json preset");
        assert!(json.structured);
    }

    #[test]
    fn long_context_prompt_is_roughly_four_thousand_tokens() {
        let preset = preset_workload("long-context-prefill").expect("preset");
        let characters = preset.prompt.chars().count();
        assert!(
            (12_000..24_000).contains(&characters),
            "expected roughly 3k–6k tokens, got {characters} characters"
        );
    }

    #[test]
    fn extra_flags_split_environment_from_arguments() {
        let (environment, arguments) = split_extra_flags(
            "JANAS_ATTN=int16 JANAS_KV=8 --cpu-moe --cache-type-k=q8_0",
        )
        .expect("flags");
        assert_eq!(
            environment,
            vec![
                ("JANAS_ATTN".into(), "int16".into()),
                ("JANAS_KV".into(), "8".into())
            ]
        );
        assert_eq!(
            arguments,
            vec!["--cpu-moe".to_string(), "--cache-type-k=q8_0".to_string()]
        );
    }

    #[test]
    fn sse_events_survive_chunk_boundaries() {
        let mut buffer = SseBuffer::default();
        assert!(buffer.push(b"data: {\"a\":1}\n").is_empty());
        let events = buffer.push(b"\ndata: [DONE]\n\n");
        assert_eq!(events, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
        assert!(buffer.finish().is_none());
    }

    #[test]
    fn payload_reports_usage_and_timings() {
        let mut acc = StreamAccum::default();
        let started = Instant::now();
        let payload = r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}],"usage":{"prompt_tokens":4000,"completion_tokens":64},"timings":{"prompt_per_second":520.5,"predicted_per_second":18.25}}"#;
        apply_payload(payload, &mut acc, started).expect("payload");
        assert_eq!(acc.tokens, 1);
        assert!(acc.ttft_ms.is_some());
        assert_eq!(acc.prompt_tokens, Some(4000));
        assert_eq!(acc.completion_tokens, Some(64));
        assert_eq!(acc.prompt_tps, Some(520.5));
        assert_eq!(acc.decode_tps, Some(18.25));
    }

    #[test]
    fn benchmark_request_disables_thinking_and_honours_the_token_cap() {
        let workload = BenchmarkWorkload {
            preset: "short-reasoning".into(),
            label: "Short Reasoning".into(),
            prompt: "Count to five.".into(),
            seed: 42,
            temperature: 0.0,
            max_tokens: 32,
            structured: false,
        };
        let target = BenchmarkTarget {
            id: "janas-1".into(),
            label: "Janas".into(),
            engine: EngineKind::Janas,
            model_path: "/tmp/model.jns".into(),
            ctx_size: 4096,
            gpu_layers: 0,
            extra_flags: String::new(),
        };
        let body = request_body(&target, &workload);
        assert_eq!(body["max_tokens"], 32);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert!(body.get("timings_per_token").is_none());

        let llama = BenchmarkTarget {
            engine: EngineKind::Llama,
            model_path: "/tmp/model.gguf".into(),
            ..target
        };
        let body = request_body(&llama, &workload);
        assert_eq!(body["timings_per_token"], true);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn janas_usage_clocks_drive_prefill_and_decode_rates() {
        let mut acc = StreamAccum::default();
        let payload = r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"length"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"janas":{"input_seconds":0.5,"output_seconds":2.5}}}"#;
        apply_payload(payload, &mut acc, Instant::now()).expect("payload");
        assert_eq!(acc.prompt_tokens, Some(100));
        assert_eq!(acc.completion_tokens, Some(50));
        assert_eq!(acc.prompt_tps, Some(200.0));
        assert_eq!(acc.decode_tps, Some(20.0));
    }

    #[test]
    fn zero_valued_llama_timings_do_not_erase_janas_clocks() {
        let mut acc = StreamAccum::default();
        let payload = r#"{"usage":{"prompt_tokens":100,"completion_tokens":50,"janas":{"input_seconds":0.5,"output_seconds":2.5}},"timings":{"prompt_per_second":0,"predicted_per_second":0,"prompt_n":0,"predicted_n":0}}"#;
        apply_payload(payload, &mut acc, Instant::now()).expect("payload");
        assert_eq!(acc.prompt_tokens, Some(100));
        assert_eq!(acc.completion_tokens, Some(50));
        assert_eq!(acc.prompt_tps, Some(200.0));
        assert_eq!(acc.decode_tps, Some(20.0));
    }

    #[test]
    fn token_cap_overruns_are_reported() {
        assert!(token_budget_warning(32, 32).is_none());
        let warning = token_budget_warning(33, 32).expect("overrun warning");
        assert!(warning.contains("33 tokens"));
        assert!(warning.contains("enable_thinking"));
    }

    #[test]
    fn live_tps_uses_the_decode_clock() {
        assert_eq!(live_decode_tps(1, 500, 600), 0.0);
        assert_eq!(live_decode_tps(11, 1_000, 2_000), 10.0);
        assert_eq!(live_decode_tps(5, 2_000, 1_500), 0.0);
    }

    #[test]
    fn target_launch_settings_isolate_each_target() {
        let mut base = AppSettings::default();
        base.model_path = "/old/model.jns".into();
        base.llama_model_path = "/old/model.gguf".into();
        base.llama_mmproj_path = "/old/mmproj.gguf".into();
        base.llama_draft_path = "/old/draft.gguf".into();
        base.llama_extra_args = "--global-flag".into();
        let target = BenchmarkTarget {
            id: "llama-1".into(),
            label: "llama Q4".into(),
            engine: EngineKind::Llama,
            model_path: "/models/new.gguf".into(),
            ctx_size: 4096,
            gpu_layers: 12,
            extra_flags: "JANAS_ATTN=float --cpu-moe".into(),
        };
        let (settings, _) = target_launch_settings(&base, &target).expect("settings");
        assert_eq!(settings.llama_ctx_size, 4096);
        assert_eq!(settings.llama_gpu_layers, 12);
        assert_eq!(settings.llama_extra_args, "--cpu-moe");
        assert!(settings.llama_mmproj_path.is_empty());
        assert!(settings.llama_draft_path.is_empty());
        assert_eq!(settings.launch_env.len(), 1);
        assert_eq!(settings.launch_env[0].0, "JANAS_ATTN");
    }

    #[test]
    fn markdown_marks_the_decode_winner() {
        let suite = sample_suite(40.0, 25.0);
        let markdown = render_markdown(&suite);
        assert!(markdown.contains("# Side-by-Side Benchmark: Short Reasoning"));
        assert!(markdown.contains("**40.00 tok/s** (best)"));
        assert!(markdown.contains("## Hardware"));
        assert!(markdown.contains("## Target configuration"));
        assert!(markdown.contains("**Warm-up pass:** none"));
        assert!(markdown.contains("Thinking is disabled"));
        assert!(markdown.contains("generation-only"));
        assert!(!markdown.contains("## Warnings and failures"));
    }

    #[test]
    fn more_ssd_streaming_wins_the_io_rows() {
        let mut suite = sample_suite(40.0, 25.0);
        suite.results[0].ssd_bytes_read = 3 * 1024 * 1024 * 1024;
        suite.results[0].peak_ssd_mbps = 1_250.0;
        suite.results[0].peak_iops = 9_000.0;
        suite.results[1].ssd_bytes_read = 1024 * 1024 * 1024;
        suite.results[1].peak_ssd_mbps = 900.0;
        suite.results[1].peak_iops = 4_000.0;

        let markdown = render_markdown(&suite);
        assert!(markdown.contains("**3.00 GiB** (best)"));
        assert!(markdown.contains("**1250 MB/s** (best)"));
        assert!(markdown.contains("**9000 read calls/s** (best)"));
    }

    #[test]
    fn equal_metric_values_are_ties_not_an_arbitrary_winner() {
        let values = [Some(0.0), Some(0.0), Some(1.0), None];
        assert_eq!(best_indices(&values, true), vec![0, 1]);
        assert_eq!(best_indices(&values, false), vec![2]);

        let suite = sample_suite(40.0, 40.0);
        let markdown = render_markdown(&suite);
        assert_eq!(markdown.matches("**40.00 tok/s** (tie)").count(), 2);
        assert!(!markdown.contains("(best)"));
    }

    #[test]
    fn markdown_escapes_pipes_in_labels() {
        let mut suite = sample_suite(10.0, 12.0);
        suite.targets[0].label = "A | B".into();
        suite.results[0].target_label = "A | B".into();
        let markdown = render_markdown(&suite);
        assert!(markdown.contains("A \\| B"));
    }

    #[test]
    fn suite_ids_are_restricted_to_safe_filenames() {
        assert!(valid_suite_id("4b2c27c0-0000-4000-8000-000000000000"));
        assert!(!valid_suite_id("../escape"));
        assert!(!valid_suite_id(""));
        assert!(!valid_suite_id("has/slash"));
    }

    #[test]
    fn benchmark_store_round_trips_and_deletes() {
        let dir = std::env::temp_dir().join(format!(
            "janas-benchmark-store-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let store = BenchmarkStore::new(dir.clone()).expect("store");
        let suite = sample_suite(30.0, 20.0);
        store.save(&suite).expect("save");
        let loaded = store.load_all();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, suite.id);
        assert_eq!(loaded[0].results.len(), 2);
        assert!(store.delete(&suite.id).expect("delete"));
        assert!(store.load_all().is_empty());
        assert!(!store.delete(&suite.id).expect("second delete"));
        let _ = fs::remove_dir_all(&dir);
    }
}
