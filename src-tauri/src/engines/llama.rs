//! llama.cpp runner and version-adaptive CLI adapter.
//!
//! The release archive is installed into the managed bin directory. Some Linux
//! builds link against `libllama.so` / `libggml*.so`; those companions are
//! installed beside the binary and `LD_LIBRARY_PATH` is scoped to the child.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::process::Command;

use crate::{
    error::{AppError, AppResult},
    settings::{self, AppSettings},
    util::now_ms,
};

use super::{
    advertises, apply_library_path, combine_warnings, managed_bin_dir, managed_binary_path,
    pid_alive, probe_spawn_error, process_counters, run_cli_probe, signal_group,
    spawn_log_reader, wait_for_exit, write_pidfile, EngineKind, EngineLaunch,
    EngineProbe, EngineRunner, EngineSupervisor, EngineTelemetry, ProbeOutput,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashSyntax {
    /// This build has no `--flash-attn` flag.
    None,
    /// Old builds accept a bare `--flash-attn` switch.
    Bare,
    /// Current builds accept `--flash-attn on|off|auto`.
    Choice,
}

#[derive(Debug, Clone)]
pub struct LlamaLayout {
    pub adapter: String,
    pub model_flag: &'static str,
    pub ctx_flag: Option<&'static str>,
    pub ngl_flag: Option<&'static str>,
    pub flash: FlashSyntax,
    pub kv_cache: bool,
    pub metrics: bool,
    pub jinja: bool,
    pub top_k: bool,
    pub alias: bool,
    /// The flag this build uses for a multimodal projector, if it advertises
    /// vision support.
    pub mmproj_flag: Option<&'static str>,
    /// This build advertises the PR #30966 MoE cache pipeline: host experts,
    /// a device-side cache, and CPU execution for the cache misses.
    pub moe_cache_cpu_share: bool,
    /// This build advertises `--spec-type`, which is where `draft-mtp` and
    /// `none` are selected. Older builds only take a draft model.
    pub spec_type: bool,
    /// The speculative draft-model flag this build uses, if it has one.
    pub draft_model_flag: Option<&'static str>,
    /// Where a draft model runs: `--spec-draft-device` or `--device-draft`.
    pub draft_device_flag: Option<&'static str>,
    /// How many draft layers live on the GPU: `--spec-draft-ngl` or an older
    /// `--gpu-layers-draft` alias.
    pub draft_ngl_flag: Option<&'static str>,
    /// `--draft-max`: how many tokens the draft model proposes per step.
    pub draft_max_flag: Option<&'static str>,
    /// `--draft-min`: the shortest accepted draft run.
    pub draft_min_flag: Option<&'static str>,
    /// `--draft-p-min`: the minimum draft probability llama.cpp keeps.
    pub draft_p_min_flag: Option<&'static str>,
}

pub struct LlamaRunner;

impl LlamaRunner {
    pub fn new() -> Self {
        Self
    }

    fn binary_path<R: Runtime>(app: &AppHandle<R>) -> AppResult<PathBuf> {
        let path = managed_binary_path(app, EngineKind::Llama)?;
        if !path.is_file() {
            return Err(AppError::message(format!(
                "llama-server is not installed at {}. Open Settings → llama.cpp sidecar, paste a llama.cpp release archive URL, and install it. Binaries live only in the managed bin folder."
            , path.display())));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path)
                .map_err(|err| super::io_error("Could not inspect llama-server", &path, err))?
                .permissions()
                .mode();
            if mode & 0o111 == 0 {
                return Err(AppError::message(format!(
                    "{} is not executable. Reinstall it from Settings or run `chmod 755 {}`.",
                    path.display(),
                    path.display()
                )));
            }
        }
        Ok(path)
    }

    async fn probe_full<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        settings: &AppSettings,
    ) -> AppResult<(ProbeOutput, LlamaLayout, EngineProbe)> {
        let path = Self::binary_path(app)?;
        let bin = managed_bin_dir(app)?;
        let output = run_cli_probe(&path, Some(&bin)).await?;
        let (layout, mut warnings) = layout_for(&output, settings);
        if output.version_line.is_none() {
            warnings.push(
                "The binary did not report a version line; the launch flags were inferred from its help text."
                    .into(),
            );
        }
        let probe = EngineProbe {
            version: output.version_line.clone(),
            build: output.build,
            adapter: layout.adapter.clone(),
            warning: combine_warnings(None, warnings),
        };
        Ok((output, layout, probe))
    }

    fn validate_port(settings: &AppSettings) -> AppResult<()> {
        let host = bind_host(settings);
        let port = settings.llama_port;
        if port == 0 {
            return Err(AppError::message(
                "The llama.cpp port must be between 1 and 65535.",
            ));
        }
        let listener = std::net::TcpListener::bind((host.as_str(), port)).map_err(|err| {
            AppError::message(format!(
                "Port {port} on {host} is not available ({err}). Stop the process using it or choose another llama.cpp port."
            ))
        })?;
        drop(listener);
        Ok(())
    }
}

#[async_trait]
impl<R: Runtime> EngineRunner<R> for LlamaRunner {
    async fn spawn_process(
        &self,
        app: &AppHandle<R>,
        settings: &AppSettings,
    ) -> AppResult<EngineLaunch> {
        Self::validate_port(settings)?;
        let path = Self::binary_path(app)?;
        let bin = managed_bin_dir(app)?;
        let (output, layout, mut probe) = self.probe_full(app, settings).await?;
        let (mut args, mut build_warnings) = build_args(settings, &layout)?;
        if let Some(warning) = guard_vulkan_swa_kv_cache(&mut args, settings, &output) {
            build_warnings.push(warning);
        }
        let context_window = if layout.ctx_flag.is_some() {
            settings.llama_ctx_size.clamp(512, 1_048_576)
        } else {
            // Without a context flag the binary keeps its own default, which
            // is the only safe value to report to the request budget.
            16_384
        };
        probe.warning = combine_warnings(probe.warning.take(), build_warnings);
        let bind_host = bind_host(settings);
        let connect_host = settings.connect_host().to_string();

        let mut command = Command::new(&path);
        command
            .args(&args)
            .current_dir(&bin)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(false);
        apply_library_path(&mut command, Some(&bin));
        for (name, value) in &settings.launch_env {
            command.env(name, value);
        }
        #[cfg(unix)]
        command.process_group(0);

        let mut child = command
            .spawn()
            .map_err(|err| probe_spawn_error(&path, Some(&bin), err))?;
        let pid = child.id().ok_or_else(|| {
            AppError::message("llama-server started but did not expose a process id.")
        })?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let (exit_tx, exit_rx) = tokio::sync::watch::channel(None);

        let launch = EngineLaunch {
            kind: EngineKind::Llama,
            pid,
            host: connect_host.clone(),
            port: settings.llama_port,
            model: settings::expand_home(settings.llama_model_path.trim())
                .to_string_lossy()
                .to_string(),
            context_window,
            started_at_ms: now_ms(),
            probe,
            metrics_enabled: layout.metrics,
            exit_rx: Some(exit_rx),
        };
        write_pidfile(app, &launch);

        if let Some(stdout) = stdout {
            spawn_log_reader(app.clone(), stdout, "stdout");
        }
        if let Some(stderr) = stderr {
            spawn_log_reader(app.clone(), stderr, "stderr");
        }

        let app_for_reaper = app.clone();
        let pid_for_reaper = pid;
        tauri::async_runtime::spawn(async move {
            let status = child.wait().await;
            let code = status.ok().and_then(|status| status.code());
            let _ = exit_tx.send(Some(code.unwrap_or(-1)));
            super::remove_pidfile(&app_for_reaper);
            if let Some(supervisor) = supervisor_handle(&app_for_reaper) {
                let _ = supervisor
                    .clear_if_pid(&app_for_reaper, pid_for_reaper, Some(code.unwrap_or(-1)))
                    .await;
            }
        });

        let _ = bind_host;
        Ok(launch)
    }

    async fn kill_process(&self, app: &AppHandle<R>, launch: &EngineLaunch) -> AppResult<()> {
        if !pid_alive(launch.pid) {
            super::remove_pidfile(app);
            return Ok(());
        }

        signal_group(launch.pid, libc::SIGTERM);
        let stopped = match &launch.exit_rx {
            Some(rx) => wait_for_exit(rx, Duration::from_secs(6)).await,
            None => false,
        };
        if !stopped {
            signal_group(launch.pid, libc::SIGKILL);
            if let Some(rx) = &launch.exit_rx {
                let _ = wait_for_exit(rx, Duration::from_secs(2)).await;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        if pid_alive(launch.pid) {
            return Err(AppError::message(format!(
                "llama-server (pid {}) did not exit after SIGTERM and SIGKILL. Check for a stuck child process before starting another engine.",
                launch.pid
            )));
        }
        super::remove_pidfile(app);
        Ok(())
    }

    async fn monitor_telemetry(
        &self,
        app: &AppHandle<R>,
        launch: EngineLaunch,
    ) -> AppResult<()> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let metrics_url = format!("http://{}:{}/metrics", launch.host, launch.port);
        let mut previous: Option<(Instant, u64, u64, u64)> = None;

        loop {
            if !pid_alive(launch.pid) {
                return Ok(());
            }
            let now = Instant::now();
            let (completion, prompt, kv) = if launch.metrics_enabled {
                fetch_metrics(&client, &metrics_url).await
            } else {
                (0, 0, 0.0)
            };
            let (read_bytes, read_ops, rss_bytes) = process_counters(launch.pid);
            let (tokens_per_second, read_bytes_per_sec, read_ops_per_sec) =
                if let Some((at, last_completion, last_read, last_ops)) = previous {
                    let seconds = now.duration_since(at).as_secs_f64().max(0.001);
                    (
                        completion.saturating_sub(last_completion) as f64 / seconds,
                        read_bytes.saturating_sub(last_read) as f64 / seconds,
                        read_ops.saturating_sub(last_ops) as f64 / seconds,
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };
            previous = Some((now, completion, read_bytes, read_ops));

            let payload = EngineTelemetry {
                engine: EngineKind::Llama,
                pid: Some(launch.pid),
                tokens_per_second: round2(tokens_per_second),
                prompt_tokens_total: prompt,
                completion_tokens_total: completion,
                kv_cache_usage: round4(kv),
                read_bytes_per_sec: read_bytes_per_sec.round() as u64,
                read_ops_per_sec: read_ops_per_sec.round() as u64,
                rss_bytes,
                sampled_at_ms: now_ms(),
            };
            let _ = app.emit("engine://telemetry", payload);
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}

/// Last llama.cpp build whose Vulkan path scheduled the KV cache of
/// sliding-window models correctly. Newer builds abort in `ggml_backend_sched`
/// before the server can answer `/health`; upstream issue #29082 tracks it.
const LAST_GOOD_VULKAN_KV_OFFLOAD_BUILD: u64 = 12_236;

/// llama.cpp's Vulkan backend cannot schedule the KV cache of a
/// sliding-window (hybrid) model when that cache lives in GPU memory:
///
/// `pre-allocated tensor (cache_k_l1 (view)) in a buffer (Vulkan0) that cannot
/// run the operation (SET_ROWS)`
///
/// The scheduler calls `GGML_ABORT`, so the server dies with SIGABRT before it
/// binds the port and the app only sees an immediate exit. No KV flag avoids
/// it, and `-nkvo` is not a free workaround: it leaves attention on the CPU
/// while the weights stay on the GPU. On the Vulkan-only hosts this affects,
/// running the whole model on the CPU measured roughly twice the throughput of
/// that split (`--device none`), so the adapter drops GPU offload instead.
/// Returns the warning that describes the substitution.
fn guard_vulkan_swa_kv_cache(
    args: &mut Vec<String>,
    settings: &AppSettings,
    output: &ProbeOutput,
) -> Option<String> {
    let advertises_flag =
        advertises(&output.help_text, "-nkvo") || advertises(&output.help_text, "--no-kv-offload");
    if !advertises_flag || !super::reports_vulkan(&output.devices) {
        return None;
    }
    // With no offloaded layers the cache never reaches a GPU buffer.
    if settings.llama_gpu_layers <= 0 {
        return None;
    }
    let risky = output
        .build
        .map(|build| build > LAST_GOOD_VULKAN_KV_OFFLOAD_BUILD)
        .unwrap_or(true);
    if !risky {
        return None;
    }
    // The user's own KV-cache or device flags always win over this fallback.
    let explicit = shell_words::split(&settings.llama_extra_args)
        .unwrap_or_default()
        .iter()
        .any(|argument| {
            matches!(
                argument.as_str(),
                "-kvo" | "--kv-offload" | "-nkvo" | "--no-kv-offload" | "--device" | "-dev"
            ) || argument.starts_with("--device=")
                || argument.starts_with("-dev=")
                || argument.starts_with("--kv-offload=")
                || argument.starts_with("--no-kv-offload=")
        });
    if explicit {
        return None;
    }
    let model = settings::expand_home(settings.llama_model_path.trim());
    if !crate::gguf::has_sliding_window(&model.to_string_lossy()) {
        return None;
    }

    let mut used_cpu = false;
    if advertises(&output.help_text, "--device") {
        args.push("--device".into());
        args.push("none".into());
        used_cpu = true;
    } else {
        args.push("-nkvo".into());
    }
    Some(match used_cpu {
        true => format!(
            "llama.cpp's Vulkan backend aborts while reserving the KV cache of sliding-window (hybrid) models when that cache is offloaded to the GPU (upstream issue #29082; b{LAST_GOOD_VULKAN_KV_OFFLOAD_BUILD} is the last known-good build). This launch runs the model on the CPU with `--device none`. Add `-nkvo` to Extra llama arguments to keep the weights on the GPU with a host KV cache, or `-kvo` to force GPU KV offload."
        ),
        false => format!(
            "llama.cpp's Vulkan backend aborts while reserving the KV cache of sliding-window (hybrid) models when that cache is offloaded to the GPU (upstream issue #29082; b{LAST_GOOD_VULKAN_KV_OFFLOAD_BUILD} is the last known-good build). This launch keeps the KV cache in system RAM with `-nkvo`. Add `-kvo` to Extra llama arguments to force GPU KV offload."
        ),
    })
}

fn bind_host(settings: &AppSettings) -> String {
    let host = settings.host.trim();
    if host.is_empty() {
        "127.0.0.1".into()
    } else {
        host.to_string()
    }
}

fn layout_for(output: &ProbeOutput, settings: &AppSettings) -> (LlamaLayout, Vec<String>) {
    let help = output.help_text.as_str();
    let help_missing = help.trim().is_empty();
    let build = output.build;
    let legacy = build.map(|value| value < 3_000).unwrap_or(false);
    let mut warnings = Vec::new();

    if help_missing {
        warnings.push(
            "`--help` returned no text; the adapter fell back to the standard llama-server flag layout."
                .into(),
        );
    }
    match build {
        None => warnings.push(
            "The llama-server build number could not be parsed; optional flags are only used when `--help` advertises them."
                .into(),
        ),
        Some(value) if !(3_000..=20_000).contains(&value) => warnings.push(format!(
            "llama-server build b{value} is outside the tested range (b3000–b20000). The adapter is using the closest known layout."
        )),
        _ => {}
    }

    let mmproj_flag = if advertises(help, "--mmproj") {
        Some("--mmproj")
    } else if advertises(help, "-mm") {
        Some("-mm")
    } else if help_missing {
        Some("--mmproj")
    } else {
        None
    };
    if mmproj_flag.is_none() && !settings.llama_mmproj_path.trim().is_empty() {
        warnings.push(
            "This llama-server build does not advertise `--mmproj`; the selected vision projector will not be loaded, so image input will stay unavailable. Update llama.cpp or remove the projector path."
                .into(),
        );
    }

    if legacy {
        warnings.push(
            "This is a legacy llama-server build. Context and GPU flags use the short forms, and KV-cache/telemetry flags are disabled."
                .into(),
        );
        return (
            LlamaLayout {
                adapter: format!("llama-server/classic{}", build_label(build)),
                model_flag: "--model",
                ctx_flag: Some("-c"),
                ngl_flag: Some("-ngl"),
                flash: FlashSyntax::None,
                kv_cache: false,
                metrics: false,
                jinja: false,
                top_k: false,
                alias: false,
                mmproj_flag,
                moe_cache_cpu_share: false,
                spec_type: false,
                draft_model_flag: None,
                draft_device_flag: None,
                draft_ngl_flag: None,
                draft_max_flag: None,
                draft_min_flag: None,
                draft_p_min_flag: None,
            },
            warnings,
        );
    }

    let model_flag = if advertises(help, "--model") || help_missing {
        "--model"
    } else if advertises(help, "-m") {
        "-m"
    } else {
        warnings.push(
            "Neither `--model` nor `-m` was found in the help text; using `--model` and falling back to the server's error if the build differs."
                .into(),
        );
        "--model"
    };
    let ctx_flag = if advertises(help, "--ctx-size") {
        Some("--ctx-size")
    } else if advertises(help, "-c") {
        Some("-c")
    } else if help_missing {
        Some("--ctx-size")
    } else {
        warnings.push(
            "No context-size flag was advertised; the context value is left to the binary's default."
                .into(),
        );
        None
    };
    let ngl_flag = if advertises(help, "--n-gpu-layers") {
        Some("--n-gpu-layers")
    } else if advertises(help, "--gpu-layers") {
        Some("--gpu-layers")
    } else if advertises(help, "-ngl") {
        Some("-ngl")
    } else if help_missing {
        Some("--n-gpu-layers")
    } else {
        warnings.push(
            "No GPU-layer flag was advertised; GPU offload is disabled for this build."
                .into(),
        );
        None
    };
    let flash = if advertises(help, "--flash-attn") || advertises(help, "-fa") {
        let tail = help
            .find("--flash-attn")
            .map(|index| &help[index..])
            .unwrap_or_default();
        let choices = tail.contains("on|off|auto")
            || tail.contains("on\\|off\\|auto")
            || tail.contains("[on")
            || (build.map(|value| value >= 6_000).unwrap_or(false) && help_missing);
        if choices {
            FlashSyntax::Choice
        } else {
            FlashSyntax::Bare
        }
    } else if help_missing && build.map(|value| value >= 6_000).unwrap_or(false) {
        FlashSyntax::Choice
    } else {
        FlashSyntax::None
    };
    let kv_cache =
        advertises(help, "--cache-type-k") && advertises(help, "--cache-type-v");
    let metrics = advertises(help, "--metrics");
    let jinja = advertises(help, "--jinja");
    let top_k = advertises(help, "--top-k");
    let alias = advertises(help, "--alias");
    let moe_cache_cpu_share = advertises(help, "--moe-cache-mib")
        && advertises(help, "--moe-cache-cpu-share");
    let spec_type = advertises(help, "--spec-type");
    let draft_model_flag = if advertises(help, "--spec-draft-model") {
        Some("--spec-draft-model")
    } else if advertises(help, "--model-draft") {
        Some("--model-draft")
    } else if advertises(help, "-md") {
        Some("-md")
    } else {
        None
    };
    let draft_device_flag = if advertises(help, "--spec-draft-device") {
        Some("--spec-draft-device")
    } else if advertises(help, "--device-draft") {
        Some("--device-draft")
    } else {
        None
    };
    let draft_ngl_flag = if advertises(help, "--spec-draft-ngl") {
        Some("--spec-draft-ngl")
    } else if advertises(help, "--gpu-layers-draft") {
        Some("--gpu-layers-draft")
    } else if advertises(help, "--n-gpu-layers-draft") {
        Some("--n-gpu-layers-draft")
    } else {
        None
    };
    if settings::normalize_llama_mtp_mode(&settings.llama_mtp_mode) == "mtp" && !spec_type {
        warnings.push(
            "This llama-server build does not advertise `--spec-type`, so MTP heads cannot be enabled. Update llama.cpp or use a plain draft model."
                .into(),
        );
    }
    let draft_configured = !settings.llama_mtp_path.trim().is_empty()
        || !settings.llama_draft_path.trim().is_empty();
    if draft_configured && draft_model_flag.is_none() {
        warnings.push(
            "This llama-server build does not advertise a speculative draft-model flag; the selected MTP head file or draft model will be ignored."
                .into(),
        );
    }
    if settings::normalize_draft_placement(&settings.llama_drafts_on) != "auto"
        && draft_device_flag.is_none()
        && draft_ngl_flag.is_none()
    {
        warnings.push(
            "This llama-server build does not advertise draft placement flags; the draft placement setting will be ignored."
                .into(),
        );
    }

    // Only emit draft tuning flags the probed build advertises. `--draft-max`
    // shipped alongside speculative decoding; older builds stop at the model.
    let draft_max_flag = advertises(help, "--draft-max").then_some("--draft-max");
    let draft_min_flag = advertises(help, "--draft-min").then_some("--draft-min");
    let draft_p_min_flag = advertises(help, "--draft-p-min").then_some("--draft-p-min");
    if draft_configured && draft_max_flag.is_none() {
        warnings.push(
            "This llama-server build does not advertise `--draft-max`, `--draft-min` or `--draft-p-min`; speculative draft length settings will be ignored."
                .into(),
        );
    }

    if !kv_cache
        && (settings::normalize_kv_cache(&settings.llama_kv_cache_k) != "f16"
            || settings::normalize_kv_cache(&settings.llama_kv_cache_v) != "f16")
    {
        warnings.push(
            "This build does not advertise KV-cache type flags; the requested cache quantization will be ignored."
                .into(),
        );
    }
    if flash == FlashSyntax::None && !settings.llama_flash_attn.eq_ignore_ascii_case("auto") {
        warnings.push(
            "This build does not advertise Flash Attention; the selected value will be ignored."
                .into(),
        );
    }
    if !metrics {
        warnings.push(
            "This build does not advertise `--metrics`; token telemetry will use process counters only."
                .into(),
        );
    }
    if !jinja {
        warnings.push(
            "This build does not advertise `--jinja`; OpenAI tool calling in the studio may be unavailable."
                .into(),
        );
    }
    if !moe_cache_cpu_share && settings.llama_direct_cpu_moe {
        warnings.push(
            "This llama-server build does not advertise `--moe-cache-mib` and `--moe-cache-cpu-share` (llama.cpp b12191 or newer). Direct CPU MoE Compute was ignored."
                .into(),
        );
    }

    let tier = if build.map(|value| value < 6_000).unwrap_or(false) {
        "transitional"
    } else if build.is_some() {
        "modern"
    } else {
        "inferred"
    };
    (
        LlamaLayout {
            adapter: format!("llama-server/{tier}{}", build_label(build)),
            model_flag,
            ctx_flag,
            ngl_flag,
            flash,
            kv_cache,
            metrics,
            jinja,
            top_k,
            alias,
            mmproj_flag,
            moe_cache_cpu_share,
            spec_type,
            draft_model_flag,
            draft_device_flag,
            draft_ngl_flag,
            draft_max_flag,
            draft_min_flag,
            draft_p_min_flag,
        },
        warnings,
    )
}

fn build_label(build: Option<u64>) -> String {
    build.map(|value| format!(" (b{value})")).unwrap_or_default()
}

/// Draft models and MTP heads are loaded as a second GGUF, so they get the
/// same existence and extension checks as the main model.
fn validate_spec_model(raw: &str, label: &str) -> AppResult<PathBuf> {
    let path = settings::expand_home(raw);
    if !path.is_file() {
        return Err(AppError::message(format!(
            "The llama.cpp {label} does not exist: {}. Choose it in Settings → llama.cpp.",
            path.display()
        )));
    }
    if !path
        .extension()
        .map(|extension| extension.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
    {
        return Err(AppError::message(format!(
            "The llama.cpp {label} must be a .gguf file that matches the selected model."
        )));
    }
    Ok(path)
}

fn build_args(
    settings: &AppSettings,
    layout: &LlamaLayout,
) -> AppResult<(Vec<String>, Vec<String>)> {
    let raw_model = settings.llama_model_path.trim();
    if raw_model.is_empty() {
        return Err(AppError::message(
            "No llama.cpp model is selected. Choose a .gguf file in Engine Settings or Settings → llama.cpp.",
        ));
    }
    let model = settings::expand_home(raw_model);
    if !model.is_file() {
        return Err(AppError::message(format!(
            "The llama.cpp model does not exist: {}",
            model.display()
        )));
    }
    if !model
        .extension()
        .map(|extension| extension.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
    {
        return Err(AppError::message(
            "llama-server needs a .gguf model. Convert or download a GGUF file first.",
        ));
    }

    let raw_mmproj = settings.llama_mmproj_path.trim();
    let mmproj = if raw_mmproj.is_empty() {
        None
    } else {
        let path = settings::expand_home(raw_mmproj);
        if !path.is_file() {
            return Err(AppError::message(format!(
                "The llama.cpp vision projector does not exist: {}. Choose the matching mmproj .gguf in Settings → llama.cpp.",
                path.display()
            )));
        }
        if !path
            .extension()
            .map(|extension| extension.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            return Err(AppError::message(
                "The llama.cpp vision projector must be a .gguf file. Choose the mmproj file that matches the selected vision model.",
            ));
        }
        if let Some(problem) = crate::companions::vision_pair_problem(
            &model.to_string_lossy(),
            &path.to_string_lossy(),
        ) {
            return Err(AppError::message(format!(
                "The llama.cpp vision projector cannot be loaded with this model: {problem}. Pair a matching mmproj in Library, or clear the projector path."
            )));
        }
        Some(path)
    };

    let mut warnings = Vec::new();
    let model_text = model.to_string_lossy().to_string();
    let host = bind_host(settings);
    let mut args = vec![
        layout.model_flag.to_string(),
        model_text.clone(),
        "--host".into(),
        host,
        "--port".into(),
        settings.llama_port.to_string(),
    ];

    if let Some(path) = mmproj.as_ref() {
        if let Some(flag) = layout.mmproj_flag {
            args.push(flag.into());
            args.push(path.to_string_lossy().to_string());
        }
    }

    let extra_args = if settings.llama_extra_args.trim().is_empty() {
        Vec::new()
    } else {
        shell_words::split(&settings.llama_extra_args).map_err(|err| {
            AppError::message(format!(
                "Invalid llama.cpp extra arguments: {err}. Use shell-style quoting."
            ))
        })?
    };
    let extra_has = |flag: &str| {
        extra_args
            .iter()
            .any(|argument| argument == flag || argument.starts_with(&format!("{flag}=")))
    };

    if let Some(flag) = layout.ctx_flag {
        let ctx = settings.llama_ctx_size.clamp(512, 1_048_576);
        args.push(flag.into());
        args.push(ctx.to_string());
    }
    if let Some(flag) = layout.ngl_flag {
        let layers = settings.llama_gpu_layers.clamp(0, 999);
        args.push(flag.into());
        args.push(layers.to_string());
    }

    let flash = settings::normalize_flash_attn(&settings.llama_flash_attn);
    let cache_k = settings::normalize_kv_cache(&settings.llama_kv_cache_k);
    let cache_v = settings::normalize_kv_cache(&settings.llama_kv_cache_v);
    let quantized_v = cache_v != "f16";

    if quantized_v && flash == "off" {
        return Err(AppError::message(
            "A quantized V cache needs Flash Attention. Set Flash Attention to `on` or `auto`, or leave the V cache at f16.",
        ));
    }
    if quantized_v && layout.flash == FlashSyntax::None {
        return Err(AppError::message(
            "This llama-server build cannot enable Flash Attention, so a quantized V cache cannot be used. Set the V cache type to f16.",
        ));
    }

    match layout.flash {
        FlashSyntax::Choice => {
            args.push("--flash-attn".into());
            args.push(flash.clone());
        }
        FlashSyntax::Bare => {
            if flash == "on" || quantized_v {
                args.push("--flash-attn".into());
            }
        }
        FlashSyntax::None => {
            if flash != "auto" {
                warnings.push(
                    "Flash Attention was requested but is not supported by this build; it was ignored."
                        .into(),
                );
            }
        }
    }

    if layout.kv_cache {
        if cache_k != "f16" {
            args.push("--cache-type-k".into());
            args.push(cache_k);
        }
        if cache_v != "f16" {
            args.push("--cache-type-v".into());
            args.push(cache_v);
        }
    }

    // Speculative decoding. Janas exposes `--mtp` / `--no-mtp` plus `--draft`
    // and `--drafts-on`; llama.cpp splits the same ideas across `--spec-type`,
    // its single draft-model slot and the draft placement flags.
    let mtp_path = settings.llama_mtp_path.trim();
    let draft_path = settings.llama_draft_path.trim();
    if !mtp_path.is_empty() && !draft_path.is_empty() {
        return Err(AppError::message(
            "Choose either an MTP head file or a draft model, not both: llama.cpp has one speculative draft slot.",
        ));
    }

    let spec_mode = settings::normalize_llama_mtp_mode(&settings.llama_mtp_mode);
    let mut draft_model: Option<PathBuf> = None;
    let mut spec_kind: Option<&str> = None;
    match spec_mode.as_str() {
        "off" => {
            if !mtp_path.is_empty() || !draft_path.is_empty() {
                warnings.push(
                    "MTP / speculation is Off, so the selected MTP head file or draft model is not used. Choose Auto or MTP to enable it."
                        .into(),
                );
            }
            spec_kind = Some("none");
        }
        "mtp" => {
            let raw = if !mtp_path.is_empty() {
                mtp_path
            } else {
                draft_path
            };
            if !raw.is_empty() {
                draft_model = Some(validate_spec_model(raw, "MTP head file")?);
            }
            spec_kind = Some("draft-mtp");
        }
        _ => {
            let raw = if !draft_path.is_empty() {
                draft_path
            } else {
                mtp_path
            };
            if !raw.is_empty() {
                let path = validate_spec_model(raw, "speculative draft model")?;
                if let Some(problem) = crate::companions::draft_pair_problem(
                    &model.to_string_lossy(),
                    &path.to_string_lossy(),
                ) {
                    return Err(AppError::message(format!(
                        "The speculative draft model cannot be loaded with this main model: {problem}. Pick a draft with the same architecture and vocabulary size in Library."
                    )));
                }
                draft_model = Some(path);
            }
        }
    }

    if layout.spec_type {
        if let Some(kind) = spec_kind {
            if !extra_has("--spec-type") {
                args.push("--spec-type".into());
                args.push(kind.into());
            }
        }
    }

    let has_draft_model = draft_model.is_some();
    if let Some(path) = draft_model {
        if let Some(flag) = layout.draft_model_flag {
            let overridden = extra_has("--spec-draft-model")
                || extra_has("-md")
                || extra_has("--model-draft");
            if !overridden {
                args.push(flag.into());
                args.push(path.to_string_lossy().to_string());
            }
        }
    }

    if has_draft_model {
        if let Some(flag) = layout.draft_max_flag {
            if !extra_has("--draft-max") {
                args.push(flag.into());
                args.push(settings.llama_draft_max.clamp(1, 64).to_string());
            }
        }
        if let Some(flag) = layout.draft_min_flag {
            if !extra_has("--draft-min") && settings.llama_draft_min > 0 {
                args.push(flag.into());
                args.push(
                    settings
                        .llama_draft_min
                        .min(settings.llama_draft_max)
                        .to_string(),
                );
            }
        }
        if let Some(flag) = layout.draft_p_min_flag {
            let p_min = settings.llama_draft_p_min.clamp(0.0, 1.0);
            if !extra_has("--draft-p-min") && p_min > 0.0 && p_min < 1.0 {
                args.push(flag.into());
                args.push(format!("{p_min:.2}"));
            }
        }
        match settings::normalize_draft_placement(&settings.llama_drafts_on).as_str() {
            "cpu" => {
                if let Some(flag) = layout.draft_device_flag {
                    if !extra_has("--spec-draft-device") && !extra_has("--device-draft") {
                        args.push(flag.into());
                        args.push("none".into());
                    }
                }
            }
            "gpu" => {
                if let Some(flag) = layout.draft_ngl_flag {
                    if !extra_has("--spec-draft-ngl")
                        && !extra_has("--gpu-layers-draft")
                        && !extra_has("--n-gpu-layers-draft")
                    {
                        args.push(flag.into());
                        args.push("all".into());
                    }
                }
            }
            _ => {}
        }
    }

    if settings.llama_direct_cpu_moe && layout.moe_cache_cpu_share {
        let host_experts = [
            "--cpu-moe",
            "-cmoe",
            "--n-cpu-moe",
            "-ncmoe",
            "--override-tensor",
            "-ot",
        ]
        .iter()
        .any(|flag| extra_has(flag));
        if !host_experts {
            args.push("--cpu-moe".into());
        }
        if !extra_has("--moe-cache-mib") {
            args.push("--moe-cache-mib".into());
            args.push("2048".into());
        }
        if !extra_has("--moe-cache-cpu-share") {
            args.push("--moe-cache-cpu-share".into());
            args.push("1.0".into());
        }
        let threads = settings.llama_cpu_threads.min(256);
        if threads > 0 && !extra_has("--threads") && !extra_has("-t") {
            args.push("--threads".into());
            args.push(threads.to_string());
        }
    }

    if layout.top_k {
        args.push("--top-k".into());
        args.push(settings.top_k.clamp(1, 100).to_string());
    }

    if layout.jinja {
        args.push("--jinja".into());
    }
    if layout.metrics {
        args.push("--metrics".into());
    }
    if layout.alias {
        if let Some(stem) = Path::new(&model_text).file_stem() {
            args.push("--alias".into());
            args.push(stem.to_string_lossy().to_string());
        }
    }

    args.extend(extra_args);

    Ok((args, warnings))
}

pub(super) fn spawn_health_probe<R: Runtime>(
    app: AppHandle<R>,
    pid: u32,
    host: String,
    port: u16,
) {
    tauri::async_runtime::spawn(async move {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let url = format!("http://{host}:{port}/health");
        let deadline = Instant::now() + Duration::from_secs(180);

        while Instant::now() < deadline {
            if !pid_alive(pid) {
                return;
            }
            if let Ok(response) = client
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                if response.status().is_success() {
                    if let Some(supervisor) = supervisor_handle(&app) {
                        // The supervisor publishes the launch before this probe
                        // starts, but a switch can still be in flight. Retry
                        // until the active engine is the one we probed; ending
                        // here would leave the app on "Loading model" while a
                        // healthy server answers requests.
                        if supervisor.mark_llama_ready(&app, pid).await {
                            return;
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(700)).await;
        }

        if pid_alive(pid) {
            if let Some(supervisor) = supervisor_handle(&app) {
                supervisor
                    .mark_llama_failed(
                        &app,
                        pid,
                        "llama-server did not answer /health within 180 seconds. Check the engine log for a model or GPU error.",
                    )
                    .await;
            }
            signal_group(pid, libc::SIGTERM);
            tokio::time::sleep(Duration::from_millis(500)).await;
            signal_group(pid, libc::SIGKILL);
        }
    });
}

fn supervisor_handle<R: Runtime>(
    app: &AppHandle<R>,
) -> Option<std::sync::Arc<EngineSupervisor<R>>> {
    app.try_state::<std::sync::Arc<EngineSupervisor<R>>>()
        .map(|state| state.inner().clone())
}

async fn fetch_metrics(client: &reqwest::Client, url: &str) -> (u64, u64, f64) {
    let Ok(response) = client.get(url).timeout(Duration::from_secs(2)).send().await else {
        return (0, 0, 0.0);
    };
    if !response.status().is_success() {
        return (0, 0, 0.0);
    }
    let Ok(text) = response.text().await else {
        return (0, 0, 0.0);
    };
    parse_metrics(&text)
}

fn parse_metrics(text: &str) -> (u64, u64, f64) {
    let mut completion = 0u64;
    let mut prompt = 0u64;
    let mut kv_usage = 0.0f64;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next().and_then(|value| value.parse::<f64>().ok()) else {
            continue;
        };
        if name.contains("tokens_predicted_total") && !name.contains("seconds") {
            completion = value.max(0.0) as u64;
        } else if name.contains("prompt_tokens_total") && !name.contains("seconds") {
            prompt = value.max(0.0) as u64;
        } else if name.contains("kv_cache_usage_ratio") {
            kv_usage = value.clamp(0.0, 1.0);
        }
    }
    (completion, prompt, kv_usage)
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_gguf(path: &Path, sliding_window: Option<u32>) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend((if sliding_window.is_some() { 1u64 } else { 0u64 }).to_le_bytes());
        if let Some(window) = sliding_window {
            let key = b"spark2_5.attention.sliding_window";
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key);
            bytes.extend(4u32.to_le_bytes()); // uint32
            bytes.extend(window.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write gguf");
    }

    fn vulkan_probe(build: u64) -> ProbeOutput {
        ProbeOutput {
            version_line: Some(format!("version: 1.0.0-dev (build {build}, commit a956a852c)")),
            build: Some(build),
            help_text: "-kvo, --kv-offload, -nkvo, --no-kv-offload\n-dev, --device DEVICE\n--list-devices\n--model".into(),
            devices: "Available devices:\n  Vulkan0: AMD Radeon 660M (RADV REMBRANDT) (16173 MiB)".into(),
        }
    }

    #[test]
    fn vulkan_swa_models_keep_the_kv_cache_on_the_host() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-swa-{}.gguf",
            std::process::id()
        ));
        write_gguf(&path, Some(512));
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_gpu_layers = 36;

        let mut args = vec!["--model".to_string(), "model.gguf".to_string()];
        let warning = guard_vulkan_swa_kv_cache(&mut args, &settings, &vulkan_probe(12_387));

        assert_eq!(
            args[args.len() - 2..].to_vec(),
            vec!["--device".to_string(), "none".to_string()]
        );
        let warning = warning.expect("warning");
        assert!(warning.contains("#29082"));
        assert!(warning.contains("--device none"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_kv_cache_guard_leaves_safe_launches_alone() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-full-attn-{}.gguf",
            std::process::id()
        ));
        write_gguf(&path, None);
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_gpu_layers = 36;

        // A full-attention model needs no workaround even on Vulkan.
        let mut args = vec!["--model".to_string()];
        assert!(guard_vulkan_swa_kv_cache(&mut args, &settings, &vulkan_probe(12_387)).is_none());
        assert_eq!(args, vec!["--model".to_string()]);

        // Neither does the last known-good build.
        write_gguf(&path, Some(512));
        assert!(guard_vulkan_swa_kv_cache(&mut args, &settings, &vulkan_probe(12_236)).is_none());

        // An explicit llama.cpp flag wins over the adapter default.
        for override_args in ["-kvo", "-nkvo", "--device Vulkan0"] {
            settings.llama_extra_args = override_args.into();
            assert!(
                guard_vulkan_swa_kv_cache(&mut args, &settings, &vulkan_probe(12_387)).is_none(),
                "{override_args} should disable the guard"
            );
        }

        // CUDA-only hosts are untouched.
        settings.llama_extra_args.clear();
        let mut cpu_probe = vulkan_probe(12_387);
        cpu_probe.devices = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090".into();
        assert!(guard_vulkan_swa_kv_cache(&mut args, &settings, &cpu_probe).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn parses_build_from_version_line() {
        assert_eq!(
            crate::engines::parse_build("version: 6640 (b4f4d4e4c)"),
            Some(6640)
        );
        assert_eq!(
            crate::engines::parse_build("llama-server b8123\nbuilt with clang"),
            Some(8123)
        );
    }

    #[test]
    fn detects_flag_tokens_without_substring_false_positives() {
        let help = "-c, --ctx-size N  context size\n-ngl, --n-gpu-layers N\n-fa, --flash-attn [on|off|auto]\n-ctk, --cache-type-k TYPE\n--metrics\n--jinja\n--top-k N\n--alias STR";
        assert!(advertises(help, "--ctx-size"));
        assert!(advertises(help, "-c"));
        assert!(!advertises("-cache", "-c"));
    }

    #[test]
    fn modern_layout_uses_choice_flash_and_kv_flags() {
        let output = ProbeOutput {
            version_line: Some("version: 6640 (hash)".into()),
            build: Some(6640),
            help_text: "--model\n--ctx-size\n--n-gpu-layers\n--flash-attn [on|off|auto]\n--cache-type-k\n--cache-type-v\n--metrics\n--jinja\n--top-k\n--alias\n--moe-cache-mib N\n--moe-cache-cpu-share F\n-mm, --mmproj FILE".into(),
            devices: String::new(),
        };
        let (layout, _) = layout_for(&output, &AppSettings::default());
        assert_eq!(layout.flash, FlashSyntax::Choice);
        assert!(layout.kv_cache);
        assert!(layout.metrics);
        assert!(layout.moe_cache_cpu_share);
        assert_eq!(layout.mmproj_flag, Some("--mmproj"));
    }

    #[test]
    fn quantized_v_requires_flash_attention() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-args-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"GGUF-test").expect("write test model");
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_kv_cache_v = "q8_0".into();
        settings.llama_flash_attn = "off".into();
        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: false,
            spec_type: false,
            draft_model_flag: None,
            draft_device_flag: None,
            draft_ngl_flag: None,
            draft_max_flag: None,
            draft_min_flag: None,
            draft_p_min_flag: None,
        };
        let result = build_args(&settings, &layout);
        assert!(result.is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn args_include_gpu_context_and_kv_flags() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-args-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"GGUF-test").expect("write test model");
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_gpu_layers = 24;
        settings.llama_ctx_size = 8192;
        settings.llama_flash_attn = "on".into();
        settings.llama_kv_cache_k = "q8_0".into();
        settings.llama_kv_cache_v = "q8_0".into();
        settings.top_k = 40;
        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: false,
            spec_type: false,
            draft_model_flag: None,
            draft_device_flag: None,
            draft_ngl_flag: None,
            draft_max_flag: None,
            draft_min_flag: None,
            draft_p_min_flag: None,
        };
        let (args, warnings) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(joined.contains("--n-gpu-layers 24"));
        assert!(joined.contains("--ctx-size 8192"));
        assert!(joined.contains("--flash-attn on"));
        assert!(joined.contains("--cache-type-k q8_0"));
        assert!(joined.contains("--cache-type-v q8_0"));
        assert!(joined.contains("--top-k 40"));
        assert!(warnings.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn args_include_vision_projector() {
        let dir = std::env::temp_dir().join(format!(
            "janas-llama-mmproj-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create test directory");
        let model = dir.join("vision-model.gguf");
        let projector = dir.join("mmproj-model-f16.gguf");
        std::fs::write(&model, b"GGUF-test").expect("write test model");
        std::fs::write(&projector, b"GGUF-mmproj").expect("write test projector");
        let mut settings = AppSettings::default();
        settings.llama_model_path = model.to_string_lossy().to_string();
        settings.llama_mmproj_path = projector.to_string_lossy().to_string();
        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: false,
            spec_type: false,
            draft_model_flag: None,
            draft_device_flag: None,
            draft_ngl_flag: None,
            draft_max_flag: None,
            draft_min_flag: None,
            draft_p_min_flag: None,
        };
        let (args, warnings) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(joined.contains(&format!("--mmproj {}", projector.to_string_lossy())));
        assert!(warnings.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn speculative_flags_follow_the_mtp_and_draft_settings() {
        let dir = std::env::temp_dir().join(format!(
            "janas-llama-spec-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create test directory");
        let model = dir.join("model.gguf");
        let draft = dir.join("draft.gguf");
        std::fs::write(&model, b"GGUF-test").expect("write test model");
        std::fs::write(&draft, b"GGUF-draft").expect("write test draft model");
        let draft_text = draft.to_string_lossy().to_string();

        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: false,
            spec_type: true,
            draft_model_flag: Some("--spec-draft-model"),
            draft_device_flag: Some("--spec-draft-device"),
            draft_ngl_flag: Some("--spec-draft-ngl"),
            draft_max_flag: Some("--draft-max"),
            draft_min_flag: Some("--draft-min"),
            draft_p_min_flag: Some("--draft-p-min"),
        };

        let mut settings = AppSettings::default();
        settings.llama_model_path = model.to_string_lossy().to_string();
        settings.llama_draft_path = draft_text.clone();
        settings.llama_drafts_on = "cpu".into();

        // Auto leaves the type to llama.cpp, which detects it from the model.
        let (args, warnings) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(!joined.contains("--spec-type"));
        assert!(joined.contains(&format!("--spec-draft-model {draft_text}")));
        assert!(joined.contains("--spec-draft-device none"));
        assert!(joined.contains("--draft-max 8"));
        assert!(joined.contains("--draft-p-min 0.75"));
        assert!(!joined.contains("--draft-min"));
        assert!(warnings.is_empty());

        // MTP keeps the heads inside the main model. Nothing is offloaded
        // separately, so no draft-model or draft-placement flag is emitted.
        settings.llama_mtp_mode = "mtp".into();
        settings.llama_draft_path.clear();
        settings.llama_drafts_on = "gpu".into();
        let (args, _) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(joined.contains("--spec-type draft-mtp"));
        assert!(!joined.contains("--spec-draft-model"));
        assert!(!joined.contains("--spec-draft-ngl"));

        // A separate MTP head file takes the draft-model slot and can be placed.
        settings.llama_mtp_path = draft_text.clone();
        let (args, _) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(joined.contains(&format!("--spec-draft-model {draft_text}")));
        assert!(joined.contains("--spec-draft-ngl all"));

        // Both slots at once is a configuration error, not a silent choice.
        settings.llama_draft_path = draft_text.clone();
        assert!(build_args(&settings, &layout).is_err());
        settings.llama_draft_path.clear();

        // Off disables speculation even when a file is still selected.
        settings.llama_mtp_mode = "off".into();
        let (args, warnings) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(joined.contains("--spec-type none"));
        assert!(!joined.contains("--spec-draft-model"));
        assert!(warnings.iter().any(|warning| warning.contains("Off")));

        // A missing draft file is reported instead of launching without it.
        settings.llama_mtp_mode = "mtp".into();
        settings.llama_mtp_path = dir.join("missing.gguf").to_string_lossy().to_string();
        assert!(build_args(&settings, &layout).is_err());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn direct_cpu_moe_adds_cache_pipeline_and_threads() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-direct-moe-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"GGUF-test").expect("write test model");
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_direct_cpu_moe = true;
        settings.llama_cpu_threads = 6;
        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: true,
            spec_type: false,
            draft_model_flag: None,
            draft_device_flag: None,
            draft_ngl_flag: None,
            draft_max_flag: None,
            draft_min_flag: None,
            draft_p_min_flag: None,
        };
        let (args, warnings) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(joined.contains("--cpu-moe"));
        assert!(joined.contains("--moe-cache-mib 2048"));
        assert!(joined.contains("--moe-cache-cpu-share 1.0"));
        assert!(joined.contains("--threads 6"));
        assert!(warnings.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn direct_cpu_moe_respects_extra_arg_overrides() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-direct-moe-overrides-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"GGUF-test").expect("write test model");
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_direct_cpu_moe = true;
        settings.llama_cpu_threads = 6;
        settings.llama_extra_args =
            "--n-cpu-moe 12 --moe-cache-mib 4096 --threads 3".into();
        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: true,
            spec_type: false,
            draft_model_flag: None,
            draft_device_flag: None,
            draft_ngl_flag: None,
            draft_max_flag: None,
            draft_min_flag: None,
            draft_p_min_flag: None,
        };
        let (args, _) = build_args(&settings, &layout).expect("args");
        let count = |flag: &str| args.iter().filter(|argument| *argument == flag).count();
        assert_eq!(count("--cpu-moe"), 0);
        assert_eq!(count("--moe-cache-mib"), 1);
        assert_eq!(count("--moe-cache-cpu-share"), 1);
        assert_eq!(count("--threads"), 1);
        let joined = args.join(" ");
        assert!(joined.contains("--moe-cache-mib 4096"));
        assert!(joined.contains("--threads 3"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn direct_cpu_moe_is_ignored_when_the_build_lacks_the_pipeline() {
        let path = std::env::temp_dir().join(format!(
            "janas-llama-direct-moe-unsupported-{}.gguf",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"GGUF-test").expect("write test model");
        let mut settings = AppSettings::default();
        settings.llama_model_path = path.to_string_lossy().to_string();
        settings.llama_direct_cpu_moe = true;
        settings.llama_cpu_threads = 6;
        let layout = LlamaLayout {
            adapter: "test".into(),
            model_flag: "--model",
            ctx_flag: Some("--ctx-size"),
            ngl_flag: Some("--n-gpu-layers"),
            flash: FlashSyntax::Choice,
            kv_cache: true,
            metrics: true,
            jinja: true,
            top_k: true,
            alias: true,
            mmproj_flag: Some("--mmproj"),
            moe_cache_cpu_share: false,
            spec_type: false,
            draft_model_flag: None,
            draft_device_flag: None,
            draft_ngl_flag: None,
            draft_max_flag: None,
            draft_min_flag: None,
            draft_p_min_flag: None,
        };
        let (args, _) = build_args(&settings, &layout).expect("args");
        let joined = args.join(" ");
        assert!(!joined.contains("--moe-cache"));
        assert!(!joined.contains("--cpu-moe"));
        assert!(!joined.contains("--threads"));
        let _ = std::fs::remove_file(path);
    }
}
