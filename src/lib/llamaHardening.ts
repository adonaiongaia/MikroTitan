import type {
  AdapterWarningEvent,
  AppSettings,
  EngineFallbackStep,
  LlamaUpdateChannel,
  SetupReport,
} from "../types";

/**
 * Older `settings.json` files predate the release channel. Anything that is
 * not exactly "latest" must resolve to the tested baseline: silently jumping
 * to a daily upstream build is the failure mode this setting exists to avoid.
 */
export function normalizeLlamaUpdateChannel(value: unknown): LlamaUpdateChannel {
  return typeof value === "string" && value.trim().toLowerCase() === "latest"
    ? "latest"
    : "baseline";
}

/** Repair a settings payload from an older backend or a stale store. */
export function normalizeSettings(next: AppSettings): AppSettings {
  const raw = next as AppSettings & { llama_update_channel?: unknown };
  const channel = normalizeLlamaUpdateChannel(raw.llama_update_channel);
  if (raw.llama_update_channel === channel) return next;
  return { ...next, llama_update_channel: channel };
}

/**
 * Fill the setup-report fields a backend without the hardening work does not
 * send. The UI can then render one shape regardless of app version.
 */
export function normalizeSetupReport(report: SetupReport): SetupReport {
  const raw = report as SetupReport & Partial<SetupReport>;
  return {
    ...report,
    llama_update_channel: normalizeLlamaUpdateChannel(raw.llama_update_channel),
    compat_applied: raw.compat_applied === true,
    compat_recommended_tag: raw.compat_recommended_tag ?? null,
    blocked_llama_tags: raw.blocked_llama_tags ?? [],
    skipped_llama_tags: raw.skipped_llama_tags ?? [],
    llama_baseline_tag: raw.llama_baseline_tag ?? "baseline",
    vulkan_kv_offload_limit: raw.vulkan_kv_offload_limit ?? null,
  };
}

export function fallbackLabel(
  step: EngineFallbackStep | null | undefined,
): string {
  switch (step) {
    case "host-kv":
      return "Host KV split (-nkvo)";
    case "cpu":
      return "Full CPU (--device none)";
    default:
      return "Full GPU";
  }
}

export interface AdapterWarningView {
  kind: "adapter" | "fallback" | "auto-rollback";
  title: string;
  detail: string;
  hint: string | null;
  dismissible: boolean;
  showSettings: boolean;
}

/**
 * Turn the raw `engine://adapter-warning` payload into copy the panel can
 * render. The three protections read very differently to a user: a rollback is
 * a recovery, a fallback is a downgrade, and an adapter warning is just the
 * launch layout being adjusted.
 */
export function adapterWarningView(
  event: AdapterWarningEvent,
): AdapterWarningView {
  const kind = event.kind ?? "adapter";

  if (kind === "auto-rollback") {
    return {
      kind,
      title: event.restored_version
        ? `Rolled back to ${event.restored_version}`
        : "Previous llama.cpp build restored",
      detail: event.message,
      hint: "Pin a release you trust in Settings → Automatic engine setup.",
      dismissible: true,
      showSettings: true,
    };
  }

  if (kind === "fallback") {
    return {
      kind,
      title: `GPU fallback active: ${fallbackLabel(event.step)}`,
      detail: event.message,
      hint: "Change it in Extra llama arguments if this machine can handle more offload.",
      dismissible: true,
      showSettings: false,
    };
  }

  return {
    kind: "adapter",
    title: "llama.cpp adapter",
    detail: event.message,
    hint: null,
    dismissible: true,
    showSettings: false,
  };
}
