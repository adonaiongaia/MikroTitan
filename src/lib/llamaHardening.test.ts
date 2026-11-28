import { describe, expect, it } from "vitest";

import {
  adapterWarningView,
  fallbackLabel,
  normalizeLlamaUpdateChannel,
  normalizeSettings,
  normalizeSetupReport,
} from "./llamaHardening";
import type { AdapterWarningEvent, AppSettings, SetupReport } from "../types";

function settings(patch: Record<string, unknown>): AppSettings {
  return patch as unknown as AppSettings;
}

function report(patch: Record<string, unknown>): SetupReport {
  return patch as unknown as SetupReport;
}

function warning(patch: Partial<AdapterWarningEvent>): AdapterWarningEvent {
  return {
    engine: "llama",
    message: "details",
    version: null,
    adapter: "llama-server/modern (b12728)",
    ...patch,
  };
}

describe("normalizeLlamaUpdateChannel", () => {
  it("defaults a missing or unknown channel to the verified baseline", () => {
    expect(normalizeLlamaUpdateChannel(undefined)).toBe("baseline");
    expect(normalizeLlamaUpdateChannel(null)).toBe("baseline");
    expect(normalizeLlamaUpdateChannel("")).toBe("baseline");
    expect(normalizeLlamaUpdateChannel("nightly")).toBe("baseline");
    expect(normalizeLlamaUpdateChannel(42)).toBe("baseline");
  });

  it("accepts latest in any casing or padding", () => {
    expect(normalizeLlamaUpdateChannel("latest")).toBe("latest");
    expect(normalizeLlamaUpdateChannel("  LATEST ")).toBe("latest");
  });
});

describe("normalizeSettings", () => {
  it("does not let an old settings.json imply the latest channel", () => {
    const old = settings({ llama_model_path: "/models/m.gguf" });
    const next = normalizeSettings(old);
    expect(next.llama_update_channel).toBe("baseline");
    expect(next.llama_model_path).toBe("/models/m.gguf");
  });

  it("keeps an explicit latest channel and reuses the original object", () => {
    const latest = settings({ llama_update_channel: "latest" });
    expect(normalizeSettings(latest)).toBe(latest);
    expect(normalizeSettings(latest).llama_update_channel).toBe("latest");
  });
});

describe("normalizeSetupReport", () => {
  it("fills manifest status for an older backend", () => {
    const next = normalizeSetupReport(report({ status: "ready", message: "ok" }));
    expect(next.llama_update_channel).toBe("baseline");
    expect(next.compat_applied).toBe(false);
    expect(next.compat_recommended_tag).toBeNull();
    expect(next.blocked_llama_tags).toEqual([]);
    expect(next.skipped_llama_tags).toEqual([]);
    expect(next.llama_baseline_tag).toBe("baseline");
    expect(next.vulkan_kv_offload_limit).toBeNull();
    expect(next.status).toBe("ready");
  });

  it("preserves a manifest-aware report", () => {
    const next = normalizeSetupReport(
      report({
        status: "ready",
        llama_update_channel: "latest",
        compat_applied: true,
        compat_recommended_tag: "b12900",
        blocked_llama_tags: ["b13000"],
        skipped_llama_tags: ["b13000"],
        llama_baseline_tag: "b12900",
        vulkan_kv_offload_limit: 12500,
      }),
    );
    expect(next.llama_update_channel).toBe("latest");
    expect(next.compat_applied).toBe(true);
    expect(next.compat_recommended_tag).toBe("b12900");
    expect(next.skipped_llama_tags).toEqual(["b13000"]);
    expect(next.vulkan_kv_offload_limit).toBe(12500);
  });
});

describe("adapter warnings", () => {
  it("labels each fallback rung", () => {
    expect(fallbackLabel("host-kv")).toBe("Host KV split (-nkvo)");
    expect(fallbackLabel("cpu")).toBe("Full CPU (--device none)");
    expect(fallbackLabel(null)).toBe("Full GPU");
  });

  it("presents an auto-rollback as a recovery with a settings link", () => {
    const view = adapterWarningView(
      warning({
        kind: "auto-rollback",
        message: "crashed during startup and was rolled back",
        failed_version: "b12728",
        restored_version: "b12236",
      }),
    );
    expect(view.kind).toBe("auto-rollback");
    expect(view.title).toBe("Rolled back to b12236");
    expect(view.showSettings).toBe(true);
    expect(view.dismissible).toBe(true);
    expect(view.hint).toContain("Automatic engine setup");
  });

  it("presents a ladder step as an active fallback", () => {
    const view = adapterWarningView(
      warning({
        kind: "fallback",
        step: "host-kv",
        message: "retrying with -nkvo",
      }),
    );
    expect(view.kind).toBe("fallback");
    expect(view.title).toContain("Host KV split (-nkvo)");
    expect(view.showSettings).toBe(false);
    expect(view.detail).toBe("retrying with -nkvo");
  });

  it("treats a legacy payload without a kind as a plain adapter notice", () => {
    const view = adapterWarningView(warning({ message: "no --metrics" }));
    expect(view.kind).toBe("adapter");
    expect(view.title).toBe("llama.cpp adapter");
    expect(view.detail).toBe("no --metrics");
    expect(view.hint).toBeNull();
  });
});
