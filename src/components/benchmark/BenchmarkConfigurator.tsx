import { useEffect, useMemo, useRef, useState } from "react";
import { Gauge, Plus, Trash2 } from "lucide-react";

import { Button } from "../ui/Button";
import { Toggle } from "../ui/Toggle";
import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { formatBytes, uid } from "../../lib/format";
import type { LibraryModel } from "../../types";
import type {
  BenchmarkEngine,
  BenchmarkPreset,
  BenchmarkTarget,
  BenchmarkWorkload,
} from "./types";

interface BenchmarkConfiguratorProps {
  presets: BenchmarkPreset[];
  running: boolean;
  busy: boolean;
  onStart: (
    targets: BenchmarkTarget[],
    workload: BenchmarkWorkload,
    warmup: boolean,
  ) => Promise<void>;
}

function engineFor(path: string): BenchmarkEngine {
  return path.toLowerCase().endsWith(".jns") ? "janas" : "llama";
}

function engineName(engine: BenchmarkEngine): string {
  return engine === "janas" ? "Janas" : "llama.cpp";
}

function targetFor(model: LibraryModel): BenchmarkTarget {
  return {
    id: `target-${uid()}`,
    label: model.name,
    engine: engineFor(model.path),
    model_path: model.path,
    ctx_size: 16_384,
    gpu_layers: 0,
    extra_flags: "",
  };
}

export function BenchmarkConfigurator({
  presets,
  running,
  busy,
  onStart,
}: BenchmarkConfiguratorProps) {
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [targets, setTargets] = useState<BenchmarkTarget[]>([]);
  const [presetId, setPresetId] = useState("short-reasoning");
  const [workloadLabel, setWorkloadLabel] = useState("Short Reasoning");
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(42);
  const [maxTokens, setMaxTokens] = useState(256);
  const [structured, setStructured] = useState(false);
  const [warmup, setWarmup] = useState(false);
  const initializedTargets = useRef(false);
  const initializedPreset = useRef(false);

  useEffect(() => {
    let disposed = false;
    void api
      .libraryModels()
      .then((next) => {
        if (!disposed) setModels(next);
      })
      .catch(() => {
        if (!disposed) {
          setModelsError(
            "The model library could not be read. Add a .jns or .gguf model in Library first.",
          );
        }
      });
    return () => {
      disposed = true;
    };
  }, []);

  useEffect(() => {
    if (initializedTargets.current || models.length === 0) return;
    initializedTargets.current = true;
    // Seed the comparison with one model per engine when both formats exist.
    const gguf = models.find((model) => engineFor(model.path) === "llama");
    const jns = models.find((model) => engineFor(model.path) === "janas");
    const picks: LibraryModel[] = [];
    if (gguf) picks.push(gguf);
    if (jns && (!gguf || jns.path !== gguf.path)) picks.push(jns);
    if (picks.length === 0) picks.push(models[0]);
    setTargets(picks.map((model) => targetFor(model)));
  }, [models]);

  useEffect(() => {
    if (initializedPreset.current || presets.length === 0) return;
    initializedPreset.current = true;
    const first = presets[0];
    setPresetId(first.id);
    setWorkloadLabel(first.label);
    setPrompt(first.prompt);
    setMaxTokens(first.max_tokens);
    setStructured(first.structured);
  }, [presets]);

  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.id === presetId),
    [presets, presetId],
  );

  const updateTarget = (index: number, patch: Partial<BenchmarkTarget>) => {
    setTargets((current) =>
      current.map((target, position) =>
        position === index ? { ...target, ...patch } : target,
      ),
    );
  };

  const chooseModel = (index: number, path: string) => {
    const model = models.find((candidate) => candidate.path === path);
    updateTarget(index, {
      model_path: path,
      engine: engineFor(path),
      ...(model ? { label: model.name } : {}),
    });
  };

  const addTarget = () => {
    if (targets.length >= 4 || models.length === 0) return;
    const unused = models.find(
      (model) => !targets.some((target) => target.model_path === model.path),
    );
    setTargets((current) => [
      ...current,
      targetFor(unused ?? models[0]),
    ]);
  };

  const removeTarget = (index: number) => {
    if (targets.length <= 2) return;
    setTargets((current) => current.filter((_, position) => position !== index));
  };

  const choosePreset = (id: string) => {
    const preset = presets.find((candidate) => candidate.id === id);
    if (!preset) {
      setPresetId("custom");
      setWorkloadLabel("Custom prompt");
      return;
    }
    setPresetId(preset.id);
    setWorkloadLabel(preset.label);
    setPrompt(preset.prompt);
    setMaxTokens(preset.max_tokens);
    setStructured(preset.structured);
  };

  const ready =
    targets.length >= 2 &&
    targets.every((target) => target.model_path.trim().length > 0) &&
    prompt.trim().length > 0 &&
    maxTokens > 0;

  const start = () => {
    if (!ready || running) return;
    const workload: BenchmarkWorkload = {
      preset: presetId,
      label: workloadLabel.trim() || "Custom prompt",
      prompt,
      seed,
      temperature: 0,
      max_tokens: maxTokens,
      structured,
    };
    void onStart(
      targets.map((target, index) => ({
        ...target,
        label: target.label.trim() || `Target ${index + 1}`,
        extra_flags: target.extra_flags.trim(),
      })),
      workload,
      warmup,
    );
  };

  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="label">Benchmark configurator</p>
          <h2 className="mt-1 text-sm font-semibold text-white">
            Side-by-side engine &amp; quantization bench
          </h2>
          <p className="mt-1 max-w-2xl text-[11px] leading-5 text-slate-500">
            Runs each target alone through the engine supervisor. The current
            engine is stopped, the target waits for <code>/health</code>, then
            one identical workload is measured. Target flags replace global
            extra arguments so runs stay comparable.
          </p>
        </div>
        <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-slate-400">
          {targets.length} / 4 targets
        </span>
      </div>

      {modelsError ? (
        <p className="mt-4 rounded-xl border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] leading-5 text-amber-200/80">
          {modelsError}
        </p>
      ) : null}

      <div className="mt-5 space-y-3">
        {targets.map((target, index) => (
          <div
            key={target.id}
            className="rounded-2xl border border-white/10 bg-ink-950/40 p-3.5"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-lg border border-white/10 bg-white/5 font-mono text-[10px] text-slate-400">
                  {index + 1}
                </span>
                <span className="text-xs font-medium text-slate-200">
                  Target {index + 1}
                </span>
                <span
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em]",
                    target.engine === "janas"
                      ? "border-acid-400/25 bg-acid-400/10 text-acid-300"
                      : "border-sky-400/25 bg-sky-400/10 text-sky-300",
                  )}
                >
                  {engineName(target.engine)}
                </span>
              </div>
              <button
                type="button"
                onClick={() => removeTarget(index)}
                disabled={targets.length <= 2 || running}
                className="rounded-lg p-1.5 text-slate-600 transition hover:bg-white/5 hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-30"
                aria-label={`Remove target ${index + 1}`}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <label className="block min-w-0">
                <span className="label">Label</span>
                <input
                  className="field mt-1.5"
                  value={target.label}
                  disabled={running}
                  onChange={(event) =>
                    updateTarget(index, { label: event.target.value })
                  }
                />
              </label>
              <label className="block min-w-0">
                <span className="label">Model</span>
                <select
                  className="field mt-1.5"
                  value={target.model_path}
                  disabled={running}
                  onChange={(event) => chooseModel(index, event.target.value)}
                >
                  {!models.some((model) => model.path === target.model_path) &&
                  target.model_path ? (
                    <option value={target.model_path}>
                      {target.model_path}
                    </option>
                  ) : null}
                  {models.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.name} · {model.format}
                      {model.quant && model.quant !== "—"
                        ? ` · ${model.quant}`
                        : ""}
                      {model.size_bytes > 0
                        ? ` · ${formatBytes(model.size_bytes)}`
                        : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="label">Context window</span>
                <input
                  type="number"
                  min={512}
                  max={1_048_576}
                  step={512}
                  className="field mt-1.5 font-mono"
                  value={target.ctx_size}
                  disabled={running}
                  onChange={(event) =>
                    updateTarget(index, {
                      ctx_size: Number(event.target.value),
                    })
                  }
                />
              </label>
              <label className="block">
                <span className="label">GPU layers</span>
                <input
                  type="number"
                  min={0}
                  max={999}
                  className="field mt-1.5 font-mono disabled:opacity-40"
                  value={target.gpu_layers}
                  disabled={running || target.engine === "janas"}
                  onChange={(event) =>
                    updateTarget(index, {
                      gpu_layers: Number(event.target.value),
                    })
                  }
                />
                <span className="mt-1 block text-[10px] text-slate-600">
                  {target.engine === "janas"
                    ? "Used by llama.cpp targets only."
                    : "0 keeps every layer on the CPU."}
                </span>
              </label>
              <label className="block md:col-span-2">
                <span className="label">Extra flags</span>
                <input
                  className="field mt-1.5 font-mono text-[11px]"
                  placeholder={
                    target.engine === "janas"
                      ? "JANAS_ATTN=int16 JANAS_KV=8 --cache 2"
                      : "--cpu-moe --threads 8 --n-cpu-moe 12"
                  }
                  value={target.extra_flags}
                  disabled={running}
                  onChange={(event) =>
                    updateTarget(index, { extra_flags: event.target.value })
                  }
                />
                <span className="mt-1 block text-[10px] leading-4 text-slate-600">
                  Space-separated. Arguments are passed to the engine;{" "}
                  <code>NAME=value</code> tokens become environment overrides
                  for this launch only.
                </span>
              </label>
            </div>
          </div>
        ))}

        <Button
          variant="ghost"
          size="sm"
          onClick={addTarget}
          disabled={running || targets.length >= 4 || models.length === 0}
          icon={<Plus className="h-3.5 w-3.5" />}
        >
          Add target
        </Button>
      </div>

      <div className="mt-5 border-t border-white/5 pt-4">
        <div className="grid gap-3 md:grid-cols-4">
          <label className="block">
            <span className="label">Workload preset</span>
            <select
              className="field mt-1.5"
              value={presetId}
              disabled={running}
              onChange={(event) => choosePreset(event.target.value)}
            >
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
              <option value="custom">Custom prompt</option>
            </select>
          </label>
          <label className="block">
            <span className="label">Seed</span>
            <input
              type="number"
              className="field mt-1.5 font-mono"
              value={seed}
              min={0}
              disabled={running}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
          </label>
          <label className="block">
            <span className="label">Max tokens</span>
            <input
              type="number"
              className="field mt-1.5 font-mono"
              value={maxTokens}
              min={1}
              max={8_192}
              disabled={running}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
            />
          </label>
          <label className="flex items-end pb-2.5">
            <span className="flex items-center gap-2 text-[11px] text-slate-400">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 accent-acid-400"
                checked={structured}
                disabled={running}
                onChange={(event) => setStructured(event.target.checked)}
              />
              JSON mode
            </span>
          </label>
        </div>

        <label className="mt-3 block">
          <span className="flex items-center justify-between gap-2">
            <span className="label">Prompt</span>
            <span className="text-[10px] text-slate-600">
              {selectedPreset?.description ??
                "A custom prompt replaces the preset text."}
            </span>
          </span>
          <textarea
            className="field mt-1.5 min-h-[92px] resize-y font-mono text-[11px] leading-5"
            value={prompt}
            disabled={running}
            onChange={(event) => {
              setPrompt(event.target.value);
              setPresetId("custom");
              setWorkloadLabel("Custom prompt");
            }}
          />
          <span className="mt-1 flex items-center justify-between text-[10px] text-slate-600">
            <span>
              Temperature 0.0, thinking disabled, same seed and prompt for
              every target.
            </span>
            <span className="font-mono">
              {prompt.length.toLocaleString()} chars
              {structured ? " · JSON mode" : ""}
            </span>
          </span>
        </label>

        <div className="mt-4 rounded-xl border border-white/10 bg-ink-950/40 p-3">
          <Toggle
            checked={warmup}
            onChange={setWarmup}
            disabled={running}
            label={
              warmup
                ? "Warm run (discarded warm-up pass)"
                : "Direct run (no warm-up pass)"
            }
            description={
              warmup
                ? "One discarded pass warms the page cache before measurement. Generation telemetry starts after that pass."
                : "Measured generation starts immediately after /health. Model loading happens before the telemetry window, so this is not a full cold-start measurement."
            }
          />
        </div>

        <Button
          className="mt-4 w-full justify-center"
          busy={busy || running}
          disabled={!ready || running}
          onClick={start}
          icon={<Gauge className="h-4 w-4" />}
        >
          {running ? "Suite running" : "Run side-by-side bench"}
        </Button>
        {!ready ? (
          <p className="mt-2 text-center text-[10px] text-slate-600">
            Choose at least two models and a non-empty prompt to start.
          </p>
        ) : null}
      </div>
    </section>
  );
}
