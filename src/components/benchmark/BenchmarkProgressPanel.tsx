import {
  Activity,
  Cpu,
  HardDrive,
  Hash,
  Loader2,
  Square,
  Timer,
} from "lucide-react";

import { Button } from "../ui/Button";
import { ProgressBar } from "../ui/ProgressBar";
import { cn } from "../../lib/cn";
import { formatBytes, formatDuration } from "../../lib/format";
import type { BenchmarkPhase, BenchmarkProgress } from "./types";

interface BenchmarkProgressPanelProps {
  progress: BenchmarkProgress | null;
  targetCount: number;
  maxTokens: number;
  busy: boolean;
  onCancel: () => void;
}

const PHASE_LABELS: Record<BenchmarkPhase, string> = {
  switching_engine: "Switching engine",
  warmup: "Warm-up pass",
  measuring: "Measuring",
};

function phaseProgress(progress: BenchmarkProgress | null, maxTokens: number) {
  if (!progress) return null;
  const within =
    progress.phase === "measuring"
      ? Math.max(0.04, Math.min(0.96, progress.tokens_generated / Math.max(1, maxTokens)))
      : progress.phase === "warmup"
        ? 0.3
        : 0.02;
  const total = (progress.target_index + within) / Math.max(1, progress.target_count);
  return Math.min(99, total * 100);
}

export function BenchmarkProgressPanel({
  progress,
  targetCount,
  maxTokens,
  busy,
  onCancel,
}: BenchmarkProgressPanelProps) {
  const value = phaseProgress(progress, maxTokens);
  const phase = progress?.phase;

  return (
    <section className="panel overflow-hidden border-acid-400/20">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/5 bg-acid-400/[0.04] px-5 py-4">
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-acid-400/25 bg-acid-400/10 text-acid-300">
            {progress ? (
              <Activity className="h-4 w-4 animate-pulse-soft" />
            ) : (
              <Loader2 className="h-4 w-4 animate-spin" />
            )}
          </span>
          <div className="min-w-0">
            <p className="text-[13px] font-semibold text-white">
              {progress
                ? progress.target_label
                : "Preparing the benchmark suite"}
            </p>
            <p className="truncate text-[10px] text-slate-500">
              {progress
                ? `Target ${progress.target_index + 1} of ${progress.target_count} · ${progress.engine === "janas" ? "Janas" : "llama.cpp"}`
                : "Stopping the current engine and building the target list."}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={cn(
              "rounded-full border px-2.5 py-1 text-[9px] font-semibold uppercase tracking-[0.14em]",
              phase === "measuring"
                ? "border-acid-400/30 bg-acid-400/10 text-acid-300"
                : "border-amber-400/25 bg-amber-400/10 text-amber-200",
            )}
          >
            {phase ? PHASE_LABELS[phase] : "Starting"}
          </span>
          <Button
            size="sm"
            variant="danger"
            busy={busy}
            onClick={onCancel}
            icon={<Square className="h-3 w-3" />}
          >
            Cancel run
          </Button>
        </div>
      </div>

      <div className="space-y-5 px-5 py-5">
        <div>
          <div className="mb-2 flex items-center justify-between text-[10px] text-slate-500">
            <span className="flex items-center gap-1.5">
              <Timer className="h-3 w-3" />
              {progress ? formatDuration(progress.elapsed_ms) : "0 ms"} in this
              target
            </span>
            <span className="font-mono">
              {value === null ? "switching" : `${Math.round(value)}%`}
            </span>
          </div>
          <ProgressBar value={value} indeterminate={value === null} />
        </div>

        <div className="flex flex-wrap gap-1.5">
          {Array.from({ length: Math.max(1, targetCount) }, (_, index) => {
            const completed =
              progress !== null && index < progress.target_index;
            const active =
              progress !== null && index === progress.target_index;
            return (
              <span
                key={index}
                className={cn(
                  "rounded-full border px-2.5 py-1 text-[10px]",
                  active
                    ? "border-acid-400/40 bg-acid-400/10 text-acid-200"
                    : completed
                      ? "border-emerald-400/25 bg-emerald-400/5 text-emerald-300/80"
                      : "border-white/10 bg-white/[0.02] text-slate-600",
                )}
              >
                {completed
                  ? `Target ${index + 1} ✓`
                  : active
                    ? `Target ${index + 1} · running`
                    : `Target ${index + 1}`}
              </span>
            );
          })}
        </div>

        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
          <Metric
            icon={<Hash className="h-3 w-3" />}
            label="Tokens"
            value={(progress?.tokens_generated ?? 0).toString()}
          />
          <Metric
            icon={<Activity className="h-3 w-3" />}
            label="Live tok/s"
            value={(progress?.live_tps ?? 0).toFixed(1)}
          />
          <Metric
            icon={<Cpu className="h-3 w-3" />}
            label="Peak RSS"
            value={formatBytes(progress?.rss_bytes ?? 0)}
          />
          <Metric
            icon={<HardDrive className="h-3 w-3" />}
            label="SSD read (gen)"
            value={`${formatBytes(progress?.read_bytes_per_sec ?? 0)}/s`}
          />
          <Metric
            icon={<HardDrive className="h-3 w-3" />}
            label="Read calls/s"
            value={Math.round(progress?.read_ops_per_sec ?? 0).toLocaleString()}
          />
        </div>

        <p className="min-h-[16px] text-[10px] leading-5 text-slate-500">
          {progress?.message ??
            "The previous engine is excluded from every measurement."}
        </p>
        <p className="text-[9.5px] leading-4 text-slate-600">
          Live tokens are stream deltas. SSD and read-call figures cover
          generation only: model loading is excluded, and cached reads count as
          calls but not as SSD bytes.
        </p>
      </div>
    </section>
  );
}

function Metric({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-ink-950/50 px-3 py-2.5">
      <p className="flex items-center gap-1.5 text-[9px] uppercase tracking-[0.14em] text-slate-600">
        {icon}
        {label}
      </p>
      <p className="mt-1 truncate font-mono text-sm text-slate-100">{value}</p>
    </div>
  );
}
