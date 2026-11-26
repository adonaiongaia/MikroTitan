import { Check, Copy, Crown, HardDrive, MemoryStick, Timer, Zap } from "lucide-react";

import { Button } from "../ui/Button";
import { cn } from "../../lib/cn";
import { formatBytes, formatDuration } from "../../lib/format";
import type { BenchmarkRunResult, BenchmarkStatus, BenchmarkSuite } from "./types";

interface BenchmarkMatrixProps {
  suite: BenchmarkSuite;
  copied: boolean;
  busy: boolean;
  onCopy: (suite: BenchmarkSuite) => void;
}

type MetricKey =
  | "ttft"
  | "prefill"
  | "decode"
  | "duration"
  | "rss"
  | "ssd_bytes"
  | "ssd_mbps"
  | "iops";

interface MetricRow {
  key: MetricKey;
  label: string;
  lower: boolean;
  format: (value: number) => string;
}

const ROWS: MetricRow[] = [
  { key: "ttft", label: "Time to first token", lower: true, format: ms },
  { key: "prefill", label: "Prompt eval", lower: false, format: tps },
  { key: "decode", label: "Generation", lower: false, format: tps },
  { key: "duration", label: "Total time", lower: true, format: ms },
  {
    key: "rss",
    label: "Peak process RSS (RAM)",
    lower: true,
    format: (value) => bytes(value),
  },
  {
    key: "ssd_bytes",
    label: "SSD bytes read (generation)",
    lower: false,
    format: (value) => bytes(value),
  },
  {
    key: "ssd_mbps",
    label: "Peak SSD read (generation)",
    lower: false,
    format: (value) => `${value.toFixed(1)} MB/s`,
  },
  {
    key: "iops",
    label: "Peak read calls/s",
    lower: false,
    format: (value) => Math.round(value).toLocaleString(),
  },
];

function ms(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${Math.round(value)} ms`;
}

function tps(value: number): string {
  return `${value.toFixed(1)} tok/s`;
}

function bytes(value: number): string {
  if (value <= 0) return "0 B";
  return formatBytes(value);
}

function metricValue(result: BenchmarkRunResult, key: MetricKey): number {
  switch (key) {
    case "ttft":
      return result.ttft_ms;
    case "prefill":
      return result.prefill_tps;
    case "decode":
      return result.decode_tps;
    case "duration":
      return result.total_duration_ms;
    case "rss":
      return result.peak_rss_bytes;
    case "ssd_bytes":
      return result.ssd_bytes_read;
    case "ssd_mbps":
      return result.peak_ssd_mbps;
    case "iops":
      return result.peak_iops;
  }
}

function bestIndices(
  results: BenchmarkRunResult[],
  key: MetricKey,
  lower: boolean,
): number[] {
  let best: number | null = null;
  const winners: number[] = [];
  results.forEach((result, index) => {
    if (result.status !== "completed") return;
    const value = metricValue(result, key);
    if (!Number.isFinite(value)) return;
    if (best === null || (lower ? value < best - 1e-9 : value > best + 1e-9)) {
      best = value;
      winners.length = 0;
      winners.push(index);
    } else if (Math.abs(value - best) <= 1e-9) {
      winners.push(index);
    }
  });
  return winners;
}

function deltaLabel(
  value: number,
  best: number,
  lower: boolean,
): string | null {
  if (!Number.isFinite(value) || !Number.isFinite(best) || best <= 0) {
    return null;
  }
  if (Math.abs(value - best) < 1e-9) return "best";
  const percent = lower
    ? ((value - best) / best) * 100
    : ((best - value) / best) * 100;
  if (Math.abs(percent) < 0.05) return "best";
  return `${percent > 0 ? "+" : ""}${percent.toFixed(1)}%`;
}

const STATUS_STYLE: Record<BenchmarkStatus, string> = {
  completed: "border-emerald-400/25 bg-emerald-400/10 text-emerald-300",
  failed: "border-rose-400/25 bg-rose-400/10 text-rose-300",
  cancelled: "border-amber-400/25 bg-amber-400/10 text-amber-200",
};

export function BenchmarkMatrix({
  suite,
  copied,
  busy,
  onCopy,
}: BenchmarkMatrixProps) {
  const completed = suite.results.filter(
    (result) => result.status === "completed",
  );
  const cards = [
    {
      key: "ttft" as MetricKey,
      title: "First token",
      note: "lower is better",
      lower: true,
      icon: Timer,
      format: ms,
    },
    {
      key: "decode" as MetricKey,
      title: "Decode speed",
      note: "higher is better",
      lower: false,
      icon: Zap,
      format: tps,
    },
    {
      key: "rss" as MetricKey,
      title: "Peak RAM",
      note: "lower is better",
      lower: true,
      icon: MemoryStick,
      format: (value: number) => bytes(value),
    },
    {
      key: "ssd_mbps" as MetricKey,
      title: "SSD throughput",
      note: "generation only · more streaming",
      lower: false,
      icon: HardDrive,
      format: (value: number) => `${value.toFixed(1)} MB/s`,
    },
  ];

  return (
    <section className="space-y-4">
      <div className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="label">Comparison matrix</p>
            <h2 className="mt-1 truncate text-lg font-semibold text-white">
              {suite.workload.label}
            </h2>
            <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px] text-slate-500">
              <span>{new Date(suite.created_at_ms).toLocaleString()}</span>
              <span>·</span>
              <span>
                {suite.warmup ? "Warm-up pass" : "No warm-up pass"}
              </span>
              <span>·</span>
              <span>{formatDuration(suite.duration_ms)} suite</span>
              <span>·</span>
              <span className={cn("rounded-full border px-2 py-0.5", STATUS_STYLE[suite.status])}>
                {suite.status}
              </span>
            </p>
          </div>
          <Button
            size="sm"
            variant="secondary"
            busy={busy}
            onClick={() => onCopy(suite)}
            icon={
              copied ? (
                <Check className="h-3.5 w-3.5 text-emerald-300" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )
            }
          >
            {copied ? "Copied report" : "Copy Markdown Report"}
          </Button>
        </div>

        {suite.error ? (
          <p className="mt-3 rounded-xl border border-rose-400/20 bg-rose-500/5 px-3 py-2 text-[11px] leading-5 text-rose-300">
            {suite.error}
          </p>
        ) : null}

        <div className="mt-4 grid gap-2.5 text-[10.5px] text-slate-500 md:grid-cols-2 xl:grid-cols-4">
          <p className="truncate">
            <span className="text-slate-600">CPU </span>
            {suite.hardware.cpu} · {suite.hardware.cores} threads
          </p>
          <p>
            <span className="text-slate-600">RAM </span>
            {bytes(suite.hardware.mem_total_bytes)}
          </p>
          <p className="truncate">
            <span className="text-slate-600">GPU </span>
            {suite.hardware.gpu.present ? suite.hardware.gpu.name : "not detected"}
          </p>
          <p className="truncate">
            <span className="text-slate-600">OS </span>
            {suite.hardware.os} / {suite.hardware.arch}
          </p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => {
          const winners = bestIndices(suite.results, card.key, card.lower);
          const winner = winners[0] ?? null;
          const tied = winners.length > 1;
          const bestValue =
            winner === null
              ? null
              : metricValue(suite.results[winner], card.key);
          const Icon = card.icon;
          return (
            <div
              key={card.key}
              className="rounded-2xl border border-white/10 bg-ink-900/70 p-4"
            >
              <div className="flex items-center justify-between gap-2">
                <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                  <Icon className="h-3 w-3" />
                  {card.title}
                </p>
                <span className="text-[9px] text-slate-600">{card.note}</span>
              </div>
              {winner === null || bestValue === null ? (
                <p className="mt-4 text-[11px] text-slate-600">
                  No completed target.
                </p>
              ) : (
                <>
                  <p className="mt-3 flex items-center gap-1.5 truncate text-[12px] font-medium text-slate-200">
                    <Crown className="h-3.5 w-3.5 shrink-0 text-acid-300" />
                    {tied ? "Tie" : suite.results[winner].target_label}
                  </p>
                  <p className="mt-1 font-mono text-2xl tracking-tight text-white">
                    {card.format(bestValue)}
                  </p>
                  <div className="mt-3 space-y-1 border-t border-white/5 pt-2">
                    {completed.map((result) => {
                      const value = metricValue(result, card.key);
                      const delta = deltaLabel(value, bestValue, card.lower);
                      const isTied =
                        tied && Math.abs(value - bestValue) <= 1e-9;
                      const label = isTied ? "tie" : delta;
                      return (
                        <div
                          key={result.target_id}
                          className="flex items-center justify-between gap-2 text-[10px]"
                        >
                          <span className="min-w-0 truncate text-slate-500">
                            {result.target_label}
                          </span>
                          <span className="shrink-0 font-mono text-slate-400">
                            {card.format(value)}
                            <span
                              className={cn(
                                "ml-2",
                                label === "best" || label === "tie"
                                  ? "text-acid-300"
                                  : "text-amber-300/70",
                              )}
                            >
                              {label}
                            </span>
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>

      <div className="panel overflow-hidden">
        <div className="border-b border-white/5 px-5 py-3.5">
          <p className="label">Every metric</p>
          <p className="mt-1 text-[10px] text-slate-600">
            Completed targets only. The bold cell is the winner for that row.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-[11px]">
            <thead>
              <tr className="border-b border-white/5 text-left">
                <th className="px-5 py-2.5 font-medium text-slate-500">Metric</th>
                {suite.results.map((result) => (
                  <th
                    key={result.target_id}
                    className="px-4 py-2.5 text-right font-medium text-slate-300"
                  >
                    <span className="block truncate">{result.target_label}</span>
                    <span className="block text-[9px] font-normal uppercase tracking-[0.1em] text-slate-600">
                      {result.engine === "janas" ? "Janas" : "llama.cpp"}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => {
                const winners = bestIndices(suite.results, row.key, row.lower);
                const tied = winners.length > 1;
                return (
                  <tr key={row.key} className="border-b border-white/[0.035]">
                    <td className="px-5 py-2.5 text-slate-500">
                      {row.label}
                      <span className="ml-2 text-[9px] text-slate-700">
                        {row.lower ? "lower" : "higher"}
                      </span>
                    </td>
                    {suite.results.map((result, index) => {
                      const isWinner = winners.includes(index);
                      const value = metricValue(result, row.key);
                      return (
                        <td
                          key={result.target_id}
                          className={cn(
                            "px-4 py-2.5 text-right font-mono",
                            result.status !== "completed"
                              ? "text-slate-700"
                              : isWinner
                                ? "bg-acid-400/[0.07] font-semibold text-acid-200"
                                : "text-slate-300",
                          )}
                        >
                          {result.status === "completed" ? row.format(value) : "—"}
                          {isWinner ? (
                            <span className="ml-2 rounded bg-acid-400/15 px-1 py-0.5 text-[8px] uppercase tracking-wide text-acid-300">
                              {tied ? "tie" : "best"}
                            </span>
                          ) : null}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
              <tr className="border-b border-white/[0.035]">
                <td className="px-5 py-2.5 text-slate-500">Tokens generated</td>
                {suite.results.map((result) => (
                  <td
                    key={result.target_id}
                    className="px-4 py-2.5 text-right font-mono text-slate-300"
                  >
                    {result.status === "completed"
                      ? result.tokens_generated.toLocaleString()
                      : "—"}
                  </td>
                ))}
              </tr>
              <tr>
                <td className="px-5 py-2.5 text-slate-500">Status</td>
                {suite.results.map((result) => (
                  <td key={result.target_id} className="px-4 py-2.5 text-right">
                    <span
                      className={cn(
                        "rounded-full border px-2 py-0.5 text-[9px]",
                        STATUS_STYLE[result.status],
                      )}
                    >
                      {result.status}
                    </span>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {suite.results.map((result, index) => {
          const target = suite.targets[index];
          return (
            <div
              key={result.target_id}
              className="rounded-2xl border border-white/10 bg-ink-900/70 p-4"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-[12px] font-semibold text-slate-200">
                    {result.target_label}
                  </p>
                  <p className="mt-0.5 text-[10px] text-slate-600">
                    {result.engine === "janas" ? "Janas" : "llama.cpp"}
                    {result.adapter ? ` · ${result.adapter}` : ""}
                    {result.engine_version ? ` · ${result.engine_version}` : ""}
                  </p>
                </div>
                <span
                  className={cn(
                    "shrink-0 rounded-full border px-2 py-0.5 text-[9px]",
                    STATUS_STYLE[result.status],
                  )}
                >
                  {result.status}
                </span>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
                <p className="text-slate-500">
                  Context{" "}
                  <span className="font-mono text-slate-300">
                    {target?.ctx_size.toLocaleString() ?? "—"}
                  </span>
                </p>
                <p className="text-slate-500">
                  GPU layers{" "}
                  <span className="font-mono text-slate-300">
                    {result.engine === "janas"
                      ? "n/a"
                      : (target?.gpu_layers ?? 0)}
                  </span>
                </p>
                <p className="col-span-2 truncate text-slate-500">
                  Flags{" "}
                  <span className="font-mono text-slate-400">
                    {target?.extra_flags || "none"}
                  </span>
                </p>
              </div>

              {result.error ? (
                <p className="mt-3 rounded-lg border border-rose-400/20 bg-rose-500/5 px-2.5 py-2 text-[10px] leading-4 text-rose-300">
                  {result.error}
                </p>
              ) : null}
              {result.adapter_warnings.length > 0 ? (
                <ul className="mt-3 space-y-1.5">
                  {result.adapter_warnings.map((warning) => (
                    <li
                      key={warning}
                      className="rounded-lg border border-amber-400/15 bg-amber-400/[0.04] px-2.5 py-1.5 text-[10px] leading-4 text-amber-200/75"
                    >
                      {warning}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          );
        })}
      </div>

      <details className="panel px-5 py-3.5">
        <summary className="cursor-pointer text-[11px] text-slate-400">
          Workload details
        </summary>
        <div className="mt-3 space-y-1.5 text-[10.5px] text-slate-500">
          <p>
            Preset{" "}
            <span className="font-mono text-slate-300">
              {suite.workload.preset}
            </span>{" "}
            · seed{" "}
            <span className="font-mono text-slate-300">
              {suite.workload.seed}
            </span>{" "}
            · temperature 0.0 · max tokens{" "}
            <span className="font-mono text-slate-300">
              {suite.workload.max_tokens}
            </span>
            {suite.workload.structured ? " · JSON mode" : ""}
            {" · thinking disabled"}
          </p>
          <p className="text-[9.5px] leading-4 text-slate-600">
            Thinking is disabled so the token cap normally bounds the whole
            reply; an overrun is reported under warnings. SSD and read-call
            figures cover measured generation only: model loading is excluded,
            and cached reads count as calls but not SSD bytes. More streaming
            wins the SSD rows: the engine that reads the most from the drive
            ranks first there.
          </p>
          <pre className="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-xl border border-white/10 bg-ink-950/70 p-3 font-mono text-[10px] leading-5 text-slate-400">
            {suite.workload.prompt}
          </pre>
        </div>
      </details>
    </section>
  );
}
