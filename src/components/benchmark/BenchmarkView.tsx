import { History, Trash2, X } from "lucide-react";

import { cn } from "../../lib/cn";
import type { BenchmarkStatus } from "./types";
import type { BenchmarkController } from "./useBenchmark";
import { BenchmarkConfigurator } from "./BenchmarkConfigurator";
import { BenchmarkMatrix } from "./BenchmarkMatrix";
import { BenchmarkProgressPanel } from "./BenchmarkProgressPanel";

interface BenchmarkViewProps {
  benchmark: BenchmarkController;
}

const DOT: Record<BenchmarkStatus, string> = {
  completed: "bg-emerald-400",
  failed: "bg-rose-400",
  cancelled: "bg-amber-300",
};

export function BenchmarkView({ benchmark }: BenchmarkViewProps) {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1500px] space-y-5 px-6 py-6">
        {benchmark.error ? (
          <div className="flex items-start gap-2 rounded-xl border border-rose-400/20 bg-rose-500/5 px-3.5 py-2.5 text-[11px] leading-5 text-rose-300">
            <span className="flex-1">{benchmark.error}</span>
            <button
              type="button"
              onClick={benchmark.clearError}
              className="text-rose-400/60 transition hover:text-rose-200"
              aria-label="Dismiss benchmark error"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : null}

        {benchmark.running ? (
          <BenchmarkProgressPanel
            progress={benchmark.progress}
            targetCount={
              benchmark.progress?.target_count ||
              benchmark.activeTargetCount ||
              2
            }
            maxTokens={benchmark.activeWorkload?.max_tokens ?? 256}
            busy={benchmark.busy}
            onCancel={() => void benchmark.cancel()}
          />
        ) : null}

        {!benchmark.running && benchmark.suite ? (
          <BenchmarkMatrix
            suite={benchmark.suite}
            copied={benchmark.copied}
            busy={benchmark.busy}
            onCopy={(suite) => void benchmark.copyMarkdown(suite)}
          />
        ) : null}

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
          <BenchmarkConfigurator
            presets={benchmark.presets}
            running={benchmark.running}
            busy={benchmark.busy}
            onStart={benchmark.start}
          />

          <aside className="panel flex min-h-0 flex-col p-4 xl:max-h-[calc(100vh-120px)]">
            <div className="flex items-center justify-between gap-2">
              <div>
                <p className="label">Run history</p>
                <p className="mt-1 text-[10px] text-slate-600">
                  Atomic JSON suites in the local data folder
                </p>
              </div>
              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-slate-400">
                {benchmark.suites.length}
              </span>
            </div>

            {benchmark.suites.length === 0 ? (
              <div className="mt-6 flex flex-1 flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/10 px-4 py-8 text-center">
                <History className="h-5 w-5 text-slate-700" />
                <p className="text-[11px] text-slate-500">
                  No saved suites yet.
                </p>
                <p className="text-[10px] leading-4 text-slate-600">
                  A finished or cancelled run is saved here so you can compare
                  it after a driver or quant update.
                </p>
              </div>
            ) : (
              <div className="mt-3 min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
                {benchmark.suites.map((suite) => {
                  const selected = benchmark.suite?.id === suite.id;
                  return (
                    <div
                      key={suite.id}
                      className={cn(
                        "flex items-stretch rounded-xl border transition",
                        selected
                          ? "border-acid-400/30 bg-acid-400/[0.06]"
                          : "border-transparent hover:border-white/10 hover:bg-white/[0.025]",
                      )}
                    >
                      <button
                        type="button"
                        disabled={benchmark.running}
                        onClick={() => benchmark.selectSuite(suite)}
                        className="min-w-0 flex-1 px-2.5 py-2 text-left disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <span className="flex items-center gap-2">
                          <span
                            className={cn(
                              "h-1.5 w-1.5 shrink-0 rounded-full",
                              DOT[suite.status],
                            )}
                          />
                          <span
                            className={cn(
                              "truncate text-[11px]",
                              selected ? "text-white" : "text-slate-300",
                            )}
                          >
                            {suite.workload.label}
                          </span>
                        </span>
                        <span className="mt-0.5 flex items-center justify-between gap-2 text-[9px] text-slate-600">
                          <span className="truncate">
                            {new Date(suite.created_at_ms).toLocaleDateString()}
                          </span>
                          <span className="shrink-0">
                            {suite.results.length}/{suite.targets.length}{" "}
                            targets
                          </span>
                        </span>
                      </button>
                      <button
                        type="button"
                        disabled={benchmark.running}
                        onClick={() => void benchmark.remove(suite.id)}
                        className="shrink-0 px-2 text-slate-600 transition hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-30"
                        aria-label={`Delete ${suite.workload.label}`}
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}

            <p className="mt-3 border-t border-white/5 pt-2.5 text-[9px] leading-4 text-slate-600">
              Historical suites stay comparable because prompt, seed,
              temperature, hardware and per-target flags are stored with every
              result.
            </p>
          </aside>
        </div>
      </div>
    </div>
  );
}
