import { useState } from "react";
import {
  AlertTriangle,
  Ban,
  Check,
  ChevronDown,
  Clock,
  Layers,
  Loader2,
  ShieldAlert,
  Sparkles,
  Undo2,
  X,
} from "lucide-react";

import { cn } from "../../lib/cn";
import type {
  AgentCompaction,
  AgentDiffPreview,
  AgentRisk,
  AgentStep,
  AgentStepStatus,
  AgentTrace,
  AgentPendingApproval,
} from "../../types";

export interface AgentActions {
  approve: (
    runId: string,
    stepId: string,
    options?: { allowEdits?: boolean },
  ) => void | Promise<void>;
  reject: (runId: string, stepId: string, feedback: string) => void | Promise<void>;
  abort: (runId: string) => void | Promise<void>;
  undo: (runId: string) => void | Promise<void>;
  undoFile: (runId: string, path: string) => void | Promise<void>;
}

interface AgentTraceProps {
  trace: AgentTrace;
  streaming: boolean;
  actions: AgentActions;
}

type TimelineItem =
  | { kind: "step"; step: AgentStep }
  | { kind: "compaction"; event: AgentCompaction };

function statusLabel(trace: AgentTrace): string {
  if (trace.pending) return "Waiting for approval…";
  switch (trace.status) {
    case "completed":
      return "Finished";
    case "verification_failed":
      return "Verification failed";
    case "max_iterations":
      return "Step limit reached";
    case "aborted":
      return "Stopped";
    case "error":
      return "Agent error";
    default:
      break;
  }
  if (trace.notice?.toLowerCase().includes("compacting")) {
    return "Compacting context…";
  }
  const last = trace.steps[trace.steps.length - 1];
  if (last?.status === "awaiting_approval") return "Waiting for approval…";
  if (last?.status === "running") return `Running \`${last.tool}\`…`;
  return "Thinking…";
}

function statusTone(trace: AgentTrace): string {
  if (trace.pending) return "bg-amber-400";
  switch (trace.status) {
    case "completed":
      return "bg-emerald-400";
    case "error":
      return "bg-rose-400";
    case "verification_failed":
      return "bg-rose-400";
    case "aborted":
      return "bg-slate-500";
    case "max_iterations":
      return "bg-amber-400";
    default:
      return "bg-acid-400 animate-pulse-soft";
  }
}

const RISK_LABEL: Record<AgentRisk, string> = {
  read_only: "read-only",
  mutating: "mutating",
  destructive: "destructive",
  irreversible: "always confirm",
};

function StepIcon({ status }: { status: AgentStepStatus }) {
  if (status === "running") {
    return <Loader2 className="h-3 w-3 shrink-0 animate-spin text-acid-300" />;
  }
  if (status === "awaiting_approval") {
    return <ShieldAlert className="h-3 w-3 shrink-0 text-amber-300" />;
  }
  if (status === "succeeded") {
    return <Check className="h-3 w-3 shrink-0 text-emerald-400" />;
  }
  if (status === "rejected") {
    return <Ban className="h-3 w-3 shrink-0 text-slate-500" />;
  }
  if (status === "blocked") {
    return <AlertTriangle className="h-3 w-3 shrink-0 text-amber-400" />;
  }
  return <X className="h-3 w-3 shrink-0 text-rose-400" />;
}

function formatDuration(ms: number | null): string {
  if (ms === null) return "";
  if (ms < 1_000) return `${ms} ms`;
  return `${(ms / 1_000).toFixed(ms < 10_000 ? 1 : 0)} s`;
}

function StepRow({ step }: { step: AgentStep }) {
  return (
    <li className="px-3 py-2">
      <details className="group">
        <summary className="flex cursor-pointer list-none items-center gap-2 [&::-webkit-details-marker]:hidden">
          <StepIcon status={step.status} />
          <span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-slate-300">
            {step.tool}
          </span>
          {step.risk !== "read_only" ? (
            <span className="rounded border border-amber-400/20 bg-amber-400/5 px-1 py-px text-[8px] uppercase tracking-wider text-amber-300/80">
              {RISK_LABEL[step.risk]}
            </span>
          ) : null}
          {step.truncated ? (
            <span className="rounded border border-white/10 bg-white/5 px-1 py-px text-[8px] uppercase tracking-wider text-slate-500">
              trimmed
            </span>
          ) : null}
          <span className="font-mono text-[9px] text-slate-600">
            {formatDuration(step.duration_ms)}
          </span>
          <ChevronDown className="h-3 w-3 shrink-0 text-slate-600 transition group-open:rotate-180" />
        </summary>

        <div className="mt-2 space-y-2 pl-5">
          {step.thought.trim() ? (
            <p className="line-clamp-4 whitespace-pre-wrap text-[10.5px] leading-5 text-slate-500">
              {step.thought}
            </p>
          ) : null}
          <div>
            <p className="mb-1 text-[8px] uppercase tracking-[0.16em] text-slate-600">
              Input
            </p>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-white/5 bg-ink-950/70 px-2.5 py-2 font-mono text-[9.5px] leading-4 text-slate-400">
              {step.arguments || "{}"}
            </pre>
          </div>
          {step.output ? (
            <div>
              <p className="mb-1 text-[8px] uppercase tracking-[0.16em] text-slate-600">
                Output
              </p>
              <pre
                className={cn(
                  "max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg border px-2.5 py-2 font-mono text-[9.5px] leading-4",
                  step.status === "succeeded"
                    ? "border-white/5 bg-ink-950/70 text-slate-400"
                    : "border-rose-400/15 bg-rose-500/5 text-rose-200/90",
                )}
              >
                {step.output}
              </pre>
            </div>
          ) : null}
        </div>
      </details>
    </li>
  );
}

function buildTimeline(trace: AgentTrace): TimelineItem[] {
  const items: TimelineItem[] = [];
  let compactionIndex = 0;
  for (const step of trace.steps) {
    while (
      compactionIndex < trace.compactions.length &&
      trace.compactions[compactionIndex].created_at <= step.created_at
    ) {
      items.push({
        kind: "compaction",
        event: trace.compactions[compactionIndex],
      });
      compactionIndex += 1;
    }
    items.push({ kind: "step", step });
  }
  for (; compactionIndex < trace.compactions.length; compactionIndex += 1) {
    items.push({ kind: "compaction", event: trace.compactions[compactionIndex] });
  }
  return items;
}

function CompactionRow({ event }: { event: AgentCompaction }) {
  return (
    <li className="flex items-center gap-2 px-3 py-1.5">
      <span className="h-px flex-1 bg-white/5" />
      <span
        className="inline-flex items-center gap-1 rounded-full border border-sky-400/15 bg-sky-400/5 px-2 py-0.5 text-[8.5px] text-sky-200/70"
        title={event.summary}
      >
        <Layers className="h-2.5 w-2.5" />
        Context compacted · {event.dropped_messages} messages → memory
      </span>
      <span className="h-px flex-1 bg-white/5" />
    </li>
  );
}

function DiffPreview({ preview }: { preview: AgentDiffPreview }) {
  if (preview.error) {
    return (
      <p className="mt-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-2.5 py-2 text-[10px] leading-4 text-amber-200/80">
        Preview unavailable: {preview.error}
      </p>
    );
  }
  return (
    <div className="mt-2 overflow-hidden rounded-lg border border-white/10 bg-ink-950/80">
      <div className="flex items-center gap-2 border-b border-white/5 px-2.5 py-1.5">
        <code className="min-w-0 flex-1 truncate font-mono text-[9.5px] text-slate-300">
          {preview.path}
        </code>
        <span className="font-mono text-[9px] text-emerald-400">
          +{preview.added}
        </span>
        <span className="font-mono text-[9px] text-rose-400">−{preview.removed}</span>
        {preview.truncated ? (
          <span className="rounded border border-white/10 bg-white/5 px-1 py-px text-[8px] uppercase tracking-wider text-slate-500">
            trimmed
          </span>
        ) : null}
      </div>
      <div className="max-h-56 overflow-auto px-2.5 py-2 font-mono text-[9.5px] leading-4">
        {preview.diff.split("\n").map((line, index) => (
          <div
            key={`${index}-${line}`}
            className={cn(
              "whitespace-pre",
              line.startsWith("+")
                ? "text-emerald-300"
                : line.startsWith("-")
                  ? "text-rose-300"
                  : line.startsWith("@@")
                    ? "text-sky-300/80"
                    : line.startsWith("...")
                      ? "text-slate-600"
                      : "text-slate-500",
            )}
          >
            {line || " "}
          </div>
        ))}
      </div>
    </div>
  );
}

function ApprovalCard({
  runId,
  pending,
  actions,
}: {
  runId: string;
  pending: AgentPendingApproval;
  actions: AgentActions;
}) {
  const [feedback, setFeedback] = useState("");
  const [allowEdits, setAllowEdits] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const run = async (label: string, action: () => void | Promise<void>) => {
    if (busy) return;
    setBusy(label);
    try {
      await action();
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="border-t border-amber-400/20 bg-amber-400/[0.04]">
      {/* The preview scrolls on its own so the feedback field and buttons stay
          in a fixed footer instead of being pushed out of view. */}
      <div className="max-h-[220px] overflow-y-auto px-3 pt-3">
        <div className="flex items-center gap-1.5">
          <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-amber-300" />
          <span className="text-[10.5px] font-medium text-amber-200">
            Waiting for approval
          </span>
          <span className="text-slate-600">·</span>
          <code className="truncate rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[9.5px] text-slate-300">
            {pending.tool}
          </code>
          <span className="ml-auto shrink-0 rounded border border-amber-400/20 bg-amber-400/5 px-1.5 py-0.5 text-[8px] uppercase tracking-wider text-amber-300/80">
            {RISK_LABEL[pending.risk]}
          </span>
        </div>
        <p className="mt-1 text-[10px] leading-4 text-amber-200/60">{pending.reason}</p>

        {pending.preview ? (
          <>
            <DiffPreview preview={pending.preview} />
            <details className="mt-3">
              <summary className="cursor-pointer text-[9px] uppercase tracking-[0.14em] text-slate-600">
                Raw arguments
              </summary>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-white/10 bg-ink-950/80 px-2.5 py-2 font-mono text-[9.5px] leading-4 text-slate-300">
                {pending.arguments}
              </pre>
            </details>
          </>
        ) : (
          <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-white/10 bg-ink-950/80 px-2.5 py-2 font-mono text-[9.5px] leading-4 text-slate-300">
            {pending.arguments}
          </pre>
        )}
      </div>

      <div className="border-t border-amber-400/10 bg-ink-900/40 px-3 py-2.5">
        <textarea
          rows={2}
          value={feedback}
          onChange={(event) => setFeedback(event.target.value)}
          placeholder="Optional feedback if you reject — it is sent back to the model…"
          className="field mt-0 block h-16 w-full resize-none !py-2 text-[10.5px] leading-5"
        />

        {pending.file_edit ? (
          <label className="mt-2 flex cursor-pointer items-center gap-1.5 text-[9.5px] text-slate-400">
            <input
              type="checkbox"
              checked={allowEdits}
              onChange={(event) => setAllowEdits(event.target.checked)}
              className="h-3 w-3 accent-acid-400"
            />
            Allow file edits for this run
            <span className="text-slate-600">
              · skips approval for write_file and edit_file only
            </span>
          </label>
        ) : null}

        <div className="mt-3 flex items-center gap-1.5">
          <button
            type="button"
            disabled={busy !== null}
            onClick={() =>
              void run("approve", () =>
                actions.approve(runId, pending.step_id, { allowEdits }),
              )
            }
            className="inline-flex h-7 flex-1 items-center justify-center gap-1.5 rounded-lg bg-acid-400 px-2 text-[10px] font-medium text-ink-950 transition hover:bg-acid-300 disabled:opacity-50"
          >
            {busy === "approve" ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Check className="h-3 w-3" />
            )}
            Approve
          </button>
          <button
            type="button"
            disabled={busy !== null}
            onClick={() =>
              void run("reject", () =>
                actions.reject(runId, pending.step_id, feedback.trim()),
              )
            }
            className="inline-flex h-7 flex-1 items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2 text-[10px] text-slate-300 transition hover:bg-white/10 disabled:opacity-50"
          >
            {busy === "reject" ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <X className="h-3 w-3" />
            )}
            Reject
          </button>
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void run("abort", () => actions.abort(runId))}
            className="inline-flex h-7 shrink-0 items-center justify-center gap-1 rounded-lg border border-rose-400/20 bg-rose-500/5 px-2 text-[10px] text-rose-300 transition hover:bg-rose-500/10 disabled:opacity-50"
          >
            {busy === "abort" ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Ban className="h-3 w-3" />
            )}
            Abort run
          </button>
        </div>
      </div>
    </div>
  );
}

export function AgentTraceView({ trace, streaming, actions }: AgentTraceProps) {
  const [undoing, setUndoing] = useState(false);
  const [restoringPath, setRestoringPath] = useState<string | null>(null);
  const items = buildTimeline(trace);
  const liveText = trace.live_text.trim();
  const liveReasoning = trace.live_reasoning.trim();
  const canUndo =
    trace.undo_available && !trace.undone && trace.status !== "running";

  const undo = async () => {
    if (undoing) return;
    setUndoing(true);
    try {
      await actions.undo(trace.run_id);
    } catch {
      // useChat records the failure on this trace.
    } finally {
      setUndoing(false);
    }
  };

  const restoreFile = async (path: string) => {
    if (restoringPath) return;
    setRestoringPath(path);
    try {
      await actions.undoFile(trace.run_id, path);
    } catch {
      // useChat records the failure on this trace.
    } finally {
      setRestoringPath(null);
    }
  };

  return (
    <div className="mb-3 overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
      <div className="flex items-center gap-2 border-b border-white/5 px-3 py-2">
        <Sparkles className="h-3 w-3 text-acid-400/80" />
        <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">
          Agent
        </span>
        <span className={cn("h-1.5 w-1.5 rounded-full", statusTone(trace))} />
        <span className="text-[10px] text-slate-300">{statusLabel(trace)}</span>
        <span className="ml-auto font-mono text-[9px] text-slate-600">
          {trace.iterations}/{trace.max_iterations} steps
        </span>
      </div>

      {trace.plan.length ? (
        <div className="border-b border-white/5 px-3 py-2">
          <p className="text-[8px] uppercase tracking-[0.16em] text-slate-600">Plan</p>
          <ol className="mt-1 space-y-0.5">
            {trace.plan.map((item, index) => (
              <li
                key={`${index}-${item}`}
                className="flex gap-1.5 text-[10px] leading-4 text-slate-400"
              >
                <span className="font-mono text-[9px] text-acid-500/70">
                  {index + 1}.
                </span>
                <span className="min-w-0">{item}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      {items.length ? (
        <ul className="max-h-[340px] divide-y divide-white/5 overflow-y-auto">
          {items.map((item) =>
            item.kind === "compaction" ? (
              <CompactionRow key={item.event.id} event={item.event} />
            ) : (
              <StepRow key={item.step.id} step={item.step} />
            ),
          )}
        </ul>
      ) : null}

      {liveText || liveReasoning ? (
        <div className="border-t border-white/5 px-3 py-2">
          {liveReasoning ? (
            <p className="mb-1 line-clamp-3 whitespace-pre-wrap text-[9.5px] leading-4 text-slate-600">
              {liveReasoning}
            </p>
          ) : null}
          {liveText ? (
            <p className="max-h-32 overflow-y-auto whitespace-pre-wrap text-[11px] leading-5 text-slate-300">
              {liveText}
              {streaming ? (
                <span className="ml-0.5 inline-block h-3 w-1 translate-y-0.5 animate-pulse-soft rounded-sm bg-acid-400/80" />
              ) : null}
            </p>
          ) : null}
        </div>
      ) : null}

      {trace.pending ? (
        <ApprovalCard
          runId={trace.run_id}
          pending={trace.pending}
          actions={actions}
        />
      ) : null}

      {trace.notice ? (
        <p className="border-t border-white/5 px-3 py-1.5 text-[9.5px] leading-4 text-slate-500">
          {trace.notice}
        </p>
      ) : null}

      {trace.error ? (
        <p className="border-t border-rose-400/15 bg-rose-500/5 px-3 py-1.5 text-[9.5px] leading-4 text-rose-300">
          {trace.error}
        </p>
      ) : null}

      {trace.changes.length ? (
        <div className="border-t border-white/5">
          <div className="flex items-center gap-2 px-3 py-2">
            <span className="text-[8px] uppercase tracking-[0.16em] text-slate-600">
              Run changes
            </span>
            <span className="font-mono text-[9px] text-slate-600">
              {trace.changes.length} file{trace.changes.length === 1 ? "" : "s"}
            </span>
            <span className="ml-auto text-[9px] text-slate-600">
              restore one file or all
            </span>
          </div>
          <ul className="divide-y divide-white/5 border-t border-white/5">
            {trace.changes.map((change) => (
              <li
                key={change.path}
                className="flex items-center gap-2 px-3 py-1.5"
              >
                <span
                  className={cn(
                    "w-12 shrink-0 rounded border px-1 py-px text-center text-[8px] uppercase tracking-wider",
                    change.kind === "created"
                      ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-300/80"
                      : change.kind === "deleted"
                        ? "border-rose-400/20 bg-rose-400/5 text-rose-300/80"
                        : "border-amber-400/20 bg-amber-400/5 text-amber-300/80",
                  )}
                >
                  {change.kind}
                </span>
                <code className="min-w-0 flex-1 truncate font-mono text-[9.5px] text-slate-300">
                  {change.path}
                </code>
                {change.restored ? (
                  <span className="inline-flex shrink-0 items-center gap-1 text-[9px] text-slate-500">
                    <Check className="h-2.5 w-2.5" />
                    restored
                  </span>
                ) : trace.undo_available ? (
                  <button
                    type="button"
                    disabled={restoringPath !== null}
                    onClick={() => void restoreFile(change.path)}
                    className="inline-flex h-6 shrink-0 items-center gap-1 rounded-md border border-white/10 bg-white/5 px-1.5 text-[9px] text-slate-300 transition hover:bg-white/10 disabled:opacity-40"
                  >
                    {restoringPath === change.path ? (
                      <Loader2 className="h-2.5 w-2.5 animate-spin" />
                    ) : (
                      <Undo2 className="h-2.5 w-2.5" />
                    )}
                    Restore
                  </button>
                ) : (
                  <span className="shrink-0 text-[9px] text-slate-600">
                    not restored
                  </span>
                )}
              </li>
            ))}
          </ul>
          <p className="border-t border-white/5 px-3 py-1.5 text-[9px] leading-4 text-slate-600">
            Restore only touches files that still match what this run left
            behind; later edits are protected.
          </p>
        </div>
      ) : null}

      {canUndo ? (
        <div className="flex items-center gap-2 border-t border-white/5 px-3 py-2">
          <button
            type="button"
            disabled={undoing}
            onClick={() => void undo()}
            className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 text-[10px] text-slate-300 transition hover:bg-white/10 disabled:opacity-50"
          >
            {undoing ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Undo2 className="h-3 w-3" />
            )}
            Restore all files
          </button>
          <span className="text-[9px] text-slate-600">
            Only files still matching what this run left behind
          </span>
        </div>
      ) : null}

      {!trace.undone && trace.undo_available && trace.undo_summary ? (
        <p className="flex items-start gap-1.5 border-t border-amber-400/15 bg-amber-400/5 px-3 py-1.5 text-[9.5px] leading-4 text-amber-200/80">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          <span>
            <span className="font-medium">Restore incomplete.</span>{" "}
            {trace.undo_summary}
          </span>
        </p>
      ) : null}

      {trace.undone ? (
        <p className="flex items-center gap-1.5 border-t border-white/5 px-3 py-1.5 text-[9.5px] leading-4 text-slate-400">
          <Undo2 className="h-3 w-3 shrink-0" />
          Undo applied
          {trace.undo_summary ? ` · ${trace.undo_summary}` : ""}
        </p>
      ) : null}

      {trace.undo_error ? (
        <p className="border-t border-rose-400/15 bg-rose-500/5 px-3 py-1.5 text-[9.5px] leading-4 text-rose-300">
          {trace.undo_error}
        </p>
      ) : null}

      {trace.status === "verification_failed" ? (
        <p className="flex items-center gap-1.5 border-t border-rose-400/15 bg-rose-500/5 px-3 py-1.5 text-[9.5px] leading-4 text-rose-300">
          <AlertTriangle className="h-3 w-3 shrink-0" />
          The configured verification command did not pass. Treat this run's
          answer as unverified.
        </p>
      ) : null}

      {trace.status === "max_iterations" ? (
        <p className="flex items-center gap-1.5 border-t border-amber-400/15 bg-amber-400/5 px-3 py-1.5 text-[9.5px] text-amber-200/80">
          <Clock className="h-3 w-3" />
          Stopped at the step limit
          {trace.compactions.length
            ? ` · ${trace.compactions.length} compaction${
                trace.compactions.length === 1 ? "" : "s"
              }`
            : ""}
        </p>
      ) : null}
    </div>
  );
}
