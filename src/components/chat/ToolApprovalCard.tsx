import { useState } from "react";
import { Ban, Check, ShieldAlert } from "lucide-react";

import { cn } from "../../lib/cn";
import type { ChatToolApproval } from "../../types";
import { Button } from "../ui/Button";

function prettyArguments(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "{}";
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2);
  } catch {
    return trimmed;
  }
}

interface ToolApprovalCardProps {
  approval: ChatToolApproval;
  onDecide: (decision: "approve" | "reject") => void | Promise<void>;
}

/**
 * Chat pauses before executing an MCP tool unless the server opted out. This
 * card is the only thing standing between the model and a filesystem or
 * shell server, so it stays visible above the composer until resolved.
 */
export function ToolApprovalCard({ approval, onDecide }: ToolApprovalCardProps) {
  const [busy, setBusy] = useState(false);
  const [showArguments, setShowArguments] = useState(false);

  const decide = async (decision: "approve" | "reject") => {
    setBusy(true);
    try {
      await onDecide(decision);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-xl border border-amber-400/25 bg-amber-400/[0.06] px-4 py-3 shadow-panel">
      <div className="flex items-start gap-2.5">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
        <div className="min-w-0 flex-1">
          <p className="text-[11.5px] font-medium leading-5 text-amber-100">
            Approval needed:{" "}
            <span className="font-mono text-amber-200">{approval.tool}</span> on{" "}
            <span className="text-amber-200">{approval.server}</span>
          </p>
          <p className="mt-0.5 text-[10px] leading-4 text-amber-200/60">
            {approval.reason}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <span
              className={cn(
                "rounded border px-1.5 py-0.5 text-[8.5px] uppercase tracking-wider",
                approval.risk === "read_only"
                  ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-300/80"
                  : "border-amber-400/25 bg-amber-400/10 text-amber-300/90",
              )}
            >
              {approval.risk === "read_only" ? "read-only" : "can change data"}
            </span>
            <button
              type="button"
              onClick={() => setShowArguments((value) => !value)}
              className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[8.5px] uppercase tracking-wider text-slate-400 transition hover:text-slate-200"
            >
              {showArguments ? "hide arguments" : "review arguments"}
            </button>
          </div>
          {showArguments ? (
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-white/5 bg-ink-950/70 px-2.5 py-2 font-mono text-[9.5px] leading-4 text-slate-400">
              {prettyArguments(approval.arguments)}
            </pre>
          ) : null}
        </div>
        <div className="flex shrink-0 gap-1.5">
          <Button
            size="sm"
            variant="primary"
            busy={busy}
            onClick={() => void decide("approve")}
            icon={<Check className="h-3 w-3" />}
          >
            Approve
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={busy}
            onClick={() => void decide("reject")}
            icon={<Ban className="h-3 w-3" />}
          >
            Deny
          </Button>
        </div>
      </div>
    </div>
  );
}
