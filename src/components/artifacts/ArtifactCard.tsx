import { ArrowUpRight, Loader2 } from "lucide-react";

import {
  artifactTypeLabel,
  type Artifact,
  type ArtifactType,
} from "../../lib/artifacts";
import { cn } from "../../lib/cn";

const TYPE_STYLES: Record<ArtifactType, string> = {
  html: "border-orange-400/25 bg-orange-400/10 text-orange-200/90",
  svg: "border-violet-400/25 bg-violet-400/10 text-violet-200/90",
  mermaid: "border-sky-400/25 bg-sky-400/10 text-sky-200/90",
  markdown: "border-slate-400/20 bg-slate-400/10 text-slate-300",
  json: "border-amber-400/25 bg-amber-400/10 text-amber-200/90",
  csv: "border-emerald-400/25 bg-emerald-400/10 text-emerald-200/90",
};

interface ArtifactCardProps {
  artifact: Artifact;
  active?: boolean;
  onOpen?: (artifact: Artifact) => void;
}

/**
 * Replaces a wall of fenced code in the reply. It stays a real button so the
 * card is reachable by keyboard and announces its type and size.
 */
export function ArtifactCard({ artifact, active, onOpen }: ArtifactCardProps) {
  return (
    <button
      type="button"
      onClick={() => onOpen?.(artifact)}
      className={cn(
        "group my-2 flex w-full items-center gap-2.5 rounded-xl border px-3 py-2 text-left transition",
        active
          ? "border-acid-400/40 bg-acid-400/[0.07] shadow-glow"
          : "border-white/10 bg-white/[0.02] hover:border-acid-400/25 hover:bg-white/[0.04]",
      )}
      aria-label={`Open ${artifactTypeLabel(artifact.type)} preview: ${artifact.title}`}
      title={`Open ${artifact.title} in the artifact pane`}
    >
      <span
        className={cn(
          "shrink-0 rounded-md border px-1.5 py-0.5 text-[8.5px] font-semibold uppercase tracking-wider",
          TYPE_STYLES[artifact.type],
        )}
      >
        {artifactTypeLabel(artifact.type)}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[11.5px] font-medium text-slate-200 group-hover:text-white">
          {artifact.title}
        </span>
        <span className="mt-0.5 flex items-center gap-1.5 text-[9px] text-slate-600">
          <span>{artifact.lineCount} lines</span>
          {artifact.version > 1 ? (
            <span className="rounded border border-white/10 bg-white/[0.03] px-1 py-px font-mono text-[8px] text-slate-400">
              v{artifact.version}
            </span>
          ) : null}
          {artifact.isStreaming ? (
            <span className="flex items-center gap-1 text-acid-300/80">
              <Loader2 className="h-2.5 w-2.5 animate-spin" />
              generating
            </span>
          ) : artifact.partial ? (
            <span className="text-amber-300/70">incomplete</span>
          ) : null}
        </span>
      </span>
      <span className="inline-flex shrink-0 items-center gap-1 rounded-md border border-white/10 bg-white/[0.03] px-1.5 py-1 text-[9px] text-slate-400 transition group-hover:border-acid-400/25 group-hover:text-acid-200">
        Open Preview
        <ArrowUpRight className="h-2.5 w-2.5" />
      </span>
    </button>
  );
}
