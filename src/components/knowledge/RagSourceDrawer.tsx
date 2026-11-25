import { useState } from "react";
import { Check, Copy, FileCode2, FolderOpen, Hash, X } from "lucide-react";

import { formatCount, formatBytes } from "../../lib/format";
import { cn } from "../../lib/cn";
import { formatRagScore } from "../../lib/rag";
import type { RagChunk } from "./types";

interface RagSourceDrawerProps {
  chunk: RagChunk | null;
  onClose: () => void;
}

/**
 * The exact snippet behind a `[D1]` pill: file, line range, both halves of the
 * hybrid score and the chunk text the model actually read.
 */
export function RagSourceDrawer({ chunk, onClose }: RagSourceDrawerProps) {
  const [copied, setCopied] = useState<"path" | "content" | null>(null);

  if (!chunk) return null;

  const copy = async (value: string, kind: "path" | "content") => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(kind);
      window.setTimeout(() => setCopied(null), 1_500);
    } catch {
      // Clipboard permissions can be unavailable; the text stays selectable.
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close the document source"
        onClick={onClose}
        className="absolute inset-0 bg-ink-950/70 backdrop-blur-sm"
      />
      <aside className="relative flex h-full w-full max-w-[560px] flex-col border-l border-white/10 bg-ink-900 shadow-panel animate-fade-in">
        <header className="flex items-start justify-between gap-3 border-b border-white/5 px-5 py-4">
          <div className="min-w-0">
            <p className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <FileCode2 className="h-3 w-3 text-acid-400" />
              Document source
            </p>
            <h2 className="mt-1 flex flex-wrap items-center gap-2 text-sm font-semibold text-white">
              <span className="rounded-md border border-acid-400/30 bg-acid-400/10 px-1.5 py-0.5 font-mono text-[10px] text-acid-200">
                [{chunk.citation}]
              </span>
              <span className="min-w-0 truncate">{chunk.relative_path}</span>
            </h2>
            <p className="mt-1 font-mono text-[10.5px] text-slate-500">
              lines {chunk.start_line}–{chunk.end_line}
              {chunk.heading ? ` · ${chunk.heading}` : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-white/10 bg-white/5 p-1.5 text-slate-400 transition hover:text-slate-100"
            aria-label="Close"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </header>

        <div className="grid grid-cols-2 gap-2 px-5 py-3 sm:grid-cols-4">
          <Metric label="Relevance" value={formatRagScore(chunk.score)} tone="acid" />
          <Metric label="BM25" value={chunk.lexical_score.toFixed(2)} />
          <Metric label="TF-IDF" value={chunk.semantic_score.toFixed(3)} />
          <Metric label="Tokens" value={formatCount(chunk.token_estimate)} />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-5">
          <div className="rounded-xl border border-white/10 bg-ink-950/60">
            <div className="flex items-center justify-between gap-2 border-b border-white/5 px-3 py-2">
              <span className="text-[9.5px] uppercase tracking-wider text-slate-600">
                Retrieved snippet
              </span>
              <button
                type="button"
                onClick={() => void copy(chunk.content, "content")}
                className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9.5px] text-slate-400 transition hover:text-slate-100"
              >
                {copied === "content" ? (
                  <Check className="h-2.5 w-2.5 text-acid-300" />
                ) : (
                  <Copy className="h-2.5 w-2.5" />
                )}
                {copied === "content" ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="max-h-[46vh] overflow-auto whitespace-pre-wrap break-words px-3 py-2.5 font-mono text-[11px] leading-5 text-slate-300">
              {chunk.content}
            </pre>
          </div>

          <div className="mt-3 space-y-2">
            <Field
              icon={<FolderOpen className="h-3 w-3" />}
              label="Collection"
              value={chunk.collection_name || chunk.collection_id}
            />
            <div className="rounded-xl border border-white/10 bg-white/[0.02] px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-[9.5px] uppercase tracking-wider text-slate-600">
                  <Hash className="h-3 w-3" />
                  File path
                </span>
                <button
                  type="button"
                  onClick={() => void copy(chunk.file_path, "path")}
                  className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9.5px] text-slate-400 transition hover:text-slate-100"
                >
                  {copied === "path" ? (
                    <Check className="h-2.5 w-2.5 text-acid-300" />
                  ) : (
                    <Copy className="h-2.5 w-2.5" />
                  )}
                  {copied === "path" ? "Copied" : "Copy path"}
                </button>
              </div>
              <p className="mt-1 break-all font-mono text-[10.5px] leading-5 text-slate-400">
                {chunk.file_path}
              </p>
            </div>
          </div>

          <p className="mt-3 text-[10px] leading-5 text-slate-600">
            Chunk {chunk.chunk_id} · {formatBytes(chunk.content.length)} of text ·
            cited in the reply as [{chunk.citation}]
          </p>
        </div>
      </aside>
    </div>
  );
}

function Metric({
  label,
  value,
  tone = "slate",
}: {
  label: string;
  value: string;
  tone?: "slate" | "acid";
}) {
  return (
    <div
      className={cn(
        "rounded-lg border px-2.5 py-1.5",
        tone === "acid"
          ? "border-acid-400/25 bg-acid-400/[0.06]"
          : "border-white/10 bg-white/[0.02]",
      )}
    >
      <p className="text-[8.5px] uppercase tracking-wider text-slate-600">{label}</p>
      <p
        className={cn(
          "mt-0.5 font-mono text-[11px]",
          tone === "acid" ? "text-acid-200" : "text-slate-300",
        )}
      >
        {value}
      </p>
    </div>
  );
}

function Field({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-3 py-2">
      <span className="text-slate-600">{icon}</span>
      <span className="text-[9.5px] uppercase tracking-wider text-slate-600">
        {label}
      </span>
      <span className="ml-auto min-w-0 truncate text-[11px] text-slate-300">
        {value}
      </span>
    </div>
  );
}
