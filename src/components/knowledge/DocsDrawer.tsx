import { useEffect } from "react";
import {
  CheckSquare,
  Database,
  FileText,
  FolderOpen,
  Loader2,
  SlidersHorizontal,
  Square,
  X,
} from "lucide-react";

import { cn } from "../../lib/cn";
import { formatCount } from "../../lib/format";
import {
  effectiveRagBudget,
  MAX_RAG_TOP_K,
  MAX_RAG_MAX_TOKENS,
  MIN_RAG_TOP_K,
  MIN_RAG_MAX_TOKENS,
  type RagSettings,
} from "../../lib/rag";
import { Button } from "../ui/Button";
import { Slider } from "../ui/Slider";
import type { KnowledgeCollection } from "./types";

interface DocsDrawerProps {
  open: boolean;
  onClose: () => void;
  collections: KnowledgeCollection[];
  loading: boolean;
  error: string | null;
  /** This chat's attached collections; `[]` means none, undefined means none. */
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  settings: RagSettings;
  onSettings: (next: RagSettings) => void;
  contextWindow: number;
  maxReply: number | null;
  promptTokensInUse: number;
  onOpenVault: () => void;
  onRefresh: () => void;
}

/**
 * Per-chat document scope and retrieval tuning, the Docs counterpart to the
 * MCP tool scope. Everything is local: checking a collection only changes which
 * vectors this chat's `rag_query` reads before the turn is sent.
 */
export function DocsDrawer({
  open,
  onClose,
  collections,
  loading,
  error,
  selectedIds,
  onChange,
  settings,
  onSettings,
  contextWindow,
  maxReply,
  promptTokensInUse,
  onOpenVault,
  onRefresh,
}: DocsDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const selected = new Set(selectedIds);
  const attached = collections.filter((collection) => selected.has(collection.id));
  const budget = effectiveRagBudget(
    contextWindow,
    maxReply,
    settings.maxTokens,
    promptTokensInUse,
  );

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    // Keep the stored order stable so an unchanged scope does not rewrite the
    // session file on every click.
    onChange(collections.filter((collection) => next.has(collection.id)).map((collection) => collection.id));
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close document settings"
        onClick={onClose}
        className="absolute inset-0 bg-ink-950/70 backdrop-blur-sm"
      />
      <aside className="relative flex h-full w-full max-w-[440px] flex-col border-l border-white/10 bg-ink-900 shadow-panel animate-fade-in">
        <header className="flex items-start justify-between gap-3 border-b border-white/5 px-5 py-4">
          <div>
            <p className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <Database className="h-3 w-3 text-acid-400" />
              Document vault
            </p>
            <h2 className="mt-1 text-sm font-semibold text-white">Docs for this chat</h2>
            <p className="mt-1 text-[10.5px] leading-5 text-slate-500">
              Attached collections are searched locally before each turn. Chunks
              arrive as [D1], [D2], … and stay inside the token budget below.
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

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-wider text-slate-600">
              Collections
            </p>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={onRefresh}
                className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9.5px] text-slate-400 transition hover:text-slate-100"
              >
                Refresh
              </button>
              <Button size="sm" variant="secondary" onClick={onOpenVault}>
                Open vault
              </Button>
            </div>
          </div>

          {error ? (
            <p className="mt-2 rounded-lg border border-rose-400/20 bg-rose-500/5 px-2.5 py-2 text-[10.5px] leading-5 text-rose-300">
              {error}
            </p>
          ) : null}

          {loading ? (
            <div className="mt-4 flex items-center gap-2 text-[11px] text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading collections…
            </div>
          ) : collections.length === 0 ? (
            <div className="mt-3 rounded-xl border border-dashed border-white/10 bg-white/[0.02] px-3 py-6 text-center">
              <FolderOpen className="mx-auto h-5 w-5 text-slate-600" />
              <p className="mt-2 text-[11px] text-slate-400">No collections yet</p>
              <p className="mt-1 text-[10px] leading-5 text-slate-600">
                Index a folder of notes, documents or code, then attach it here.
              </p>
              <Button
                size="sm"
                variant="primary"
                className="mt-3"
                onClick={onOpenVault}
              >
                Create a collection
              </Button>
            </div>
          ) : (
            <ul className="mt-2 space-y-1.5">
              {collections.map((collection) => {
                const checked = selected.has(collection.id);
                return (
                  <li key={collection.id}>
                    <button
                      type="button"
                      onClick={() => toggle(collection.id)}
                      className={cn(
                        "flex w-full items-start gap-2.5 rounded-xl border px-2.5 py-2 text-left transition",
                        checked
                          ? "border-acid-400/30 bg-acid-400/[0.06]"
                          : "border-white/10 bg-white/[0.02] hover:border-white/20",
                      )}
                      aria-pressed={checked}
                    >
                      <span
                        className={cn(
                          "mt-0.5 shrink-0",
                          checked ? "text-acid-300" : "text-slate-600",
                        )}
                      >
                        {checked ? (
                          <CheckSquare className="h-3.5 w-3.5" />
                        ) : (
                          <Square className="h-3.5 w-3.5" />
                        )}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[11.5px] font-medium text-slate-200">
                          {collection.name}
                        </span>
                        <span className="mt-0.5 block truncate text-[9.5px] text-slate-600">
                          {formatCount(collection.file_count)} files ·{" "}
                          {formatCount(collection.chunk_count)} chunks ·{" "}
                          {formatCount(collection.token_count)} tokens
                        </span>
                      </span>
                      <FileText className="mt-0.5 h-3 w-3 shrink-0 text-slate-700" />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          <div className="mt-5 rounded-xl border border-white/10 bg-white/[0.02] p-3">
            <p className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-600">
              <SlidersHorizontal className="h-3 w-3" />
              Retrieval budget
            </p>
            <div className="mt-3 space-y-4">
              <Slider
                label="Top-K chunks"
                value={settings.topK}
                min={MIN_RAG_TOP_K}
                max={MAX_RAG_TOP_K}
                step={1}
                onChange={(topK) => onSettings({ ...settings, topK })}
                formatValue={(value) => `${value}`}
                leftHint="Fewer"
                rightHint="More candidates"
                description="BM25 and TF-IDF each rank this many candidates before Reciprocal Rank Fusion picks the final order."
              />
              <Slider
                label="Max RAG tokens"
                value={settings.maxTokens}
                min={MIN_RAG_MAX_TOKENS}
                max={MAX_RAG_MAX_TOKENS}
                step={64}
                onChange={(maxTokens) => onSettings({ ...settings, maxTokens })}
                formatValue={(value) => formatCount(value)}
                leftHint={`${MIN_RAG_MAX_TOKENS}`}
                rightHint={`${formatCount(MAX_RAG_MAX_TOKENS)}`}
                description="A hard ceiling for the injected [D#] block. When the context window is tight the backend clamps it further and returns fewer chunks instead of overflowing."
              />
            </div>
            <div className="mt-3 flex items-center justify-between rounded-lg border border-white/5 bg-ink-950/50 px-2.5 py-1.5 text-[9.5px]">
              <span className="text-slate-500">Effective in this window</span>
              <span className="font-mono text-acid-200">
                {formatCount(budget)} tok
              </span>
            </div>
          </div>

        </div>

        <footer className="border-t border-white/5 px-5 py-3">
          <div className="flex items-center justify-between text-[10px] text-slate-600">
            <span>
              {attached.length
                ? `${attached.length} collection${attached.length === 1 ? "" : "s"} attached`
                : "No collections attached"}
            </span>
            <Button size="sm" variant="primary" onClick={onClose}>
              Done
            </Button>
          </div>
        </footer>
      </aside>
    </div>
  );
}
