/**
 * Shared helpers for the local document vault.
 *
 * The backend re-clamps every budget against the live `--ctx`, so nothing here
 * is a security boundary: these are the numbers the composer shows and the
 * preferences it stores per chat.
 */

import type { KnowledgeCollection, RagChunk } from "../types";

export const DEFAULT_RAG_TOP_K = 6;
export const DEFAULT_RAG_MAX_TOKENS = 1_400;
export const MIN_RAG_MAX_TOKENS = 128;
export const MAX_RAG_MAX_TOKENS = 4_000;
export const MIN_RAG_TOP_K = 1;
export const MAX_RAG_TOP_K = 16;

const REQUEST_OVERHEAD_TOKENS = 512;
const MIN_PROMPT_BUDGET = 256;

export interface RagSettings {
  topK: number;
  maxTokens: number;
}

export const DEFAULT_RAG_SETTINGS: RagSettings = {
  topK: DEFAULT_RAG_TOP_K,
  maxTokens: DEFAULT_RAG_MAX_TOKENS,
};

/** Mirrors `outputReserve` in contextBudget.ts so both views agree. */
function outputReserve(contextWindow: number, maxReply?: number | null): number {
  const requested =
    typeof maxReply === "number" && maxReply > 0
      ? maxReply
      : Math.min(4096, Math.max(1024, Math.floor(contextWindow * 0.25)));
  return Math.min(requested, Math.max(128, Math.floor(contextWindow * 0.5)));
}

/** Input room the engine has left for this turn, before the prompt is built. */
export function promptBudgetTokens(
  contextWindow: number,
  maxReply?: number | null,
): number {
  const window = Math.max(
    512,
    Math.floor(Number.isFinite(contextWindow) && contextWindow > 0 ? contextWindow : 16_384),
  );
  const overhead = Math.min(
    REQUEST_OVERHEAD_TOKENS,
    Math.max(64, Math.floor(window * 0.05)),
  );
  return Math.max(
    MIN_PROMPT_BUDGET,
    window - outputReserve(window, maxReply) - overhead,
  );
}

/**
 * What the composer advertises for a configured ceiling. The Rust side does
 * the authoritative clamp after subtracting the system prompt, tool schemas
 * and this turn's messages; this only keeps the slider honest about a window
 * that is too small for the requested block.
 */
export function effectiveRagBudget(
  contextWindow: number,
  maxReply: number | null | undefined,
  configured: number,
  promptTokensInUse = 0,
): number {
  const room = promptBudgetTokens(contextWindow, maxReply) - promptTokensInUse;
  const ceiling = Math.max(0, Math.floor(room * 0.5));
  return Math.max(0, Math.min(normalizeRagTokens(configured), ceiling));
}

export function normalizeRagTokens(value: number | undefined): number {
  if (!Number.isFinite(value ?? NaN)) return DEFAULT_RAG_MAX_TOKENS;
  return Math.round(
    Math.min(MAX_RAG_MAX_TOKENS, Math.max(MIN_RAG_MAX_TOKENS, value as number)),
  );
}

export function normalizeRagTopK(value: number | undefined): number {
  if (!Number.isFinite(value ?? NaN)) return DEFAULT_RAG_TOP_K;
  return Math.round(
    Math.min(MAX_RAG_TOP_K, Math.max(MIN_RAG_TOP_K, value as number)),
  );
}

/** The per-chat preference, falling back to the defaults. */
export function ragSettingsFor(
  topK: number | undefined,
  maxTokens: number | undefined,
): RagSettings {
  return {
    topK: normalizeRagTopK(topK),
    maxTokens: normalizeRagTokens(maxTokens),
  };
}

export function ragScopeLabel(
  attachedIds: string[] | undefined,
  collections: KnowledgeCollection[],
): string | null {
  const ids = Array.isArray(attachedIds) ? attachedIds : [];
  if (!ids.length) return null;
  const names = ids
    .map((id) => collections.find((collection) => collection.id === id)?.name)
    .filter((name): name is string => Boolean(name));
  if (!names.length) return `${ids.length} collection${ids.length === 1 ? "" : "s"}`;
  if (names.length <= 2) return names.join(", ");
  return `${names[0]} +${names.length - 1}`;
}

/**
 * Drop attachment ids whose collection no longer exists. A deleted collection
 * must not keep every later turn failing its `rag_query` call.
 */
export function reconcileCollectionScope(
  ids: string[] | undefined,
  collections: KnowledgeCollection[],
): { ids: string[] | undefined; changed: boolean } {
  if (!Array.isArray(ids)) return { ids, changed: false };
  const known = new Set(collections.map((collection) => collection.id));
  const kept = ids.filter((id) => known.has(id));
  return {
    ids: kept,
    changed: kept.length !== ids.length || kept.some((id, index) => id !== ids[index]),
  };
}

export function citationNumber(tag: string): number {
  const match = /^\[?D(\d+)\]?$/i.exec(tag.trim());
  return match ? Number(match[1]) : 0;
}

/** Sort once so `[D1]` is always the best-ranked chunk. */
export function sortByCitation(chunks: RagChunk[]): RagChunk[] {
  return [...chunks].sort(
    (left, right) => citationNumber(left.citation) - citationNumber(right.citation),
  );
}

export function ragChunkForCitation(
  chunks: RagChunk[] | undefined,
  citation: string,
): RagChunk | null {
  const wanted = citationNumber(citation);
  if (!wanted || !Array.isArray(chunks)) return null;
  return (
    chunks.find((chunk) => citationNumber(chunk.citation) === wanted) ?? null
  );
}

/**
 * Rewrite bare `[D1]` markers into markdown links so ReactMarkdown can render
 * them as clickable pills. Links the model already wrote (`[D1](…)`) and code
 * spans are left alone; a marker with no matching chunk stays literal text.
 */
export function linkifyDocCitations(
  markdown: string,
  chunks: RagChunk[] | undefined,
): string {
  if (!markdown || !Array.isArray(chunks) || !chunks.length) return markdown;
  const known = new Set(
    chunks.map((chunk) => citationNumber(chunk.citation)).filter(Boolean),
  );
  if (!known.size) return markdown;

  let result = "";
  let index = 0;
  let inFence = false;
  let inInlineCode = false;

  while (index < markdown.length) {
    if (markdown.startsWith("```", index)) {
      inFence = !inFence;
      result += "```";
      index += 3;
      continue;
    }
    if (!inFence && markdown[index] === "`") {
      inInlineCode = !inInlineCode;
      result += "`";
      index += 1;
      continue;
    }
    if (!inFence && !inInlineCode && markdown[index] === "[") {
      const match = /^\[D(\d+)\](?!\()/.exec(markdown.slice(index));
      if (match) {
        const number = Number(match[1]);
        if (known.has(number)) {
          result += `[D${number}](rag://D${number})`;
          index += match[0].length;
          continue;
        }
      }
    }
    result += markdown[index];
    index += 1;
  }
  return result;
}

export function ragFileLabel(chunk: RagChunk): string {
  const path = chunk.relative_path || chunk.file_path;
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export function formatRagScore(score: number): string {
  return `${Math.round(Math.min(1, Math.max(0, score)) * 100)}%`;
}
