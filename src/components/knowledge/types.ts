/**
 * Types shared by the Knowledge Vault view and the composer's Docs drawer.
 * The wire shapes live in `src/types.ts` next to every other IPC payload; this
 * module re-exports the subset the vault owns so its components import from
 * one place.
 */

import type {
  IndexedKnowledgeFile,
  KnowledgeCollection,
  KnowledgeCollectionDetail,
  RagChunk,
  RagIndexProgress,
  RagIndexSummary,
  RagQueryResult,
} from "../../types";

export type {
  IndexedKnowledgeFile,
  KnowledgeCollection,
  KnowledgeCollectionDetail,
  RagChunk,
  RagIndexProgress,
  RagIndexSummary,
  RagQueryResult,
};

export type { RagSettings } from "../../lib/rag";
export {
  DEFAULT_RAG_SETTINGS,
  MAX_RAG_MAX_TOKENS,
  MAX_RAG_TOP_K,
  MIN_RAG_MAX_TOKENS,
  MIN_RAG_TOP_K,
} from "../../lib/rag";

/** An index job the UI is currently watching. */
export interface RagIndexJob {
  collectionId: string;
  phase: RagIndexProgress["phase"];
  filesTotal: number;
  filesDone: number;
  chunksDone: number;
  currentFile: string;
  percent: number;
  message: string;
  cancelling: boolean;
}

export function jobFromProgress(progress: RagIndexProgress): RagIndexJob {
  return {
    collectionId: progress.collection_id,
    phase: progress.phase,
    filesTotal: progress.files_total,
    filesDone: progress.files_done,
    chunksDone: progress.chunks_done,
    currentFile: progress.current_file,
    percent: progress.percent,
    message: progress.message,
    cancelling: false,
  };
}

export function isActiveJob(phase: string): boolean {
  return phase === "scanning" || phase === "indexing" || phase === "writing";
}
