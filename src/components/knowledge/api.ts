/**
 * The vault's typed IPC facade. Commands themselves live in `src/lib/api.ts`
 * beside every other invocation; this narrows them for the knowledge UI so a
 * component never hand-writes a command name.
 */

import { api } from "../../lib/api";
import type {
  KnowledgeCollection,
  KnowledgeCollectionDetail,
  RagIndexSummary,
  RagQueryResult,
} from "./types";

export const knowledgeApi = {
  list: (): Promise<KnowledgeCollection[]> => api.ragListCollections(),
  detail: (id: string): Promise<KnowledgeCollectionDetail> =>
    api.ragCollectionDetail(id),
  create: (
    name: string,
    description: string,
    sourcePaths: string[] = [],
  ): Promise<KnowledgeCollection> =>
    api.ragCreateCollection(name, description, sourcePaths),
  remove: (id: string): Promise<boolean> => api.ragDeleteCollection(id),
  indexPaths: (collectionId: string, paths: string[]): Promise<RagIndexSummary> =>
    api.ragIndexPaths(collectionId, paths),
  reindex: (collectionId: string): Promise<RagIndexSummary> =>
    api.ragReindexCollection(collectionId),
  cancel: (collectionId: string): Promise<boolean> =>
    api.ragCancelIndex(collectionId),
  query: async (
    collectionIds: string[],
    query: string,
    topK: number,
    maxTokens: number,
  ): Promise<RagQueryResult> =>
    api.ragQuery(collectionIds, query, topK, maxTokens),
  removePath: (collectionId: string, path: string): Promise<KnowledgeCollection> =>
    api.ragRemovePath(collectionId, path),
};

export type KnowledgeApi = typeof knowledgeApi;
