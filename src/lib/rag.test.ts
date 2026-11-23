import { describe, expect, it } from "vitest";

import {
  effectiveRagBudget,
  linkifyDocCitations,
  normalizeRagTokens,
  normalizeRagTopK,
  promptBudgetTokens,
  ragChunkForCitation,
  ragScopeLabel,
  reconcileCollectionScope,
  sortByCitation,
} from "./rag";
import type { KnowledgeCollection, RagChunk } from "../types";

function chunk(citation: string, extra: Partial<RagChunk> = {}): RagChunk {
  return {
    chunk_id: `chunk-${citation}`,
    collection_id: "c1",
    collection_name: "Notes",
    file_path: `/docs/${citation}.md`,
    relative_path: `docs/${citation}.md`,
    start_line: 1,
    end_line: 10,
    content: "text",
    heading: "",
    token_estimate: 40,
    score: 0.8,
    lexical_score: 3,
    semantic_score: 0.4,
    citation,
    ...extra,
  };
}

function collection(id: string, name: string): KnowledgeCollection {
  return {
    id,
    name,
    description: "",
    source_paths: [],
    file_count: 0,
    chunk_count: 0,
    token_count: 0,
    byte_count: 0,
    created_at: 0,
    updated_at: 0,
  };
}

describe("linkifyDocCitations", () => {
  it("turns known markers into rag links", () => {
    const chunks = [chunk("D1"), chunk("D2")];
    expect(linkifyDocCitations("See [D1] and [D2].", chunks)).toBe(
      "See [D1](rag://D1) and [D2](rag://D2).",
    );
  });

  it("leaves unknown markers, code spans and fences alone", () => {
    const chunks = [chunk("D1")];
    expect(linkifyDocCitations("missing [D7]", chunks)).toBe("missing [D7]");
    expect(linkifyDocCitations("`[D1]`", chunks)).toBe("`[D1]`");
    expect(linkifyDocCitations("```\n[D1]\n```", chunks)).toBe("```\n[D1]\n```");
  });

  it("does not double-link a marker the model already wrote as a link", () => {
    const chunks = [chunk("D1")];
    expect(linkifyDocCitations("[D1](https://example.com)", chunks)).toBe(
      "[D1](https://example.com)",
    );
  });

  it("is a no-op without chunks", () => {
    expect(linkifyDocCitations("[D1]", [])).toBe("[D1]");
    expect(linkifyDocCitations("[D1]", undefined)).toBe("[D1]");
  });
});

describe("citation helpers", () => {
  it("looks up a chunk by tag and sorts by number", () => {
    const chunks = [chunk("D2"), chunk("D1")];
    expect(ragChunkForCitation(chunks, "D1")?.chunk_id).toBe("chunk-D1");
    expect(ragChunkForCitation(chunks, "[D2]")?.chunk_id).toBe("chunk-D2");
    expect(ragChunkForCitation(chunks, "D9")).toBeNull();
    expect(sortByCitation(chunks).map((entry) => entry.citation)).toEqual([
      "D1",
      "D2",
    ]);
  });
});

describe("scope helpers", () => {
  it("drops ids for deleted collections", () => {
    const collections = [collection("a", "Alpha"), collection("b", "Beta")];
    expect(reconcileCollectionScope(["a", "gone"], collections)).toEqual({
      ids: ["a"],
      changed: true,
    });
    expect(reconcileCollectionScope(["a"], collections).changed).toBe(false);
    expect(reconcileCollectionScope(undefined, collections)).toEqual({
      ids: undefined,
      changed: false,
    });
  });

  it("labels a chat's scope with names, not ids", () => {
    const collections = [collection("a", "Alpha"), collection("b", "Beta")];
    expect(ragScopeLabel(["a"], collections)).toBe("Alpha");
    expect(ragScopeLabel(["a", "b"], collections)).toBe("Alpha, Beta");
    expect(ragScopeLabel([], collections)).toBeNull();
  });
});

describe("token budget", () => {
  it("clamps stored preferences to the slider range", () => {
    expect(normalizeRagTokens(undefined)).toBe(1_400);
    expect(normalizeRagTokens(10)).toBe(128);
    expect(normalizeRagTokens(99_999)).toBe(4_000);
    expect(normalizeRagTopK(0)).toBe(1);
    expect(normalizeRagTopK(999)).toBe(16);
  });

  it("never advertises more than half of the free prompt room", () => {
    const room = promptBudgetTokens(16_384, 2_048);
    expect(room).toBeGreaterThan(8_000);
    expect(effectiveRagBudget(16_384, 2_048, 4_000)).toBe(4_000);
    expect(effectiveRagBudget(2_048, 1_024, 4_000)).toBeLessThan(4_000);
    expect(effectiveRagBudget(2_048, 1_024, 4_000)).toBeGreaterThanOrEqual(0);
  });
});
