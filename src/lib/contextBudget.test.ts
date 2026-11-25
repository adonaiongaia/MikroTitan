import { describe, expect, it } from "vitest";

import {
  estimateTokens,
  fitContext,
  projectBundleFragments,
  selectProjectContext,
  type BudgetMessage,
} from "./contextBudget";

function historyMessage(index: number, size: number): BudgetMessage {
  return {
    id: `m${index}`,
    role: index % 2 === 0 ? "user" : "assistant",
    content: `turn ${index} ${"x".repeat(size)}`,
  };
}

const current: BudgetMessage = {
  id: "current",
  role: "user",
  content: "what should I do next?",
};

describe("fitContext with pinned turns", () => {
  it("drops old unpinned turns before a pinned one", () => {
    const prior = Array.from({ length: 8 }, (_, index) =>
      historyMessage(index, 1_500),
    );
    const pinnedTurn = { ...prior[1], pinned: true };
    const messages = [...prior.slice(0, 1), pinnedTurn, ...prior.slice(2), current];

    const result = fitContext({
      contextWindow: 4_096,
      maxTokens: 1_024,
      systemPrompt: "system instructions",
      messages,
    });

    expect(result.error).toBeNull();
    expect(result.messages.some((message) => message.id === "m1")).toBe(true);
    expect(result.messages.some((message) => message.id === "m0")).toBe(false);
    expect(result.trimmedMessages).toBeGreaterThan(0);
    expect(result.pinnedTokens).toBeGreaterThan(0);
  });

  it("excludes evicted turns without removing them from the input", () => {
    const messages: BudgetMessage[] = [
      historyMessage(0, 400),
      { ...historyMessage(1, 400), evicted: true },
      current,
    ];
    const result = fitContext({
      contextWindow: 8_192,
      maxTokens: 512,
      systemPrompt: "system",
      messages,
    });
    expect(result.error).toBeNull();
    expect(result.messages.some((message) => message.id === "m1")).toBe(false);
    expect(result.messages.some((message) => message.id === "m0")).toBe(true);
  });

  it("recalculates prompt tokens when a turn is pinned or evicted", () => {
    const base = {
      contextWindow: 8_192,
      maxTokens: 512,
      systemPrompt: "system",
    };
    const messages: BudgetMessage[] = [
      { id: "u1", role: "user", content: "a".repeat(3_000) },
      { id: "a1", role: "assistant", content: "b".repeat(3_000) },
      { id: "u2", role: "user", content: "c".repeat(3_000) },
      current,
    ];
    const normal = fitContext({ ...base, messages });
    const evicted = fitContext({
      ...base,
      messages: messages.map((message) =>
        message.id === "a1" ? { ...message, evicted: true } : message,
      ),
    });
    const pinned = fitContext({
      ...base,
      messages: messages.map((message) =>
        message.id === "a1" ? { ...message, pinned: true } : message,
      ),
    });

    expect(evicted.promptTokens).toBeLessThan(normal.promptTokens);
    expect(pinned.pinnedTokens).toBeGreaterThan(0);
    expect(pinned.messages.some((message) => message.id === "a1")).toBe(true);
    expect(pinned.promptTokens).toBeGreaterThanOrEqual(evicted.promptTokens);
  });
});

describe("fitContext with pinned attached context", () => {
  it("keeps the pinned block while clipping the unpinned bundle", () => {
    const pinnedContext = `=== src/pinned.rs ===\n${"p".repeat(3_000)}`;
    const result = fitContext({
      contextWindow: 8_192,
      maxTokens: 1_024,
      systemPrompt: "system",
      pinnedContext,
      extraContext: `=== src/huge.rs ===\n${"u".repeat(60_000)}`,
      messages: [current],
    });

    expect(result.error).toBeNull();
    expect(result.pinnedContext).toBe(pinnedContext);
    // Pinned context is carried separately so Rust appends it exactly once.
    expect(result.systemPrompt).not.toContain("=== src/pinned.rs ===");
    expect(result.systemPrompt).toContain("attached context truncated");
  });

  it("refuses a turn whose pinned items cannot fit, naming them", () => {
    const result = fitContext({
      contextWindow: 2_048,
      maxTokens: 1_024,
      systemPrompt: "system",
      pinnedContext: "p".repeat(20_000),
      pinnedLabels: ["src/huge.rs", "memory: design notes"],
      messages: [current],
    });

    expect(result.error).toContain("Pinned context needs");
    expect(result.error).toContain("src/huge.rs");
    expect(result.error).toContain("Unpin or evict");
  });
});

describe("project bundle fragments", () => {
  it("splits an ingested bundle into per-file items", () => {
    const bundle =
      "\n=== src/main.rs ===\nfn main() {}\n\n=== README.md ===\n# Hello\n";
    const fragments = projectBundleFragments(bundle);
    expect(fragments.map((fragment) => fragment.id)).toEqual([
      "file:src/main.rs",
      "file:README.md",
    ]);
    expect(fragments[0].tokens).toBeGreaterThan(0);
  });

  it("pins one file and evicts another", () => {
    const bundle =
      "\n=== src/main.rs ===\nfn main() {}\n\n=== README.md ===\n# Hello\n";
    const selection = selectProjectContext(
      bundle,
      new Set(["file:README.md"]),
      new Set(["file:src/main.rs"]),
    );
    expect(selection.pinnedText).toContain("=== README.md ===");
    expect(selection.unpinnedText).toBe("");
    expect(selection.evictedTokens).toBeGreaterThan(0);
    expect(selection.pinnedTokens).toBeGreaterThan(0);
  });

  it("treats an unmarked bundle as one fragment", () => {
    const fragments = projectBundleFragments("just some text");
    expect(fragments).toHaveLength(1);
    expect(fragments[0].id).toBe("file:bundle");
    expect(fragments[0].tokens).toBe(estimateTokens("just some text"));
  });
});
