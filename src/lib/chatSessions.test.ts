import { describe, expect, it } from "vitest";

import {
  appendTurn,
  backfillTree,
  branchPrefix,
  evictedContextIds,
  forkAssistantTurn,
  forkUserTurn,
  inheritPinnedContext,
  isMessageEvicted,
  isMessagePinned,
  normalizeRagChunk,
  normalizeSession,
  pinnedContextIds,
  pruneMcpContextMarkers,
  restoreEvictedContext,
  siblingsFor,
  switchMessageVersion,
  toggleEvictedContextItem,
  togglePinnedContextItem,
} from "./chatSessions";
import type { ChatMessage, ChatSession } from "../types";

function message(
  id: string,
  role: "user" | "assistant",
  content: string,
  created_at: number,
): ChatMessage {
  return {
    id,
    role,
    content,
    reasoning: "",
    created_at,
  };
}

function session(messages: ChatMessage[]): ChatSession {
  return {
    id: "chat",
    title: "Test chat",
    created_at: 1,
    updated_at: 2,
    messages,
    archived_messages: [],
  };
}

function linearSession(): ChatSession {
  const value = session([
    message("u1", "user", "first", 1),
    message("a1", "assistant", "first answer", 2),
    message("u2", "user", "second", 3),
    message("a2", "assistant", "second answer", 4),
  ]);
  backfillTree(value.messages);
  return value;
}

describe("normalizeSession", () => {
  it("backfills parents and active children for legacy chats", () => {
    const normalized = normalizeSession({
      id: "legacy",
      title: "Legacy",
      created_at: 1,
      updated_at: 2,
      messages: [
        { id: "u1", role: "user", content: "hi", created_at: 1 },
        { id: "a1", role: "assistant", content: "hello", created_at: 2 },
      ],
    });
    expect(normalized).not.toBeNull();
    expect(normalized?.messages[0].parent_id).toBeNull();
    expect(normalized?.messages[0].active_child_id).toBe("a1");
    expect(normalized?.messages[1].parent_id).toBe("u1");
    expect(normalized?.messages[1].active_child_id).toBeNull();
  });

  it("preserves archived branches, drafts and project context", () => {
    const normalized = normalizeSession({
      id: "rich",
      title: "Rich",
      created_at: 1,
      updated_at: 2,
      messages: [{ id: "u1", role: "user", content: "new", created_at: 3 }],
      archived_messages: [
        { id: "old", role: "user", content: "old", created_at: 1 },
      ],
      draft: {
        text: "half written",
        images: [{ name: "shot.png", data_url: "data:image/png;base64,AA==", tokens: 3 }],
        updated_at: 4,
      },
      project: {
        root: "/tmp/project",
        name: "project",
        files: 2,
        bytes: 20,
        tokens: 5,
        tree: "src",
        bundle: "bundle",
        truncated: false,
      },
      workspace_root: "/tmp/workspace",
      mcp_server_ids: ["mcp-1"],
      mcp_server_names: { "mcp-1": "Files" },
      system_prompt: "You are a careful reviewer.",
    });
    expect(normalized?.archived_messages?.[0].content).toBe("old");
    expect(normalized?.draft?.text).toBe("half written");
    expect(normalized?.draft?.images[0].data_url).toContain("base64");
    expect(normalized?.project?.bundle).toBe("bundle");
    expect(normalized?.workspace_root).toBe("/tmp/workspace");
    expect(normalized?.mcp_server_ids).toEqual(["mcp-1"]);
    expect(normalized?.mcp_server_names).toEqual({ "mcp-1": "Files" });
    expect(normalized?.system_prompt).toBe("You are a careful reviewer.");
    expect(normalizeSession(session([]))?.system_prompt).toBeUndefined();
  });

  it("keeps an empty MCP scope distinct from all servers", () => {
    const normalized = normalizeSession({
      ...session([]),
      mcp_server_ids: [],
      mcp_server_names: { "mcp-1": "Files" },
    });
    expect(normalized?.mcp_server_ids).toEqual([]);
    expect(normalized?.mcp_server_names).toBeUndefined();
  });
});

describe("branching turns", () => {
  it("links appended turns into the active path", () => {
    const base = session([]);
    const next = appendTurn(
      base,
      message("u1", "user", "hello", 10),
      message("a1", "assistant", "hi", 11),
    );
    expect(next.messages.map((item) => item.id)).toEqual(["u1", "a1"]);
    expect(next.messages[0].parent_id).toBeNull();
    expect(next.messages[0].active_child_id).toBe("a1");
    expect(next.messages[1].parent_id).toBe("u1");
    expect(next.archived_messages).toEqual([]);
  });

  it("editing an earlier prompt keeps the old continuation archived", () => {
    const base = linearSession();
    const edited = message("u1b", "user", "edited first", 10);
    const answer = message("a1b", "assistant", "new answer", 11);
    const next = forkUserTurn(base, 0, edited, answer);

    expect(next.messages.map((item) => item.id)).toEqual(["u1b", "a1b"]);
    expect(next.archived_messages?.map((item) => item.id)).toEqual([
      "u1",
      "a1",
      "u2",
      "a2",
    ]);
    expect(siblingsFor(next, next.messages[0]).map((item) => item.id)).toEqual([
      "u1",
      "u1b",
    ]);
  });

  it("switching back restores the archived continuation", () => {
    const base = linearSession();
    const next = forkUserTurn(
      base,
      0,
      message("u1b", "user", "edited first", 10),
      message("a1b", "assistant", "new answer", 11),
    );
    const restored = switchMessageVersion(next, "u1b", "u1");
    expect(restored.messages.map((item) => item.id)).toEqual([
      "u1",
      "a1",
      "u2",
      "a2",
    ]);
    expect(restored.archived_messages?.map((item) => item.id)).toEqual([
      "u1b",
      "a1b",
    ]);
  });

  it("regenerating preserves the previous answer as a version", () => {
    const base = linearSession();
    const next = forkAssistantTurn(
      base,
      3,
      message("a2b", "assistant", "regenerated", 20),
    );
    expect(next.messages.map((item) => item.id)).toEqual([
      "u1",
      "a1",
      "u2",
      "a2b",
    ]);
    expect(siblingsFor(next, next.messages[3]).map((item) => item.id)).toEqual([
      "a2",
      "a2b",
    ]);
    const restored = switchMessageVersion(next, "a2b", "a2");
    expect(restored.messages[3].id).toBe("a2");
  });

  it("branchPrefix copies only the path to the chosen message", () => {
    const prefix = branchPrefix(linearSession(), "a1");
    expect(prefix.map((item) => item.id)).toEqual(["u1", "a1"]);
    expect(prefix[1].active_child_id).toBeNull();
  });
});

describe("pinned and evicted context", () => {
  it("treats a chat without pinned_context as having no pins", () => {
    const normalized = normalizeSession({
      id: "legacy-pins",
      title: "Legacy",
      created_at: 1,
      updated_at: 2,
      messages: [
        { id: "u1", role: "user", content: "hi", created_at: 1 },
        { id: "a1", role: "assistant", content: "hello", created_at: 2 },
      ],
    });
    expect(normalized).not.toBeNull();
    expect(normalized?.pinned_context).toBeUndefined();
    expect(normalized?.evicted_context_ids).toBeUndefined();
    expect(pinnedContextIds(normalized).size).toBe(0);
    expect(evictedContextIds(normalized).size).toBe(0);
    expect(isMessagePinned(normalized, "a1")).toBe(false);
    expect(isMessageEvicted(normalized, "a1")).toBe(false);
  });

  it("normalizes valid items and drops duplicates and malformed entries", () => {
    const normalized = normalizeSession({
      ...session([]),
      pinned_context: [
        {
          id: "file:src/main.rs",
          kind: "file",
          label: "src/main.rs",
          tokens: 12,
          added_at: 5,
        },
        {
          id: "file:src/main.rs",
          kind: "file",
          label: "duplicate",
          tokens: 12,
          added_at: 5,
        },
        { id: "rag:chunk-1" },
        { id: "", kind: "file" },
      ],
      evicted_context_ids: ["message:m1", "message:m1", 7],
    });
    expect(normalized?.pinned_context?.map((item) => item.id)).toEqual([
      "file:src/main.rs",
      "rag:chunk-1",
    ]);
    // A missing kind is recovered from the id prefix.
    expect(normalized?.pinned_context?.[1].kind).toBe("rag");
    expect(normalized?.evicted_context_ids).toEqual(["message:m1"]);
  });

  it("pinning restores an evicted item and evicting unpins it", () => {
    let value = linearSession();
    value = toggleEvictedContextItem(value, "message:a1");
    expect(isMessageEvicted(value, "a1")).toBe(true);

    value = togglePinnedContextItem(value, {
      id: "message:a1",
      kind: "message",
      label: "first answer",
      tokens: 9,
      added_at: 1,
    });
    expect(isMessagePinned(value, "a1")).toBe(true);
    expect(isMessageEvicted(value, "a1")).toBe(false);

    value = toggleEvictedContextItem(value, "message:a1");
    expect(isMessagePinned(value, "a1")).toBe(false);
    expect(isMessageEvicted(value, "a1")).toBe(true);

    value = restoreEvictedContext(value);
    expect(value.evicted_context_ids).toBeUndefined();
  });

  it("branches inherit file pins and only surviving message pins", () => {
    const source = {
      ...linearSession(),
      pinned_context: [
        {
          id: "file:src/main.rs",
          kind: "file" as const,
          label: "src/main.rs",
          tokens: 4,
          added_at: 1,
        },
        {
          id: "message:a2",
          kind: "message" as const,
          label: "second answer",
          tokens: 4,
          added_at: 1,
        },
      ],
      evicted_context_ids: ["message:u2", "file:src/old.rs"],
    };
    const branch = session([
      message("u1", "user", "first", 1),
      message("a1", "assistant", "first answer", 2),
    ]);
    const inherited = inheritPinnedContext(source, branch);
    expect(inherited.pinned_context?.map((item) => item.id)).toEqual([
      "file:src/main.rs",
    ]);
    expect(inherited.evicted_context_ids).toEqual(["file:src/old.rs"]);
  });
});

describe("pruneMcpContextMarkers", () => {
  it("drops pins and evictions for a deleted server", () => {
    const base = session([message("m1", "user", "hi", 1)]);
    const withMarkers: ChatSession = {
      ...base,
      pinned_context: [
        {
          id: "mcp:gone",
          kind: "mcp",
          label: "Gone",
          tokens: 12,
          added_at: 1,
        },
        {
          id: "mcp:kept",
          kind: "mcp",
          label: "Kept",
          tokens: 7,
          added_at: 1,
        },
        {
          id: "message:m1",
          kind: "message",
          label: "hi",
          tokens: 3,
          added_at: 1,
        },
      ],
      evicted_context_ids: ["mcp:gone", "mcp:kept", "message:m1"],
    };
    const pruned = pruneMcpContextMarkers(withMarkers, [{ id: "kept" }]);
    expect(pruned.pinned_context?.map((item) => item.id)).toEqual([
      "mcp:kept",
      "message:m1",
    ]);
    expect(pruned.evicted_context_ids).toEqual(["mcp:kept", "message:m1"]);
  });

  it("returns the same session when every marker is still valid", () => {
    const base = session([message("m1", "user", "hi", 1)]);
    const scoped: ChatSession = {
      ...base,
      evicted_context_ids: ["mcp:kept"],
    };
    expect(pruneMcpContextMarkers(scoped, [{ id: "kept" }])).toBe(scoped);
  });
});

describe("stored document sources", () => {
  it("normalizes legacy bracketed citations", () => {
    const chunk = {
      chunk_id: "c1",
      content: "text",
      file_path: "/tmp/a.md",
    };
    expect(normalizeRagChunk({ ...chunk, citation: "[D2]" })?.citation).toBe("D2");
    expect(normalizeRagChunk({ ...chunk, citation: "D3" })?.citation).toBe("D3");
  });
});
