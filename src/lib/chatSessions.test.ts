import { describe, expect, it } from "vitest";

import {
  appendTurn,
  backfillTree,
  branchPrefix,
  forkAssistantTurn,
  forkUserTurn,
  normalizeSession,
  siblingsFor,
  switchMessageVersion,
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
    });
    expect(normalized?.archived_messages?.[0].content).toBe("old");
    expect(normalized?.draft?.text).toBe("half written");
    expect(normalized?.draft?.images[0].data_url).toContain("base64");
    expect(normalized?.project?.bundle).toBe("bundle");
    expect(normalized?.workspace_root).toBe("/tmp/workspace");
    expect(normalized?.mcp_server_ids).toEqual(["mcp-1"]);
    expect(normalized?.mcp_server_names).toEqual({ "mcp-1": "Files" });
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
