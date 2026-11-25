import { describe, expect, it } from "vitest";

import { normalizeSession } from "./chatSessions";
import {
  mcpScopeNames,
  mcpServerInputs,
  mcpServersInScope,
  reconcileMcpScope,
  toggleServerInScope,
} from "./mcpScope";

const servers = [
  { id: "mcp-1", name: "Files", command: "files", args: [], enabled: true },
  {
    id: "mcp-2",
    name: "Search",
    command: "search",
    args: ["--fast"],
    enabled: true,
  },
  { id: "mcp-off", name: "Off", command: "off", args: [], enabled: false },
  { id: "mcp-blank", name: "Blank", command: "  ", args: [], enabled: true },
];

describe("mcpServersInScope", () => {
  it("returns every enabled, runnable server when no scope is stored", () => {
    expect(mcpServersInScope(servers, undefined).map((server) => server.id)).toEqual([
      "mcp-1",
      "mcp-2",
    ]);
  });

  it("returns no servers for an empty scope", () => {
    expect(mcpServersInScope(servers, [])).toEqual([]);
  });

  it("returns only selected, runnable servers", () => {
    expect(
      mcpServersInScope(servers, ["mcp-2", "mcp-off", "mcp-blank"]).map(
        (server) => server.id,
      ),
    ).toEqual(["mcp-2"]);
  });
});

describe("mcpServerInputs", () => {
  it("sends no MCP servers for an empty scope", () => {
    expect(mcpServerInputs(servers, [])).toEqual([]);
  });

  it("sends no MCP servers after an empty scope is normalized", () => {
    const stored = normalizeSession({
      id: "scoped",
      title: "Scoped",
      created_at: 1,
      updated_at: 2,
      messages: [],
      mcp_server_ids: [],
    });
    expect(stored?.mcp_server_ids).toEqual([]);
    expect(mcpServerInputs(servers, stored?.mcp_server_ids)).toEqual([]);
  });

  it("sends every enabled server when the chat has no scope", () => {
    expect(
      mcpServerInputs(servers, undefined).map((server) => server.name),
    ).toEqual(["Files", "Search"]);
  });

  it("sends the request shape for selected servers", () => {
    expect(mcpServerInputs(servers, ["mcp-2"])).toEqual([
      {
        name: "Search",
        command: "search",
        args: ["--fast"],
        env: "",
        cwd: "",
        auto_approve: false,
      },
    ]);
  });
});

describe("mcpScopeNames", () => {
  it("captures names for the selected servers", () => {
    expect(mcpScopeNames(["mcp-1", "mcp-2"], servers)).toEqual({
      "mcp-1": "Files",
      "mcp-2": "Search",
    });
  });

  it("has no names for all-servers or empty scopes", () => {
    expect(mcpScopeNames(undefined, servers)).toBeUndefined();
    expect(mcpScopeNames([], servers)).toBeUndefined();
  });
});

describe("reconcileMcpScope", () => {
  const current = [
    { id: "mcp-1", name: "Files" },
    { id: "mcp-2", name: "Search" },
  ];

  it("keeps valid entries and captures their names", () => {
    const result = reconcileMcpScope(["mcp-1"], undefined, current);
    expect(result.ids).toEqual(["mcp-1"]);
    expect(result.names).toEqual({ "mcp-1": "Files" });
    expect(result.changed).toBe(true);
    expect(result.removed).toBe(0);
    expect(reconcileMcpScope(result.ids, result.names, current).changed).toBe(
      false,
    );
  });

  it("drops servers that were removed", () => {
    const result = reconcileMcpScope(
      ["mcp-1", "gone"],
      { gone: "Old" },
      current,
    );
    expect(result.ids).toEqual(["mcp-1"]);
    expect(result.removed).toBe(1);
    expect(result.changed).toBe(true);
  });

  it("remaps a server re-created under the same name", () => {
    const result = reconcileMcpScope(["old"], { old: "search" }, current);
    expect(result.ids).toEqual(["mcp-2"]);
    expect(result.names).toEqual({ "mcp-2": "Search" });
    expect(result.remapped).toBe(1);
    expect(result.changed).toBe(true);
  });

  it("does not guess when a re-created name is ambiguous", () => {
    const result = reconcileMcpScope(["old"], { old: "Shared" }, [
      { id: "a", name: "Shared" },
      { id: "b", name: "shared" },
    ]);
    expect(result.ids).toEqual([]);
    expect(result.removed).toBe(1);
  });

  it("keeps an explicit empty scope and reports no change", () => {
    const result = reconcileMcpScope([], undefined, current);
    expect(result.ids).toEqual([]);
    expect(result.names).toBeUndefined();
    expect(result.changed).toBe(false);
  });

  it("leaves the all-servers scope untouched", () => {
    const result = reconcileMcpScope(undefined, undefined, current);
    expect(result.ids).toBeUndefined();
    expect(result.names).toBeUndefined();
    expect(result.changed).toBe(false);
  });
});

describe("toggleServerInScope", () => {
  const available = ["mcp-1", "mcp-2"];

  it("materialises the all-servers scope when one server is evicted", () => {
    expect(toggleServerInScope(undefined, "mcp-1", available)).toEqual([
      "mcp-2",
    ]);
  });

  it("evicts the last scoped server as an explicit empty scope", () => {
    expect(toggleServerInScope(["mcp-1"], "mcp-1", available)).toEqual([]);
  });

  it("restores an evicted server", () => {
    expect(toggleServerInScope(["mcp-2"], "mcp-1", available)).toEqual([
      "mcp-2",
      "mcp-1",
    ]);
  });
});
