import type { McpServerInput } from "../types";

interface ScopedServer {
  id: string;
  enabled: boolean;
  command: string;
}

interface RequestServer extends ScopedServer {
  name: string;
  args: string[];
  env?: string;
  cwd?: string;
  autoApprove?: boolean;
}

export interface McpScopeReconciliation {
  ids: string[] | undefined;
  names: Record<string, string> | undefined;
  changed: boolean;
  removed: number;
  remapped: number;
}

/**
 * Enabled, runnable MCP servers a chat may use. No scope means every server;
 * an empty scope means none, so unchecking everything really disables tools.
 */
export function mcpServersInScope<T extends ScopedServer>(
  servers: T[],
  scope: string[] | undefined,
): T[] {
  return servers.filter(
    (server) =>
      server.enabled &&
      server.command.trim() &&
      (scope === undefined || scope.includes(server.id)),
  );
}

/** The request payload for the servers inside a chat's stored scope. */
export function mcpServerInputs(
  servers: RequestServer[],
  scope: string[] | undefined,
): McpServerInput[] {
  return mcpServersInScope(servers, scope).map((server) => ({
    name: server.name,
    command: server.command,
    args: server.args,
    env: server.env ?? "",
    cwd: server.cwd ?? "",
    auto_approve: Boolean(server.autoApprove),
  }));
}

/** Names captured with a scope, used to remap servers re-created under a new id. */
export function mcpScopeNames(
  ids: string[] | undefined,
  servers: Array<{ id: string; name: string }>,
): Record<string, string> | undefined {
  if (!ids?.length) return undefined;
  const names: Record<string, string> = {};
  for (const id of ids) {
    const server = servers.find((candidate) => candidate.id === id);
    if (server?.name.trim()) names[id] = server.name;
  }
  return Object.keys(names).length ? names : undefined;
}

/**
 * Drop scope entries for servers that no longer exist. If a server was
 * re-created under a new id with the same name, move the entry to the new id
 * so the chat keeps its tools.
 */
export function reconcileMcpScope(
  scope: string[] | undefined,
  names: Record<string, string> | undefined,
  servers: Array<{ id: string; name: string }>,
): McpScopeReconciliation {
  if (scope === undefined) {
    return {
      ids: undefined,
      names: undefined,
      changed: names !== undefined,
      removed: 0,
      remapped: 0,
    };
  }

  const byId = new Map(servers.map((server) => [server.id, server]));
  const idsByName = new Map<string, string[]>();
  for (const server of servers) {
    const key = server.name.trim().toLowerCase();
    if (!key) continue;
    const ids = idsByName.get(key);
    if (ids) ids.push(server.id);
    else idsByName.set(key, [server.id]);
  }

  const ids: string[] = [];
  const nextNames: Record<string, string> = {};
  let removed = 0;
  let remapped = 0;

  for (const id of scope) {
    let server = byId.get(id);
    if (!server) {
      const key = names?.[id]?.trim().toLowerCase();
      const candidates = key ? idsByName.get(key) : undefined;
      if (candidates?.length === 1) {
        server = byId.get(candidates[0]);
        remapped += 1;
      } else {
        removed += 1;
      }
    }
    if (!server || ids.includes(server.id)) continue;
    ids.push(server.id);
    nextNames[server.id] = server.name;
  }

  const scopedNames = Object.keys(nextNames).length ? nextNames : undefined;
  const changed =
    scope.length !== ids.length ||
    scope.some((id, index) => id !== ids[index]) ||
    !sameNames(names, scopedNames);

  return { ids, names: scopedNames, changed, removed, remapped };
}

function sameNames(
  left: Record<string, string> | undefined,
  right: Record<string, string> | undefined,
): boolean {
  const leftEntries = Object.entries(left ?? {});
  return (
    leftEntries.length === Object.keys(right ?? {}).length &&
    leftEntries.every(([id, name]) => right?.[id] === name)
  );
}
