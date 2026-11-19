import { useEffect, useState } from "react";
import { Terminal, Trash2, Wrench, X } from "lucide-react";

import { cn } from "../../lib/cn";
import {
  clearMcpLog,
  readMcpLog,
  subscribeMcpLog,
  type McpLogEntry,
} from "../../lib/mcpLog";
import type { McpServerConfig } from "../../types";

interface McpLogDrawerProps {
  open: boolean;
  onClose: () => void;
  servers: McpServerConfig[];
  selectedIds?: string[];
  onChangeScope: (ids: string[] | undefined) => void;
}

export function McpLogDrawer({
  open,
  onClose,
  servers,
  selectedIds,
  onChangeScope,
}: McpLogDrawerProps) {
  const [log, setLog] = useState<McpLogEntry[]>(readMcpLog);

  useEffect(() => {
    setLog(readMcpLog());
    return subscribeMcpLog(() => setLog(readMcpLog()));
  }, []);

  if (!open) return null;

  return (
    <div className="fixed bottom-16 right-4 z-40 w-[380px] overflow-hidden rounded-2xl border border-white/10 bg-ink-900 shadow-panel animate-fade-in">
      <div className="flex items-center gap-2 border-b border-white/5 px-3.5 py-2.5">
        <Terminal className="h-3.5 w-3.5 text-acid-400/80" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300">
          MCP tool activity
        </span>
        <span className="text-[9px] text-slate-600">{log.length}</span>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={() => {
              clearMcpLog();
              setLog([]);
            }}
            className="rounded p-1 text-slate-600 transition hover:text-slate-300"
            aria-label="Clear MCP log"
          >
            <Trash2 className="h-3 w-3" />
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-slate-600 transition hover:text-slate-200"
            aria-label="Close MCP log"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      </div>
      <div className="border-b border-white/5 px-3.5 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">
            <Wrench className="h-3 w-3 text-acid-500/70" />
            Tools for this chat
          </div>
          {servers.length ? (
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onChangeScope(undefined)}
                disabled={selectedIds === undefined}
                title="Use every enabled MCP server, including ones added later."
                className="rounded border border-white/10 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wider text-slate-500 transition hover:border-white/20 hover:text-slate-300 disabled:cursor-not-allowed disabled:opacity-30"
              >
                All
              </button>
              <button
                type="button"
                onClick={() => onChangeScope([])}
                disabled={selectedIds?.length === 0}
                title="Send no MCP tools for this chat."
                className="rounded border border-white/10 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wider text-slate-500 transition hover:border-white/20 hover:text-slate-300 disabled:cursor-not-allowed disabled:opacity-30"
              >
                None
              </button>
            </div>
          ) : null}
        </div>
        {servers.length && selectedIds?.length === 0 ? (
          <p className="mt-2 rounded-md border border-amber-400/20 bg-amber-400/5 px-2 py-1.5 text-[9.5px] leading-4 text-amber-200/80">
            No tools are enabled for this chat. The model will not receive any
            MCP tools until you select a server.
          </p>
        ) : null}
        {servers.length ? (
          <div className="mt-2 space-y-1">
            {servers.map((server) => {
              const checked =
                !selectedIds || selectedIds.includes(server.id);
              return (
                <label
                  key={server.id}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-[10px] text-slate-400 transition hover:bg-white/[0.03]"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => {
                      const active = new Set(
                        selectedIds ?? servers.map((item) => item.id),
                      );
                      if (checked) active.delete(server.id);
                      else active.add(server.id);
                      const next = servers
                        .map((item) => item.id)
                        .filter((id) => active.has(id));
                      onChangeScope(
                        next.length === servers.length ? undefined : next,
                      );
                    }}
                    className="h-3 w-3 accent-[rgb(var(--accent-400))]"
                  />
                  <span className="min-w-0 flex-1 truncate">
                    {server.name || server.command}
                  </span>
                  {server.autoApprove ? (
                    <span className="rounded border border-amber-400/20 bg-amber-400/5 px-1 py-px text-[8px] uppercase tracking-wider text-amber-300/80">
                      auto
                    </span>
                  ) : null}
                </label>
              );
            })}
          </div>
        ) : (
          <p className="mt-1.5 text-[10px] leading-4 text-slate-600">
            No enabled MCP servers. Configure them in the MCP Hub, then choose
            which ones this chat may use here.
          </p>
        )}
      </div>
      <div className="max-h-[300px] overflow-y-auto p-2">
        {log.length ? (
          log.map((entry) => (
            <div
              key={entry.id}
              className="rounded-lg px-2 py-1.5 hover:bg-white/[0.02]"
            >
              <div className="flex items-center gap-1.5 text-[9px] text-slate-600">
                <span className="font-mono">
                  {new Date(entry.at).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <span className="text-slate-500">{entry.server}</span>
                <span className="uppercase tracking-wider text-slate-700">
                  {entry.kind}
                </span>
              </div>
              <p
                className={cn(
                  "mt-0.5 break-words font-mono text-[9.5px] leading-4",
                  entry.kind === "error" ? "text-rose-300/80" : "text-slate-400",
                )}
              >
                {entry.message}
              </p>
            </div>
          ))
        ) : (
          <p className="px-2 py-4 text-center text-[10px] leading-5 text-slate-600">
            No MCP activity yet. Register and probe a server in the MCP Hub; its
            handshake and tool calls are logged here, inside the chat.
          </p>
        )}
      </div>
    </div>
  );
}
