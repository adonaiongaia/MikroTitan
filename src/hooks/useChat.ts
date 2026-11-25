import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import { api, Events } from "../lib/api";
import {
  allSessionMessages,
  appendTurn,
  evictedContextIds,
  forkAssistantTurn,
  forkUserTurn,
  inheritPinnedContext,
  messageContextId,
  normalizeSession as normalizeStoredSession,
  pinnedContextIds,
  pruneMcpContextMarkers,
  restoreEvictedContext as restoreEvictedContextState,
  switchMessageVersion,
  toggleEvictedContextItem,
  togglePinnedContextItem,
  updateStep,
} from "../lib/chatSessions";
import { estimateTokens, fitContext } from "../lib/contextBudget";
import { mcpScopeNames, reconcileMcpScope } from "../lib/mcpScope";
import { errorMessage, uid } from "../lib/format";
import { appendMcpLog } from "../lib/mcpLog";
import type {
  AgentApprovalEvent,
  AgentCompactEvent,
  AgentDoneEvent,
  AgentErrorEvent,
  AgentNoticeEvent,
  AgentPlanEvent,
  AgentPreferences,
  AgentStep,
  AgentStepStatus,
  AgentThoughtEvent,
  AgentToolResultEvent,
  AgentToolStartEvent,
  AgentTrace,
  AgentUndoFile,
  AgentUndoResult,
  ChatDelta,
  ChatDone,
  ChatDraftImage,
  ChatErrorEvent,
  ChatMessage,
  ChatNotice,
  ChatSession,
  ChatToolApproval,
  ChatUsage,
  Citation,
  McpActivityEvent,
  McpServerConfig,
  McpServerInput,
  PinnedContextItem,
  ProjectIngest,
  RagSourcesEvent,
} from "../types";
import { ragSettingsFor } from "../lib/rag";

/** Pre-disk-store history. Read once for migration and then removed. */
const LEGACY_STORAGE_KEY = "janas.sessions.v1";
const ACTIVE_KEY = "janas.activeSession.v1";

export interface SendOptions {
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
  thinking: boolean;
  structuredOutput: boolean;
  contextPrefetch: number;
  maxTokens: number | null;
  contextWindow: number;
  imageTokens?: number;
  images?: string[];
  extraContext?: string;
  /** Pinned project/memory context; never trimmed by the budget. */
  pinnedContext?: string;
  /** Human labels for the pinned items, used in an oversized-context error. */
  pinnedLabels?: string[];
  /** MCP server names whose tool schemas must survive a tight budget. */
  pinnedMcpServers?: string[];
  citations?: Citation[];
  mcpServers?: McpServerInput[];
  /** Called once the token budget accepts the turn and adds it to the session. */
  onAccepted?: () => void;
}

export interface AgentSendOptions extends SendOptions {
  agent: AgentPreferences;
  /** Whether the built-in `web_search` tool is exposed to the agent. */
  webSearch: boolean;
}

function createSession(): ChatSession {
  const now = Date.now();
  return {
    id: uid(),
    title: "New chat",
    created_at: now,
    updated_at: now,
    messages: [],
  };
}

type ActiveRequest = {
  requestId: string;
  sessionId: string;
  mode: "chat" | "agent";
};

async function cancelRequest(request: ActiveRequest) {
  if (request.mode === "agent") return api.abortAgent(request.requestId);
  return api.cancelChat(request.requestId);
}

/** Resolve stored image references for a turn the user is re-running. */
async function imageDataUrls(
  images: ChatMessage["images"] | undefined,
): Promise<string[]> {
  if (!images?.length) return [];
  const resolved = await Promise.all(
    images.map(async (image) => {
      if (image.data_url) return image.data_url;
      if (!image.asset) return null;
      try {
        return await api.chatAssetDataUrl(image.asset);
      } catch {
        return null;
      }
    }),
  );
  return resolved.filter((url): url is string => Boolean(url));
}

function undoSummary(result: AgentUndoResult): string {
  const parts: string[] = [];
  const files = result.files;
  const restored = files.length
    ? files.filter((file) => file.state === "restored").length
    : result.restored;
  const kept = files.filter((file) => file.state === "kept").length;
  const unresolved = files.filter(
    (file) =>
      file.state === "pending" ||
      file.state === "skipped" ||
      file.state === "failed",
  ).length;
  if (restored > 0) {
    parts.push(`${restored} restored`);
  } else if (result.restored > 0) {
    // The journal fallback only lists skipped paths, so its restored count
    // cannot be derived from `files`.
    parts.push(`${result.restored} restored`);
  }
  if (result.deleted > 0) {
    parts.push(
      `${result.deleted} new file${result.deleted === 1 ? "" : "s"} removed`,
    );
  }
  if (kept > 0) parts.push(`${kept} kept`);
  // A journal result lists only failed entries; its successful restores are
  // counted from result.restored.
  if (result.complete) {
    const skipped = files.filter(
      (file) => file.state === "skipped" || file.state === "failed",
    ).length;
    if (skipped > 0) parts.push(`${skipped} skipped`);
  } else if (unresolved > 0) {
    parts.push(`${unresolved} pending`);
  }
  if (result.incomplete) parts.push("some changes were not recorded");
  return parts.length ? parts.join(" · ") : "nothing needed reverting";
}

/** The old localStorage blob, read once so existing chats survive the move. */
function readLegacySessions(): ChatSession[] {
  try {
    const raw = window.localStorage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map(normalizeStoredSession)
      .filter((session): session is ChatSession => session !== null);
  } catch {
    return [];
  }
}

function loadActive(sessions: ChatSession[]): string {
  try {
    const stored = window.localStorage.getItem(ACTIVE_KEY);
    if (stored) {
      const id = JSON.parse(stored) as string;
      if (sessions.some((session) => session.id === id)) return id;
    }
  } catch {
    // fall through to the first session
  }
  return sessions[0]?.id ?? "";
}

export function useChat() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [hydrated, setHydrated] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamingSessionId, setStreamingSessionId] = useState<string | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  // Informational lines from the Rust turn (currently document retrieval).
  // Kept apart from `error` so a successful vault lookup never renders as a
  // failure banner.
  const [ragNotice, setRagNotice] = useState<string | null>(null);
  const [usage, setUsage] = useState<ChatUsage | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [firstTokenMs, setFirstTokenMs] = useState<number | null>(null);
  const [finishReason, setFinishReason] = useState<string | null>(null);
  const [liveBytes, setLiveBytes] = useState(0);
  const [startedAtMs, setStartedAtMs] = useState<number | null>(null);
  const [toolApproval, setToolApproval] = useState<ChatToolApproval | null>(
    null,
  );
  const activeRequest = useRef<ActiveRequest | null>(null);
  const streamStartedAt = useRef<number | null>(null);
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const savedRef = useRef<Map<string, ChatSession>>(new Map());
  const saveTimers = useRef<Map<string, number>>(new Map());
  const deletedRef = useRef<Set<string>>(new Set());

  // Hydrate from the Rust store. Existing localStorage history is migrated
  // once, then the old blob is removed so it can never go stale.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      let loaded: ChatSession[] = [];
      let loadError: string | null = null;
      try {
        const stored = await api.chatSessionsList();
        loaded = Array.isArray(stored)
          ? stored
              .map(normalizeStoredSession)
              .filter((session): session is ChatSession => session !== null)
          : [];
      } catch (err) {
        loadError = errorMessage(err);
      }

      // Migrate anything the disk store does not already have, even when the
      // shelf is not empty. A partial write must not strand the rest of the
      // old localStorage history.
      const legacy = readLegacySessions();
      if (legacy.length) {
        const known = new Set(loaded.map((session) => session.id));
        let complete = true;
        for (const session of legacy) {
          if (known.has(session.id)) continue;
          try {
            await api.chatSessionSave(session);
            loaded.push(session);
            known.add(session.id);
          } catch {
            complete = false;
          }
        }
        if (complete) {
          try {
            window.localStorage.removeItem(LEGACY_STORAGE_KEY);
          } catch {
            // ignore
          }
        }
      }

      if (!loaded.length) loaded = [createSession()];
      if (cancelled) return;
      savedRef.current = new Map(loaded.map((session) => [session.id, session]));
      setSessions(loaded);
      setActiveId(loadActive(loaded));
      setHydrated(true);
      if (loadError) {
        setError(`Chat history could not be loaded from disk: ${loadError}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const flushSession = useCallback(async (sessionId: string) => {
    const timer = saveTimers.current.get(sessionId);
    if (timer) {
      window.clearTimeout(timer);
      saveTimers.current.delete(sessionId);
    }
    const session = sessionsRef.current.find(
      (candidate) => candidate.id === sessionId,
    );
    if (!session || deletedRef.current.has(sessionId)) return;
    savedRef.current.set(session.id, session);
    try {
      await api.chatSessionSave(session);
    } catch (err) {
      savedRef.current.delete(session.id);
      setError(`Chat history could not be saved: ${errorMessage(err)}`);
    }
  }, []);

  // Only changed sessions are written. Streaming deltas replace one session
  // object, so the debounce coalesces a burst of tokens into one small file
  // write instead of serializing the whole shelf on every token.
  useEffect(() => {
    if (!hydrated) return;
    for (const session of sessions) {
      if (savedRef.current.get(session.id) === session) continue;
      savedRef.current.set(session.id, session);
      const existing = saveTimers.current.get(session.id);
      if (existing) window.clearTimeout(existing);
      const timer = window.setTimeout(() => {
        saveTimers.current.delete(session.id);
        if (deletedRef.current.has(session.id)) return;
        void api.chatSessionSave(session).catch((err) => {
          savedRef.current.delete(session.id);
          setError(`Chat history could not be saved: ${errorMessage(err)}`);
        });
      }, 500);
      saveTimers.current.set(session.id, timer);
    }
  }, [sessions, hydrated]);

  useEffect(() => {
    if (!hydrated || !activeId) return;
    try {
      window.localStorage.setItem(ACTIVE_KEY, JSON.stringify(activeId));
    } catch {
      // ignore
    }
  }, [activeId, hydrated]);

  // Settings can import or restore chats while the chat view is unmounted.
  // Refresh the rail without dropping a live stream or an unsaved draft.
  useEffect(() => {
    const reload = () => {
      if (activeRequest.current) return;
      void (async () => {
        try {
          const stored = await api.chatSessionsList();
          const normalized = stored
            .map(normalizeStoredSession)
            .filter((session): session is ChatSession => session !== null);
          const byId = new Map(normalized.map((session) => [session.id, session]));
          setSessions((previous) => {
            const next = previous.map((session) => byId.get(session.id) ?? session);
            for (const session of normalized) {
              if (!next.some((candidate) => candidate.id === session.id)) {
                next.unshift(session);
              }
            }
            return next;
          });
          for (const session of normalized) {
            const timer = saveTimers.current.get(session.id);
            if (timer) {
              window.clearTimeout(timer);
              saveTimers.current.delete(session.id);
            }
            savedRef.current.set(session.id, session);
          }
        } catch (err) {
          setError(`Chat history could not be refreshed: ${errorMessage(err)}`);
        }
      })();
    };
    window.addEventListener("janas:chat-history-changed", reload);
    return () =>
      window.removeEventListener("janas:chat-history-changed", reload);
  }, []);

  // Checkpoints live on disk and survive a restart, while the in-memory
  // journal does not. Ask Rust which persisted traces are still undoable once
  // the initially loaded sessions are available.
  const availabilityChecked = useRef(false);
  useEffect(() => {
    if (!hydrated || availabilityChecked.current) return;
    availabilityChecked.current = true;
    const runIds = Array.from(
      new Set(
        sessions.flatMap((session) =>
          session.messages
            .map((message) => message.agentTrace?.run_id)
            .filter((runId): runId is string => Boolean(runId)),
        ),
      ),
    );
    if (!runIds.length) return;
    // Traces that already carry per-file state do not need another read; the
    // saved state is what the tray rendered before the restart.
    const needsFiles = new Set(
      sessions.flatMap((session) =>
        session.messages.flatMap((message) => {
          const trace = message.agentTrace;
          return trace && trace.undo_files.length === 0 ? [trace.run_id] : [];
        }),
      ),
    );
    void api
      .agentUndoAvailability(runIds)
      .then(async (available) => {
        const canUndo = new Set(available);
        // A restored or skipped run keeps a checkpoint, so load its per-file
        // state too: a restart should not hide why a file is still pending.
        const filesByRun = new Map<string, AgentUndoFile[]>();
        await Promise.all(
          available.filter((runId) => needsFiles.has(runId)).map(async (runId) => {
            try {
              const state = await api.agentUndoState(runId);
              if (state.files.length) filesByRun.set(runId, state.files);
            } catch {
              // The undo button still works without the preview.
            }
          }),
        );
        setSessions((previous) =>
          previous.map((session) => ({
            ...session,
            messages: session.messages.map((message) => {
              const trace = message.agentTrace;
              if (!trace) return message;
              const next = canUndo.has(trace.run_id) && !trace.undone;
              const undoFiles = filesByRun.get(trace.run_id);
              if (trace.undo_available === next && !undoFiles) return message;
              return {
                ...message,
                agentTrace: {
                  ...trace,
                  undo_available: next,
                  undo_files: undoFiles ?? trace.undo_files,
                },
              };
            }),
          })),
        );
      })
      .catch(() => {
        // Older backends do not expose availability; the button stays hidden.
      });
  }, [sessions, hydrated]);

  useEffect(() => {
    let disposed = false;
    const unlisteners: UnlistenFn[] = [];

    const updateMessage = (
      requestId: string,
      sessionId: string,
      updater: (message: ChatMessage) => ChatMessage,
    ) => {
      setSessions((previous) =>
        previous.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                updated_at: Date.now(),
                messages: session.messages.map((message) =>
                  message.id === requestId ? updater(message) : message,
                ),
              }
            : session,
        ),
      );
    };

    const updateTrace = (
      requestId: string,
      sessionId: string,
      updater: (trace: AgentTrace) => AgentTrace,
    ) => {
      updateMessage(requestId, sessionId, (message) =>
        message.agentTrace
          ? { ...message, agentTrace: updater(message.agentTrace) }
          : message,
      );
    };

    const attach = async () => {
      const deltaUnlisten = await listen<ChatDelta>(
        Events.chatDelta,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.request_id !== request.requestId) return;
          const delta =
            typeof event.payload.delta === "string" ? event.payload.delta : "";
          const reasoning =
            typeof event.payload.reasoning === "string"
              ? event.payload.reasoning
              : "";
          if (!delta && !reasoning) return;
          // A context/tool notice is not a failed turn. Clear it as soon as
          // the model produces output so a successful reply does not leave a
          // red error banner behind.
          setError(null);
          setLiveBytes((bytes) => bytes + delta.length);
          updateMessage(request.requestId, request.sessionId, (message) => ({
            ...message,
            content: (message.content ?? "") + delta,
            reasoning: (message.reasoning ?? "") + reasoning,
          }));
        },
      );
      const doneUnlisten = await listen<ChatDone>(Events.chatDone, (event) => {
        const request = activeRequest.current;
        if (!request || event.payload.request_id !== request.requestId) return;
        setStreaming(false);
        setStreamingSessionId(null);
        setUsage(event.payload.usage);
        setElapsedMs(event.payload.elapsed_ms);
        setFirstTokenMs(event.payload.first_token_ms);
        setFinishReason(event.payload.finish_reason);
        setError(null);
        setToolApproval(null);
        activeRequest.current = null;
        flushSession(request.sessionId);
      });
      const errorUnlisten = await listen<ChatErrorEvent>(
        Events.chatError,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.request_id !== request.requestId) return;
          setError(event.payload.message);
          setStreaming(false);
          setStreamingSessionId(null);
          setToolApproval(null);
          activeRequest.current = null;
          flushSession(request.sessionId);
          updateMessage(request.requestId, request.sessionId, (message) =>
            message.content
              ? message
              : {
                  ...message,
                  content: `_${event.payload.message}_`,
                },
          );
        },
      );
      const noticeUnlisten = await listen<ChatNotice>(
        Events.chatNotice,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.request_id !== request.requestId) return;
          if (event.payload.level === "info") {
            setRagNotice(event.payload.message);
          } else {
            setError(event.payload.message);
          }
        },
      );
      // The Rust turn attaches the retrieved chunks to the message it is
      // streaming, so [D1] pills and their drawer survive a reload.
      const ragSourcesUnlisten = await listen<RagSourcesEvent>(
        Events.chatRagSources,
        (event) => {
          const request = activeRequest.current;
          if (!request) return;
          const chunks = Array.isArray(event.payload.chunks)
            ? event.payload.chunks
            : [];
          updateMessage(request.requestId, request.sessionId, (message) => ({
            ...message,
            rag_sources: chunks,
          }));
        },
      );
      const mcpActivityUnlisten = await listen<McpActivityEvent>(
        Events.mcpActivity,
        (event) => {
          appendMcpLog({
            server: event.payload.server || "MCP",
            kind: event.payload.level === "error" ? "error" : "tool",
            message: event.payload.message,
          });
        },
      );
      const toolApprovalUnlisten = await listen<ChatToolApproval>(
        Events.chatToolApproval,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.request_id !== request.requestId) return;
          setToolApproval(event.payload);
        },
      );

      const planUnlisten = await listen<AgentPlanEvent>(Events.agentPlan, (event) => {
        const request = activeRequest.current;
        if (!request || event.payload.run_id !== request.requestId) return;
        updateTrace(request.requestId, request.sessionId, (trace) => ({
          ...trace,
          plan: Array.isArray(event.payload.plan)
            ? event.payload.plan.slice(0, 20)
            : trace.plan,
        }));
      });

      const thoughtUnlisten = await listen<AgentThoughtEvent>(
        Events.agentThought,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          const delta = typeof event.payload.delta === "string" ? event.payload.delta : "";
          const reasoning =
            typeof event.payload.reasoning === "string" ? event.payload.reasoning : "";
          if (!delta && !reasoning) return;
          updateTrace(request.requestId, request.sessionId, (trace) => ({
            ...trace,
            live_text: (trace.live_text + delta).slice(-8_000),
            live_reasoning: (trace.live_reasoning + reasoning).slice(-4_000),
            notice: null,
          }));
        },
      );

      const toolStartUnlisten = await listen<AgentToolStartEvent>(
        Events.agentToolStart,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          updateTrace(request.requestId, request.sessionId, (trace) => {
            const status: AgentStepStatus = event.payload.requires_approval
              ? "awaiting_approval"
              : "running";
            const step: AgentStep = {
              id: event.payload.step_id,
              iteration: event.payload.iteration,
              thought: (trace.live_text || trace.live_reasoning).slice(-4_000),
              tool: event.payload.tool,
              arguments:
                typeof event.payload.arguments === "string"
                  ? event.payload.arguments
                  : "{}",
              risk: event.payload.risk,
              status,
              output: null,
              duration_ms: null,
              truncated: false,
              created_at: Date.now(),
            };
            return {
              ...trace,
              steps: [...trace.steps, step].slice(-60),
              live_text: "",
              live_reasoning: "",
              notice: null,
            };
          });
        },
      );

      const approvalUnlisten = await listen<AgentApprovalEvent>(
        Events.agentApprovalNeeded,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          updateTrace(request.requestId, request.sessionId, (trace) => ({
            ...trace,
            pending: {
              step_id: event.payload.step_id,
              tool: event.payload.tool,
              arguments: event.payload.arguments,
              risk: event.payload.risk,
              reason: event.payload.reason,
              file_edit: Boolean(event.payload.file_edit),
              can_always_allow: Boolean(event.payload.can_always_allow),
              preview: event.payload.preview ?? null,
              workspace_root:
                event.payload.workspace_root ?? trace.workspace_root,
            },
            steps: updateStep(trace.steps, event.payload.step_id, {
              status: "awaiting_approval",
            }),
            notice: null,
          }));
        },
      );

      const toolResultUnlisten = await listen<AgentToolResultEvent>(
        Events.agentToolResult,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          updateTrace(request.requestId, request.sessionId, (trace) => {
            const status: AgentStepStatus = event.payload.ok
              ? "succeeded"
              : event.payload.rejected
                ? "rejected"
                : event.payload.blocked
                  ? "blocked"
                  : "failed";
            return {
              ...trace,
              pending:
                trace.pending?.step_id === event.payload.step_id
                  ? null
                  : trace.pending,
              steps: updateStep(trace.steps, event.payload.step_id, {
                status,
                output: event.payload.output,
                duration_ms: event.payload.duration_ms,
                truncated: Boolean(event.payload.truncated),
              }),
            };
          });
        },
      );

      const compactUnlisten = await listen<AgentCompactEvent>(
        Events.agentContextCompacted,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          updateTrace(request.requestId, request.sessionId, (trace) => ({
            ...trace,
            compactions: [
              ...trace.compactions,
              {
                id: `${event.payload.run_id}-${Date.now()}`,
                dropped_messages: event.payload.dropped_messages,
                estimated_tokens: event.payload.estimated_tokens,
                context_window: event.payload.context_window,
                summary: event.payload.summary,
                created_at: Date.now(),
              },
            ].slice(-12),
          }));
        },
      );

      const agentNoticeUnlisten = await listen<AgentNoticeEvent>(
        Events.agentNotice,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          updateTrace(request.requestId, request.sessionId, (trace) => ({
            ...trace,
            notice: event.payload.message,
          }));
        },
      );

      const agentErrorUnlisten = await listen<AgentErrorEvent>(
        Events.agentError,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          setError(event.payload.message);
          // An error is terminal for the generation even if checkpoint cleanup
          // delays `agent:done`; do not leave the composer spinning meanwhile.
          setStreaming(false);
          setStreamingSessionId(null);
          setToolApproval(null);
          updateTrace(request.requestId, request.sessionId, (trace) => ({
            ...trace,
            error: event.payload.message,
          }));
        },
      );

      const agentDoneUnlisten = await listen<AgentDoneEvent>(
        Events.agentDone,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.run_id !== request.requestId) return;
          setStreaming(false);
          setStreamingSessionId(null);
          setFinishReason(event.payload.status);
          if (event.payload.status !== "error") setError(null);
          updateMessage(request.requestId, request.sessionId, (message) => ({
            ...message,
            content: event.payload.final_content || message.content,
            // Chunks the agent looked up with search_knowledge_base. Attaching
            // them here makes its [D#] tags render as pills after a reload too.
            rag_sources:
              Array.isArray(event.payload.rag_sources) &&
              event.payload.rag_sources.length
                ? event.payload.rag_sources
                : message.rag_sources,
            agentTrace: message.agentTrace
              ? {
                  ...message.agentTrace,
                  status: event.payload.status,
                  iterations: event.payload.iterations,
                  finished_at: Date.now(),
                  pending: null,
                  live_text: "",
                  live_reasoning: "",
                  notice: null,
                  undo_available: event.payload.changed_files > 0,
                  undo_files: [],
                  workspace_root:
                    event.payload.workspace_root ??
                    message.agentTrace.workspace_root,
                  changes:
                    Array.isArray(event.payload.changes) &&
                    event.payload.changes.length
                      ? event.payload.changes
                      : message.agentTrace.changes,
                }
              : null,
          }));
          // The checkpoint is finished before this event, so the file list is
          // available right away. A failure only hides the preview; the undo
          // button still works.
          if (event.payload.changed_files > 0) {
            void api
              .agentUndoState(event.payload.run_id)
              .then((state) => applyUndoState(event.payload.run_id, state))
              .catch(() => undefined);
          }
          setToolApproval(null);
          activeRequest.current = null;
          flushSession(request.sessionId);
        },
      );

      if (disposed) {
        deltaUnlisten();
        doneUnlisten();
        errorUnlisten();
        noticeUnlisten();
        ragSourcesUnlisten();
        mcpActivityUnlisten();
        toolApprovalUnlisten();
        planUnlisten();
        thoughtUnlisten();
        toolStartUnlisten();
        approvalUnlisten();
        toolResultUnlisten();
        compactUnlisten();
        agentNoticeUnlisten();
        agentErrorUnlisten();
        agentDoneUnlisten();
      } else {
        unlisteners.push(
          deltaUnlisten,
          doneUnlisten,
          errorUnlisten,
          noticeUnlisten,
          ragSourcesUnlisten,
          mcpActivityUnlisten,
          toolApprovalUnlisten,
          planUnlisten,
          thoughtUnlisten,
          toolStartUnlisten,
          approvalUnlisten,
          toolResultUnlisten,
          compactUnlisten,
          agentNoticeUnlisten,
          agentErrorUnlisten,
          agentDoneUnlisten,
        );
      }
    };

    void attach().catch((err) => {
      setError(`Chat events could not be connected: ${errorMessage(err)}`);
    });
    return () => {
      disposed = true;
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, []);

  const runChatTurn = useCallback(
    async (input: {
      sessionId: string;
      nextSession: ChatSession;
      requestMessages: ChatMessage[];
      assistantId: string;
      options: SendOptions;
      images?: string[];
    }) => {
      if (activeRequest.current) return false;
      const { sessionId, nextSession, requestMessages, assistantId, options } =
        input;
      const turnWindow = Math.max(1, options.contextPrefetch) * 2 + 1;
      const pinnedIds = pinnedContextIds(nextSession);
      const evictedIds = evictedContextIds(nextSession);
      // Evicted turns stay in the transcript but never reach the prompt, so
      // filter before the prefetch window picks the turns to send.
      const windowed = requestMessages
        .filter((message) => !evictedIds.has(messageContextId(message.id)))
        .slice(-turnWindow)
        .filter(
          (message) =>
            message.role === "user" ||
            message.content.trim().length > 0 ||
            message.reasoning.trim().length > 0,
        );
      const current = requestMessages[requestMessages.length - 1];
      const imageTokens =
        options.imageTokens ??
        (current?.images ?? []).reduce(
          (total, image) => total + (image.tokens || 0),
          0,
        );

      const fitted = fitContext({
        contextWindow: options.contextWindow,
        maxTokens: options.maxTokens,
        systemPrompt: options.systemPrompt,
        extraContext: options.extraContext,
        pinnedContext: options.pinnedContext,
        pinnedLabels: options.pinnedLabels,
        messages: windowed.map((message) => ({
          role: message.role,
          content: message.content,
          id: message.id,
          pinned: pinnedIds.has(messageContextId(message.id)),
        })),
        imageTokens,
      });
      if (fitted.error) {
        setError(fitted.error);
        return false;
      }

      const now = Date.now();
      setSessions((previous) =>
        previous.map((candidate) =>
          candidate.id === sessionId ? nextSession : candidate,
        ),
      );
      options.onAccepted?.();
      setStreaming(true);
      setStreamingSessionId(sessionId);
      setError(null);
      setRagNotice(null);
      setUsage(null);
      setElapsedMs(null);
      setFirstTokenMs(null);
      setFinishReason(null);
      setLiveBytes(0);
      setStartedAtMs(now);
      activeRequest.current = {
        requestId: assistantId,
        sessionId,
        mode: "chat",
      };
      streamStartedAt.current = Date.now();

      const rag = ragSettingsFor(
        nextSession.rag_top_k,
        nextSession.rag_max_tokens,
      );
      try {
        await api.streamChat({
          request_id: assistantId,
          messages: fitted.messages,
          system_prompt: fitted.systemPrompt || null,
          temperature: options.temperature,
          top_p: options.topP,
          top_k: Math.round(Math.min(100, Math.max(1, options.topK))),
          min_p: options.minP,
          repeat_penalty: options.repeatPenalty,
          structured: options.structuredOutput,
          thinking: options.thinking,
          images: input.images ?? options.images ?? [],
          max_tokens: options.maxTokens,
          // The budget already sized this block; Rust appends it after the
          // instructions so it can protect it during a context retry.
          pinned_context: fitted.pinnedContext || null,
          mcp_servers: options.mcpServers ?? [],
          rag_collections: Array.isArray(nextSession.attached_collection_ids)
            ? nextSession.attached_collection_ids
            : [],
          rag_top_k: rag.topK,
          rag_max_tokens: rag.maxTokens,
          // Chunks the user evicted are still ranked by retrieval, then
          // removed here so the turn is not sent with them.
          rag_excluded_chunks: [...evictedIds]
            .filter((id) => id.startsWith("rag:"))
            .map((id) => id.slice("rag:".length)),
          pinned_mcp_servers: options.pinnedMcpServers ?? [],
        });
      } catch (err) {
        setError(errorMessage(err));
        setStreaming(false);
        setStreamingSessionId(null);
        setToolApproval(null);
        activeRequest.current = null;
        setSessions((previous) =>
          previous.map((candidate) =>
            candidate.id === sessionId
              ? {
                  ...candidate,
                  messages: candidate.messages.map((message) =>
                    message.id === assistantId && !message.content
                      ? {
                          ...message,
                          content: "_The engine did not return a response._",
                        }
                      : message,
                  ),
                }
              : candidate,
          ),
        );
      }
      return true;
    },
    [],
  );

  const send = useCallback(
    async (text: string, options: SendOptions) => {
      const trimmed = text.trim();
      if (!trimmed || activeRequest.current) return false;
      if (!sessionsRef.current.length) {
        setError("Chat history is still loading. Try again in a moment.");
        return false;
      }

      const requestedSessionId = activeIdRef.current;
      // Fall back to the first chat if the stored active id is stale, so a
      // message is never dropped without feedback.
      const session =
        sessionsRef.current.find(
          (candidate) => candidate.id === requestedSessionId,
        ) ?? sessionsRef.current[0];
      if (!session) return false;
      const sessionId = session.id;
      if (sessionId !== requestedSessionId) setActiveId(sessionId);

      const now = Date.now();
      const userMessage: ChatMessage = {
        id: uid(),
        role: "user",
        content: trimmed,
        reasoning: "",
        created_at: now,
        images: options.images?.length
          ? options.images.map((dataUrl, index) => ({
              name: `image-${index + 1}`,
              data_url: dataUrl,
              tokens: 0,
            }))
          : undefined,
      };
      const assistantMessage: ChatMessage = {
        id: uid(),
        role: "assistant",
        content: "",
        reasoning: "",
        created_at: now,
        citations: options.citations,
      };

      const nextSession = appendTurn(session, userMessage, assistantMessage);
      const titled =
        session.messages.length === 0
          ? { ...nextSession, title: trimmed.slice(0, 56) }
          : nextSession;
      return runChatTurn({
        sessionId,
        nextSession: titled,
        requestMessages: titled.messages.slice(0, -1),
        assistantId: assistantMessage.id,
        options,
        images: options.images,
      });
    },
    [runChatTurn],
  );

  const regenerate = useCallback(
    async (messageId: string, options: SendOptions) => {
      if (activeRequest.current) return false;
      const requestedSessionId = activeIdRef.current;
      const session =
        sessionsRef.current.find(
          (candidate) => candidate.id === requestedSessionId,
        ) ?? sessionsRef.current[0];
      if (!session) return false;
      const index = session.messages.findIndex(
        (message) => message.id === messageId,
      );
      if (index <= 0) return false;
      const target = session.messages[index];
      const user = session.messages[index - 1];
      if (
        !target ||
        target.role !== "assistant" ||
        !user ||
        user.role !== "user"
      ) {
        return false;
      }

      const assistant: ChatMessage = {
        id: uid(),
        role: "assistant",
        content: "",
        reasoning: "",
        created_at: Date.now(),
        citations: target.citations,
      };
      const nextSession = forkAssistantTurn(session, index, assistant);
      return runChatTurn({
        sessionId: session.id,
        nextSession,
        requestMessages: nextSession.messages.slice(0, -1),
        assistantId: assistant.id,
        options,
        images: await imageDataUrls(user.images),
      });
    },
    [runChatTurn],
  );

  const editAndResend = useCallback(
    async (messageId: string, text: string, options: SendOptions) => {
      const trimmed = text.trim();
      if (!trimmed || activeRequest.current) return false;
      const requestedSessionId = activeIdRef.current;
      const session =
        sessionsRef.current.find(
          (candidate) => candidate.id === requestedSessionId,
        ) ?? sessionsRef.current[0];
      if (!session) return false;
      const index = session.messages.findIndex(
        (message) => message.id === messageId,
      );
      if (index < 0) return false;
      const message = session.messages[index];
      if (!message || message.role !== "user") return false;

      const edited: ChatMessage = {
        ...message,
        id: uid(),
        content: trimmed,
        reasoning: "",
      };
      const assistant: ChatMessage = {
        id: uid(),
        role: "assistant",
        content: "",
        reasoning: "",
        created_at: Date.now(),
      };
      const nextSession = forkUserTurn(session, index, edited, assistant);
      const titled =
        index === 0
          ? { ...nextSession, title: trimmed.slice(0, 56) }
          : nextSession;
      return runChatTurn({
        sessionId: session.id,
        nextSession: titled,
        requestMessages: titled.messages.slice(0, -1),
        assistantId: assistant.id,
        options,
        images: await imageDataUrls(edited.images),
      });
    },
    [runChatTurn],
  );

  const sendAgent = useCallback(
    async (text: string, options: AgentSendOptions) => {
      const trimmed = text.trim();
      if (!trimmed || activeRequest.current) return false;
      if (!sessionsRef.current.length) {
        setError("Chat history is still loading. Try again in a moment.");
        return false;
      }

      const requestedSessionId = activeIdRef.current;
      const session =
        sessionsRef.current.find(
          (candidate) => candidate.id === requestedSessionId,
        ) ?? sessionsRef.current[0];
      if (!session) return false;
      const sessionId = session.id;
      if (sessionId !== requestedSessionId) setActiveId(sessionId);

      const requestId = uid();
      const now = Date.now();
      const userMessage: ChatMessage = {
        id: uid(),
        role: "user",
        content: trimmed,
        reasoning: "",
        created_at: now,
      };
      const trace: AgentTrace = {
        run_id: requestId,
        status: "running",
        iterations: 0,
        max_iterations: Math.min(25, Math.max(1, options.agent.maxIterations)),
        plan: [],
        steps: [],
        compactions: [],
        pending: null,
        live_text: "",
        live_reasoning: "",
        notice: null,
        error: null,
        undo_available: false,
        undone: false,
        undo_error: null,
        undo_summary: null,
        undo_files: [],
        changes: [],
        workspace_root: options.agent.workspaceRoot.trim(),
        started_at: now,
        finished_at: null,
      };
      const assistantMessage: ChatMessage = {
        id: requestId,
        role: "assistant",
        content: "",
        reasoning: "",
        created_at: now,
        agentTrace: trace,
      };

      const nextSession = appendTurn(session, userMessage, assistantMessage);
      const titled =
        session.messages.length === 0
          ? { ...nextSession, title: trimmed.slice(0, 56) }
          : nextSession;

      // The Rust loop appends the goal itself; only prior turns are history.
      const pinnedIds = pinnedContextIds(titled);
      const evictedIds = evictedContextIds(titled);
      const history = session.messages
        .filter((message) => !evictedIds.has(messageContextId(message.id)))
        .slice(-Math.max(1, options.contextPrefetch) * 2)
        .filter(
          (message) =>
            (message.role === "user" || message.role === "assistant") &&
            message.content.trim().length > 0,
        )
        .map((message) => ({
          role: message.role,
          content: message.content,
          pinned: pinnedIds.has(messageContextId(message.id)),
        }));

      setSessions((previous) =>
        previous.map((candidate) =>
          candidate.id === sessionId ? titled : candidate,
        ),
      );
      options.onAccepted?.();
      setStreaming(true);
      setStreamingSessionId(sessionId);
      setError(null);
      setRagNotice(null);
      setUsage(null);
      setElapsedMs(null);
      setFirstTokenMs(null);
      setFinishReason(null);
      setLiveBytes(0);
      setStartedAtMs(now);
      activeRequest.current = { requestId, sessionId, mode: "agent" };
      streamStartedAt.current = Date.now();

      const rag = ragSettingsFor(
        titled.rag_top_k,
        titled.rag_max_tokens,
      );
      try {
        await api.startAgent({
          run_id: requestId,
          conversation_id: sessionId,
          goal: trimmed,
          history,
          system_prompt: options.systemPrompt || null,
          temperature: options.temperature,
          top_p: options.topP,
          top_k: Math.round(Math.min(100, Math.max(1, options.topK))),
          min_p: options.minP,
          repeat_penalty: options.repeatPenalty,
          thinking: options.thinking,
          max_tokens: options.maxTokens,
          max_iterations: trace.max_iterations,
          approval_policy: options.agent.approvalPolicy,
          workspace_root: options.agent.workspaceRoot.trim(),
          planning_step: options.agent.planningStep,
          auto_compact: options.agent.autoCompact,
          compaction_threshold: Math.min(
            0.9,
            Math.max(0.5, options.agent.compactionThreshold),
          ),
          web_search: options.webSearch,
          verify_command: (options.agent.verifyCommand ?? "").trim(),
          mcp_servers: options.mcpServers ?? [],
          rag_collections: Array.isArray(titled.attached_collection_ids)
            ? titled.attached_collection_ids
            : [],
          rag_top_k: rag.topK,
          rag_max_tokens: rag.maxTokens,
        });
      } catch (err) {
        const message = errorMessage(err);
        setError(message);
        setStreaming(false);
        setStreamingSessionId(null);
        setToolApproval(null);
        activeRequest.current = null;
        flushSession(sessionId);
        setSessions((previous) =>
          previous.map((candidate) =>
            candidate.id === sessionId
              ? {
                  ...candidate,
                  messages: candidate.messages.map((item) =>
                    item.id === requestId && item.agentTrace
                      ? {
                          ...item,
                          agentTrace: {
                            ...item.agentTrace,
                            status: "error",
                            error: message,
                            finished_at: Date.now(),
                          },
                        }
                      : item,
                  ),
                }
              : candidate,
          ),
        );
      }
      return true;
    },
    [flushSession],
  );

  const stop = useCallback(async () => {
    const request = activeRequest.current;
    if (!request) return;
    setToolApproval(null);
    if (request.mode === "agent") {
      // Abort reaches the run asynchronously, so stop the UI now. Clearing the
      // active request also makes a late `agent:done` a no-op, which is the
      // point: a stalled run must not keep the composer spinning.
      setStreaming(false);
      setStreamingSessionId(null);
      setSessions((previous) =>
        previous.map((session) =>
          session.id === request.sessionId
            ? {
                ...session,
                messages: session.messages.map((message) =>
                  message.id === request.requestId && message.agentTrace
                    ? {
                        ...message,
                        agentTrace: {
                          ...message.agentTrace,
                          status: "aborted",
                          pending: null,
                          live_text: "",
                          live_reasoning: "",
                        },
                      }
                    : message,
                ),
              }
            : session,
        ),
      );
      activeRequest.current = null;
    }
    try {
      await cancelRequest(request);
    } catch (err) {
      if (request.mode === "chat") {
        setError(errorMessage(err));
        setStreaming(false);
        setStreamingSessionId(null);
        activeRequest.current = null;
      }
    }
    flushSession(request.sessionId);
  }, [flushSession]);

  const clear = useCallback(async () => {
    if (activeRequest.current) {
      try {
        await cancelRequest(activeRequest.current);
      } catch {
        // The request may have completed between the click and the cancel.
      }
    }
    activeRequest.current = null;
    setSessions((previous) =>
      previous.map((session) =>
        session.id === activeIdRef.current
          ? {
              ...session,
              messages: [],
              title: "New chat",
              // Pins and evictions only make sense for turns that exist.
              pinned_context: undefined,
              evicted_context_ids: undefined,
            }
          : session,
      ),
    );
    setStreaming(false);
    setStreamingSessionId(null);
    setUsage(null);
    setElapsedMs(null);
    setFirstTokenMs(null);
    setFinishReason(null);
    setError(null);
    setLiveBytes(0);
    setToolApproval(null);
  }, []);

  const newChat = useCallback(() => {
    const session = createSession();
    setSessions((previous) => [session, ...previous]);
    setActiveId(session.id);
    setUsage(null);
    setElapsedMs(null);
    setFirstTokenMs(null);
    setFinishReason(null);
    setError(null);
    return session.id;
  }, []);

  const deleteChat = useCallback(async (id: string) => {
    if (activeRequest.current?.sessionId === id) {
      try {
        await cancelRequest(activeRequest.current);
      } catch {
        // ignore
      }
      activeRequest.current = null;
      setStreaming(false);
      setStreamingSessionId(null);
    }
    const timer = saveTimers.current.get(id);
    if (timer) {
      window.clearTimeout(timer);
      saveTimers.current.delete(id);
    }
    savedRef.current.delete(id);
    deletedRef.current.add(id);
    void api.chatSessionDelete(id).catch((err) => {
      setError(`That chat could not be deleted: ${errorMessage(err)}`);
    });
    setSessions((previous) => {
      const remaining = previous.filter((session) => session.id !== id);
      const next = remaining.length ? remaining : [createSession()];
      setActiveId((current) =>
        current === id ? next[0].id : current,
      );
      return next;
    });
  }, []);

  const deleteMessage = useCallback((messageId: string) => {
    const id = messageContextId(messageId);
    setSessions((previous) =>
      previous.map((session) => {
        if (session.id !== activeIdRef.current) return session;
        const pinned = (session.pinned_context ?? []).filter(
          (item) => item.id !== id,
        );
        const evicted = (session.evicted_context_ids ?? []).filter(
          (candidate) => candidate !== id,
        );
        return {
          ...session,
          messages: session.messages.filter(
            (message) => message.id !== messageId,
          ),
          pinned_context: pinned.length ? pinned : undefined,
          evicted_context_ids: evicted.length ? evicted : undefined,
          updated_at: Date.now(),
        };
      }),
    );
  }, []);

  const renameChat = useCallback((id: string, title: string) => {
    const trimmed = title.trim();
    if (!trimmed) return;
    setSessions((previous) =>
      previous.map((session) =>
        session.id === id
          ? { ...session, title: trimmed, updated_at: Date.now() }
          : session,
      ),
    );
  }, []);

  const togglePin = useCallback((id: string) => {
    setSessions((previous) =>
      previous.map((session) =>
        session.id === id
          ? { ...session, pinned: !session.pinned }
          : session,
      ),
    );
  }, []);

  const importChat = useCallback(async (path: string) => {
    const imported = normalizeStoredSession(await api.chatImportSession(path));
    if (!imported) {
      throw new Error("The imported chat could not be read.");
    }
    savedRef.current.set(imported.id, imported);
    deletedRef.current.delete(imported.id);
    setSessions((previous) => [imported, ...previous]);
    setActiveId(imported.id);
    return imported.id;
  }, []);

  const exportChat = useCallback(async (id: string, path: string) => {
    const session = sessionsRef.current.find(
      (candidate) => candidate.id === id,
    );
    // Export reads the file on disk, so write the latest in-memory turn first.
    if (session && !deletedRef.current.has(id)) {
      await api.chatSessionSave(session);
      savedRef.current.set(id, session);
    }
    await api.chatExportSession(id, path);
  }, []);

  const updateActiveSession = useCallback(
    (updater: (session: ChatSession) => ChatSession) => {
      const id = activeIdRef.current;
      if (!id) return;
      setSessions((previous) =>
        previous.map((session) =>
          session.id === id ? updater(session) : session,
        ),
      );
    },
    [],
  );

  const setDraft = useCallback(
    (text: string) => {
      updateActiveSession((session) => ({
        ...session,
        draft: {
          text,
          images: session.draft?.images ?? [],
          updated_at: Date.now(),
        },
      }));
    },
    [updateActiveSession],
  );

  const setDraftImages = useCallback(
    (images: ChatDraftImage[]) => {
      updateActiveSession((session) => ({
        ...session,
        draft: {
          text: session.draft?.text ?? "",
          images,
          updated_at: Date.now(),
        },
      }));
    },
    [updateActiveSession],
  );

  const clearDraft = useCallback(() => {
    updateActiveSession((session) => ({
      ...session,
      draft: { text: "", images: [], updated_at: Date.now() },
    }));
  }, [updateActiveSession]);

  const setSessionProject = useCallback(
    (project: ProjectIngest | null) => {
      updateActiveSession((session) => ({
        ...session,
        project: project ?? undefined,
      }));
    },
    [updateActiveSession],
  );

  const setSessionWorkspace = useCallback(
    (workspaceRoot: string) => {
      updateActiveSession((session) => ({
        ...session,
        workspace_root: workspaceRoot.trim() || undefined,
      }));
    },
    [updateActiveSession],
  );

  /** `null` clears the override so the chat falls back to the default prompt. */
  const setSessionSystemPrompt = useCallback(
    (prompt: string | null) => {
      updateActiveSession((session) => {
        const next = { ...session };
        if (prompt === null) delete next.system_prompt;
        else next.system_prompt = prompt;
        return next;
      });
    },
    [updateActiveSession],
  );

  const setSessionMcpServerIds = useCallback(
    (ids: string[] | undefined, servers: McpServerConfig[] = []) => {
      updateActiveSession((session) => ({
        ...session,
        mcp_server_ids: ids,
        mcp_server_names: mcpScopeNames(ids, servers),
      }));
    },
    [updateActiveSession],
  );

  /** Replace this chat's attached collections. `[]` means "query nothing". */
  const setSessionCollectionIds = useCallback(
    (ids: string[]) => {
      updateActiveSession((session) => ({
        ...session,
        attached_collection_ids: ids,
      }));
    },
    [updateActiveSession],
  );

  const setSessionRagTuning = useCallback(
    (tuning: { topK?: number; maxTokens?: number }) => {
      updateActiveSession((session) => ({
        ...session,
        rag_top_k: tuning.topK ?? session.rag_top_k,
        rag_max_tokens: tuning.maxTokens ?? session.rag_max_tokens,
      }));
    },
    [updateActiveSession],
  );

  /** Pin or unpin one file, chunk, memory or turn in the active chat. */
  const toggleContextPin = useCallback(
    (item: PinnedContextItem) => {
      updateActiveSession((session) => togglePinnedContextItem(session, item));
    },
    [updateActiveSession],
  );

  /** Drop an item from the prompt immediately, or restore it when evicted. */
  const toggleContextEvict = useCallback(
    (id: string) => {
      updateActiveSession((session) => toggleEvictedContextItem(session, id));
    },
    [updateActiveSession],
  );

  const restoreEvictedContext = useCallback(() => {
    updateActiveSession((session) => restoreEvictedContextState(session));
  }, [updateActiveSession]);

  const toggleMessagePin = useCallback(
    (messageId: string) => {
      updateActiveSession((session) => {
        const message = allSessionMessages(session).find(
          (candidate) => candidate.id === messageId,
        );
        if (!message) return session;
        const text = message.content.trim();
        return togglePinnedContextItem(session, {
          id: messageContextId(messageId),
          kind: "message",
          label: text
            ? text.slice(0, 72)
            : message.role === "user"
              ? "User turn"
              : "Assistant reply",
          tokens: estimateTokens(message.content) + 8,
          added_at: Date.now(),
        });
      });
    },
    [updateActiveSession],
  );

  const toggleMessageEvict = useCallback(
    (messageId: string) => {
      updateActiveSession((session) =>
        toggleEvictedContextItem(session, messageContextId(messageId)),
      );
    },
    [updateActiveSession],
  );

  const reconcileMcpScopes = useCallback((servers: McpServerConfig[]) => {
    setSessions((previous) => {
      let changed = false;
      const next = previous.map((session) => {
        const reconciled = reconcileMcpScope(
          session.mcp_server_ids,
          session.mcp_server_names,
          servers,
        );
        const scoped = reconciled.changed
          ? {
              ...session,
              mcp_server_ids: reconciled.ids,
              mcp_server_names: reconciled.names,
            }
          : session;
        // Deleting a server must also retire its pins and eviction markers,
        // not just its scope entry.
        const pruned = pruneMcpContextMarkers(scoped, servers);
        if (pruned === session) return session;
        changed = true;
        return pruned;
      });
      return changed ? next : previous;
    });
  }, []);

  const switchVersion = useCallback((messageId: string, siblingId: string) => {
    setSessions((previous) =>
      previous.map((session) =>
        session.id === activeIdRef.current
          ? switchMessageVersion(session, messageId, siblingId)
          : session,
      ),
    );
  }, []);

  const branchFromMessage = useCallback(async (messageId: string) => {
    const sessionId = activeIdRef.current;
    if (!sessionId) return null;
    const stored = normalizeStoredSession(
      await api.chatBranchSession(sessionId, messageId),
    );
    if (!stored) throw new Error("That branch could not be read.");
    const source = sessionsRef.current.find(
      (candidate) => candidate.id === sessionId,
    );
    // The backend copies pins when it writes the branch file; applying the
    // same inheritance here also covers branches produced by older builds.
    const branch = source ? inheritPinnedContext(source, stored) : stored;
    savedRef.current.set(branch.id, branch);
    deletedRef.current.delete(branch.id);
    setSessions((previous) => [branch, ...previous]);
    setActiveId(branch.id);
    return branch.id;
  }, []);

  /** Force every dirty session to disk, for window close and app shutdown. */
  const flushAll = useCallback(async () => {
    const pending = sessionsRef.current.filter(
      (session) =>
        saveTimers.current.has(session.id) ||
        savedRef.current.get(session.id) !== session,
    );
    await Promise.all(pending.map((session) => flushSession(session.id)));
  }, [flushSession]);

  const resolveToolApproval = useCallback(
    async (decision: "approve" | "reject") => {
      const pending = toolApproval;
      if (!pending) return;
      setToolApproval(null);
      try {
        const accepted = await api.resolveChatToolApproval(
          pending.request_id,
          pending.call_id,
          decision,
        );
        if (!accepted) {
          setError("That tool approval was no longer waiting.");
        }
      } catch (err) {
        setError(errorMessage(err));
      }
    },
    [toolApproval],
  );

  const updateTraceByRun = useCallback(
    (runId: string, updater: (trace: AgentTrace) => AgentTrace) => {
      setSessions((previous) =>
        previous.map((session) => ({
          ...session,
          messages: session.messages.map((message) =>
            message.agentTrace?.run_id === runId
              ? { ...message, agentTrace: updater(message.agentTrace) }
              : message,
          ),
        })),
      );
    },
    [],
  );

  const applyUndoState = useCallback(
    (runId: string, result: AgentUndoResult) => {
      if (!result.files.length) return;
      updateTraceByRun(runId, (trace) =>
        // A restore or keep may finish while this read-only query is in
        // flight; never replace newer per-file state with the earlier snapshot.
        trace.undo_files.length ? trace : { ...trace, undo_files: result.files },
      );
    },
    [updateTraceByRun],
  );

  const applyUndoResult = useCallback(
    (runId: string, result: AgentUndoResult) => {
      updateTraceByRun(runId, (trace) => ({
        ...trace,
        // The run is complete exactly when no file still needs a restore or a
        // keep decision, which is also when the backend clears the checkpoint.
        undone: result.complete,
        undo_available: !result.complete,
        undo_error: null,
        undo_summary: undoSummary(result),
        undo_files: result.files.length ? result.files : trace.undo_files,
        changes: trace.changes.map((change) => {
          const file = result.files.find((entry) => entry.path === change.path);
          if (file) {
            return {
              ...change,
              restored: file.state === "restored" || file.state === "kept",
            };
          }
          return result.complete ? { ...change, restored: true } : change;
        }),
      }));
    },
    [updateTraceByRun],
  );

  const runUndo = useCallback(
    async (
      runId: string,
      action: () => Promise<AgentUndoResult>,
    ): Promise<AgentUndoResult> => {
      updateTraceByRun(runId, (trace) => ({
        ...trace,
        undo_error: null,
      }));
      try {
        const result = await action();
        applyUndoResult(runId, result);
        return result;
      } catch (err) {
        const message = errorMessage(err);
        // Keep undo available after a refusal (for example while another run
        // holds the workspace) so the user can stop that run and retry.
        updateTraceByRun(runId, (trace) => ({
          ...trace,
          undo_error: message,
        }));
        throw err;
      }
    },
    [applyUndoResult, updateTraceByRun],
  );

  const undoAgent = useCallback(
    (runId: string) => runUndo(runId, () => api.undoAgent(runId)),
    [runUndo],
  );

  const undoAgentFile = useCallback(
    (runId: string, path: string) =>
      runUndo(runId, () => api.undoAgentFile(runId, path)),
    [runUndo],
  );

  const keepAgentFile = useCallback(
    (runId: string, path: string) =>
      runUndo(runId, () => api.keepAgentFile(runId, path)),
    [runUndo],
  );

  const activeSession =
    sessions.find((session) => session.id === activeId) ?? sessions[0];
  const messages = activeSession?.messages ?? [];
  const activeStreaming = streaming && streamingSessionId === activeId;

  return {
    sessions,
    activeId,
    activeSession: activeSession ?? null,
    archivedMessages: activeSession?.archived_messages ?? [],
    draft: activeSession?.draft ?? { text: "", images: [], updated_at: 0 },
    project: activeSession?.project ?? null,
    workspaceRoot: activeSession?.workspace_root ?? "",
    mcpServerIds: activeSession?.mcp_server_ids,
    collectionIds: activeSession?.attached_collection_ids,
    ragTopK: activeSession?.rag_top_k,
    ragMaxTokens: activeSession?.rag_max_tokens,
    ragNotice,
    messages,
    loading: !hydrated,
    streaming: activeStreaming,
    busy: streaming,
    startedAtMs,
    streamingSessionId,
    error,
    usage,
    elapsedMs,
    firstTokenMs,
    finishReason,
    liveBytes,
    streamStartedAt,
    setActiveId,
    send,
    regenerate,
    editAndResend,
    sendAgent,
    toolApproval,
    resolveToolApproval,
    renameChat,
    togglePin,
    setDraft,
    setDraftImages,
    clearDraft,
    setSessionProject,
    setSessionWorkspace,
    setSessionSystemPrompt,
    setSessionMcpServerIds,
    setSessionCollectionIds,
    setSessionRagTuning,
    toggleContextPin,
    toggleContextEvict,
    restoreEvictedContext,
    toggleMessagePin,
    toggleMessageEvict,
    reconcileMcpScopes,
    switchVersion,
    branchFromMessage,
    flushAll,
    importChat,
    exportChat,
    undoAgent,
    undoAgentFile,
    keepAgentFile,
    stop,
    clear,
    newChat,
    deleteChat,
    deleteMessage,
    setError,
    dismissRagNotice: () => setRagNotice(null),
  };
}

export type ChatController = ReturnType<typeof useChat>;
