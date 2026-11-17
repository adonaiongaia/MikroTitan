import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import { api, Events } from "../lib/api";
import { fitContext } from "../lib/contextBudget";
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
  AgentUndoAttempt,
  AgentUndoFile,
  AgentUndoResult,
  AgentUndoState,
  ChatDelta,
  ChatDone,
  ChatErrorEvent,
  ChatMessage,
  ChatNotice,
  ChatSession,
  ChatUsage,
  Citation,
  McpActivityEvent,
  McpServerInput,
} from "../types";

const STORAGE_KEY = "janas.sessions.v1";
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

function normalizeCitation(value: unknown): Citation | null {
  if (!value || typeof value !== "object") return null;
  const citation = value as Partial<Citation>;
  if (
    typeof citation.url !== "string" ||
    !/^https?:\/\//i.test(citation.url)
  ) {
    return null;
  }
  return {
    title:
      typeof citation.title === "string" && citation.title.trim()
        ? citation.title
        : citation.url,
    url: citation.url,
    snippet: typeof citation.snippet === "string" ? citation.snippet : undefined,
  };
}

function normalizeRisk(value: unknown): AgentStep["risk"] {
  return value === "mutating" ||
    value === "destructive" ||
    value === "irreversible"
    ? value
    : "read_only";
}

function updateStep(
  steps: AgentStep[],
  stepId: string,
  patch: Partial<AgentStep>,
): AgentStep[] {
  return steps.map((step) => (step.id === stepId ? { ...step, ...patch } : step));
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

const UNDO_STATES: AgentUndoState[] = [
  "pending",
  "restored",
  "skipped",
  "failed",
  "kept",
];

function normalizeUndoFile(value: unknown): AgentUndoFile | null {
  if (!value || typeof value !== "object") return null;
  const file = value as Partial<AgentUndoFile>;
  if (typeof file.path !== "string" || !file.path) return null;
  const state = UNDO_STATES.includes(file.state as AgentUndoState)
    ? (file.state as AgentUndoState)
    : "pending";
  const kind =
    file.kind === "created" || file.kind === "deleted" ? file.kind : "modified";
  const history: AgentUndoAttempt[] = Array.isArray(file.history)
    ? file.history
        .filter((attempt) => Boolean(attempt) && typeof attempt === "object")
        .map((attempt) => {
          const entry = attempt as Partial<AgentUndoAttempt>;
          return {
            action:
              entry.action === "keep" ? ("keep" as const) : ("restore" as const),
            state: UNDO_STATES.includes(entry.state as AgentUndoState)
              ? (entry.state as AgentUndoState)
              : state,
            reason: typeof entry.reason === "string" ? entry.reason : null,
            at: typeof entry.at === "number" ? entry.at : Date.now(),
          };
        })
        .slice(-20)
    : [];
  return {
    path: file.path,
    kind,
    state,
    reason: typeof file.reason === "string" ? file.reason : null,
    attempts:
      typeof file.attempts === "number" ? file.attempts : history.length,
    history,
  };
}

function normalizeAgentTrace(value: unknown): AgentTrace | undefined {
  if (!value || typeof value !== "object") return undefined;
  const trace = value as Partial<AgentTrace>;
  if (!Array.isArray(trace.steps)) return undefined;
  const stepStatuses: AgentStepStatus[] = [
    "running",
    "awaiting_approval",
    "succeeded",
    "failed",
    "rejected",
    "blocked",
  ];
  const steps: AgentStep[] = trace.steps
    .filter((step): step is AgentStep => Boolean(step) && typeof step === "object")
    .map((step) => ({
      id: typeof step.id === "string" ? step.id : uid(),
      iteration: typeof step.iteration === "number" ? step.iteration : 0,
      thought: typeof step.thought === "string" ? step.thought : "",
      tool: typeof step.tool === "string" ? step.tool : "tool",
      arguments: typeof step.arguments === "string" ? step.arguments : "{}",
      risk: normalizeRisk(step.risk),
      status: stepStatuses.includes(step.status) ? step.status : "succeeded",
      output: typeof step.output === "string" ? step.output : null,
      duration_ms: typeof step.duration_ms === "number" ? step.duration_ms : null,
      truncated: Boolean(step.truncated),
      created_at:
        typeof step.created_at === "number" ? step.created_at : Date.now(),
    }))
    .slice(-60);
  const status =
    trace.status === "completed" ||
    trace.status === "verification_failed" ||
    trace.status === "max_iterations" ||
    trace.status === "aborted" ||
    trace.status === "error"
      ? trace.status
      : // A reloaded "running" trace never resumes; it was interrupted.
        "aborted";
  return {
    run_id: typeof trace.run_id === "string" ? trace.run_id : "",
    status,
    iterations: typeof trace.iterations === "number" ? trace.iterations : 0,
    max_iterations:
      typeof trace.max_iterations === "number" ? trace.max_iterations : 10,
    plan: Array.isArray(trace.plan)
      ? trace.plan.filter((item): item is string => typeof item === "string").slice(0, 20)
      : [],
    steps,
    compactions: Array.isArray(trace.compactions)
      ? trace.compactions
          .filter((event) => Boolean(event) && typeof event === "object")
          .map((event) => ({
            id: typeof event.id === "string" ? event.id : uid(),
            dropped_messages:
              typeof event.dropped_messages === "number" ? event.dropped_messages : 0,
            estimated_tokens:
              typeof event.estimated_tokens === "number" ? event.estimated_tokens : 0,
            context_window:
              typeof event.context_window === "number" ? event.context_window : 0,
            summary: typeof event.summary === "string" ? event.summary : "",
            created_at:
              typeof event.created_at === "number" ? event.created_at : Date.now(),
          }))
          .slice(-12)
      : [],
    pending: null,
    live_text: "",
    live_reasoning: "",
    notice: null,
    error: typeof trace.error === "string" ? trace.error : null,
    // The startup availability query fills this in for traces whose on-disk
    // checkpoint still exists; live runs set it when agent:done arrives.
    undo_available: false,
    undone: Boolean(trace.undone),
    undo_error:
      typeof trace.undo_error === "string" ? trace.undo_error : null,
    undo_summary:
      typeof trace.undo_summary === "string" ? trace.undo_summary : null,
    undo_files: Array.isArray(trace.undo_files)
      ? trace.undo_files
          .map(normalizeUndoFile)
          .filter((file): file is AgentUndoFile => file !== null)
      : [],
    workspace_root:
      typeof trace.workspace_root === "string" ? trace.workspace_root : "",
    changes: Array.isArray(trace.changes)
      ? trace.changes
          .filter((change) => Boolean(change) && typeof change === "object")
          .map((change) => ({
            path: typeof change.path === "string" ? change.path : "",
            kind:
              typeof change.kind === "string" && change.kind
                ? change.kind
                : "modified",
            restored: Boolean(change.restored),
          }))
          .filter((change) => change.path.length > 0)
          .slice(0, 500)
      : [],
    started_at: typeof trace.started_at === "number" ? trace.started_at : Date.now(),
    finished_at:
      typeof trace.finished_at === "number" ? trace.finished_at : null,
  };
}

function normalizeMessage(value: unknown): ChatMessage | null {
  if (!value || typeof value !== "object") return null;
  const message = value as Partial<ChatMessage>;
  if (message.role !== "user" && message.role !== "assistant") return null;
  const images = Array.isArray(message.images)
    ? message.images
        .filter(
          (image): image is { name: string; data_url: string; tokens: number } =>
            Boolean(image) &&
            typeof image.name === "string" &&
            typeof image.data_url === "string",
        )
        .map((image) => ({
          ...image,
          tokens: typeof image.tokens === "number" ? image.tokens : 0,
        }))
    : undefined;
  return {
    id: typeof message.id === "string" ? message.id : uid(),
    role: message.role,
    content: typeof message.content === "string" ? message.content : "",
    reasoning: typeof message.reasoning === "string" ? message.reasoning : "",
    created_at:
      typeof message.created_at === "number" ? message.created_at : Date.now(),
    citations: Array.isArray(message.citations)
      ? message.citations
          .map(normalizeCitation)
          .filter((citation): citation is Citation => citation !== null)
      : undefined,
    images,
    agentTrace: normalizeAgentTrace(message.agentTrace),
  };
}

function loadSessions(): ChatSession[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length) {
        const sessions = parsed
          .filter(
            (session): session is ChatSession =>
              Boolean(session) &&
              typeof session === "object" &&
              typeof (session as ChatSession).id === "string" &&
              Array.isArray((session as ChatSession).messages),
          )
          .map((session) => ({
            ...session,
            title:
              typeof session.title === "string" ? session.title : "New chat",
            created_at:
              typeof session.created_at === "number"
                ? session.created_at
                : Date.now(),
            updated_at:
              typeof session.updated_at === "number"
                ? session.updated_at
                : Date.now(),
            messages: session.messages
              .map(normalizeMessage)
              .filter((message): message is ChatMessage => message !== null),
          }));
        if (sessions.length) return sessions;
      }
    }
  } catch {
    // A corrupt store should never stop the studio from opening.
  }
  return [createSession()];
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
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions);
  // Resolve the active chat against the sessions the state actually holds.
  // Calling loadSessions() again here would mint a second, unrelated session
  // on a fresh profile, leaving activeId pointing at a chat that is not in
  // the list. Sending then had no session to append to and returned silently.
  const [activeId, setActiveId] = useState<string>(() =>
    loadActive(sessions),
  );
  const [streaming, setStreaming] = useState(false);
  const [streamingSessionId, setStreamingSessionId] = useState<string | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [usage, setUsage] = useState<ChatUsage | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [firstTokenMs, setFirstTokenMs] = useState<number | null>(null);
  const [finishReason, setFinishReason] = useState<string | null>(null);
  const [liveBytes, setLiveBytes] = useState(0);
  const [startedAtMs, setStartedAtMs] = useState<number | null>(null);
  const activeRequest = useRef<ActiveRequest | null>(null);
  const streamStartedAt = useRef<number | null>(null);
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;

  useEffect(() => {
    try {
      // Keep the newest 60 sessions; messages are plain text.
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(sessions.slice(0, 60)),
      );
    } catch {
      // Storage can be full or disabled. The session still works in memory.
    }
  }, [sessions]);

  useEffect(() => {
    try {
      window.localStorage.setItem(ACTIVE_KEY, JSON.stringify(activeId));
    } catch {
      // ignore
    }
  }, [activeId]);

  // Checkpoints live on disk and survive a restart, while the in-memory
  // journal does not. Ask Rust which persisted traces are still undoable once
  // the initially loaded sessions are available.
  const availabilityChecked = useRef(false);
  useEffect(() => {
    if (availabilityChecked.current) return;
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
  }, [sessions]);

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
        activeRequest.current = null;
      });
      const errorUnlisten = await listen<ChatErrorEvent>(
        Events.chatError,
        (event) => {
          const request = activeRequest.current;
          if (!request || event.payload.request_id !== request.requestId) return;
          setError(event.payload.message);
          setStreaming(false);
          setStreamingSessionId(null);
          activeRequest.current = null;
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
          setError(event.payload.message);
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
          activeRequest.current = null;
        },
      );

      if (disposed) {
        deltaUnlisten();
        doneUnlisten();
        errorUnlisten();
        noticeUnlisten();
        mcpActivityUnlisten();
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
          mcpActivityUnlisten,
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

  const send = useCallback(
    async (text: string, options: SendOptions) => {
      const trimmed = text.trim();
      if (!trimmed || activeRequest.current) return false;

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

      const requestId = uid();
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
        id: requestId,
        role: "assistant",
        content: "",
        reasoning: "",
        created_at: now,
        citations: options.citations,
      };

      const history = [...session.messages, userMessage];
      const turnWindow = Math.max(1, options.contextPrefetch) * 2 + 1;
      const windowed = history.slice(-turnWindow).filter(
        (message) =>
          message.role === "user" ||
          message.content.trim().length > 0 ||
          message.reasoning.trim().length > 0,
      );

      const fitted = fitContext({
        contextWindow: options.contextWindow,
        maxTokens: options.maxTokens,
        systemPrompt: options.systemPrompt,
        extraContext: options.extraContext,
        messages: windowed.map((message) => ({
          role: message.role,
          content: message.content,
        })),
        imageTokens: options.imageTokens ?? 0,
      });
      if (fitted.error) {
        setError(fitted.error);
        return false;
      }

      setSessions((previous) =>
        previous.map((candidate) =>
          candidate.id === sessionId
            ? {
                ...candidate,
                title:
                  candidate.messages.length === 0
                    ? trimmed.slice(0, 56)
                    : candidate.title,
                updated_at: now,
                messages: [...candidate.messages, userMessage, assistantMessage],
              }
            : candidate,
        ),
      );
      options.onAccepted?.();
      setStreaming(true);
      setStreamingSessionId(sessionId);
      setError(null);
      setUsage(null);
      setElapsedMs(null);
      setFirstTokenMs(null);
      setFinishReason(null);
      setLiveBytes(0);
      setStartedAtMs(now);
      activeRequest.current = { requestId, sessionId, mode: "chat" };
      streamStartedAt.current = Date.now();

      try {
        await api.streamChat({
          request_id: requestId,
          messages: fitted.messages,
          system_prompt: fitted.systemPrompt || null,
          temperature: options.temperature,
          top_p: options.topP,
          top_k: Math.round(Math.min(100, Math.max(1, options.topK))),
          min_p: options.minP,
          repeat_penalty: options.repeatPenalty,
          structured: options.structuredOutput,
          thinking: options.thinking,
          images: options.images ?? [],
          max_tokens: options.maxTokens,
          mcp_servers: options.mcpServers ?? [],
        });
      } catch (err) {
        setError(errorMessage(err));
        setStreaming(false);
        setStreamingSessionId(null);
        activeRequest.current = null;
        setSessions((previous) =>
          previous.map((candidate) =>
            candidate.id === sessionId
              ? {
                  ...candidate,
                  messages: candidate.messages.map((message) =>
                    message.id === requestId && !message.content
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

  const sendAgent = useCallback(
    async (text: string, options: AgentSendOptions) => {
      const trimmed = text.trim();
      if (!trimmed || activeRequest.current) return false;

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

      // The Rust loop appends the goal itself; only prior turns are history.
      const history = session.messages
        .slice(-Math.max(1, options.contextPrefetch) * 2)
        .filter(
          (message) =>
            (message.role === "user" || message.role === "assistant") &&
            message.content.trim().length > 0,
        )
        .map((message) => ({
          role: message.role,
          content: message.content,
        }));

      setSessions((previous) =>
        previous.map((candidate) =>
          candidate.id === sessionId
            ? {
                ...candidate,
                title:
                  candidate.messages.length === 0
                    ? trimmed.slice(0, 56)
                    : candidate.title,
                updated_at: now,
                messages: [...candidate.messages, userMessage, assistantMessage],
              }
            : candidate,
        ),
      );
      options.onAccepted?.();
      setStreaming(true);
      setStreamingSessionId(sessionId);
      setError(null);
      setUsage(null);
      setElapsedMs(null);
      setFirstTokenMs(null);
      setFinishReason(null);
      setLiveBytes(0);
      setStartedAtMs(now);
      activeRequest.current = { requestId, sessionId, mode: "agent" };
      streamStartedAt.current = Date.now();

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
        });
      } catch (err) {
        const message = errorMessage(err);
        setError(message);
        setStreaming(false);
        setStreamingSessionId(null);
        activeRequest.current = null;
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
    [],
  );

  const stop = useCallback(async () => {
    const request = activeRequest.current;
    if (!request) return;
    try {
      await cancelRequest(request);
    } catch (err) {
      setError(errorMessage(err));
      if (request.mode === "agent") {
        setStreaming(false);
        setStreamingSessionId(null);
      }
    }
  }, []);

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
          ? { ...session, messages: [], title: "New chat" }
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
  }, []);

  const newChat = useCallback(() => {
    const session = createSession();
    setSessions((previous) => [session, ...previous].slice(0, 60));
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
    setSessions((previous) =>
      previous.map((session) =>
        session.id === activeIdRef.current
          ? {
              ...session,
              messages: session.messages.filter(
                (message) => message.id !== messageId,
              ),
              updated_at: Date.now(),
            }
          : session,
      ),
    );
  }, []);

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
    messages,
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
    sendAgent,
    undoAgent,
    undoAgentFile,
    keepAgentFile,
    stop,
    clear,
    newChat,
    deleteChat,
    deleteMessage,
    setError,
  };
}

export type ChatController = ReturnType<typeof useChat>;
