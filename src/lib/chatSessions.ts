import { uid } from "./format";
import type {
  AgentStep,
  AgentStepStatus,
  AgentTrace,
  AgentUndoAttempt,
  AgentUndoFile,
  AgentUndoState,
  ChatDraft,
  ChatDraftImage,
  ChatMessage,
  ChatSession,
  Citation,
  ProjectIngest,
  RagChunk,
} from "../types";

const UNDO_STATES: AgentUndoState[] = [
  "pending",
  "restored",
  "skipped",
  "failed",
  "kept",
];

export function normalizeCitation(value: unknown): Citation | null {
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
    snippet:
      typeof citation.snippet === "string" ? citation.snippet : undefined,
  };
}

/**
 * A retrieved document chunk persisted with a reply. The content is stored with
 * the message so the `[D1]` drawer still works after the collection is
 * re-indexed or deleted.
 */
export function normalizeRagChunk(value: unknown): RagChunk | null {
  if (!value || typeof value !== "object") return null;
  const chunk = value as Partial<RagChunk>;
  if (
    typeof chunk.chunk_id !== "string" ||
    typeof chunk.content !== "string" ||
    typeof chunk.file_path !== "string"
  ) {
    return null;
  }
  const number = (candidate: unknown, fallback = 0): number =>
    typeof candidate === "number" && Number.isFinite(candidate)
      ? candidate
      : fallback;
  return {
    chunk_id: chunk.chunk_id,
    collection_id:
      typeof chunk.collection_id === "string" ? chunk.collection_id : "",
    collection_name:
      typeof chunk.collection_name === "string" ? chunk.collection_name : "",
    file_path: chunk.file_path,
    relative_path:
      typeof chunk.relative_path === "string" && chunk.relative_path
        ? chunk.relative_path
        : chunk.file_path,
    start_line: Math.max(1, Math.round(number(chunk.start_line, 1))),
    end_line: Math.max(1, Math.round(number(chunk.end_line, 1))),
    content: chunk.content,
    heading: typeof chunk.heading === "string" ? chunk.heading : "",
    token_estimate: Math.max(0, Math.round(number(chunk.token_estimate))),
    score: Math.min(1, Math.max(0, number(chunk.score))),
    lexical_score: Math.max(0, number(chunk.lexical_score)),
    semantic_score: Math.max(0, number(chunk.semantic_score)),
    // Sessions saved before the citation contract changed stored `[D1]`.
    citation:
      typeof chunk.citation === "string"
        ? chunk.citation.replace(/^\[|\]$/g, "")
        : "D1",
  };
}

export function normalizeRisk(value: unknown): AgentStep["risk"] {
  return value === "mutating" ||
    value === "destructive" ||
    value === "irreversible"
    ? value
    : "read_only";
}

export function updateStep(
  steps: AgentStep[],
  stepId: string,
  patch: Partial<AgentStep>,
): AgentStep[] {
  return steps.map((step) =>
    step.id === stepId ? { ...step, ...patch } : step,
  );
}

export function normalizeUndoFile(value: unknown): AgentUndoFile | null {
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

export function normalizeAgentTrace(value: unknown): AgentTrace | undefined {
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
    .filter(
      (step): step is AgentStep =>
        Boolean(step) && typeof step === "object",
    )
    .map((step) => ({
      id: typeof step.id === "string" ? step.id : uid(),
      iteration: typeof step.iteration === "number" ? step.iteration : 0,
      thought: typeof step.thought === "string" ? step.thought : "",
      tool: typeof step.tool === "string" ? step.tool : "tool",
      arguments:
        typeof step.arguments === "string" ? step.arguments : "{}",
      risk: normalizeRisk(step.risk),
      status: stepStatuses.includes(step.status) ? step.status : "succeeded",
      output: typeof step.output === "string" ? step.output : null,
      duration_ms:
        typeof step.duration_ms === "number" ? step.duration_ms : null,
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
      ? trace.plan
          .filter((item): item is string => typeof item === "string")
          .slice(0, 20)
      : [],
    steps,
    compactions: Array.isArray(trace.compactions)
      ? trace.compactions
          .filter((event) => Boolean(event) && typeof event === "object")
          .map((event) => ({
            id: typeof event.id === "string" ? event.id : uid(),
            dropped_messages:
              typeof event.dropped_messages === "number"
                ? event.dropped_messages
                : 0,
            estimated_tokens:
              typeof event.estimated_tokens === "number"
                ? event.estimated_tokens
                : 0,
            context_window:
              typeof event.context_window === "number"
                ? event.context_window
                : 0,
            summary: typeof event.summary === "string" ? event.summary : "",
            created_at:
              typeof event.created_at === "number"
                ? event.created_at
                : Date.now(),
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
    started_at:
      typeof trace.started_at === "number" ? trace.started_at : Date.now(),
    finished_at:
      typeof trace.finished_at === "number" ? trace.finished_at : null,
  };
}

function normalizeDraftImage(value: unknown): ChatDraftImage | null {
  if (!value || typeof value !== "object") return null;
  const image = value as Partial<ChatDraftImage>;
  if (
    typeof image.data_url !== "string" &&
    typeof image.asset !== "string"
  ) {
    return null;
  }
  return {
    name: typeof image.name === "string" ? image.name : "image",
    data_url:
      typeof image.data_url === "string" ? image.data_url : undefined,
    asset: typeof image.asset === "string" ? image.asset : undefined,
    tokens: typeof image.tokens === "number" ? image.tokens : 0,
  };
}

export function normalizeDraft(value: unknown): ChatDraft | undefined {
  if (!value || typeof value !== "object") return undefined;
  const draft = value as Partial<ChatDraft>;
  const images = Array.isArray(draft.images)
    ? draft.images
        .map(normalizeDraftImage)
        .filter((image): image is ChatDraftImage => image !== null)
        .slice(0, 4)
    : [];
  return {
    text: typeof draft.text === "string" ? draft.text : "",
    images,
    updated_at:
      typeof draft.updated_at === "number" ? draft.updated_at : Date.now(),
  };
}

function normalizeProject(value: unknown): ProjectIngest | undefined {
  if (!value || typeof value !== "object") return undefined;
  const project = value as Partial<ProjectIngest>;
  if (
    typeof project.root !== "string" ||
    typeof project.bundle !== "string"
  ) {
    return undefined;
  }
  return {
    root: project.root,
    name: typeof project.name === "string" ? project.name : "project",
    files: typeof project.files === "number" ? project.files : 0,
    bytes: typeof project.bytes === "number" ? project.bytes : 0,
    tokens: typeof project.tokens === "number" ? project.tokens : 0,
    tree: typeof project.tree === "string" ? project.tree : "",
    bundle: project.bundle,
    truncated: Boolean(project.truncated),
  };
}

export function normalizeMessage(value: unknown): ChatMessage | null {
  if (!value || typeof value !== "object") return null;
  const message = value as Partial<ChatMessage>;
  if (message.role !== "user" && message.role !== "assistant") return null;
  const rawImages: unknown[] = Array.isArray(message.images)
    ? (message.images as unknown[])
    : [];
  const images = rawImages.length
    ? rawImages
        .filter(
          (image): image is Record<string, unknown> =>
            Boolean(image) && typeof image === "object",
        )
        .map((image) => ({
          name: typeof image.name === "string" ? image.name : "image",
          data_url:
            typeof image.data_url === "string" ? image.data_url : undefined,
          asset: typeof image.asset === "string" ? image.asset : undefined,
          tokens: typeof image.tokens === "number" ? image.tokens : 0,
        }))
        .filter((image) => Boolean(image.data_url || image.asset))
    : undefined;
  return {
    id: typeof message.id === "string" ? message.id : uid(),
    role: message.role,
    content: typeof message.content === "string" ? message.content : "",
    reasoning:
      typeof message.reasoning === "string" ? message.reasoning : "",
    created_at:
      typeof message.created_at === "number"
        ? message.created_at
        : Date.now(),
    citations: Array.isArray(message.citations)
      ? message.citations
          .map(normalizeCitation)
          .filter((citation): citation is Citation => citation !== null)
      : undefined,
    rag_sources: Array.isArray(message.rag_sources)
      ? message.rag_sources
          .map(normalizeRagChunk)
          .filter((chunk): chunk is RagChunk => chunk !== null)
      : undefined,
    images,
    agentTrace: normalizeAgentTrace(message.agentTrace),
    // Leave missing links undefined so backfillTree can infer the legacy
    // linear path. An explicit null is a real root/leaf marker.
    parent_id:
      message.parent_id === null
        ? null
        : typeof message.parent_id === "string"
          ? message.parent_id
          : undefined,
    active_child_id:
      message.active_child_id === null
        ? null
        : typeof message.active_child_id === "string"
          ? message.active_child_id
          : undefined,
  };
}

/**
 * Backfill the tree links for chats written before branching existed. The
 * active path is linear, so each message's parent is the previous message and
 * its active child is the next one. Archived messages keep whatever links they
 * already carry.
 */
export function backfillTree(messages: ChatMessage[]): void {
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message.parent_id === undefined) {
      message.parent_id = index > 0 ? messages[index - 1].id : null;
    }
    message.active_child_id =
      index + 1 < messages.length ? messages[index + 1].id : null;
  }
}

export function normalizeSession(value: unknown): ChatSession | null {
  if (!value || typeof value !== "object") return null;
  const session = value as Partial<ChatSession>;
  if (typeof session.id !== "string" || !Array.isArray(session.messages)) {
    return null;
  }
  const now = Date.now();
  const messages = session.messages
    .map(normalizeMessage)
    .filter((message): message is ChatMessage => message !== null);
  backfillTree(messages);
  const archived = Array.isArray(session.archived_messages)
    ? session.archived_messages
        .map(normalizeMessage)
        .filter((message): message is ChatMessage => message !== null)
    : [];
  const mcpServerIds = Array.isArray(session.mcp_server_ids)
    ? session.mcp_server_ids.filter(
        (id): id is string => typeof id === "string",
      )
    : undefined;
  // An empty array is meaningful here: it means "no collections", not "all",
  // so it must survive a reload instead of collapsing to undefined.
  const attachedCollectionIds = Array.isArray(session.attached_collection_ids)
    ? session.attached_collection_ids.filter(
        (id): id is string => typeof id === "string" && id.length > 0,
      )
    : undefined;
  return {
    id: session.id,
    schema_version:
      typeof session.schema_version === "number"
        ? session.schema_version
        : undefined,
    title: typeof session.title === "string" ? session.title : "New chat",
    created_at:
      typeof session.created_at === "number" ? session.created_at : now,
    updated_at:
      typeof session.updated_at === "number" ? session.updated_at : now,
    pinned: Boolean(session.pinned),
    messages,
    archived_messages: archived.length ? archived : undefined,
    draft: normalizeDraft(session.draft),
    project: normalizeProject(session.project),
    workspace_root:
      typeof session.workspace_root === "string"
        ? session.workspace_root
        : undefined,
    mcp_server_ids: mcpServerIds,
    mcp_server_names:
      mcpServerIds?.length
        ? normalizeMcpServerNames(session.mcp_server_names)
        : undefined,
    system_prompt:
      typeof session.system_prompt === "string"
        ? session.system_prompt
        : undefined,
    attached_collection_ids: attachedCollectionIds,
    rag_top_k:
      typeof session.rag_top_k === "number" && Number.isFinite(session.rag_top_k)
        ? session.rag_top_k
        : undefined,
    rag_max_tokens:
      typeof session.rag_max_tokens === "number" &&
      Number.isFinite(session.rag_max_tokens)
        ? session.rag_max_tokens
        : undefined,
  };
}

function normalizeMcpServerNames(
  value: unknown,
): Record<string, string> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  const entries = Object.entries(value).flatMap(([id, name]) =>
    typeof name === "string" && name.trim()
      ? [[id, name.trim()] as const]
      : [],
  );
  return entries.length ? Object.fromEntries(entries) : undefined;
}

export function createSession(): ChatSession {
  const now = Date.now();
  return {
    id: uid(),
    title: "New chat",
    created_at: now,
    updated_at: now,
    messages: [],
    archived_messages: [],
  };
}

/** Every node in a session, active path first, without duplicate ids. */
export function allSessionMessages(session: ChatSession): ChatMessage[] {
  const byId = new Map<string, ChatMessage>();
  for (const message of session.messages) byId.set(message.id, message);
  for (const message of session.archived_messages ?? []) {
    if (!byId.has(message.id)) byId.set(message.id, message);
  }
  return [...byId.values()];
}

/** Messages that share a parent are versions of the same position. */
export function siblingsFor(
  session: ChatSession,
  message: ChatMessage,
): ChatMessage[] {
  const parentId = message.parent_id ?? null;
  return allSessionMessages(session)
    .filter((candidate) => (candidate.parent_id ?? null) === parentId)
    .sort(
      (left, right) =>
        left.created_at - right.created_at || left.id.localeCompare(right.id),
    );
}

function followActiveChildren(
  all: ChatMessage[],
  start: ChatMessage,
): ChatMessage[] {
  const byId = new Map(all.map((message) => [message.id, message]));
  const path: ChatMessage[] = [];
  const seen = new Set<string>([start.id]);
  let nextId = start.active_child_id;
  while (nextId && !seen.has(nextId)) {
    const next = byId.get(nextId);
    if (!next) break;
    path.push(next);
    seen.add(nextId);
    nextId = next.active_child_id;
  }
  return path;
}

function finalizePath(
  session: ChatSession,
  rawPath: ChatMessage[],
  all: ChatMessage[],
): ChatSession {
  const seen = new Set<string>();
  const path: ChatMessage[] = [];
  for (const message of rawPath) {
    if (seen.has(message.id)) continue;
    seen.add(message.id);
    path.push(message);
  }
  const linked = path.map((message, index) => ({
    ...message,
    active_child_id: path[index + 1]?.id ?? null,
  }));
  const archived = all.filter((message) => !seen.has(message.id));
  return {
    ...session,
    messages: linked,
    archived_messages: archived,
    updated_at: Date.now(),
  };
}

/** Append a new user/assistant pair to the active path. */
export function appendTurn(
  session: ChatSession,
  user: ChatMessage,
  assistant: ChatMessage,
): ChatSession {
  const parent = session.messages[session.messages.length - 1] ?? null;
  const nextUser: ChatMessage = {
    ...user,
    parent_id: parent?.id ?? null,
    active_child_id: assistant.id,
  };
  const nextAssistant: ChatMessage = {
    ...assistant,
    parent_id: nextUser.id,
    active_child_id: null,
  };
  const prefix = parent
    ? session.messages.map((message) =>
        message.id === parent.id
          ? { ...message, active_child_id: nextUser.id }
          : message,
      )
    : [];
  const all = [
    ...allSessionMessages(session),
    nextUser,
    nextAssistant,
  ];
  return finalizePath(
    session,
    [...prefix, nextUser, nextAssistant],
    all,
  );
}

/**
 * Edit a user message without deleting the old continuation. The edited
 * message becomes a sibling of the original; the old suffix stays archived.
 */
export function forkUserTurn(
  session: ChatSession,
  index: number,
  edited: ChatMessage,
  assistant: ChatMessage,
): ChatSession {
  const prefix = session.messages.slice(0, index);
  const parent = prefix[prefix.length - 1] ?? null;
  const nextUser: ChatMessage = {
    ...edited,
    parent_id: parent?.id ?? null,
    active_child_id: assistant.id,
  };
  const nextAssistant: ChatMessage = {
    ...assistant,
    parent_id: nextUser.id,
    active_child_id: null,
  };
  const nextPrefix = parent
    ? prefix.map((message) =>
        message.id === parent.id
          ? { ...message, active_child_id: nextUser.id }
          : message,
      )
    : prefix;
  const all = [
    ...allSessionMessages(session),
    nextUser,
    nextAssistant,
  ];
  return finalizePath(
    session,
    [...nextPrefix, nextUser, nextAssistant],
    all,
  );
}

/** Regenerate a reply while keeping the previous reply as a sibling version. */
export function forkAssistantTurn(
  session: ChatSession,
  index: number,
  assistant: ChatMessage,
): ChatSession {
  const prefix = session.messages.slice(0, index);
  const parent = prefix[prefix.length - 1] ?? null;
  if (!parent) return session;
  const nextAssistant: ChatMessage = {
    ...assistant,
    parent_id: parent.id,
    active_child_id: null,
  };
  const nextPrefix = prefix.map((message) =>
    message.id === parent.id
      ? { ...message, active_child_id: nextAssistant.id }
      : message,
  );
  const all = [...allSessionMessages(session), nextAssistant];
  return finalizePath(session, [...nextPrefix, nextAssistant], all);
}

/** Select a sibling at `messageId` and restore the path that followed it. */
export function switchMessageVersion(
  session: ChatSession,
  messageId: string,
  siblingId: string,
): ChatSession {
  const index = session.messages.findIndex(
    (message) => message.id === messageId,
  );
  if (index < 0) return session;
  const all = allSessionMessages(session);
  const target = all.find((message) => message.id === siblingId);
  if (!target) return session;
  const prefix = session.messages.slice(0, index);
  const parent = prefix[prefix.length - 1] ?? null;
  const nextPrefix = parent
    ? prefix.map((message) =>
        message.id === parent.id
          ? { ...message, active_child_id: target.id }
          : message,
      )
    : prefix;
  const nextPath = [
    ...nextPrefix,
    target,
    ...followActiveChildren(all, target),
  ];
  return finalizePath(session, nextPath, all);
}

/** Copy the active path up to a message into a new session object. */
export function branchPrefix(
  session: ChatSession,
  messageId: string,
): ChatMessage[] {
  const index = session.messages.findIndex(
    (message) => message.id === messageId,
  );
  if (index < 0) return [];
  return session.messages.slice(0, index + 1).map((message) => ({
    ...message,
    active_child_id:
      message.id === messageId ? null : message.active_child_id ?? null,
  }));
}
