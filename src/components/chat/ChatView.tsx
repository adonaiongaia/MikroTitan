import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertTriangle,
  Cpu,
  FolderOpen,
  Loader2,
  PanelRight,
  Play,
  Terminal,
  Trash2,
} from "lucide-react";

import type { ChatController } from "../../hooks/useChat";
import type { EngineController } from "../../hooks/useEngine";
import { useAgentPreferences } from "../../hooks/useAgentPreferences";
import {
  DEFAULT_ENGINE_PREFERENCES,
  useEnginePreferences,
} from "../../hooks/useEnginePreferences";
import { useKeyboardShortcuts } from "../../hooks/useKeyboardShortcuts";
import { usePersistentState } from "../../hooks/usePersistentState";
import { usePersonas } from "../../hooks/usePersonas";
import { useSettings } from "../../hooks/useSettings";
import { api } from "../../lib/api";
import {
  artifactGroupFor,
  collectChatArtifacts,
  type Artifact,
} from "../../lib/artifacts";
import { cn } from "../../lib/cn";
import { errorMessage, basename } from "../../lib/format";
import {
  ragFileLabel,
  ragScopeLabel,
  ragSettingsFor,
  reconcileCollectionScope,
} from "../../lib/rag";
import { recallFacts, useMemory } from "../../lib/memory";
import {
  ARTIFACT_TAGS_INSTRUCTION,
  buildWebContext,
  CORE_SYSTEM_PROMPT,
} from "../../lib/prompts";
import {
  evictedContextIds,
  messageContextId,
  pinnedContextIds,
  siblingsFor,
} from "../../lib/chatSessions";
import { estimateTokens, selectProjectContext } from "../../lib/contextBudget";
import {
  mcpServerInputs,
  mcpServersInScope,
  toggleServerInScope,
} from "../../lib/mcpScope";
import type {
  AppSettings,
  Citation,
  CompanionPairing,
  EngineKind,
  KnowledgeCollection,
  LibraryModel,
  LocalModel,
  McpServerConfig,
  RagChunk,
  StorageBenchmark,
  WhisperStatus,
} from "../../types";
import { ArtifactPane } from "../artifacts/ArtifactPane";
import { knowledgeApi } from "../knowledge/api";
import { DocsDrawer } from "../knowledge/DocsDrawer";
import { RagSourceDrawer } from "../knowledge/RagSourceDrawer";
import { Logo } from "../Logo";
import { Button } from "../ui/Button";
import type { AgentActions } from "./AgentTrace";
import { Composer, type ComposerImage } from "./Composer";
import type {
  ContextBreakdownCategory,
  ContextBreakdownItem,
} from "./ContextBreakdownPopover";
import { EnginePanel } from "./EnginePanel";
import { McpLogDrawer } from "./McpLogDrawer";
import { MessageBoundary } from "../MessageBoundary";
import { MessageBubble } from "./MessageBubble";
import { SessionsRail } from "./SessionsRail";
import { TelemetryPanel } from "./TelemetryPanel";
import { ToolApprovalCard } from "./ToolApprovalCard";

interface ChatViewProps {
  engine: EngineController;
  chat: ChatController;
  /** Bumped by Mod+K so the sessions rail search takes focus. */
  focusSearchSignal: number;
  /** Clears the request once the rail has focused its search field. */
  onFocusSearchHandled: () => void;
  /** Opens the Knowledge Vault view from the Docs drawer. */
  onOpenKnowledge: () => void;
  /** Opens Settings so the mic can install Whisper. */
  onOpenSettings: () => void;
}

const SUGGESTIONS = [
  {
    title: "Explain a trade-off",
    prompt:
      "Explain the trade-offs between keeping mixture-of-experts weights in RAM and streaming them from an NVMe SSD.",
  },
  {
    title: "Read my project",
    prompt:
      "Summarise the architecture of this project from the attached folder, then point out the riskiest module.",
  },
  {
    title: "Search and verify",
    prompt:
      "Search the web for the latest Janas release notes and summarise what changed, with sources.",
  },
];

export function ChatView({
  engine,
  chat,
  focusSearchSignal,
  onFocusSearchHandled,
  onOpenKnowledge,
  onOpenSettings,
}: ChatViewProps) {
  const { settings, save } = useSettings();
  const [whisperStatus, setWhisperStatus] = useState<WhisperStatus | null>(null);
  const refreshWhisper = useCallback(async () => {
    try {
      setWhisperStatus(await api.getWhisperStatus());
    } catch {
      setWhisperStatus(null);
    }
  }, []);

  useEffect(() => {
    void refreshWhisper();
  }, [
    refreshWhisper,
    settings?.voice_input_enabled,
    settings?.whisper_model,
    settings?.whisper_language,
  ]);
  const { preferences, update, setPreferences } = useEnginePreferences();
  const agent = useAgentPreferences();
  const memory = useMemory();

  const [defaultSystemPrompt, setDefaultSystemPrompt] = usePersistentState(
    "janas.systemPrompt",
    "",
  );
  const [explicitArtifactTags, setExplicitArtifactTags] = usePersistentState(
    "janas.artifacts.explicitTags.v1",
    false,
  );
  const personas = usePersonas();
  // A chat owns its instructions when it has an explicit value; otherwise it
  // follows the saved default, which is what new chats start from too.
  const chatSystemPrompt = chat.activeSession?.system_prompt;
  const systemPrompt = chatSystemPrompt ?? defaultSystemPrompt;
  const usingDefaultPrompt = chatSystemPrompt === undefined;
  const setSystemPrompt = useCallback(
    (value: string) => chat.setSessionSystemPrompt(value),
    [chat.setSessionSystemPrompt],
  );
  const [approvalRulesVersion, setApprovalRulesVersion] = useState(0);
  const [mcpServers] = usePersistentState<McpServerConfig[]>(
    "janas.mcpServers.v1",
    [],
  );
  const activeMcpServers = useMemo(
    () => mcpServersInScope(mcpServers, chat.mcpServerIds),
    [chat.mcpServerIds, mcpServers],
  );
  const availableMcpServers = useMemo(
    () => mcpServersInScope(mcpServers, undefined),
    [mcpServers],
  );
  const mcpScopeLabel =
    availableMcpServers.length === 0
      ? null
      : chat.mcpServerIds === undefined
        ? "All"
        : activeMcpServers.length === 0
          ? "None"
          : `${activeMcpServers.length}/${availableMcpServers.length}`;
  const mcpScopeHint =
    mcpScopeLabel === null
      ? undefined
      : mcpScopeLabel === "None"
        ? "No MCP tools are enabled for this chat."
        : mcpScopeLabel === "All"
          ? "This chat can use every enabled MCP server."
          : `${activeMcpServers.length} of ${availableMcpServers.length} enabled MCP servers are enabled for this chat.`;

  useEffect(() => {
    if (chat.loading) return;
    chat.reconcileMcpScopes(mcpServers);
  }, [
    chat.loading,
    chat.reconcileMcpScopes,
    chat.sessions.length,
    mcpServers,
  ]);

  const input = chat.draft.text;
  const setInput = chat.setDraft;
  const images = chat.draft.images.filter(
    (image): image is ComposerImage => Boolean(image.data_url),
  );
  const [models, setModels] = useState<LocalModel[]>([]);
  const [llamaModels, setLlamaModels] = useState<LibraryModel[]>([]);
  const [switchingModel, setSwitchingModel] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [visionGuard, setVisionGuard] = useState<CompanionPairing | null>(null);
  const [visionBusy, setVisionBusy] = useState(false);
  const [sessionsCollapsed, setSessionsCollapsed] = usePersistentState(
    "janas.ui.sessionsCollapsed",
    false,
  );
  const [engineCollapsed, setEngineCollapsed] = usePersistentState(
    "janas.ui.engineCollapsed",
    false,
  );
  const [telemetryCollapsed, setTelemetryCollapsed] = usePersistentState(
    "janas.ui.telemetryCollapsed",
    false,
  );
  const [artifactPaneOpen, setArtifactPaneOpen] = usePersistentState(
    "janas.ui.artifactPaneOpen",
    false,
  );
  // The selected artifact follows the chat, not the window. Switching chats
  // restores that chat's canvas; switching siblings for a persisted root that
  // is not on the new branch falls back to the newest artifact in the branch.
  const [artifactSelection, setArtifactSelection] = usePersistentState<
    Record<string, { rootId: string; versionId: string | null }>
  >("janas.artifacts.active.v2", {});

  const artifactIndex = useMemo(
    () =>
      collectChatArtifacts(chat.messages, {
        streamingMessageId: chat.streaming
          ? (chat.messages[chat.messages.length - 1]?.id ?? null)
          : null,
      }),
    [chat.messages, chat.streaming],
  );
  const activeArtifactSelection = chat.activeId
    ? artifactSelection[chat.activeId]
    : undefined;
  const selectedArtifactRootId = useMemo(() => {
    if (
      activeArtifactSelection &&
      artifactGroupFor(artifactIndex, activeArtifactSelection.rootId)
    ) {
      return activeArtifactSelection.rootId;
    }
    return artifactIndex.latest?.rootId ?? null;
  }, [activeArtifactSelection, artifactIndex]);
  // The exact revision the user opened. A sibling branch may not contain it, so
  // the pane falls back to that group's latest version there.
  const selectedArtifactVersionId =
    activeArtifactSelection &&
    activeArtifactSelection.rootId === selectedArtifactRootId
      ? activeArtifactSelection.versionId
      : null;

  const openArtifact = useCallback(
    (artifact: Artifact) => {
      if (chat.activeId) {
        setArtifactSelection((previous) => ({
          ...previous,
          [chat.activeId]: {
            rootId: artifact.rootId,
            versionId: artifact.id,
          },
        }));
      }
      setArtifactPaneOpen(true);
    },
    [chat.activeId, setArtifactPaneOpen, setArtifactSelection],
  );

  // The artifact pane takes the space the engine panel was using, so the
  // transcript and composer keep a usable width instead of being squeezed.
  useEffect(() => {
    if (artifactPaneOpen) setEngineCollapsed(true);
  }, [artifactPaneOpen, setEngineCollapsed]);

  // Mod+K may arrive while the rail is collapsed; expanding it lets the rail
  // itself place focus on the search field.
  useEffect(() => {
    if (focusSearchSignal) setSessionsCollapsed(false);
  }, [focusSearchSignal, setSessionsCollapsed]);

  const project = chat.project;
  const [projectBusy, setProjectBusy] = useState(false);
  const [activity, setActivity] = useState<string | null>(null);
  const [benchmark, setBenchmark] = useState<StorageBenchmark | null>(null);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [mcpOpen, setMcpOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [collections, setCollections] = useState<KnowledgeCollection[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);
  const [sourceChunk, setSourceChunk] = useState<RagChunk | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const workspaceLocked = useMemo(
    () =>
      chat.sessions.some((session) =>
        session.messages.some((message) => {
          const trace = message.agentTrace;
          return Boolean(
            trace &&
              (trace.status === "running" ||
                (trace.undo_available && !trace.undone)),
          );
        }),
      ),
    [chat.sessions],
  );

  const attachedCollectionIds = useMemo(
    () =>
      Array.isArray(chat.collectionIds)
        ? chat.collectionIds.filter((id) =>
            collections.some((collection) => collection.id === id),
          )
        : [],
    [chat.collectionIds, collections],
  );
  const ragSettings = ragSettingsFor(chat.ragTopK, chat.ragMaxTokens);
  const docsScopeLabel = ragScopeLabel(attachedCollectionIds, collections);

  // Pins and evictions live on the chat session, so they survive reloads and
  // are shared by the composer ring, the transcript and the send path.
  const pinnedIds = useMemo(
    () => pinnedContextIds(chat.activeSession),
    [chat.activeSession],
  );
  const evictedIds = useMemo(
    () => evictedContextIds(chat.activeSession),
    [chat.activeSession],
  );
  const projectSelection = useMemo(
    () => selectProjectContext(project?.bundle ?? "", pinnedIds, evictedIds),
    [evictedIds, pinnedIds, project?.bundle],
  );
  // MCP pins are stored by server id but the engine matches display names, so
  // resolve the current name instead of trusting a label captured at pin time.
  const pinnedMcpServers = useMemo(() => {
    // Only servers that are actually in scope are connected for the turn, so
    // sending a pin for a disabled or out-of-scope server would be noise.
    const activeIds = new Set(activeMcpServers.map((server) => server.id));
    return (chat.activeSession?.pinned_context ?? [])
      .filter((item) => item.kind === "mcp")
      .map((item) => {
        const id = item.id.startsWith("mcp:")
          ? item.id.slice("mcp:".length)
          : "";
        return activeIds.has(id)
          ? activeMcpServers.find((server) => server.id === id)
          : undefined;
      })
      .filter((server): server is McpServerConfig => Boolean(server))
      .map((server) => server.name.trim())
      .filter(Boolean);
  }, [activeMcpServers, chat.activeSession]);

  const refreshCollections = useCallback(async () => {
    setCollectionsLoading(true);
    try {
      const next = await knowledgeApi.list();
      setCollections(Array.isArray(next) ? next : []);
      setCollectionsError(null);
    } catch (err) {
      setCollectionsError(errorMessage(err));
    } finally {
      setCollectionsLoading(false);
    }
  }, []);

  // Load once so the Docs badge can show a real count before the drawer opens.
  useEffect(() => {
    void refreshCollections();
  }, [refreshCollections]);

  useEffect(() => {
    if (docsOpen) void refreshCollections();
  }, [docsOpen, refreshCollections]);

  // A collection deleted in the vault must not stay attached to this chat, or
  // every later turn would report a missing collection.
  useEffect(() => {
    if (chat.loading || collectionsLoading || !collections.length) return;
    const reconciled = reconcileCollectionScope(chat.collectionIds, collections);
    if (reconciled.changed) {
      chat.setSessionCollectionIds(reconciled.ids ?? []);
    }
  }, [
    chat.collectionIds,
    chat.loading,
    chat.setSessionCollectionIds,
    collections,
    collectionsLoading,
  ]);

  const ready = engine.snapshot.status === "ready";
  const activeKind = engine.active?.kind ?? null;
  const workspaceRoot =
    chat.workspaceRoot || agent.preferences.workspaceRoot;
  const currentModelPath = settings?.model_path ?? "";
  const llamaModelPath = settings?.llama_model_path ?? "";
  const selectedModelPath =
    activeKind === "llama"
      ? llamaModelPath
      : activeKind === "janas"
        ? currentModelPath
        : "";
  const modelName = selectedModelPath
    ? basename(selectedModelPath)
    : activeKind === "llama"
      ? "No GGUF selected"
      : "No model selected";

  const loadModels = useCallback(async () => {
    try {
      setModels(await api.scanLocalModels());
    } catch {
      setModels([]);
    }
    try {
      const library = await api.libraryModels();
      setLlamaModels(
        library.filter((model) => model.path.toLowerCase().endsWith(".gguf")),
      );
    } catch {
      setLlamaModels([]);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels, currentModelPath, llamaModelPath]);

  // Streaming replaces the messages array on every delta. Coalesce those into
  // one scroll per frame instead of forcing a layout per token.
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      endRef.current?.scrollIntoView({ block: "end" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [chat.messages, chat.streaming]);

  // Pre-flight image guardrail: a multimodal GGUF only accepts image content
  // parts once llama.cpp has loaded its matching `--mmproj` projector. Inspect
  // the active model as soon as an image is queued instead of letting the
  // request fail on the server.
  useEffect(() => {
    if (!images.length || activeKind !== "llama" || !llamaModelPath) {
      setVisionGuard(null);
      return;
    }
    let cancelled = false;
    api
      .inspectModelCompanions(llamaModelPath)
      .then((pairing) => {
        if (cancelled) return;
        setVisionGuard(
          pairing.main.role === "vision_main" && !pairing.vision.bound_path
            ? pairing
            : null,
        );
      })
      .catch(() => {
        if (!cancelled) setVisionGuard(null);
      });
    return () => {
      cancelled = true;
    };
  }, [images.length, activeKind, llamaModelPath, settings?.llama_mmproj_path]);

  const chooseModel = useCallback(
    async (path: string) => {
      if (!settings || !path) return;
      setSwitchingModel(true);
      setModelError(null);
      try {
        await save({ ...settings, model_path: path });
        await loadModels();
        if (engine.snapshot.pid !== null) await engine.stop();
        await engine.switchTo("janas");
      } catch (err) {
        setModelError(errorMessage(err));
      } finally {
        setSwitchingModel(false);
      }
    },
    [engine, loadModels, save, settings],
  );

  const chooseLlamaModel = useCallback(
    async (path: string) => {
      if (!settings || !path) return;
      setSwitchingModel(true);
      setModelError(null);
      try {
        await save({ ...settings, llama_model_path: path });
        await loadModels();
        if (engine.snapshot.pid !== null) await engine.stop();
        await engine.switchTo("llama");
      } catch (err) {
        setModelError(errorMessage(err));
      } finally {
        setSwitchingModel(false);
      }
    },
    [engine, loadModels, save, settings],
  );

  const chooseAnyModel = useCallback(
    (path: string) => {
      if (!path) return;
      if (path.toLowerCase().endsWith(".gguf")) {
        void chooseLlamaModel(path);
      } else {
        void chooseModel(path);
      }
    },
    [chooseLlamaModel, chooseModel],
  );

  const attachProject = useCallback(
    async (path: string) => {
      setProjectBusy(true);
      setActivity("Parsing the project tree and building a semantic bundle…");
      try {
        const ingest = await api.ingestProject(path);
        chat.setSessionProject(ingest);
      } catch (err) {
        chat.setError(errorMessage(err));
      } finally {
        setProjectBusy(false);
        setActivity(null);
      }
    },
    [chat],
  );

  const pickProject = useCallback(async () => {
    try {
      const picked = await open({ multiple: false, directory: true });
      if (typeof picked === "string") await attachProject(picked);
    } catch (err) {
      chat.setError(errorMessage(err));
    }
  }, [attachProject, chat]);

  // The path picker only chooses a folder; Rust canonicalises it and enforces
  // it as the sandbox root for every built-in tool.
  const pickWorkspace = useCallback(async () => {
    if (workspaceLocked) {
      chat.setError(
        "The agent workspace is locked while a run or pending undo exists.",
      );
      return;
    }
    try {
      const picked = await open({ multiple: false, directory: true });
      if (typeof picked === "string") {
        // Canonicalise and verify with Rust before pinning the folder, so a
        // path that vanished between the dialog and this point is rejected
        // now instead of failing every later run.
        const canonical = await api.agentValidateWorkspace(picked);
        if (canonical) {
          chat.setSessionWorkspace(canonical);
          agent.update("workspaceRoot", canonical);
        }
      }
    } catch (err) {
      chat.setError(errorMessage(err));
    }
  }, [agent, chat, workspaceLocked]);

  const clearWorkspace = useCallback(() => {
    if (workspaceLocked) {
      chat.setError(
        "The agent workspace is locked while a run or pending undo exists.",
      );
      return;
    }
    // A chat can pin its own workspace, which takes precedence over the
    // default. Clear both so no stale path survives a later run.
    chat.setSessionWorkspace("");
    agent.update("workspaceRoot", "");
  }, [agent, chat, workspaceLocked]);

  const agentActions = useMemo<AgentActions>(
    () => ({
      approve: async (runId, stepId, options) => {
        try {
          await api.resolveAgentApproval({
            run_id: runId,
            step_id: stepId,
            decision: "approve",
            feedback: "",
            allow_file_edits: options?.allowEdits ?? false,
            always_allow: options?.alwaysAllow ?? false,
          });
          if (options?.alwaysAllow) {
            setApprovalRulesVersion((value) => value + 1);
          }
        } catch (err) {
          chat.setError(errorMessage(err));
        }
      },
      reject: async (runId, stepId, feedback) => {
        try {
          await api.resolveAgentApproval({
            run_id: runId,
            step_id: stepId,
            decision: "reject",
            feedback,
          });
        } catch (err) {
          chat.setError(errorMessage(err));
        }
      },
      abort: async (runId) => {
        try {
          await api.abortAgent(runId);
        } catch (err) {
          chat.setError(errorMessage(err));
        }
      },
      undo: async (runId) => {
        try {
          await chat.undoAgent(runId);
        } catch {
          // useChat records the failure on the run's trace.
        }
      },
      undoFile: async (runId, path) => {
        try {
          await chat.undoAgentFile(runId, path);
        } catch {
          // useChat records the failure on the run's trace.
        }
      },
      keepFile: async (runId, path) => {
        try {
          await chat.keepAgentFile(runId, path);
        } catch {
          // useChat records the failure on the run's trace.
        }
      },
    }),
    [chat],
  );

  // A folder dropped anywhere on the window is ingested without a dialog.
  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | undefined;
    void import("@tauri-apps/api/webview")
      .then(({ getCurrentWebview }) =>
        getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type !== "drop") return;
          const path = event.payload.paths?.[0];
          if (path) void attachProject(path);
        }),
      )
      .then((stop) => {
        if (cancelled) stop();
        else unlisten = stop;
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [attachProject]);

  const runBenchmark = useCallback(async () => {
    setBenchmarkBusy(true);
    try {
      setBenchmark(await api.benchmarkStorage(settings?.models_dir));
    } catch (err) {
      chat.setError(errorMessage(err));
    } finally {
      setBenchmarkBusy(false);
    }
  }, [chat, settings?.models_dir]);

  const applyEngineSettings = useCallback(
    async (patch: Partial<AppSettings>) => {
      if (!settings) return;
      await save({ ...settings, ...patch });
    },
    [save, settings],
  );

  const restartEngine = useCallback(async () => {
    try {
      await engine.stop();
      await engine.start();
    } catch (err) {
      chat.setError(errorMessage(err));
    }
  }, [chat, engine]);

  const pairVisionProjector = useCallback(async () => {
    const projector = visionGuard?.vision.recommended?.path;
    if (!visionGuard || !projector) return;
    setVisionBusy(true);
    try {
      const next = await api.pairModelCompanions(
        visionGuard.main.path,
        projector,
        settings?.llama_draft_path.trim() || null,
        settings?.llama_draft_max ?? 8,
      );
      await save(next);
      setVisionGuard(null);
      if (engine.snapshot.pid !== null) await engine.stop();
      await engine.switchTo("llama");
    } catch (err) {
      chat.setError(errorMessage(err));
    } finally {
      setVisionBusy(false);
    }
  }, [chat, engine, save, settings, visionGuard]);

  const downloadVisionProjector = useCallback(() => {
    const repo = visionGuard?.repo;
    if (!repo) {
      chat.setError(
        "No download source is recorded for this model. Download an mmproj for its family in Models, then pair it in Library.",
      );
      return;
    }
    void api
      .openExternal(`https://huggingface.co/${repo}/tree/main`)
      .catch((err) => chat.setError(errorMessage(err)));
  }, [chat, visionGuard]);

  const resetEnginePanel = useCallback(
    async (runtime: Partial<AppSettings>) => {
      if (!settings) return;
      try {
        await save({ ...settings, ...runtime });
        setPreferences((previous) => ({
          ...DEFAULT_ENGINE_PREFERENCES,
          internet: previous.internet,
        }));
        chat.setSessionSystemPrompt(null);
        setDefaultSystemPrompt("");
      } catch (err) {
        chat.setError(errorMessage(err));
      }
    },
    [
      chat,
      save,
      setDefaultSystemPrompt,
      setPreferences,
      settings,
    ],
  );

  const switchEngine = useCallback(
    async (kind: EngineKind) => {
      await engine.switchTo(kind);
    },
    [engine],
  );

  const configuredContextMax =
    activeKind === "llama"
      ? settings?.llama_ctx_size ?? 16384
      : settings?.ctx_size ?? 16384;
  // Settings can change while the old child is still running. Budget against
  // the context the active child actually launched with; the settings value
  // only becomes real after a restart.
  const contextMax = engine.active?.context_window || configuredContextMax;
  const imageTokens = images.reduce(
    (total, image) => total + image.tokens,
    0,
  );

  /**
   * Split the attached context into the pinned block (never trimmed) and the
   * unpinned block (trimmed first). Memory facts are context items too, so they
   * can be pinned or evicted like a file.
   */
  const buildContextParts = useCallback(
    (text: string, webContext = "") => {
      const recalled = recallFacts(memory.facts, text).filter(
        (fact) => !evictedIds.has(`memory:${fact.id}`),
      );
      const memoryItems = recalled.map((fact) => ({
        id: `memory:${fact.id}`,
        label: fact.text.slice(0, 72),
        line: `- ${fact.text}`,
        tokens: estimateTokens(fact.text),
        pinned: pinnedIds.has(`memory:${fact.id}`),
      }));
      const pinnedMemory = memoryItems.filter((item) => item.pinned);
      const unpinnedMemory = memoryItems.filter((item) => !item.pinned);
      const memoryHeader =
        "Facts you remember about this project. Use them when relevant, and ignore any that no longer apply:";
      const memoryBlock = (items: typeof memoryItems) =>
        items.length
          ? `${memoryHeader}\n${items.map((item) => item.line).join("\n")}`
          : "";
      // A pinned [D#] chunk is re-sent verbatim. Retrieval happens per turn in
      // Rust, so this is what keeps the exact snippet in the prompt.
      const pinnedChunks = new Map<string, RagChunk>();
      for (const message of chat.messages) {
        for (const chunk of message.rag_sources ?? []) {
          if (pinnedIds.has(`rag:${chunk.chunk_id}`)) {
            pinnedChunks.set(chunk.chunk_id, chunk);
          }
        }
      }
      const pinnedChunkList = [...pinnedChunks.values()];
      const pinnedRagText = pinnedChunkList.length
        ? `Pinned document snippets. Cite their [D#] labels when you use them:\n${pinnedChunkList
            .map(
              (chunk) =>
                `[${chunk.citation}] ${chunk.relative_path} (lines ${chunk.start_line}–${chunk.end_line})\n${chunk.content}`,
            )
            .join("\n\n")}`
        : "";
      return {
        systemPrompt: [
          CORE_SYSTEM_PROMPT,
          systemPrompt,
          explicitArtifactTags ? ARTIFACT_TAGS_INSTRUCTION : "",
        ]
          .filter(Boolean)
          .join("\n\n"),
        pinnedContext: [
          projectSelection.pinnedText,
          pinnedRagText,
          memoryBlock(pinnedMemory),
        ]
          .filter(Boolean)
          .join("\n\n"),
        extraContext: [
          projectSelection.unpinnedText,
          webContext,
          memoryBlock(unpinnedMemory),
        ]
          .filter(Boolean)
          .join("\n\n"),
        pinnedLabels: [
          ...projectSelection.fragments
            .filter((fragment) => pinnedIds.has(fragment.id))
            .map((fragment) => fragment.label),
          ...pinnedChunkList.map(
            (chunk) => `[${chunk.citation}] ${chunk.relative_path}`,
          ),
          ...pinnedMemory.map((item) => `memory: ${item.label}`),
        ],
        memoryItems,
      };
    },
    [
      chat.messages,
      evictedIds,
      explicitArtifactTags,
      memory.facts,
      pinnedIds,
      projectSelection,
      systemPrompt,
    ],
  );

  const buildOptions = useCallback(
    (
      text: string,
      extras: {
        images?: string[];
        imageTokens?: number;
        citations?: Citation[];
        extraContext?: string;
      } = {},
    ) => {
      const parts = buildContextParts(text, extras.extraContext ?? "");
      const mcpInputs = mcpServerInputs(mcpServers, chat.mcpServerIds);
      return {
        systemPrompt: parts.systemPrompt,
        temperature: preferences.temperature,
        topP: preferences.topP,
        topK: preferences.topK ?? 40,
        minP: preferences.minP,
        repeatPenalty: preferences.repeatPenalty,
        thinking: preferences.thinking,
        structuredOutput: preferences.structuredOutput,
        contextPrefetch: preferences.contextPrefetch,
        maxTokens: preferences.maxTokens,
        contextWindow: contextMax,
        images: extras.images,
        imageTokens: extras.imageTokens,
        extraContext: parts.extraContext,
        pinnedContext: parts.pinnedContext,
        pinnedLabels: parts.pinnedLabels,
        pinnedMcpServers,
        citations: extras.citations,
        mcpServers: mcpInputs,
      };
    },
    [
      buildContextParts,
      chat.mcpServerIds,
      contextMax,
      mcpServers,
      pinnedMcpServers,
      preferences,
    ],
  );

  const handleSend = useCallback(async (override?: string) => {
    const text = (override ?? input).trim();
    if (!text || chat.busy || !ready) return;
    if (images.length && visionGuard) {
      chat.setError(
        "This vision model needs an mmproj projector before it can read images. Use the pairing banner under the composer, then send again.",
      );
      return;
    }

    let citations: Citation[] | undefined;
    // Only web evidence travels here; the attached project is split into
    // pinned and unpinned blocks inside buildOptions.
    let extraContext = "";

    // In Agent Mode the agent decides when to search; pre-fetching here would
    // hide the tool call from the trace.
    if (preferences.internet && !agent.preferences.enabled) {
      setActivity(`Searching DuckDuckGo for “${text.slice(0, 64)}”…`);
      try {
        const search = await api.webSearch(text, 3);
        setActivity(
          `Ingesting ${search.results.length} page${
            search.results.length === 1 ? "" : "s"
          } · ${search.bytes > 0 ? `${(search.bytes / 1024).toFixed(0)} KB` : ""} → prefill layer…`,
        );
        if (!search.results.length) {
          setActivity(null);
          chat.setError("The web search returned no readable pages.");
          return;
        }
        citations = search.results.map((result) => ({
          title: result.title,
          url: result.url,
          snippet: result.snippet,
        }));
        extraContext = buildWebContext(text, search.results);
      } catch (err) {
        setActivity(null);
        chat.setError(errorMessage(err));
        return;
      }
    }

    setActivity(null);
    if (agent.preferences.enabled) {
      try {
        await chat.sendAgent(text, {
          ...buildOptions(text, { extraContext: "" }),
          agent: { ...agent.preferences, workspaceRoot },
          webSearch: preferences.internet,
          onAccepted: () => {
            chat.clearDraft();
          },
        });
      } catch (err) {
        chat.setError(errorMessage(err));
      }
      return;
    }

    try {
      await chat.send(text, {
        ...buildOptions(text, {
          images: images.map((image) => image.data_url),
          imageTokens,
          citations,
          extraContext,
        }),
        onAccepted: () => {
          chat.clearDraft();
        },
      });
    } catch (err) {
      chat.setError(errorMessage(err));
    }
  }, [
    agent.preferences,
    buildOptions,
    chat,
    images,
    imageTokens,
    input,
    preferences.internet,
    ready,
    visionGuard,
    workspaceRoot,
  ]);

  // Chat-local shortcuts run before the app-wide handlers. Escape dismisses
  // the drawers in priority order and declines when none is open, so the
  // global handler can stop the stream instead.
  useKeyboardShortcuts(
    {
      "toggle-sessions": () => setSessionsCollapsed((value) => !value),
      "toggle-engine": () => setEngineCollapsed((value) => !value),
      "toggle-agent": () => agent.update("enabled", !agent.preferences.enabled),
      "toggle-internet": () => update("internet", !preferences.internet),
      send: () => {
        void handleSend();
      },
      dismiss: () => {
        if (docsOpen) {
          setDocsOpen(false);
          return true;
        }
        if (sourceChunk) {
          setSourceChunk(null);
          return true;
        }
        if (mcpOpen) {
          setMcpOpen(false);
          return true;
        }
        return false;
      },
    },
    { priority: 60 },
  );

  const inputRef = useRef(input);
  inputRef.current = input;

  const insertVoiceTranscript = useCallback(
    (transcript: string) => {
      const text = transcript.trim();
      if (!text) return;
      const current = inputRef.current;
      const next = current.trim()
        ? `${current.replace(/\s+$/, "")} ${text}`
        : text;
      inputRef.current = next;
      setInput(next);
    },
    [setInput],
  );

  const sendVoiceTranscript = useCallback(
    (transcript: string) => {
      // Never discard a draft or queued images silently: only auto-send when
      // the composer is empty and the engine can accept a turn right now.
      if (!ready || chat.busy || input.trim() || images.length) {
        insertVoiceTranscript(transcript);
        return;
      }
      inputRef.current = transcript;
      setInput(transcript);
      void handleSend(transcript);
    },
    [chat.busy, handleSend, images.length, input, insertVoiceTranscript, ready, setInput],
  );

  const setVoiceAutoSend = useCallback(
    async (value: boolean) => {
      if (!settings) return;
      try {
        await save({ ...settings, voice_auto_send: value });
      } catch (err) {
        chat.setError(errorMessage(err));
      }
    },
    [chat, save, settings],
  );

  const handleRegenerateMessage = useCallback(
    async (messageId: string) => {
      if (!ready) return;
      const index = chat.messages.findIndex(
        (message) => message.id === messageId,
      );
      const previousUser = index > 0 ? chat.messages[index - 1] : undefined;
      try {
        await chat.regenerate(
          messageId,
          buildOptions(previousUser?.content ?? "", {
            citations: chat.messages[index]?.citations,
          }),
        );
      } catch (err) {
        chat.setError(errorMessage(err));
      }
    },
    [buildOptions, chat, ready],
  );

  const handleEditMessage = useCallback(
    async (messageId: string, text: string) => {
      if (!ready) return;
      try {
        await chat.editAndResend(messageId, text, buildOptions(text));
      } catch (err) {
        chat.setError(errorMessage(err));
      }
    },
    [buildOptions, chat, ready],
  );

  // Everything the breakdown popover can pin or evict, grouped the way the
  // prompt is assembled. Items are sorted by token cost inside the popover.
  const contextCategories = useMemo<ContextBreakdownCategory[]>(() => {
    const flags = (id: string) => ({
      pinned: pinnedIds.has(id),
      evicted: evictedIds.has(id),
    });
    const projectItems: ContextBreakdownItem[] =
      projectSelection.fragments.map((fragment) => ({
        id: fragment.id,
        kind: "file",
        label: fragment.label,
        tokens: fragment.tokens,
        ...flags(fragment.id),
      }));
    const messageItems: ContextBreakdownItem[] = chat.messages
      .filter((message) => message.content.trim().length > 0)
      .map((message) => {
        const id = messageContextId(message.id);
        return {
          id,
          kind: "message",
          label: `${message.role === "user" ? "You" : "Janas"} · ${message.content.trim().slice(0, 64)}`,
          tokens: estimateTokens(message.content) + 8,
          detail: new Date(message.created_at).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          ...flags(id),
        };
      });
    const ragChunks = new Map<string, RagChunk>();
    for (const message of chat.messages) {
      for (const chunk of message.rag_sources ?? []) {
        if (!ragChunks.has(chunk.chunk_id)) ragChunks.set(chunk.chunk_id, chunk);
      }
    }
    const ragItems: ContextBreakdownItem[] = [...ragChunks.values()].map(
      (chunk) => {
        const id = `rag:${chunk.chunk_id}`;
        return {
          id,
          kind: "rag",
          label: `[${chunk.citation}] ${ragFileLabel(chunk)}`,
          tokens: chunk.token_estimate,
          detail: `lines ${chunk.start_line}–${chunk.end_line}`,
          ...flags(id),
        };
      },
    );
    const memoryItems: ContextBreakdownItem[] = memory.facts.map((fact) => {
      const id = `memory:${fact.id}`;
      return {
        id,
        kind: "memory",
        label: fact.text.slice(0, 72),
        tokens: estimateTokens(fact.text),
        detail: "recalled when relevant",
        ...flags(id),
      };
    });
    // Every enabled server that is either in this chat's scope or explicitly
    // evicted. Evicting removes it from the scope (a real prompt change) and
    // keeps the row so it can be restored from here.
    const mcpItems: ContextBreakdownItem[] = availableMcpServers
      .filter((server) => {
        const inScope =
          chat.mcpServerIds === undefined ||
          chat.mcpServerIds.includes(server.id);
        return inScope || evictedIds.has(`mcp:${server.id}`);
      })
      .map((server) => {
        const id = `mcp:${server.id}`;
        const evicted = evictedIds.has(id);
        const toolCount = server.toolCount;
        const toolLabel =
          toolCount === undefined
            ? "tool count unknown"
            : `${toolCount} tool${toolCount === 1 ? "" : "s"}`;
        return {
          id,
          kind: "mcp",
          label: server.name,
          // An evicted server is out of scope, so its schemas are not sent.
          tokens: evicted ? 0 : (server.toolSchemaTokens ?? 0),
          detail: evicted
            ? "evicted: tool schemas are not sent"
            : toolCount === undefined
              ? "not probed yet — probe it in MCP Hub to price its schemas"
              : pinnedIds.has(id)
                ? `${toolLabel} · pinned: schemas survive a tight budget`
                : `${toolLabel} · schemas are sent with every turn`,
          ...flags(id),
        };
      });
    const unprobedMcp = mcpItems.filter((item) => {
      const server = availableMcpServers.find(
        (candidate) => `mcp:${candidate.id}` === item.id,
      );
      return !item.evicted && server?.toolCount === undefined;
    }).length;
    return [
      { key: "file", label: "Attached project", items: projectItems },
      { key: "message", label: "Chat history", items: messageItems },
      { key: "rag", label: "Docs / RAG", items: ragItems },
      { key: "memory", label: "Memory", items: memoryItems },
      {
        key: "mcp",
        label: "MCP tools",
        note: `${mcpItems.length} server${mcpItems.length === 1 ? "" : "s"}${
          unprobedMcp ? ` · ${unprobedMcp} not probed` : ""
        }`,
        items: mcpItems,
      },
    ].filter((category) => category.items.length > 0);
  }, [
    availableMcpServers,
    chat.mcpServerIds,
    chat.messages,
    evictedIds,
    memory.facts,
    pinnedIds,
    projectSelection.fragments,
  ]);

  const pinnedTokens = useMemo(
    () =>
      contextCategories.reduce(
        (total, category) =>
          total +
          category.items
            .filter((item) => item.pinned)
            .reduce((sum, item) => sum + item.tokens, 0),
        0,
      ),
    [contextCategories],
  );

  // The ring follows pin/evict immediately. It mirrors the send budget instead
  // of waiting for the engine's usage report, which only arrives after a turn.
  const previewText =
    input.trim() ||
    [...chat.messages].reverse().find((message) => message.role === "user")
      ?.content ||
    "";
  const liveContextTokens = useMemo(() => {
    const parts = buildContextParts(previewText);
    const history = chat.messages
      .filter(
        (message) =>
          !evictedIds.has(messageContextId(message.id)) &&
          message.content.trim().length > 0,
      )
      .slice(-(Math.max(1, preferences.contextPrefetch) * 2))
      .reduce(
        (total, message) => total + estimateTokens(message.content) + 8,
        0,
      );
    const prompt = [
      parts.systemPrompt,
      parts.pinnedContext,
      parts.extraContext,
    ]
      .filter(Boolean)
      .join("\n\n");
    // In-scope MCP servers send their tool schemas with every request. Evicted
    // servers are out of scope, so they are not counted here.
    const mcpTokens = activeMcpServers.reduce(
      (total, server) => total + (server.toolSchemaTokens ?? 0),
      0,
    );
    return (
      estimateTokens(prompt) +
      history +
      estimateTokens(input) +
      images.reduce((total, image) => total + image.tokens, 0) +
      mcpTokens
    );
  }, [
    activeMcpServers,
    buildContextParts,
    chat.messages,
    evictedIds,
    imageTokens,
    images,
    input,
    preferences.contextPrefetch,
    previewText,
  ]);
  const contextUsed = chat.streaming
    ? (chat.usage?.total_tokens ?? liveContextTokens)
    : liveContextTokens;

  const handleToggleContextPin = useCallback(
    (item: ContextBreakdownItem) => {
      chat.toggleContextPin({
        id: item.id,
        kind: item.kind,
        label: item.label,
        tokens: item.tokens,
        added_at: Date.now(),
      });
    },
    [chat],
  );

  const handleToggleContextEvict = useCallback(
    (item: ContextBreakdownItem) => {
      if (item.kind !== "mcp") {
        chat.toggleContextEvict(item.id);
        return;
      }
      const serverId = item.id.startsWith("mcp:")
        ? item.id.slice("mcp:".length)
        : "";
      if (!serverId) return;
      const scope = chat.mcpServerIds;
      if (item.evicted) {
        // Restore. Decide from the marker, not from scope membership: a stale
        // marker must not turn a restore click into an eviction. An undefined
        // scope already includes every enabled server.
        const known = availableMcpServers.some(
          (server) => server.id === serverId,
        );
        if (scope !== undefined && known && !scope.includes(serverId)) {
          chat.setSessionMcpServerIds([...scope, serverId], mcpServers);
        }
        chat.toggleContextEvict(item.id);
        return;
      }
      // Evict: remove the server from this chat's scope and keep a marker so
      // the row can be restored. An undefined scope means "all enabled
      // servers", so this has to materialise the list first.
      chat.setSessionMcpServerIds(
        toggleServerInScope(
          scope,
          serverId,
          availableMcpServers.map((server) => server.id),
        ),
        mcpServers,
      );
      chat.toggleContextEvict(item.id);
    },
    [availableMcpServers, chat, mcpServers],
  );

  const handleRestoreEvictedContext = useCallback(() => {
    // A deleted server's marker is retired by reconciliation; restoring must
    // not re-add its id to the scope on the way out.
    const availableIds = new Set(
      availableMcpServers.map((server) => server.id),
    );
    const evictedMcp = (chat.activeSession?.evicted_context_ids ?? [])
      .filter((id) => id.startsWith("mcp:"))
      .map((id) => id.slice("mcp:".length))
      .filter((id) => availableIds.has(id));
    chat.restoreEvictedContext();
    if (!evictedMcp.length) return;
    const scope = chat.mcpServerIds;
    // An undefined scope already includes every enabled server.
    if (scope === undefined) return;
    chat.setSessionMcpServerIds(
      [...new Set([...scope, ...evictedMcp])],
      mcpServers,
    );
  }, [availableMcpServers, chat, mcpServers]);

  return (
    <div className="flex h-full min-h-0">
      <SessionsRail
        sessions={chat.sessions}
        activeId={chat.activeId}
        streamingSessionId={chat.streamingSessionId}
        collapsed={sessionsCollapsed}
        focusSearchSignal={focusSearchSignal}
        onFocusSearchHandled={onFocusSearchHandled}
        onToggle={() => setSessionsCollapsed((value) => !value)}
        onSelect={chat.setActiveId}
        onNew={chat.newChat}
        onDelete={(id) => void chat.deleteChat(id)}
        onRename={chat.renameChat}
        onTogglePin={chat.togglePin}
        onImport={chat.importChat}
        onExport={chat.exportChat}
        loading={chat.loading}
        memory={memory}
      />

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-white/5 px-4">
          <div className="flex min-w-0 items-center gap-2 text-[11px] text-slate-500">
            <Cpu className="h-3.5 w-3.5 shrink-0 text-slate-600" />
            <select
              value={selectedModelPath}
              disabled={switchingModel || engine.busy}
              onChange={(event) => chooseAnyModel(event.target.value)}
              className="max-w-[260px] truncate rounded-lg border border-white/10 bg-ink-900 px-2 py-1 text-[11px] text-slate-300 outline-none transition hover:border-white/20 focus:border-acid-400/40 disabled:opacity-50"
              title={selectedModelPath || "Select a local model"}
            >
              <option value="">Select a local model…</option>
              {models.length ? (
                <optgroup label="Janas (.jns)">
                  {models.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              {llamaModels.length ? (
                <optgroup label="llama.cpp (.gguf)">
                  {llamaModels.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              {selectedModelPath &&
              !models.some((model) => model.path === selectedModelPath) &&
              !llamaModels.some((model) => model.path === selectedModelPath) ? (
                <option value={selectedModelPath}>
                  {basename(selectedModelPath)} (custom)
                </option>
              ) : null}
            </select>
            <span className="text-slate-700">·</span>
            <span className="shrink-0">
              {chat.messages.filter((message) => message.role === "user").length}{" "}
              turns
            </span>
            {chat.finishReason ? (
              <>
                <span className="text-slate-700">·</span>
                <span className="shrink-0 font-mono text-[10px] text-slate-600">
                  {chat.finishReason}
                </span>
              </>
            ) : null}
            {projectBusy ? (
              <span className="flex shrink-0 items-center gap-1 text-acid-300/80">
                <Loader2 className="h-3 w-3 animate-spin" />
                ingesting
              </span>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            {artifactIndex.groups.length ? (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setArtifactPaneOpen((value) => !value)}
                icon={<PanelRight className="h-3.5 w-3.5" />}
                title="Toggle the live artifact workspace pane"
              >
                Artifacts
                <span className="ml-0.5 rounded border border-acid-400/25 bg-acid-400/10 px-1 py-px text-[8px] font-semibold text-acid-200/90">
                  {artifactIndex.groups.length}
                </span>
              </Button>
            ) : null}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void pickProject()}
              disabled={projectBusy}
              icon={<FolderOpen className="h-3.5 w-3.5" />}
            >
              Attach project
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setMcpOpen((value) => !value)}
              icon={<Terminal className="h-3.5 w-3.5" />}
              title={mcpScopeHint}
            >
              Tools
              {mcpScopeLabel ? (
                <span
                  className={cn(
                    "ml-0.5 rounded border px-1 py-px text-[8px] font-semibold uppercase tracking-wider",
                    mcpScopeLabel === "None"
                      ? "border-rose-400/25 bg-rose-400/10 text-rose-300/80"
                      : mcpScopeLabel === "All"
                        ? "border-white/10 bg-white/5 text-slate-500"
                        : "border-acid-400/25 bg-acid-400/10 text-acid-200/80",
                  )}
                >
                  {mcpScopeLabel}
                </span>
              ) : null}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void chat.clear()}
              disabled={!chat.messages.length}
              icon={<Trash2 className="h-3.5 w-3.5" />}
            >
              Clear
            </Button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-6 py-6">
            {!ready ? (
              <div className="mb-5 flex flex-wrap items-center gap-3 rounded-xl border border-amber-400/20 bg-amber-400/5 px-4 py-3">
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" />
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-amber-200">
                    {activeKind === null
                      ? "No engine selected."
                      : `The ${
                          activeKind === "llama" ? "llama.cpp" : "Janas"
                        } engine is not ready.`}
                  </p>
                  <p className="mt-0.5 truncate text-[10px] text-amber-200/60">
                    {activeKind === null
                      ? "Select an engine in the Engine panel, then start it."
                      : engine.snapshot.detail}
                  </p>
                </div>
                {activeKind === null ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setEngineCollapsed(false)}
                    icon={<Cpu className="h-3 w-3" />}
                  >
                    Select engine
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    variant="secondary"
                    busy={engine.busy}
                    onClick={() => void engine.start()}
                    icon={<Play className="h-3 w-3" />}
                  >
                    Start engine
                  </Button>
                )}
              </div>
            ) : null}

            {chat.loading ? (
              <div className="flex flex-col items-center gap-3 py-24 text-center">
                <Loader2 className="h-5 w-5 animate-spin text-slate-600" />
                <p className="text-[11px] text-slate-500">
                  Loading your chats from disk…
                </p>
              </div>
            ) : !chat.messages.length ? (
              <EmptyState
                modelName={modelName}
                ready={ready}
                onSuggestion={(prompt) => setInput(prompt)}
              />
            ) : (
              <div className="space-y-7">
                {chat.messages.map((message, index) => {
                  const siblings = chat.activeSession
                    ? siblingsFor(chat.activeSession, message)
                    : [message];
                  return (
                    <MessageBoundary
                      key={message.id}
                      text={message.content}
                      resetKey={message.id}
                    >
                    <MessageBubble
                      message={message}
                      streaming={
                        chat.streaming &&
                        index === chat.messages.length - 1 &&
                        message.role === "assistant"
                      }
                      onDelete={chat.deleteMessage}
                      onRegenerate={
                        message.role === "assistant" &&
                        index === chat.messages.length - 1
                          ? () => handleRegenerateMessage(message.id)
                          : undefined
                      }
                      onEdit={
                        message.role === "user"
                          ? (text) => handleEditMessage(message.id, text)
                          : undefined
                      }
                      regenerable={
                        message.role === "assistant" &&
                        index === chat.messages.length - 1 &&
                        !chat.streaming &&
                        !message.agentTrace &&
                        !agent.preferences.enabled
                      }
                      editable={
                        message.role === "user" &&
                        !chat.streaming &&
                        !agent.preferences.enabled
                      }
                      versionIndex={siblings.findIndex(
                        (sibling) => sibling.id === message.id,
                      )}
                      versionIds={siblings.map((sibling) => sibling.id)}
                      onSelectVersion={(siblingId) =>
                        chat.switchVersion(message.id, siblingId)
                      }
                      onBranch={
                        message.role === "user" && !chat.streaming
                          ? () => void chat.branchFromMessage(message.id)
                          : undefined
                      }
                      agentActions={agentActions}
                      onOpenRagSource={setSourceChunk}
                      artifacts={artifactIndex.byMessage[message.id]}
                      onOpenArtifact={openArtifact}
                      activeArtifactRootId={selectedArtifactRootId}
                      pinned={pinnedIds.has(messageContextId(message.id))}
                      evicted={evictedIds.has(messageContextId(message.id))}
                      onTogglePin={() => chat.toggleMessagePin(message.id)}
                      onToggleEvict={() => chat.toggleMessageEvict(message.id)}
                    />
                    </MessageBoundary>
                  );
                })}
              </div>
            )}
            <div ref={endRef} className="h-2" />
          </div>
        </div>

        {chat.toolApproval ? (
          <div className="shrink-0 px-6 pt-3">
            <div className="mx-auto max-w-3xl">
              <ToolApprovalCard
                approval={chat.toolApproval}
                onDecide={chat.resolveToolApproval}
              />
            </div>
          </div>
        ) : null}

        <Composer
          value={input}
          onChange={setInput}
          onSend={() => void handleSend()}
          onStop={() => void chat.stop()}
          streaming={chat.busy}
          ready={ready}
          contextUsed={contextUsed}
          contextMax={contextMax}
          pinnedTokens={pinnedTokens}
          contextCategories={contextCategories}
          onToggleContextPin={handleToggleContextPin}
          onToggleContextEvict={handleToggleContextEvict}
          onRestoreEvictedContext={handleRestoreEvictedContext}
          internet={preferences.internet}
          onInternetChange={(value) => update("internet", value)}
          agentMode={agent.preferences.enabled}
          onAgentModeChange={(value) => agent.update("enabled", value)}
          images={images}
          onAddImages={(next) => chat.setDraftImages(next)}
          onRemoveImage={(index) =>
            chat.setDraftImages(
              images.filter((_, imageIndex) => imageIndex !== index),
            )
          }
          projectName={project?.name ?? null}
          projectTokens={project?.tokens ?? 0}
          workspaceRoot={workspaceRoot}
          workspaceLocked={workspaceLocked}
          onPickProject={() => void pickProject()}
          onClearProject={() => chat.setSessionProject(null)}
          activity={activity}
          docsCount={attachedCollectionIds.length}
          docsLabel={docsScopeLabel}
          onOpenDocs={() => setDocsOpen(true)}
          ragNotice={chat.ragNotice}
          onDismissRagNotice={chat.dismissRagNotice}
          error={chat.error ?? modelError}
          onDismissError={() => {
            chat.setError(null);
            setModelError(null);
          }}
          voice={{
            enabled: Boolean(settings?.voice_input_enabled),
            ready: Boolean(
              whisperStatus?.installed && whisperStatus.active_model,
            ),
            status: whisperStatus,
            activation:
              settings?.voice_activation === "hold" ? "hold" : "toggle",
            autoSend: Boolean(settings?.voice_auto_send),
            stripNonSpeech: settings?.voice_strip_non_speech !== false,
            onAutoSendChange: (value) => void setVoiceAutoSend(value),
            onTranscript: insertVoiceTranscript,
            onAutoSendTranscript: sendVoiceTranscript,
            onOpenSettings,
          }}
          visionWarning={
            visionGuard
              ? {
                  modelName: visionGuard.main.file_name,
                  family: visionGuard.main.vision_family,
                  candidateName:
                    visionGuard.vision.recommended?.file_name ?? null,
                  canDownload: Boolean(visionGuard.repo),
                  busy: visionBusy,
                  onPair: () => void pairVisionProjector(),
                  onDownload: () => downloadVisionProjector(),
                }
              : null
          }
        />

        <TelemetryPanel
          collapsed={telemetryCollapsed}
          onToggle={() => setTelemetryCollapsed((value) => !value)}
          pid={engine.snapshot.pid}
          benchmark={benchmark}
          benchmarkBusy={benchmarkBusy}
          onBenchmark={() => void runBenchmark()}
          tokensGenerated={
            chat.usage?.completion_tokens ?? Math.round(chat.liveBytes / 4)
          }
          streaming={chat.busy}
        />
      </section>

      <ArtifactPane
        open={artifactPaneOpen}
        index={artifactIndex}
        selectedRootId={selectedArtifactRootId}
        selectedVersionId={selectedArtifactVersionId}
        onSelectArtifact={(rootId, versionId) => {
          if (!chat.activeId) return;
          setArtifactSelection((previous) => ({
            ...previous,
            [chat.activeId]: { rootId, versionId },
          }));
        }}
        onOpen={() => setArtifactPaneOpen(true)}
        onClose={() => setArtifactPaneOpen(false)}
        workspaceRoot={workspaceRoot}
        onError={chat.setError}
      />

      <EnginePanel
        collapsed={engineCollapsed}
        onToggle={() => setEngineCollapsed((value) => !value)}
        preferences={preferences}
        onUpdate={update}
        settings={settings}
        activeEngine={activeKind}
        engineTelemetry={engine.telemetry}
        engineRunning={engine.snapshot.pid !== null}
        engineBusy={engine.busy}
        engineStatus={engine.snapshot.status}
        onSwitchEngine={switchEngine}
        onApplyEngineSettings={applyEngineSettings}
        onRestartEngine={restartEngine}
        onReset={resetEnginePanel}
        agentPreferences={agent.preferences}
        onAgentUpdate={agent.update}
        onPickWorkspace={pickWorkspace}
        onClearWorkspace={clearWorkspace}
        workspaceLocked={workspaceLocked}
        systemPrompt={systemPrompt}
        onSystemPromptChange={setSystemPrompt}
        explicitArtifactTags={explicitArtifactTags}
        onExplicitArtifactTagsChange={setExplicitArtifactTags}
        usingDefaultPrompt={usingDefaultPrompt}
        onUseDefaultPrompt={() => chat.setSessionSystemPrompt(null)}
        onMakeDefaultPrompt={() => setDefaultSystemPrompt(systemPrompt)}
        personas={personas.personas}
        onAddPersona={personas.add}
        onDeletePersona={personas.remove}
        workspaceRoot={workspaceRoot}
        approvalRulesVersion={approvalRulesVersion}
        usage={chat.usage}
        elapsedMs={chat.elapsedMs}
        firstTokenMs={chat.firstTokenMs}
        liveBytes={chat.liveBytes}
        streaming={chat.streaming}
        startedAtMs={chat.startedAtMs}
      />

      <DocsDrawer
        open={docsOpen}
        onClose={() => setDocsOpen(false)}
        collections={collections}
        loading={collectionsLoading}
        error={collectionsError}
        selectedIds={attachedCollectionIds}
        onChange={chat.setSessionCollectionIds}
        settings={ragSettings}
        onSettings={(next) => chat.setSessionRagTuning(next)}
        contextWindow={contextMax}
        maxReply={preferences.maxTokens}
        promptTokensInUse={contextUsed}
        onOpenVault={() => {
          setDocsOpen(false);
          onOpenKnowledge();
        }}
        onRefresh={() => void refreshCollections()}
      />

      <RagSourceDrawer chunk={sourceChunk} onClose={() => setSourceChunk(null)} />

      <McpLogDrawer
        open={mcpOpen}
        onClose={() => setMcpOpen(false)}
        servers={mcpServers.filter(
          (server) => server.enabled && server.command.trim(),
        )}
        selectedIds={chat.mcpServerIds}
        onChangeScope={(ids) =>
          chat.setSessionMcpServerIds(ids, mcpServers)
        }
      />
    </div>
  );
}

function EmptyState({
  modelName,
  ready,
  onSuggestion,
}: {
  modelName: string;
  ready: boolean;
  onSuggestion: (prompt: string) => void;
}) {
  return (
    <div className="flex flex-col items-center py-10 text-center">
      <div className="relative">
        <div className="absolute inset-0 -z-10 rounded-full bg-acid-400/10 blur-2xl" />
        <Logo className="h-16 w-16" />
      </div>
      <h2 className="mt-6 text-xl font-semibold tracking-tight text-white">
        Your own inference bench
      </h2>
      <p className="mt-2 max-w-md text-[12.5px] leading-6 text-slate-500">
        {ready
          ? `${modelName} is loaded and listening. Every panel collapses; the centre stays pure text.`
          : "Select a model above to start its engine. Everything runs locally through the selected engine."}
      </p>

      <div className="mt-8 grid w-full max-w-2xl gap-2 sm:grid-cols-3">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.title}
            type="button"
            onClick={() => onSuggestion(suggestion.prompt)}
            disabled={!ready}
            className="group rounded-xl border border-white/10 bg-white/[0.02] p-3.5 text-left transition hover:border-acid-400/25 hover:bg-white/[0.04] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <p className="text-[11px] font-medium text-slate-300 group-hover:text-white">
              {suggestion.title}
            </p>
            <p className="mt-1 line-clamp-3 text-[10px] leading-4 text-slate-600">
              {suggestion.prompt}
            </p>
          </button>
        ))}
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-[10px] text-slate-600">
        {[
          "Prompt / reply ticker",
          "Context ring",
          "Web + citations",
          "Local memory",
          "SSD telemetry",
          "Expert heatmap",
        ].map((chip) => (
          <span
            key={chip}
            className="rounded-full border border-white/5 bg-white/[0.02] px-2.5 py-1"
          >
            {chip}
          </span>
        ))}
      </div>
    </div>
  );
}
