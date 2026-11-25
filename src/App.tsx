import { useEffect, useRef, useState } from "react";
import { Loader2, Play, Square, X } from "lucide-react";

import { BenchmarkView } from "./components/benchmark/BenchmarkView";
import { useBenchmark } from "./components/benchmark/useBenchmark";
import { ChatView } from "./components/chat/ChatView";
import { ConverterView } from "./components/converter/ConverterView";
import { KnowledgeVaultView } from "./components/knowledge/KnowledgeVaultView";
import { McpView } from "./components/mcp/McpView";
import { DownloadsTray } from "./components/models/DownloadsTray";
import { ModelLibraryView } from "./components/models/ModelLibraryView";
import { ModelsView } from "./components/models/ModelsView";
import { SettingsView } from "./components/settings/SettingsView";
import { Sidebar } from "./components/Sidebar";
import { Button } from "./components/ui/Button";
import { StatusPill } from "./components/ui/StatusPill";
import { useChat } from "./hooks/useChat";
import { useConverter } from "./hooks/useConverter";
import { useDownloads } from "./hooks/useDownloads";
import { useEngine } from "./hooks/useEngine";
import { api } from "./lib/api";
import { engineLabel } from "./lib/engine-controller";
import { errorMessage } from "./lib/format";
import type { ViewId } from "./types";

const HEADINGS: Record<ViewId, { title: string; subtitle: string }> = {
  chat: {
    title: "Chat Workspace",
    subtitle: "Streaming local inference over Janas or llama.cpp",
  },
  models: {
    title: "Model Explorer",
    subtitle: "Discover GGUF builds on the Hugging Face Hub",
  },
  library: {
    title: "Model Library",
    subtitle: "The models on this machine, their architecture and quantization",
  },
  compiler: {
    title: "GGUF → JNS Compiler",
    subtitle: "Rewrite GGUF weights into Janas's native format",
  },
  benchmark: {
    title: "Side-by-Side Bench",
    subtitle: "Engine and quantization trade-offs on this exact machine",
  },
  knowledge: {
    title: "Knowledge Vault",
    subtitle: "Persistent local document collections for retrieval",
  },
  mcp: {
    title: "MCP Server Hub",
    subtitle: "Register, probe and inspect local tool servers",
  },
  settings: {
    title: "Settings",
    subtitle: "Runtime binaries, model paths, and the local server",
  },
};

export default function App() {
  const engine = useEngine();
  // Long-lived async state lives above the tabs so switching views never
  // interrupts a stream, a conversion, or an in-flight model download.
  const chat = useChat();
  const converter = useConverter();
  const downloads = useDownloads();
  const benchmark = useBenchmark();
  const [view, setView] = useState<ViewId>("chat");
  const [toast, setToast] = useState<string | null>(null);
  const closing = useRef(false);

  // A debounced save can still be waiting when the window closes. Flush the
  // dirty sessions on blur/visibility and, for Tauri, wait for the write before
  // letting the close request through.
  useEffect(() => {
    const flush = () => {
      void chat.flushAll();
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") flush();
    };
    window.addEventListener("beforeunload", flush);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("beforeunload", flush);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [chat.flushAll]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => {
        const window = getCurrentWindow();
        return window.onCloseRequested(async (event) => {
          if (closing.current) return;
          closing.current = true;
          event.preventDefault();
          try {
            // close() can re-enter this listener on some Linux window
            // managers. Give the flush a short budget, then destroy the
            // window so a failed save can never trap the app open.
            await Promise.race([
              chat.flushAll(),
              new Promise<void>((resolve) =>
                globalThis.setTimeout(resolve, 1_500),
              ),
            ]);
          } finally {
            try {
              await window.destroy();
            } catch {
              await window.close();
            }
          }
        });
      })
      .then((stop) => {
        if (disposed) stop();
        else unlisten = stop;
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [chat.flushAll]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4_000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const useModel = async (path: string) => {
    const isGguf = path.toLowerCase().endsWith(".gguf");
    try {
      const current = await api.getSettings();
      // Activating a library model re-applies its saved bundle in the same
      // click: the projector and draft model follow the main weights instead
      // of whatever the previous model left behind.
      const bundle = isGguf
        ? current.model_bundles.find(
            (candidate) =>
              candidate.main_model_path.trim().toLowerCase() ===
              path.trim().toLowerCase(),
          )
        : undefined;
      await api.saveSettings(
        isGguf
          ? {
              ...current,
              llama_model_path: path,
              ...(bundle
                ? {
                    llama_mmproj_path: bundle.mmproj_path,
                    llama_draft_path: bundle.draft_model_path,
                    llama_draft_max: bundle.draft_max_tokens,
                    llama_draft_min: bundle.draft_min_tokens,
                    llama_draft_p_min: bundle.draft_p_min,
                    llama_mtp_mode: bundle.draft_model_path
                      ? "auto"
                      : current.llama_mtp_mode,
                    llama_mtp_path: bundle.draft_model_path
                      ? ""
                      : current.llama_mtp_path,
                    active_bundle_id: bundle.id,
                  }
                : {}),
            }
          : { ...current, model_path: path },
      );
      if (engine.snapshot.pid !== null) await engine.stop();
      await engine.switchTo(isGguf ? "llama" : "janas");
      setView("chat");
      setToast(
        isGguf
          ? "GGUF selected. llama.cpp is loading it."
          : "Model selected. Janas is loading it.",
      );
    } catch (err) {
      setToast(errorMessage(err));
    }
  };

  const running = engine.snapshot.status !== "disconnected";
  const heading = HEADINGS[view];

  return (
    <div className="flex h-screen overflow-hidden bg-ink-950 text-slate-200">
      <Sidebar view={view} onNavigate={setView} engine={engine} />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[68px] shrink-0 items-center justify-between gap-4 border-b border-white/5 bg-ink-900/30 px-6 backdrop-blur">
          <div className="min-w-0">
            <h1 className="truncate text-[15px] font-semibold tracking-tight text-white">
              {heading.title}
            </h1>
            <p className="mt-0.5 truncate text-[10.5px] text-slate-500">
              {heading.subtitle}
              {engine.active ? ` · ${engineLabel(engine.active.kind)} active` : ""}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-3">
            <StatusPill
              status={engine.snapshot.status}
              detail={engine.snapshot.detail}
            />
            <Button
              size="sm"
              variant={running ? "secondary" : "primary"}
              busy={engine.busy}
              onClick={() => {
                if (running) void engine.stop();
                else void engine.start();
              }}
              icon={
                engine.busy ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : running ? (
                  <Square className="h-3 w-3" />
                ) : (
                  <Play className="h-3 w-3" />
                )
              }
            >
              {running ? "Stop engine" : "Start engine"}
            </Button>
          </div>
        </header>

        {engine.error ? (
          <div className="flex shrink-0 items-start gap-2 border-b border-rose-400/15 bg-rose-500/5 px-6 py-2.5 text-[10.5px] leading-5 text-rose-300">
            <span className="flex-1">{engine.error}</span>
            <button
              type="button"
              onClick={engine.clearError}
              className="text-rose-400/60 transition hover:text-rose-200"
              aria-label="Dismiss engine error"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : null}

        <div className="min-h-0 flex-1 overflow-hidden">
          {view === "chat" ? (
            <ChatView
              engine={engine}
              chat={chat}
              onOpenKnowledge={() => setView("knowledge")}
              onOpenSettings={() => setView("settings")}
            />
          ) : null}
          {view === "models" ? (
            <ModelsView
              downloads={downloads.downloads}
              onStart={downloads.start}
              onCancel={downloads.cancel}
              onPause={downloads.pause}
              onResume={downloads.resume}
            />
          ) : null}
          {view === "library" ? (
            <ModelLibraryView onUseModel={useModel} />
          ) : null}
          {view === "compiler" ? (
            <ConverterView converter={converter} onUseModel={useModel} />
          ) : null}
          {view === "benchmark" ? (
            <BenchmarkView benchmark={benchmark} />
          ) : null}
          {view === "knowledge" ? <KnowledgeVaultView /> : null}
          {view === "mcp" ? <McpView /> : null}
          {view === "settings" ? <SettingsView engine={engine} /> : null}
        </div>
      </main>

      <DownloadsTray
        downloads={downloads.downloads}
        onCancel={(id) => void downloads.cancel(id)}
        onPause={(id) => void downloads.pause(id)}
        onResume={(id) => void downloads.resume(id)}
        onClearFinished={downloads.clearFinished}
      />

      {toast ? (
        <div className="fixed bottom-5 left-1/2 z-50 -translate-x-1/2 animate-fade-in rounded-xl border border-acid-400/25 bg-ink-850 px-4 py-2.5 text-[11px] text-slate-200 shadow-panel">
          {toast}
        </div>
      ) : null}
    </div>
  );
}
