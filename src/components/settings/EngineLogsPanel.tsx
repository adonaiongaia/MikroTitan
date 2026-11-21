import { useEffect, useRef, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { Check, ClipboardCopy, Copy, Terminal, Trash2 } from "lucide-react";

import type { EngineController } from "../../hooks/useEngine";
import { useSettings } from "../../hooks/useSettings";
import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage } from "../../lib/format";
import type { EngineLog } from "../../types";
import { Button } from "../ui/Button";

type Filter = "all" | "stdout" | "stderr";

function timeLabel(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function logLine(log: EngineLog): string {
  return `[${timeLabel(log.timestamp_ms)}] ${log.stream}: ${log.line}`;
}

/**
 * The Rust supervisor already forwards every engine line as `engine://log`;
 * nothing in the webview used to read it. This panel is the difference between
 * "the process exited" and an actionable stderr trace.
 */
export function EngineLogsPanel({ engine }: { engine: EngineController }) {
  const { settings } = useSettings();
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const visible = engine.logs.filter((log) => {
    if (filter !== "all" && log.stream !== filter) return false;
    const needle = query.trim().toLowerCase();
    return needle ? log.line.toLowerCase().includes(needle) : true;
  });

  useEffect(() => {
    if (autoScroll) endRef.current?.scrollIntoView({ block: "end" });
  }, [visible.length, autoScroll]);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(null), 1_800);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const copyText = async (text: string, label: string) => {
    setError(null);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
    } catch (err) {
      setError(`Clipboard unavailable: ${errorMessage(err)}`);
    }
  };

  const copyLogs = () =>
    copyText(visible.map(logLine).join("\n"), "logs");

  const copyDiagnostics = async () => {
    setError(null);
    let appVersion = "unknown";
    try {
      appVersion = await getVersion();
    } catch {
      // The app API may be unavailable in a stripped-down build.
    }
    let setup = "unavailable";
    try {
      setup = JSON.stringify(await api.getEngineSetupStatus(), null, 2);
    } catch (err) {
      setup = errorMessage(err);
    }
    const redactedSettings = settings
      ? {
          ...settings,
          api_key: settings.api_key ? "[redacted]" : "",
          hf_token: settings.hf_token ? "[redacted]" : "",
        }
      : null;
    const text = [
      "Studio for Janas diagnostics",
      `app_version: ${appVersion}`,
      `snapshot: ${JSON.stringify(engine.snapshot, null, 2)}`,
      `active_engine: ${JSON.stringify(engine.active, null, 2)}`,
      `adapter_warning: ${JSON.stringify(engine.adapterWarning, null, 2)}`,
      "settings:",
      JSON.stringify(redactedSettings, null, 2),
      "engine_setup:",
      setup,
      `engine_logs (last ${Math.min(visible.length, 200)}):`,
      visible.slice(-200).map(logLine).join("\n"),
    ].join("\n");
    await copyText(text, "diagnostics");
  };

  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-1.5 text-[12px] font-semibold text-white">
            <Terminal className="h-3.5 w-3.5 text-slate-500" />
            Engine logs
          </h3>
          <p className="mt-1 text-[10px] leading-4 text-slate-600">
            stdout, stderr and adapter warnings from the running child. The
            buffer keeps the newest 500 lines.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void copyLogs()}
            disabled={!visible.length}
            icon={
              copied === "logs" ? (
                <Check className="h-3 w-3 text-emerald-400" />
              ) : (
                <Copy className="h-3 w-3" />
              )
            }
          >
            Copy logs
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void copyDiagnostics()}
            icon={
              copied === "diagnostics" ? (
                <Check className="h-3 w-3 text-emerald-400" />
              ) : (
                <ClipboardCopy className="h-3 w-3" />
              )
            }
          >
            Copy diagnostics
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              engine.clearLogs();
              setError(null);
            }}
            disabled={!engine.logs.length}
            icon={<Trash2 className="h-3 w-3" />}
          >
            Clear
          </Button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg border border-white/10 bg-ink-950/60 p-0.5">
          {(["all", "stdout", "stderr"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setFilter(value)}
              className={cn(
                "rounded-md px-2 py-1 text-[9px] font-medium uppercase tracking-wider transition",
                filter === value
                  ? "bg-white/[0.08] text-slate-200"
                  : "text-slate-600 hover:text-slate-400",
              )}
            >
              {value}
            </button>
          ))}
        </div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter lines…"
          className="field !w-48 !py-1.5 text-[10.5px]"
        />
        <label className="flex items-center gap-1.5 text-[9.5px] text-slate-500">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(event) => setAutoScroll(event.target.checked)}
            className="accent-acid-400"
          />
          Follow
        </label>
        <span className="ml-auto font-mono text-[9px] text-slate-600">
          {visible.length} / {engine.logs.length}
        </span>
      </div>

      {error ? (
        <p className="mt-2 text-[10px] text-rose-300">{error}</p>
      ) : null}

      {engine.adapterWarning ? (
        <div className="mt-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-2.5 py-2 text-[10px] leading-4 text-amber-200/80">
          Adapter warning: {engine.adapterWarning.message}
          {engine.adapterWarning.version
            ? ` (${engine.adapterWarning.version})`
            : ""}
        </div>
      ) : null}

      <div className="mt-2 max-h-80 overflow-y-auto rounded-lg border border-white/5 bg-ink-950/70 px-3 py-2">
        {visible.length ? (
          <div className="space-y-0.5 font-mono text-[9.5px] leading-4">
            {visible.map((log, index) => (
              <p
                key={`${log.timestamp_ms}-${index}`}
                className={cn(
                  "whitespace-pre-wrap break-words",
                  log.stream === "stderr"
                    ? "text-rose-300/80"
                    : "text-slate-500",
                )}
              >
                <span className="mr-1.5 text-slate-700">
                  {timeLabel(log.timestamp_ms)}
                </span>
                <span
                  className={cn(
                    "mr-1.5 uppercase",
                    log.stream === "stderr"
                      ? "text-rose-400/60"
                      : "text-slate-700",
                  )}
                >
                  {log.stream}
                </span>
                {log.line}
              </p>
            ))}
            <div ref={endRef} />
          </div>
        ) : (
          <p className="py-4 text-center text-[10px] text-slate-600">
            {engine.logs.length
              ? "No line matches that filter."
              : "No engine output yet. Start the engine to capture its logs."}
          </p>
        )}
      </div>
    </section>
  );
}
