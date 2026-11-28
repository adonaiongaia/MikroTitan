import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  RefreshCw,
  ShieldCheck,
  Terminal,
} from "lucide-react";

import { api, Events } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage } from "../../lib/format";
import type {
  AppSettings,
  GatewayStatus,
  KnowledgeCollection,
} from "../../types";
import { Button } from "../ui/Button";
import { Toggle } from "../ui/Toggle";

interface GatewayPanelProps {
  form: AppSettings;
  update: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  save: (next: AppSettings) => Promise<AppSettings>;
  onError: (message: string | null) => void;
}

/**
 * Settings card for the loopback OpenAI-compatible gateway. The enable switch
 * applies immediately (persist, then start/stop); the other switches save with
 * the form and are read by the gateway on the next request.
 */
export function GatewayPanel({
  form,
  update,
  save,
  onError,
}: GatewayPanelProps) {
  const [status, setStatus] = useState<GatewayStatus | null>(null);
  const [collections, setCollections] = useState<KnowledgeCollection[]>([]);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showSnippet, setShowSnippet] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.getGatewayStatus());
    } catch {
      // An older backend without the gateway keeps the card in its stopped
      // state instead of breaking the Settings view.
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void listen<GatewayStatus>(Events.gatewayStatus, (event) => {
      setStatus(event.payload);
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
  }, []);

  useEffect(() => {
    let disposed = false;
    void api
      .ragListCollections()
      .then((list) => {
        if (!disposed) setCollections(list);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, []);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 2_000);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const apiBase = status?.running
    ? status.api_base
    : `http://127.0.0.1:${form.gateway_port}/v1`;

  const setEnabled = async (enabled: boolean) => {
    setBusy(true);
    onError(null);
    try {
      update("gateway_enabled", enabled);
      await save({ ...form, gateway_enabled: enabled });
      setStatus(enabled ? await api.startGateway() : await api.stopGateway());
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const restart = async () => {
    setBusy(true);
    onError(null);
    try {
      await save(form);
      await api.stopGateway();
      setStatus(await api.startGateway());
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const copyBaseUrl = async () => {
    try {
      await navigator.clipboard.writeText(apiBase);
      setCopied(true);
    } catch (err) {
      onError(errorMessage(err));
    }
  };

  const toggleCollection = (id: string) => {
    const selected = form.gateway_default_rag_collections ?? [];
    update(
      "gateway_default_rag_collections",
      selected.includes(id)
        ? selected.filter((value) => value !== id)
        : [...selected, id],
    );
  };

  const apiKey = form.api_key.trim() ? "<your Local API key>" : "local";
  const model = status?.active_model?.trim() || "default";
  const snippet = `# Cline / Continue.dev / Cursor
baseURL: ${apiBase}
apiKey: ${apiKey}
model: ${model}

# OpenAI Python SDK
from openai import OpenAI
client = OpenAI(base_url="${apiBase}", api_key="${apiKey}")
`;

  return (
    <section className="panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-[12px] font-semibold text-white">
            Local API Gateway
          </h3>
          <p className="mt-1 text-[10px] leading-4 text-slate-600">
            One OpenAI-compatible endpoint for external IDEs and scripts. The
            gateway follows whichever engine MikroTitan has running and stays
            bound to 127.0.0.1.
          </p>
        </div>
        <span
          className={cn(
            "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-[9px] font-medium",
            status?.running
              ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
              : "border-white/10 bg-white/5 text-slate-500",
          )}
        >
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              status?.running ? "bg-emerald-400" : "bg-slate-600",
            )}
          />
          {status?.running ? "Running" : "Stopped"}
        </span>
      </div>

      <div className="mt-4 space-y-4">
        <Toggle
          checked={form.gateway_enabled}
          onChange={(value) => void setEnabled(value)}
          label="Enable local API gateway"
          description="Starts the loopback server now and again on app launch. No traffic leaves this machine."
          disabled={busy}
        />

        <div className="grid gap-2 border-t border-white/5 pt-4 md:grid-cols-[180px_minmax(0,1fr)] md:gap-5">
          <div>
            <p className="text-[11px] font-medium text-slate-300">Port</p>
            <p className="mt-0.5 text-[9.5px] leading-4 text-slate-600">
              Loopback only. Defaults to 11435 so it never collides with an
              engine port.
            </p>
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="number"
                min={1}
                max={65_535}
                value={form.gateway_port}
                onChange={(event) =>
                  update(
                    "gateway_port",
                    Math.min(65_535, Math.max(1, Number(event.target.value) || 1)),
                  )
                }
                className="field w-32 font-mono text-[11px]"
              />
              <Button
                size="sm"
                variant="secondary"
                busy={busy}
                onClick={() => void restart()}
                icon={<RefreshCw className="h-3.5 w-3.5" />}
              >
                Restart gateway
              </Button>
            </div>
            {status?.running && status.port !== form.gateway_port ? (
              <p className="mt-1.5 text-[9.5px] leading-4 text-amber-300/80">
                The running server is still on port {status.port}. Restart the
                gateway to move it.
              </p>
            ) : null}
          </div>
        </div>

        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-slate-300">
              {status?.running
                ? `Running on ${apiBase}`
                : "Stopped — enable the gateway to accept external requests"}
            </span>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void copyBaseUrl()}
              icon={
                copied ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <Copy className="h-3 w-3" />
                )
              }
            >
              {copied ? "Copied" : "Copy Base URL"}
            </Button>
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] text-slate-600">
            <span className="inline-flex items-center gap-1">
              <ShieldCheck className="h-3 w-3" />
              {form.api_key.trim()
                ? "Bearer key required"
                : "No API key configured"}
            </span>
            {status?.active_engine ? (
              <span>
                Engine: {status.active_engine === "llama" ? "llama.cpp" : "Janas"}
              </span>
            ) : null}
            {status?.active_model ? (
              <span className="truncate font-mono">{status.active_model}</span>
            ) : null}
            {status ? <span>{status.request_count} request(s)</span> : null}
          </div>
          {status?.last_error ? (
            <p className="mt-2 flex items-start gap-1.5 text-[9.5px] leading-4 text-rose-300/90">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              <span className="min-w-0 break-words">{status.last_error}</span>
            </p>
          ) : null}
          {status?.last_context_warning ? (
            <p className="mt-2 flex items-start gap-1.5 text-[9.5px] leading-4 text-amber-300/90">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              <span className="min-w-0 break-words">
                {status.last_context_warning}
              </span>
            </p>
          ) : null}
        </div>

        <Toggle
          checked={form.gateway_auto_switch_models}
          onChange={(value) => update("gateway_auto_switch_models", value)}
          label="Auto-switch models on request"
          description="When a request names a different local .gguf or .jns model, stop the current engine, load it, and wait until it is ready."
        />

        <Toggle
          checked={form.gateway_inject_memory}
          onChange={(value) => update("gateway_inject_memory", value)}
          label="Inject Persistent Memory"
          description="Prepend up to 40 Memory Bank facts, tagged [M1]…[Mn], to the system prompt of external requests."
        />

        <div className="border-t border-white/5 pt-4">
          <p className="text-[11px] font-medium text-slate-300">
            Default Document Vault collections
          </p>
          <p className="mt-0.5 text-[9.5px] leading-4 text-slate-600">
            Opt-in collections whose chunks are retrieved as [D1], [D2], … for
            the latest user message. Retrieval is budgeted against the running
            engine&apos;s context window.
          </p>
          {collections.length ? (
            <div className="mt-2 max-h-36 space-y-1 overflow-y-auto pr-1">
              {collections.map((collection) => {
                const checked = (
                  form.gateway_default_rag_collections ?? []
                ).includes(collection.id);
                return (
                  <label
                    key={collection.id}
                    className="flex cursor-pointer items-center gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-1.5 hover:bg-white/[0.04]"
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleCollection(collection.id)}
                      className="h-3.5 w-3.5 accent-acid-400"
                    />
                    <span className="min-w-0 flex-1 truncate text-[10.5px] text-slate-300">
                      {collection.name}
                    </span>
                    <span className="shrink-0 text-[9px] text-slate-600">
                      {collection.chunk_count} chunks
                    </span>
                  </label>
                );
              })}
            </div>
          ) : (
            <p className="mt-2 rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-2 text-[10px] text-slate-600">
              No Document Vault collections yet. Create one in the Knowledge
              view to expose it here.
            </p>
          )}
        </div>

        <div className="border-t border-white/5 pt-4">
          <button
            type="button"
            onClick={() => setShowSnippet((value) => !value)}
            className="flex w-full items-center gap-2 text-left"
          >
            {showSnippet ? (
              <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-slate-500" />
            )}
            <span className="flex items-center gap-1.5 text-[11px] font-medium text-slate-300">
              <Terminal className="h-3.5 w-3.5 text-slate-500" />
              Client Setup Snippet
            </span>
          </button>
          {showSnippet ? (
            <pre className="mt-2 overflow-x-auto rounded-lg border border-white/5 bg-black/30 p-3 font-mono text-[10px] leading-5 text-slate-300">
              {snippet}
            </pre>
          ) : null}
        </div>
      </div>
    </section>
  );
}
