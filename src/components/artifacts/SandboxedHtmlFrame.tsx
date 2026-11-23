import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Loader2, Monitor, ShieldCheck, ShieldOff, Smartphone } from "lucide-react";

import { isTauriRuntime } from "../../lib/artifactExport";
import { cn } from "../../lib/cn";
import { errorMessage } from "../../lib/format";

export interface ArtifactConsoleEntry {
  id: string;
  level: "error" | "warn";
  message: string;
  source?: string;
  line?: number | null;
  at: number;
}

interface SandboxedHtmlFrameProps {
  html: string;
  title: string;
  streaming?: boolean;
  onConsole: (entry: ArtifactConsoleEntry) => void;
}

/**
 * The bridge runs inside the sandbox before any model script. It reports
 * runtime errors and console failures to the pane footer, but it has no access
 * to the parent DOM: the frame is sandboxed without allow-same-origin.
 */
const CONSOLE_BRIDGE = `<script>(function(){
  function report(kind, payload) {
    try {
      parent.postMessage(Object.assign({ __janasArtifact: true, kind: kind }, payload), "*");
    } catch (error) { /* the parent may be gone while the pane closes */ }
  }
  function text(value) {
    if (value instanceof Error) return value.message || String(value);
    if (typeof value === "string") return value;
    try { return JSON.stringify(value); } catch (error) { return String(value); }
  }
  window.addEventListener("error", function (event) {
    report("error", {
      message: event.message || "Script error",
      source: event.filename || "",
      line: typeof event.lineno === "number" ? event.lineno : null
    });
  });
  window.addEventListener("unhandledrejection", function (event) {
    var reason = event.reason;
    report("error", {
      message: "Unhandled rejection: " + text(reason && reason.message ? reason.message : reason)
    });
  });
  ["error", "warn"].forEach(function (level) {
    var original = console[level];
    console[level] = function () {
      var message = Array.prototype.map.call(arguments, text).join(" ");
      report("console", { level: level, message: message });
      if (typeof original === "function") original.apply(console, arguments);
    };
  });
})();</script>`;

export const STRICT_ARTIFACT_CSP = [
  "default-src 'none'",
  "script-src 'unsafe-inline'",
  "style-src 'unsafe-inline'",
  "img-src data: blob:",
  "font-src data:",
  "media-src data: blob:",
  "connect-src 'none'",
  "form-action 'none'",
  "base-uri 'none'",
  "object-src 'none'",
  "worker-src 'none'",
].join("; ");

export const NETWORK_ARTIFACT_CSP = [
  "default-src 'none'",
  "script-src 'unsafe-inline' https: http:",
  "style-src 'unsafe-inline' https: http:",
  "img-src data: blob: https: http:",
  "font-src data: https: http:",
  "media-src data: blob: https: http:",
  "connect-src https: http:",
  "form-action 'none'",
  "base-uri 'none'",
  "object-src 'none'",
  "worker-src 'none'",
].join("; ");

/**
 * Build a complete document. The CSP meta repeats the custom-scheme header and
 * is the primary policy in the browser fallback; the frame's opaque origin and
 * the parent's Tauri CSP remain in force regardless.
 */
export function buildArtifactDocument(
  html: string,
  allowNetwork: boolean,
): string {
  const csp = allowNetwork ? NETWORK_ARTIFACT_CSP : STRICT_ARTIFACT_CSP;
  const injection =
    `<meta charset="utf-8">` +
    `<meta http-equiv="Content-Security-Policy" content="${csp}">` +
    CONSOLE_BRIDGE;
  if (/<head[^>]*>/i.test(html)) {
    return html.replace(/<head([^>]*)>/i, `<head$1>${injection}`);
  }
  if (/<html[^>]*>/i.test(html)) {
    return html.replace(/<html([^>]*)>/i, `<html$1><head>${injection}</head>`);
  }
  return `<!doctype html><html><head>${injection}</head><body>${html}</body></html>`;
}

const IS_WINDOWS =
  typeof navigator !== "undefined" && /Windows/i.test(navigator.userAgent);

/**
 * Tauri serves the registered document from this scheme with its own CSP
 * header. A `srcdoc` document would inherit the app's `script-src 'self'` and
 * could never run a generated script, so the packaged app uses a real origin
 * that is still sandboxed into an opaque one by the iframe attribute.
 */
export function artifactFrameUrl(token: string): string {
  const origin = IS_WINDOWS
    ? "http://janas-artifact.localhost"
    : "janas-artifact://localhost";
  // Network permission lives in the token's server-side record. The frame
  // cannot add a query parameter to grant itself network access.
  return `${origin}/frame/${token}`;
}

export function SandboxedHtmlFrame({
  html,
  title,
  streaming = false,
  onConsole,
}: SandboxedHtmlFrameProps) {
  const [viewport, setViewport] = useState<"desktop" | "mobile">("desktop");
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [rendered, setRendered] = useState(html);
  const [frameToken, setFrameToken] = useState<string | null>(null);
  const [frameError, setFrameError] = useState<string | null>(null);
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const tauri = isTauriRuntime();

  // A live stream changes the document several times a second. Debouncing
  // keeps the frame readable instead of reloading it on every token, and the
  // final document is applied immediately when the fence closes.
  useEffect(() => {
    if (!streaming) {
      setRendered(html);
      return;
    }
    const timer = window.setTimeout(() => setRendered(html), 450);
    return () => window.clearTimeout(timer);
  }, [html, streaming]);

  const documentMarkup = useMemo(
    () => buildArtifactDocument(rendered, allowNetwork),
    [allowNetwork, rendered],
  );

  // Only the packaged app needs the custom scheme. Registering the exact
  // document the pane is showing lets Rust apply its per-request CSP header.
  useEffect(() => {
    if (!tauri) return;
    let cancelled = false;
    setFrameError(null);
    void invoke<string>("artifact_register_frame", {
      html: documentMarkup,
      allow_network: allowNetwork,
    })
      .then((token) => {
        if (!cancelled) setFrameToken(token);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setFrameError(errorMessage(error));
          setFrameToken(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [allowNetwork, documentMarkup, tauri]);

  const frameSrc = useMemo(
    () => (tauri && frameToken ? artifactFrameUrl(frameToken) : null),
    [frameToken, tauri],
  );

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      if (event.source !== frameRef.current?.contentWindow) return;
      const data = event.data as Record<string, unknown> | null;
      if (!data || data.__janasArtifact !== true) return;
      if (data.kind !== "error" && data.kind !== "console") return;
      const message =
        typeof data.message === "string" ? data.message.slice(0, 4_000) : "";
      if (!message) return;
      onConsole({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        level: data.level === "warn" ? "warn" : "error",
        message,
        source:
          typeof data.source === "string"
            ? data.source.slice(0, 500)
            : undefined,
        line: typeof data.line === "number" ? data.line : null,
        at: Date.now(),
      });
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [onConsole]);

  const showFrame = !tauri || Boolean(frameSrc);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-white/5 px-2 py-1.5">
        <div className="flex items-center rounded-lg border border-white/10 bg-white/[0.03] p-0.5">
          <button
            type="button"
            onClick={() => setViewport("desktop")}
            aria-pressed={viewport === "desktop"}
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] transition",
              viewport === "desktop"
                ? "bg-acid-400/15 text-acid-200"
                : "text-slate-500 hover:text-slate-300",
            )}
            title="Desktop width"
          >
            <Monitor className="h-2.5 w-2.5" />
            Desktop
          </button>
          <button
            type="button"
            onClick={() => setViewport("mobile")}
            aria-pressed={viewport === "mobile"}
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] transition",
              viewport === "mobile"
                ? "bg-acid-400/15 text-acid-200"
                : "text-slate-500 hover:text-slate-300",
            )}
            title="Mobile width, 390 px"
          >
            <Smartphone className="h-2.5 w-2.5" />
            Mobile
          </button>
        </div>
        <span className="ml-auto flex items-center gap-1.5">
          {streaming ? (
            <span className="flex items-center gap-1 text-[9px] text-acid-300/80">
              <span className="h-1 w-1 animate-pulse-soft rounded-full bg-acid-400" />
              live
            </span>
          ) : null}
          <button
            type="button"
            onClick={() => {
              setAllowNetwork((value) => !value);
              // Drop the old frame immediately so a network-enabled preview
              // cannot keep running while the strict replacement registers.
              setFrameToken(null);
            }}
            aria-pressed={allowNetwork}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[9px] transition",
              allowNetwork
                ? "border-amber-400/30 bg-amber-400/10 text-amber-200"
                : "border-white/10 bg-white/[0.02] text-slate-500 hover:text-slate-300",
            )}
            title={
              allowNetwork
                ? "Network access is allowed inside this preview. The webview CSP may still block it."
                : "External network requests are blocked inside this preview."
            }
          >
            {allowNetwork ? (
              <ShieldOff className="h-2.5 w-2.5" />
            ) : (
              <ShieldCheck className="h-2.5 w-2.5" />
            )}
            {allowNetwork ? "Network on" : "Network off"}
          </button>
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto bg-ink-950/40 p-2">
        {showFrame ? (
          <iframe
            ref={frameRef}
            title={`${title} live preview`}
            sandbox="allow-scripts"
            referrerPolicy="no-referrer"
            // In Tauri the dedicated scheme carries its own CSP. Everywhere
            // else the srcdoc document itself carries the injected meta policy.
            src={frameSrc ?? undefined}
            srcDoc={frameSrc ? undefined : documentMarkup}
            className={cn(
              "h-full min-h-[220px] rounded-lg border border-white/10 bg-white shadow-inner",
              viewport === "mobile"
                ? "mx-auto w-[390px] max-w-full"
                : "w-full",
            )}
          />
        ) : (
          <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-white/10 bg-ink-950/30 px-4 text-center">
            {frameError ? (
              <>
                <p className="text-[11px] text-rose-300/80">
                  The sandboxed preview could not be created.
                </p>
                <p className="max-w-sm font-mono text-[9px] leading-4 text-slate-500">
                  {frameError}
                </p>
              </>
            ) : (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-slate-600" />
                <p className="text-[10px] text-slate-500">
                  Preparing the sandboxed preview…
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
