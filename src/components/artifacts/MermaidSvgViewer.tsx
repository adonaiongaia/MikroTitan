import { useEffect, useMemo, useRef, useState } from "react";
import {
  ImageDown,
  Loader2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

import {
  pngFilename,
  saveDataUrlFile,
  saveTextFile,
  svgFilename,
} from "../../lib/artifactExport";
import { sanitizeSvg } from "../../lib/artifacts";
import { cn } from "../../lib/cn";
import { errorMessage } from "../../lib/format";

interface MermaidSvgViewerProps {
  kind: "mermaid" | "svg";
  code: string;
  title: string;
  streaming?: boolean;
  workspaceRoot: string;
}

let renderCounter = 0;

function cssColor(name: string, fallback: string): string {
  try {
    const raw = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
    if (!raw) return fallback;
    const channels = raw.split(/\s+/);
    return channels.length === 3 ? `rgb(${channels.join(", ")})` : fallback;
  } catch {
    return fallback;
  }
}

/**
 * The SVG is passive, so it renders in an iframe with no allow-scripts at all.
 * That keeps even a hand-written `<svg onload>` away from the Tauri IPC bridge,
 * while the injected CSP blocks every network request.
 */
function svgDocument(svg: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:"><style>html,body{height:100%;margin:0;background:transparent;overflow:hidden}body{display:flex;align-items:center;justify-content:center;padding:18px;box-sizing:border-box}svg{max-width:100%;max-height:100%;height:auto;width:auto}</style></head><body>${svg}</body></html>`;
}

export function MermaidSvgViewer({
  kind,
  code,
  title,
  streaming = false,
  workspaceRoot,
}: MermaidSvgViewerProps) {
  const [svg, setSvg] = useState("");
  const [busy, setBusy] = useState(kind === "mermaid");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{
    x: number;
    y: number;
    originX: number;
    originY: number;
  } | null>(null);
  const noticeTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (noticeTimer.current) window.clearTimeout(noticeTimer.current);
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setZoom(1);
    setOffset({ x: 0, y: 0 });

    if (streaming) {
      setSvg("");
      setBusy(false);
      return;
    }
    if (!code.trim()) {
      setSvg("");
      setBusy(false);
      setError("This artifact is empty.");
      return;
    }

    if (kind === "svg") {
      const cleaned = sanitizeSvg(code);
      if (!/<svg[\s>]/i.test(cleaned)) {
        setSvg("");
        setBusy(false);
        setError("This block does not contain an <svg> root element.");
        return;
      }
      setSvg(cleaned);
      setBusy(false);
      return;
    }

    setBusy(true);
    void import("mermaid")
      .then(async ({ default: mermaid }) => {
        const accent = cssColor("--accent-400", "rgb(205, 247, 90)");
        const ink = cssColor("--ink-750", "rgb(24, 29, 40)");
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "base",
          fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
          themeVariables: {
            darkMode: true,
            background: "transparent",
            primaryColor: ink,
            primaryTextColor: cssColor("--accent-200", "rgb(239, 255, 194)"),
            primaryBorderColor: accent,
            lineColor: cssColor("--ink-500", "rgb(58, 69, 88)"),
            secondaryColor: cssColor("--ink-700", "rgb(29, 36, 49)"),
            tertiaryColor: cssColor("--ink-800", "rgb(20, 24, 33)"),
            fontSize: "14px",
          },
          flowchart: { htmlLabels: false, useMaxWidth: true },
          sequence: { useMaxWidth: true },
          er: { useMaxWidth: true },
        });
        renderCounter += 1;
        const id = `janas-mermaid-${Date.now()}-${renderCounter}`;
        const result = await mermaid.render(id, code);
        if (!cancelled) {
          setSvg(sanitizeSvg(result.svg));
          setBusy(false);
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setSvg("");
        setBusy(false);
        setError(errorMessage(err));
      });

    return () => {
      cancelled = true;
    };
  }, [code, kind, streaming]);

  const svgFrame = useMemo(() => svgDocument(svg), [svg]);

  const flash = (message: string) => {
    setNotice(message);
    if (noticeTimer.current) window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setNotice(null), 4_000);
  };

  const exportSvg = async () => {
    if (!svg) return;
    try {
      const saved = await saveTextFile(
        svg,
        svgFilename(title),
        "image/svg+xml;charset=utf-8",
        workspaceRoot || undefined,
        "svg",
      );
      if (saved) flash(`Saved ${svgFilename(title)} to ${saved}`);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const exportPng = async () => {
    if (!svg) return;
    let url: string | null = null;
    try {
      const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
      const objectUrl = URL.createObjectURL(blob);
      url = objectUrl;
      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () =>
          reject(new Error("The SVG could not be rasterised."));
        image.src = objectUrl;
      });
      // A hostile SVG can declare an enormous intrinsic size; keep the
      // raster target bounded even before the scale is applied.
      const width = Math.max(1, Math.min(4_000, image.naturalWidth || 1_200));
      const height = Math.max(1, Math.min(4_000, image.naturalHeight || 800));
      const scale = Math.min(3, Math.max(1, 1_800 / width));
      const canvas = documentCanvas(width, height, scale);
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas rendering is unavailable.");
      context.fillStyle = kind === "mermaid" ? "#0b0d11" : "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const saved = await saveDataUrlFile(
        canvas.toDataURL("image/png"),
        pngFilename(title),
        workspaceRoot || undefined,
      );
      if (saved) flash(`Saved ${pngFilename(title)} to ${saved}`);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  };

  if (streaming) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <Loader2 className="h-4 w-4 animate-spin text-slate-600" />
        <p className="text-[11px] text-slate-500">
          The {kind === "mermaid" ? "diagram" : "SVG"} is still arriving. It
          renders as soon as the block closes.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-white/5 px-2 py-1.5">
        <div className="flex items-center rounded-lg border border-white/10 bg-white/[0.03] p-0.5">
          <button
            type="button"
            onClick={() => setZoom((value) => Math.max(0.25, value / 1.25))}
            className="rounded-md p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
            title="Zoom out"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-3 w-3" />
          </button>
          <span className="w-9 text-center font-mono text-[9px] text-slate-500">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setZoom((value) => Math.min(6, value * 1.25))}
            className="rounded-md p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
            title="Zoom in"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-3 w-3" />
          </button>
          <button
            type="button"
            onClick={() => {
              setZoom(1);
              setOffset({ x: 0, y: 0 });
            }}
            className="rounded-md p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
            title="Reset zoom and pan"
            aria-label="Reset zoom and pan"
          >
            <RotateCcw className="h-3 w-3" />
          </button>
        </div>
        <span className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={() => void exportSvg()}
            disabled={!svg}
            className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/[0.03] px-1.5 py-1 text-[9px] text-slate-400 transition hover:text-slate-200 disabled:opacity-40"
            title="Export SVG"
          >
            SVG
          </button>
          <button
            type="button"
            onClick={() => void exportPng()}
            disabled={!svg}
            className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/[0.03] px-1.5 py-1 text-[9px] text-slate-400 transition hover:text-slate-200 disabled:opacity-40"
            title="Export PNG"
          >
            <ImageDown className="h-2.5 w-2.5" />
            PNG
          </button>
        </span>
      </div>

      {busy ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-slate-600" />
          <p className="text-[10px] text-slate-500">Laying out the diagram…</p>
        </div>
      ) : error ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
          <p className="text-[11px] font-medium text-rose-300/80">
            {kind === "mermaid" ? "Diagram" : "SVG"} preview failed
          </p>
          <p className="max-w-sm whitespace-pre-wrap font-mono text-[9.5px] leading-4 text-slate-500">
            {error}
          </p>
          <p className="text-[9px] text-slate-600">
            The raw source is still available in the Code tab.
          </p>
        </div>
      ) : (
        <div
          className={cn(
            "relative min-h-0 flex-1 overflow-hidden bg-ink-950/30",
            zoom > 1 ? "cursor-grab active:cursor-grabbing" : "cursor-grab",
          )}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            dragRef.current = {
              x: event.clientX,
              y: event.clientY,
              originX: offset.x,
              originY: offset.y,
            };
          }}
          onPointerMove={(event) => {
            const drag = dragRef.current;
            if (!drag) return;
            setOffset({
              x: drag.originX + (event.clientX - drag.x),
              y: drag.originY + (event.clientY - drag.y),
            });
          }}
          onPointerUp={() => {
            dragRef.current = null;
          }}
          onPointerCancel={() => {
            dragRef.current = null;
          }}
        >
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              transformOrigin: "center center",
            }}
          >
            <iframe
              title={`${title} ${kind === "mermaid" ? "diagram" : "SVG"}`}
              // No allow-scripts: SVG is passive, and a model-provided event
              // handler must never reach the Tauri bridge.
              sandbox=""
              referrerPolicy="no-referrer"
              srcDoc={svgFrame}
              className="pointer-events-none h-full w-full border-0 bg-transparent"
            />
          </div>
        </div>
      )}

      {notice ? (
        <p className="shrink-0 border-t border-emerald-400/15 bg-emerald-400/5 px-3 py-1.5 text-[9.5px] text-emerald-200/80">
          {notice}
        </p>
      ) : null}
    </div>
  );
}

function documentCanvas(width: number, height: number, scale: number) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  return canvas;
}
