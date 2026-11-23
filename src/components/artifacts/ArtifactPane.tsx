import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  AlertTriangle,
  Check,
  Code2,
  Copy,
  Download,
  FolderDown,
  GitCompare,
  Loader2,
  MonitorPlay,
  PanelRightClose,
  PanelRightOpen,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { usePersistentState } from "../../hooks/usePersistentState";
import {
  artifactFilename,
  artifactGroupFor,
  artifactTypeLabel,
  diffLines,
  diffStats,
  latestArtifactGroup,
  type Artifact,
  type ArtifactGroup,
  type ChatArtifactIndex,
} from "../../lib/artifacts";
import {
  isTauriRuntime,
  saveArtifactFile,
  saveArtifactToWorkspace,
} from "../../lib/artifactExport";
import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage } from "../../lib/format";
import { DataTableViewer } from "./DataTableViewer";
import { MermaidSvgViewer } from "./MermaidSvgViewer";
import {
  SandboxedHtmlFrame,
  type ArtifactConsoleEntry,
} from "./SandboxedHtmlFrame";

interface ArtifactPaneProps {
  open: boolean;
  index: ChatArtifactIndex;
  selectedRootId: string | null;
  selectedVersionId: string | null;
  onSelectArtifact: (rootId: string, versionId: string | null) => void;
  onOpen: () => void;
  onClose: () => void;
  workspaceRoot: string;
  onError: (message: string) => void;
}

type ArtifactTab = "preview" | "code" | "diff";
type SaveAction = "workspace" | "download" | null;

const TABS: Array<{
  id: ArtifactTab;
  label: string;
  icon: typeof MonitorPlay;
}> = [
  { id: "preview", label: "Preview", icon: MonitorPlay },
  { id: "code", label: "Code", icon: Code2 },
  { id: "diff", label: "Diff", icon: GitCompare },
];

export function ArtifactPane({
  open,
  index,
  selectedRootId,
  selectedVersionId,
  onSelectArtifact,
  onOpen,
  onClose,
  workspaceRoot,
  onError,
}: ArtifactPaneProps) {
  const [width, setWidth] = usePersistentState(
    "janas.ui.artifactPaneWidth",
    480,
  );
  const [tab, setTab] = useState<ArtifactTab>("preview");
  const [consoleEntries, setConsoleEntries] = useState<ArtifactConsoleEntry[]>(
    [],
  );
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState<SaveAction>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const dragRef = useRef<{
    x: number;
    width: number;
  } | null>(null);

  const group: ArtifactGroup | null = useMemo(
    () =>
      artifactGroupFor(index, selectedRootId) ??
      latestArtifactGroup(index) ??
      index.groups[index.groups.length - 1] ??
      null,
    [index, selectedRootId],
  );

  const artifact: Artifact | null = group
    ? (group.versions.find((version) => version.id === selectedVersionId) ??
      group.latest)
    : null;

  const previous: Artifact | null =
    group && artifact?.previousId
      ? (group.versions.find(
          (version) => version.id === artifact.previousId,
        ) ?? null)
      : null;

  const diff = useMemo(
    () =>
      previous && artifact ? diffLines(previous.content, artifact.content) : [],
    [artifact, previous],
  );
  const stats = useMemo(() => diffStats(diff), [diff]);

  useEffect(() => {
    setTab("preview");
  }, [group?.rootId]);

  useEffect(() => {
    setConsoleEntries([]);
    setCopied(false);
  }, [artifact?.id]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 4_500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const pushConsole = useCallback((entry: ArtifactConsoleEntry) => {
    setConsoleEntries((current) => {
      const last = current[current.length - 1];
      if (last && last.message === entry.message && last.level === entry.level) {
        return current;
      }
      return [...current.slice(-9), entry];
    });
  }, []);

  const clampWidth = (value: number) => {
    const viewport =
      typeof window === "undefined" ? 1440 : Math.max(320, window.innerWidth);
    // Leave room for the sessions rail, a collapsed engine rail and a usable
    // chat column, so the composer is never squeezed by the preview pane.
    return Math.round(Math.max(340, Math.min(900, value, viewport - 700)));
  };

  const copy = async () => {
    if (!artifact) return;
    try {
      await navigator.clipboard.writeText(artifact.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_600);
    } catch {
      onError("The clipboard is unavailable.");
    }
  };

  const saveWorkspace = async () => {
    if (!artifact) return;
    setSaving("workspace");
    try {
      const path = await saveArtifactToWorkspace(artifact, workspaceRoot);
      setNotice(`Saved ${artifactFilename(artifact)} to ${path}`);
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setSaving(null);
    }
  };

  const download = async () => {
    if (!artifact) return;
    setSaving("download");
    try {
      const path = await saveArtifactFile(artifact, workspaceRoot || undefined);
      if (path) setNotice(`Saved ${artifactFilename(artifact)} to ${path}`);
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setSaving(null);
    }
  };

  if (!open) {
    if (!index.groups.length) return null;
    return (
      <aside className="flex w-10 shrink-0 flex-col items-center gap-3 border-l border-white/5 bg-ink-900/40 py-3">
        <button
          type="button"
          onClick={onOpen}
          className="rounded-lg p-1.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
          aria-label="Open the artifact preview pane"
          title="Open artifacts"
        >
          <PanelRightOpen className="h-4 w-4" />
        </button>
        <span className="text-[8px] uppercase tracking-[0.2em] text-slate-600 [writing-mode:vertical-rl]">
          artifacts
        </span>
        <span className="rounded-full border border-acid-400/25 bg-acid-400/10 px-1.5 py-0.5 font-mono text-[8px] text-acid-200/90">
          {index.groups.length}
        </span>
      </aside>
    );
  }

  const tauri = isTauriRuntime();
  const canSaveToWorkspace = Boolean(workspaceRoot.trim()) && tauri;
  const workspaceHint = !workspaceRoot.trim()
    ? "Choose a workspace root in Engine settings to save artifacts into it."
    : !tauri
      ? "Workspace writes are available in the desktop app."
      : `Save into ${workspaceRoot}`;

  return (
    <aside
      className="relative flex shrink-0 flex-col border-l border-white/5 bg-ink-900/40"
      style={{
        width: clampWidth(width),
        maxWidth: "calc(100vw - 700px)",
        minWidth: 300,
      }}
      aria-label="Artifact preview pane"
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the artifact pane"
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { x: event.clientX, width };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          setWidth(clampWidth(drag.width + (drag.x - event.clientX)));
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
        onLostPointerCapture={() => {
          dragRef.current = null;
        }}
        className="absolute -left-1 top-0 z-20 h-full w-2 cursor-col-resize bg-transparent transition hover:bg-acid-400/20"
        title="Drag to resize"
      />

      <div className="flex h-12 shrink-0 items-center gap-2 border-b border-white/5 px-3">
        <PanelRightClose className="h-3.5 w-3.5 shrink-0 text-slate-500" />
        {artifact ? (
          <span className="shrink-0 rounded border border-acid-400/25 bg-acid-400/10 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wider text-acid-200/90">
            {artifactTypeLabel(artifact.type)}
          </span>
        ) : null}
        <span
          className="min-w-0 flex-1 truncate text-[11px] font-medium text-slate-200"
          title={group?.title}
        >
          {group?.title ?? "Artifacts"}
        </span>
        {artifact?.isStreaming ? (
          <span className="flex shrink-0 items-center gap-1 text-[9px] text-acid-300/80">
            <Loader2 className="h-2.5 w-2.5 animate-spin" />
            generating
          </span>
        ) : null}
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded-md p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
          aria-label="Collapse the artifact pane"
          title="Collapse artifact pane"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {artifact ? (
        <>
          <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-white/5 px-2 py-1.5">
            <div className="flex items-center rounded-lg border border-white/10 bg-white/[0.03] p-0.5">
              {TABS.map((entry) => {
                const Icon = entry.icon;
                return (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => setTab(entry.id)}
                    className={cn(
                      "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] transition",
                      tab === entry.id
                        ? "bg-acid-400/15 text-acid-200"
                        : "text-slate-500 hover:text-slate-300",
                    )}
                    aria-pressed={tab === entry.id}
                  >
                    <Icon className="h-2.5 w-2.5" />
                    {entry.label}
                  </button>
                );
              })}
            </div>

            {group && group.versions.length > 1 ? (
              <select
                value={artifact.id}
                onChange={(event) => {
                  const next = group.versions.find(
                    (version) => version.id === event.target.value,
                  );
                  if (next) onSelectArtifact(group.rootId, next.id);
                }}
                className="h-7 min-w-0 rounded-lg border border-white/10 bg-ink-900 px-1.5 font-mono text-[9px] text-slate-300 outline-none transition hover:border-white/20 focus:border-acid-400/40"
                aria-label="Artifact version"
                title="Switch artifact version"
              >
                {group.versions.map((version) => (
                  <option key={version.id} value={version.id}>
                    v{version.version} · {version.lineCount} lines
                  </option>
                ))}
              </select>
            ) : (
              <span className="rounded border border-white/10 bg-white/[0.02] px-1 py-0.5 font-mono text-[8.5px] text-slate-500">
                v{artifact.version}
              </span>
            )}

            <span className="ml-auto flex items-center gap-0.5">
              <IconAction
                label={copied ? "Copied" : "Copy"}
                onClick={() => void copy()}
                icon={
                  copied ? (
                    <Check className="h-3 w-3 text-emerald-400" />
                  ) : (
                    <Copy className="h-3 w-3" />
                  )
                }
              />
              <IconAction
                label={saving === "workspace" ? "Saving…" : "Save to workspace"}
                onClick={() => void saveWorkspace()}
                disabled={!canSaveToWorkspace || saving !== null}
                title={workspaceHint}
                icon={
                  saving === "workspace" ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <FolderDown className="h-3 w-3" />
                  )
                }
              />
              <IconAction
                label={saving === "download" ? "Saving…" : "Download"}
                onClick={() => void download()}
                disabled={saving !== null}
                title={`Choose a location for ${artifactFilename(artifact)}`}
                icon={
                  saving === "download" ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <Download className="h-3 w-3" />
                  )
                }
              />
            </span>
          </div>

          {notice ? (
            <p className="shrink-0 truncate border-b border-emerald-400/15 bg-emerald-400/5 px-3 py-1.5 text-[9.5px] text-emerald-200/80">
              {notice}
            </p>
          ) : null}

          <div className="min-h-0 flex-1 overflow-hidden">
            {tab === "preview" ? (
              <ArtifactPreview
                artifact={artifact}
                workspaceRoot={workspaceRoot}
                onConsole={pushConsole}
              />
            ) : tab === "code" ? (
              <pre className="h-full overflow-auto whitespace-pre p-3 font-mono text-[11px] leading-5 text-slate-300">
                <code>{artifact.content}</code>
              </pre>
            ) : (
              <DiffView
                artifact={artifact}
                previous={previous}
                diff={diff}
                added={stats.added}
                removed={stats.removed}
              />
            )}
          </div>

          {artifact.type === "html" &&
          tab === "preview" &&
          consoleEntries.length ? (
            <div className="max-h-32 shrink-0 overflow-y-auto border-t border-amber-400/15 bg-amber-400/[0.04] px-3 py-2">
              <div className="flex items-center gap-1.5">
                <AlertTriangle className="h-3 w-3 shrink-0 text-amber-400/80" />
                <span className="text-[9px] font-semibold uppercase tracking-wider text-amber-200/70">
                  Preview console
                </span>
                <button
                  type="button"
                  onClick={() => setConsoleEntries([])}
                  className="ml-auto text-[9px] text-slate-500 transition hover:text-slate-300"
                >
                  Clear
                </button>
              </div>
              <div className="mt-1 space-y-1">
                {consoleEntries.slice(-4).map((entry) => (
                  <p
                    key={entry.id}
                    className={cn(
                      "break-words font-mono text-[9px] leading-4",
                      entry.level === "warn"
                        ? "text-amber-200/70"
                        : "text-rose-300/80",
                    )}
                  >
                    <span className="uppercase">{entry.level}</span>{" "}
                    {entry.message}
                    {entry.line ? (
                      <span className="text-slate-600">
                        {" "}
                        · line {entry.line}
                      </span>
                    ) : null}
                  </p>
                ))}
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
          <MonitorPlay className="h-5 w-5 text-slate-600" />
          <p className="text-[11px] text-slate-500">
            No renderable artifact in this reply yet.
          </p>
          <p className="max-w-xs text-[9.5px] leading-4 text-slate-600">
            HTML, SVG, Mermaid, Markdown, JSON and CSV content appears here when
            a reply includes a complete tagged document or a substantial
            fenced block.
          </p>
        </div>
      )}
    </aside>
  );
}

function ArtifactPreview({
  artifact,
  workspaceRoot,
  onConsole,
}: {
  artifact: Artifact;
  workspaceRoot: string;
  onConsole: (entry: ArtifactConsoleEntry) => void;
}) {
  switch (artifact.type) {
    case "html":
      return (
        <SandboxedHtmlFrame
          key={artifact.id}
          html={artifact.content}
          title={artifact.title}
          streaming={artifact.isStreaming}
          onConsole={onConsole}
        />
      );
    case "svg":
      return (
        <MermaidSvgViewer
          key={artifact.id}
          kind="svg"
          code={artifact.content}
          title={artifact.title}
          streaming={artifact.isStreaming}
          workspaceRoot={workspaceRoot}
        />
      );
    case "mermaid":
      return (
        <MermaidSvgViewer
          key={artifact.id}
          kind="mermaid"
          code={artifact.content}
          title={artifact.title}
          streaming={artifact.isStreaming}
          workspaceRoot={workspaceRoot}
        />
      );
    case "markdown":
      return (
        <MarkdownArtifactView
          key={artifact.id}
          content={artifact.content}
          streaming={artifact.isStreaming}
        />
      );
    default:
      return <DataTableViewer key={artifact.id} artifact={artifact} />;
  }
}

function MarkdownArtifactView({
  content,
  streaming,
}: {
  content: string;
  streaming: boolean;
}) {
  return (
    <div className="h-full overflow-y-auto px-5 py-4">
      <div className="md">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          // Raw HTML is not enabled. URLs are restricted to the schemes this
          // view can actually render; links are still opened only in the
          // user's browser through the Tauri opener.
          urlTransform={safeArtifactUrl}
          components={{
            a: ({ href, children }) => {
              const web = Boolean(href && /^https?:\/\//i.test(href));
              return (
                <a
                  href={web ? href : undefined}
                  onClick={(event) => {
                    if (!web || !href) return;
                    event.preventDefault();
                    void api.openExternal(href).catch(() => undefined);
                  }}
                  rel="noreferrer noopener"
                >
                  {children}
                </a>
              );
            },
            img: ({ src, alt }) => {
              const safe = Boolean(
                src &&
                  (/^https?:\/\//i.test(src) ||
                    /^data:image\//i.test(src) ||
                    src.startsWith("asset:")),
              );
              if (!safe) return null;
              return (
                <img
                  src={src}
                  alt={alt ?? ""}
                  loading="lazy"
                  className="my-2 max-w-full rounded-lg border border-white/10"
                />
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>
        {streaming ? (
          <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 animate-pulse-soft rounded-sm bg-acid-400/80" />
        ) : null}
      </div>
    </div>
  );
}

function safeArtifactUrl(url: string): string {
  const trimmed = url.trim();
  if (/^(?:https?:|data:image\/|asset:)/i.test(trimmed)) return trimmed;
  if (trimmed.startsWith("#")) return trimmed;
  return "";
}

function DiffView({
  artifact,
  previous,
  diff,
  added,
  removed,
}: {
  artifact: Artifact;
  previous: Artifact | null;
  diff: ReturnType<typeof diffLines>;
  added: number;
  removed: number;
}) {
  if (!previous) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <GitCompare className="h-5 w-5 text-slate-600" />
        <p className="text-[11px] text-slate-500">
          This is the first version of this artifact.
        </p>
        <p className="max-w-xs text-[9.5px] leading-4 text-slate-600">
          Ask for a change in a later turn, or regenerate the reply, to compare
          revisions line by line.
        </p>
      </div>
    );
  }
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-white/5 px-2 py-1.5 font-mono text-[9px] text-slate-500">
        <span>
          v{previous.version} → v{artifact.version}
        </span>
        <span className="text-emerald-400/80">+{added}</span>
        <span className="text-rose-400/80">−{removed}</span>
        <span className="ml-auto text-slate-600">
          {diff.length.toLocaleString()} lines
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {diff.map((line, index) => (
          <div
            key={index}
            className={cn(
              "grid grid-cols-[2.5rem_2.5rem_minmax(0,1fr)] border-b border-white/[0.025] font-mono text-[10px] leading-5",
              line.kind === "add"
                ? "bg-emerald-400/[0.06]"
                : line.kind === "remove"
                  ? "bg-rose-400/[0.06]"
                  : "",
            )}
          >
            <span className="select-none border-r border-white/5 px-1 text-right text-slate-700">
              {line.oldLine ?? ""}
            </span>
            <span className="select-none border-r border-white/5 px-1 text-right text-slate-700">
              {line.newLine ?? ""}
            </span>
            <span
              className={cn(
                "whitespace-pre-wrap break-all px-2",
                line.kind === "add"
                  ? "text-emerald-200/90"
                  : line.kind === "remove"
                    ? "text-rose-200/80"
                    : "text-slate-400",
              )}
            >
              {line.kind === "add" ? "+" : line.kind === "remove" ? "−" : " "}
              {line.text || " "}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function IconAction({
  icon,
  label,
  title,
  onClick,
  disabled,
}: {
  icon: ReactNode;
  label: string;
  title?: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title ?? label}
      aria-label={label}
      className="inline-flex h-6 w-6 items-center justify-center rounded-md text-slate-500 transition hover:bg-white/5 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-35"
    >
      {icon}
    </button>
  );
}
