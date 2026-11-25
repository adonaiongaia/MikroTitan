import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  Database,
  FileText,
  FolderOpen,
  FolderPlus,
  Loader2,
  Plus,
  RefreshCw,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react";

import { Events } from "../../lib/api";
import { cn } from "../../lib/cn";
import {
  errorMessage,
  formatBytes,
  formatCount,
  formatRelativeDate,
} from "../../lib/format";
import { Button } from "../ui/Button";
import { ProgressBar } from "../ui/ProgressBar";
import { knowledgeApi } from "./api";
import {
  isActiveJob,
  jobFromProgress,
  type KnowledgeCollection,
  type KnowledgeCollectionDetail,
  type RagIndexJob,
  type RagIndexProgress,
} from "./types";

/**
 * The Knowledge Vault: persistent, local collections of documents.
 *
 * Indexing is a Rust blocking task, so this view only owns presentation: it
 * listens to `rag://index-progress`, asks for cancellation, and refreshes the
 * manifest once a job reports `done` or `cancelled`.
 */
export function KnowledgeVaultView() {
  const [collections, setCollections] = useState<KnowledgeCollection[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<KnowledgeCollectionDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftDescription, setDraftDescription] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmFile, setConfirmFile] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, RagIndexJob>>({});
  const [refreshToken, setRefreshToken] = useState(0);
  const selectedIdRef = useRef<string | null>(null);
  selectedIdRef.current = selectedId;

  const selected = useMemo(
    () => collections.find((collection) => collection.id === selectedId) ?? null,
    [collections, selectedId],
  );

  const loadCollections = useCallback(async () => {
    try {
      const next = await knowledgeApi.list();
      setCollections(Array.isArray(next) ? next : []);
      setError(null);
      setSelectedId((current) => {
        if (current && next.some((collection) => collection.id === current)) {
          return current;
        }
        return next[0]?.id ?? null;
      });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCollections();
  }, [loadCollections]);

  // Reload the manifest after a job finishes or a file is removed.
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    void knowledgeApi
      .detail(selectedId)
      .then((next) => {
        if (!cancelled) setDetail(next);
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, refreshToken]);

  // Progress is per collection, so an index run on another collection still
  // shows its percentage in the rail.
  useEffect(() => {
    let disposed = false;
    let unlisten: UnlistenFn | undefined;
    void listen<RagIndexProgress>(Events.ragIndexProgress, (event) => {
      if (disposed) return;
      const progress = event.payload;
      setJobs((previous) => ({
        ...previous,
        [progress.collection_id]: jobFromProgress(progress),
      }));
      if (progress.phase === "done" || progress.phase === "cancelled") {
        setSummary(progress.message);
        void loadCollections();
        setRefreshToken((value) => value + 1);
      }
      if (progress.phase === "error") setError(progress.message);
    })
      .then((stop) => {
        if (disposed) stop();
        else unlisten = stop;
      })
      .catch((err) => {
        if (!disposed) setError(`Index progress could not be connected: ${errorMessage(err)}`);
      });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [loadCollections]);

  const runIndex = useCallback(
    async (collectionId: string, paths: string[]) => {
      if (!paths.length) return;
      const job = jobs[collectionId];
      if (job && isActiveJob(job.phase)) {
        setError("That collection is already indexing. Stop the current job first.");
        return;
      }
      setBusy("index");
      setError(null);
      setSummary(null);
      try {
        const result = await knowledgeApi.indexPaths(collectionId, paths);
        setSummary(
          `${result.files_indexed} indexed · ${result.files_unchanged} unchanged · ${formatCount(
            result.chunks,
          )} chunks · ${formatCount(result.tokens)} tokens${
            result.cancelled ? " (stopped)" : ""
          }`,
        );
        if (result.warnings.length) {
          setError(result.warnings.slice(0, 4).join(" "));
        }
        await loadCollections();
        setRefreshToken((value) => value + 1);
      } catch (err) {
        // A rejected invoke might not have emitted a progress event; clear the
        // optimistic job so the vault cannot stay stuck on "Collecting files…".
        setJobs((previous) => {
          const next = { ...previous };
          delete next[collectionId];
          return next;
        });
        setError(errorMessage(err));
      } finally {
        setBusy(null);
      }
    },
    [jobs, loadCollections],
  );

  // A folder or files dropped anywhere in the vault indexes into the selected
  // collection. ChatView owns its own listener and is unmounted here.
  useEffect(() => {
    let disposed = false;
    let unlisten: UnlistenFn | undefined;
    void import("@tauri-apps/api/webview")
      .then(({ getCurrentWebview }) =>
        getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type !== "drop") return;
          const paths = event.payload.paths ?? [];
          const target = selectedIdRef.current;
          if (!target) {
            setError("Select a collection (or create one) before dropping files.");
            return;
          }
          void runIndex(target, paths);
        }),
      )
      .then((stop) => {
        if (disposed) stop();
        else unlisten = stop;
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [runIndex]);

  const pickFiles = async () => {
    if (!selectedId) return;
    try {
      const picked = await open({
        multiple: true,
        directory: false,
        title: "Add documents to this collection",
      });
      if (!picked) return;
      await runIndex(selectedId, Array.isArray(picked) ? picked : [picked]);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const pickFolder = async () => {
    if (!selectedId) return;
    try {
      const picked = await open({
        multiple: true,
        directory: true,
        title: "Index a folder into this collection",
      });
      if (!picked) return;
      await runIndex(selectedId, Array.isArray(picked) ? picked : [picked]);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const reindex = async () => {
    if (!selectedId) return;
    setBusy("reindex");
    setError(null);
    setSummary(null);
    try {
      const result = await knowledgeApi.reindex(selectedId);
      setSummary(
        `${result.files_indexed} changed · ${result.files_unchanged} unchanged · ${formatCount(
          result.chunks,
        )} chunks`,
      );
      if (result.warnings.length) setError(result.warnings.slice(0, 4).join(" "));
      await loadCollections();
      setRefreshToken((value) => value + 1);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const cancel = async () => {
    if (!selectedId) return;
    setJobs((previous) => {
      const job = previous[selectedId];
      return job
        ? { ...previous, [selectedId]: { ...job, cancelling: true } }
        : previous;
    });
    try {
      await knowledgeApi.cancel(selectedId);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const createCollection = async () => {
    if (!draftName.trim()) {
      setError("Give the collection a name.");
      return;
    }
    setBusy("create");
    setError(null);
    try {
      const created = await knowledgeApi.create(
        draftName.trim(),
        draftDescription.trim(),
      );
      setCollections((previous) => [created, ...previous]);
      setSelectedId(created.id);
      setCreating(false);
      setDraftName("");
      setDraftDescription("");
      setSummary("Collection created. Add files or drop a folder to index it.");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const deleteCollection = async () => {
    if (!selectedId) return;
    setBusy("delete");
    setError(null);
    try {
      await knowledgeApi.remove(selectedId);
      setConfirmDelete(false);
      setDetail(null);
      setSelectedId(null);
      setJobs((previous) => {
        const next = { ...previous };
        delete next[selectedId];
        return next;
      });
      await loadCollections();
      setSummary("Collection deleted. Its index and chunks were removed from disk.");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const removeFile = async (path: string) => {
    if (!selectedId) return;
    setBusy("file");
    setError(null);
    try {
      await knowledgeApi.removePath(selectedId, path);
      setConfirmFile(null);
      await loadCollections();
      setRefreshToken((value) => value + 1);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const job = selectedId ? jobs[selectedId] : undefined;
  const jobActive = Boolean(job && isActiveJob(job.phase));
  const working = busy !== null || jobActive;

  return (
    <div className="flex h-full min-h-0">
      {/* Collections rail */}
      <div className="flex w-[310px] shrink-0 flex-col border-r border-white/5 bg-ink-900/30">
        <div className="flex items-center justify-between gap-2 px-4 py-3.5">
          <div>
            <p className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-slate-600">
              <Database className="h-3 w-3 text-acid-400" />
              Knowledge vault
            </p>
            <p className="mt-1 text-[10px] text-slate-600">
              {collections.length} collection{collections.length === 1 ? "" : "s"} ·{" "}
              {formatCount(
                collections.reduce(
                  (total, collection) => total + collection.chunk_count,
                  0,
                ),
              )}{" "}
              chunks
            </p>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => void loadCollections()}
              className="rounded-lg border border-white/10 bg-white/5 p-1.5 text-slate-400 transition hover:text-slate-100"
              aria-label="Refresh collections"
              title="Refresh collections"
            >
              <RefreshCw className="h-3 w-3" />
            </button>
            <Button
              size="sm"
              variant="primary"
              icon={<Plus className="h-3 w-3" />}
              onClick={() => setCreating(true)}
            >
              New
            </Button>
          </div>
        </div>

        <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto px-3 pb-3">
          {loading ? (
            <div className="flex items-center gap-2 px-2 py-6 text-[11px] text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Reading the vault…
            </div>
          ) : collections.length === 0 ? (
            <div className="rounded-xl border border-dashed border-white/10 bg-white/[0.02] px-3 py-6 text-center">
              <FolderPlus className="mx-auto h-5 w-5 text-slate-600" />
              <p className="mt-2 text-[11px] text-slate-400">The vault is empty</p>
              <p className="mt-1 text-[10px] leading-5 text-slate-600">
                Create a collection, then drop Markdown, text, code, JSON, TOML,
                CSV or PDF files into it.
              </p>
            </div>
          ) : (
            collections.map((collection) => {
              const active = collection.id === selectedId;
              const itemJob = jobs[collection.id];
              const running = Boolean(itemJob && isActiveJob(itemJob.phase));
              return (
                <button
                  key={collection.id}
                  type="button"
                  onClick={() => {
                    setSelectedId(collection.id);
                    setConfirmDelete(false);
                    setConfirmFile(null);
                  }}
                  className={cn(
                    "w-full rounded-xl border px-3 py-2.5 text-left transition",
                    active
                      ? "border-acid-400/30 bg-acid-400/[0.06]"
                      : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]",
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[12px] font-medium text-slate-200">
                        {collection.name}
                      </span>
                      {collection.description ? (
                        <span className="mt-0.5 block truncate text-[10px] text-slate-600">
                          {collection.description}
                        </span>
                      ) : null}
                    </span>
                    {running ? (
                      <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin text-sky-300" />
                    ) : (
                      <FileText className="mt-0.5 h-3 w-3 shrink-0 text-slate-700" />
                    )}
                  </div>
                  <div className="mt-1.5 flex items-center gap-2 text-[9.5px] text-slate-600">
                    <span className="font-mono">
                      {collection.file_count} files
                    </span>
                    <span>·</span>
                    <span className="font-mono">
                      {formatCount(collection.chunk_count)} chunks
                    </span>
                    {running && itemJob ? (
                      <span className="ml-auto font-mono text-sky-300/80">
                        {Math.round(itemJob.percent)}%
                      </span>
                    ) : (
                      <span className="ml-auto">
                        {formatRelativeDate(
                          new Date(collection.updated_at).toISOString(),
                        )}
                      </span>
                    )}
                  </div>
                  {running && itemJob ? (
                    <ProgressBar
                      value={itemJob.percent}
                      tone="sky"
                      className="mt-1.5"
                    />
                  ) : null}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Detail */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
          {error ? (
            <div className="mb-3 flex items-start gap-2 rounded-xl border border-rose-400/20 bg-rose-500/5 px-3 py-2 text-[10.5px] leading-5 text-rose-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="flex-1">{error}</span>
              <button
                type="button"
                onClick={() => setError(null)}
                className="text-rose-400/60 transition hover:text-rose-200"
                aria-label="Dismiss"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ) : null}

          {summary ? (
            <div className="mb-3 flex items-start gap-2 rounded-xl border border-acid-400/20 bg-acid-400/[0.05] px-3 py-2 text-[10.5px] leading-5 text-acid-200/90">
              <span className="flex-1">{summary}</span>
              <button
                type="button"
                onClick={() => setSummary(null)}
                className="text-acid-400/60 transition hover:text-acid-100"
                aria-label="Dismiss"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ) : null}

          {creating ? (
            <div className="mb-4 rounded-2xl border border-acid-400/25 bg-acid-400/[0.04] p-4">
              <p className="text-[11px] font-medium text-acid-100">
                New knowledge collection
              </p>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <input
                  autoFocus
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void createCollection();
                    if (event.key === "Escape") setCreating(false);
                  }}
                  placeholder="Collection name, e.g. Studio docs"
                  className="rounded-lg border border-white/10 bg-ink-950 px-2.5 py-1.5 text-[11.5px] text-slate-200 outline-none placeholder:text-slate-600 focus:border-acid-400/40"
                />
                <input
                  value={draftDescription}
                  onChange={(event) => setDraftDescription(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void createCollection();
                    if (event.key === "Escape") setCreating(false);
                  }}
                  placeholder="What is in it? (optional)"
                  className="rounded-lg border border-white/10 bg-ink-950 px-2.5 py-1.5 text-[11.5px] text-slate-200 outline-none placeholder:text-slate-600 focus:border-acid-400/40"
                />
              </div>
              <div className="mt-2.5 flex items-center gap-2">
                <Button
                  size="sm"
                  variant="primary"
                  busy={busy === "create"}
                  onClick={() => void createCollection()}
                >
                  Create collection
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setCreating(false);
                    setDraftName("");
                    setDraftDescription("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : null}

          {!selected ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Database className="h-8 w-8 text-slate-700" />
              <p className="mt-3 text-[12.5px] text-slate-400">
                Select a collection to inspect its index
              </p>
              <p className="mt-1 max-w-sm text-[10.5px] leading-5 text-slate-600">
                Each collection keeps its own chunks under the app data folder, so
                deleting one never touches your documents and never affects another
                chat's scope.
              </p>
              <Button
                size="sm"
                variant="primary"
                className="mt-4"
                icon={<Plus className="h-3 w-3" />}
                onClick={() => setCreating(true)}
              >
                New collection
              </Button>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="truncate text-[15px] font-semibold text-white">
                    {selected.name}
                  </h2>
                  {selected.description ? (
                    <p className="mt-0.5 max-w-2xl text-[10.5px] leading-5 text-slate-500">
                      {selected.description}
                    </p>
                  ) : null}
                  <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[9.5px] text-slate-600">
                    <Chip>{selected.file_count} files</Chip>
                    <Chip>{formatCount(selected.chunk_count)} chunks</Chip>
                    <Chip>{formatCount(selected.token_count)} tokens</Chip>
                    <Chip>{formatBytes(selected.byte_count)}</Chip>
                    <span>
                      updated{" "}
                      {formatRelativeDate(new Date(selected.updated_at).toISOString())}
                    </span>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <Button
                    size="sm"
                    variant="secondary"
                    icon={<Upload className="h-3 w-3" />}
                    disabled={working}
                    onClick={() => void pickFiles()}
                  >
                    Add files
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    icon={<FolderOpen className="h-3 w-3" />}
                    disabled={working}
                    onClick={() => void pickFolder()}
                  >
                    Add folder
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    busy={busy === "reindex"}
                    icon={<RefreshCw className="h-3 w-3" />}
                    disabled={jobActive}
                    onClick={() => void reindex()}
                  >
                    Re-index
                  </Button>
                  {confirmDelete ? (
                    <>
                      <Button
                        size="sm"
                        variant="danger"
                        busy={busy === "delete"}
                        onClick={() => void deleteCollection()}
                      >
                        Confirm delete
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setConfirmDelete(false)}
                      >
                        Cancel
                      </Button>
                    </>
                  ) : (
                    <Button
                      size="sm"
                      variant="danger"
                      icon={<Trash2 className="h-3 w-3" />}
                      onClick={() => setConfirmDelete(true)}
                    >
                      Delete
                    </Button>
                  )}
                </div>
              </div>

              {selected.source_paths.length ? (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {selected.source_paths.slice(0, 6).map((path) => (
                    <span
                      key={path}
                      title={path}
                      className="inline-flex max-w-[280px] items-center gap-1 rounded-full border border-white/10 bg-white/[0.03] px-2 py-0.5 font-mono text-[9px] text-slate-500"
                    >
                      <FolderOpen className="h-2.5 w-2.5 shrink-0" />
                      <span className="truncate">{path}</span>
                    </span>
                  ))}
                  {selected.source_paths.length > 6 ? (
                    <span className="text-[9px] text-slate-600">
                      +{selected.source_paths.length - 6} more
                    </span>
                  ) : null}
                </div>
              ) : null}

              {job ? (
                <div className="mt-3 rounded-xl border border-sky-400/25 bg-sky-400/[0.05] px-3 py-2.5">
                  <div className="flex items-center gap-2 text-[10.5px] text-sky-100">
                    {jobActive ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <span className="h-1.5 w-1.5 rounded-full bg-sky-300" />
                    )}
                    <span className="min-w-0 flex-1 truncate">{job.message}</span>
                    <span className="font-mono text-[10px] text-sky-300/80">
                      {Math.round(job.percent)}%
                    </span>
                  </div>
                  <ProgressBar value={job.percent} tone="sky" className="mt-2" />
                  <div className="mt-1.5 flex items-center gap-2 text-[9.5px] text-sky-200/60">
                    <span className="min-w-0 flex-1 truncate font-mono">
                      {job.currentFile || "—"}
                    </span>
                    <span className="shrink-0 font-mono">
                      {job.filesDone}/{job.filesTotal} files · {job.chunksDone} chunks
                    </span>
                    {jobActive ? (
                      <Button
                        size="sm"
                        variant="secondary"
                        icon={<Square className="h-2.5 w-2.5" />}
                        busy={job.cancelling}
                        onClick={() => void cancel()}
                      >
                        Stop
                      </Button>
                    ) : null}
                  </div>
                </div>
              ) : null}

              <div className="mt-4 rounded-2xl border border-white/10 bg-ink-900/40">
                <div className="flex items-center justify-between border-b border-white/5 px-3.5 py-2.5">
                  <p className="text-[10px] uppercase tracking-wider text-slate-600">
                    Indexed files
                  </p>
                  <p className="text-[9.5px] text-slate-600">
                    {detailLoading
                      ? "reading manifest…"
                      : `${detail?.files.length ?? 0} entries`}
                  </p>
                </div>

                {!detail || detail.files.length === 0 ? (
                  <div className="px-4 py-8 text-center">
                    <Upload className="mx-auto h-5 w-5 text-slate-700" />
                    <p className="mt-2 text-[11px] text-slate-400">
                      Nothing indexed yet
                    </p>
                    <p className="mt-1 text-[10px] leading-5 text-slate-600">
                      Drop files or a folder anywhere in this view, or use Add
                      files. Markdown headings and code blocks are kept whole so a
                      chunk rarely starts mid-function.
                    </p>
                  </div>
                ) : (
                  <ul className="divide-y divide-white/5">
                    {detail.files.map((file) => (
                      <li
                        key={file.path}
                        className="flex items-center gap-3 px-3.5 py-2"
                      >
                        <FileText className="h-3 w-3 shrink-0 text-slate-600" />
                        <div className="min-w-0 flex-1">
                          <p
                            className="truncate text-[11px] text-slate-300"
                            title={file.path}
                          >
                            {file.relative_path}
                          </p>
                          <p className="mt-0.5 truncate font-mono text-[9px] text-slate-600">
                            {formatBytes(file.bytes)} · {file.chunks} chunks ·{" "}
                            {formatCount(file.tokens)} tok ·{" "}
                            {formatRelativeDate(new Date(file.modified_ms).toISOString())}
                          </p>
                          {file.skipped_reason ? (
                            <p className="mt-0.5 text-[9px] leading-4 text-amber-300/80">
                              {file.skipped_reason}
                            </p>
                          ) : null}
                        </div>
                        {confirmFile === file.path ? (
                          <div className="flex shrink-0 items-center gap-1">
                            <Button
                              size="sm"
                              variant="danger"
                              busy={busy === "file"}
                              onClick={() => void removeFile(file.path)}
                            >
                              Remove
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setConfirmFile(null)}
                            >
                              Cancel
                            </Button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => setConfirmFile(file.path)}
                            className="shrink-0 rounded-md border border-white/5 p-1 text-slate-600 transition hover:border-rose-400/30 hover:text-rose-300"
                            aria-label={`Remove ${file.relative_path}`}
                            title="Remove from the index"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <p className="mt-3 text-[9.5px] leading-5 text-slate-600">
                Text chunks, term frequencies and a hashed TF-IDF vector are
                stored in{" "}
                <span className="font-mono text-slate-500">
                  app_local_data_dir()/knowledge/
                </span>
                . Nothing leaves this machine, and no collection is queried by a
                chat that has not attached it.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-white/10 bg-white/[0.03] px-2 py-0.5 font-mono text-[9px] text-slate-500">
      {children}
    </span>
  );
}
