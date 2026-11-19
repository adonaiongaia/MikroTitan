import { useCallback, useEffect, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  Archive,
  CheckCircle2,
  Download,
  Loader2,
  RotateCcw,
  Trash2,
  Upload,
} from "lucide-react";

import { api } from "../../lib/api";
import { errorMessage, formatBytes } from "../../lib/format";
import type { ChatBackupInfo, ChatStoreStats } from "../../types";
import { Button } from "../ui/Button";

export function ChatHistoryPanel() {
  const [stats, setStats] = useState<ChatStoreStats | null>(null);
  const [backups, setBackups] = useState<ChatBackupInfo[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextStats, nextBackups] = await Promise.all([
        api.chatStoreStats(),
        api.chatBackupList(),
      ]);
      setStats(nextStats);
      setBackups(nextBackups);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const exportAll = async () => {
    setError(null);
    setMessage(null);
    try {
      const path = await save({
        defaultPath: `janas-chat-backup-${new Date()
          .toISOString()
          .slice(0, 10)}.json`,
        filters: [{ name: "Janas chat backup", extensions: ["json"] }],
      });
      if (typeof path !== "string") return;
      setBusy("export");
      const count = await api.chatExportAll(path);
      setMessage(`Exported ${count} chat${count === 1 ? "" : "s"}.`);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const importAll = async () => {
    setError(null);
    setMessage(null);
    try {
      const path = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Janas chat backup", extensions: ["json"] }],
      });
      if (typeof path !== "string") return;
      setBusy("import");
      const summary = await api.chatImportAll(path);
      setMessage(
        `Imported ${summary.imported} chat${
          summary.imported === 1 ? "" : "s"
        }${summary.failed ? `; ${summary.failed} failed` : ""}.`,
      );
      if (summary.errors.length) setError(summary.errors.join("\n"));
      await refresh();
      window.dispatchEvent(new Event("janas:chat-history-changed"));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const cleanAssets = async () => {
    setError(null);
    setMessage(null);
    setBusy("gc");
    try {
      const freed = await api.chatStoreGc();
      setMessage(
        freed > 0
          ? `Removed ${formatBytes(freed)} of orphaned images.`
          : "No orphaned images were found.",
      );
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const restore = async (backup: ChatBackupInfo) => {
    setError(null);
    setMessage(null);
    setBusy(backup.id);
    try {
      const restored = await api.chatRestoreBackup(backup.id);
      setMessage(`Restored “${restored.title}”. The chat rail has been refreshed.`);
      await refresh();
      window.dispatchEvent(new Event("janas:chat-history-changed"));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-[12px] font-semibold text-white">
            Chat history and backups
          </h3>
          <p className="mt-1 text-[10px] leading-4 text-slate-600">
            Sessions live as one JSON file per chat. Each save keeps the
            previous version, and images are content-addressed on disk.
          </p>
        </div>
        <Archive className="mt-0.5 h-4 w-4 shrink-0 text-slate-600" />
      </div>

      {stats ? (
        <div className="mt-4 grid grid-cols-2 gap-2 text-[10px] text-slate-500 sm:grid-cols-4">
          <Stat label="Chats" value={String(stats.sessions)} />
          <Stat label="History" value={formatBytes(stats.session_bytes)} />
          <Stat label="Images" value={formatBytes(stats.asset_bytes)} />
          <Stat label="Backups" value={String(stats.backups)} />
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          size="sm"
          variant="secondary"
          busy={busy === "export"}
          onClick={() => void exportAll()}
          icon={<Download className="h-3 w-3" />}
        >
          Export all chats
        </Button>
        <Button
          size="sm"
          variant="secondary"
          busy={busy === "import"}
          onClick={() => void importAll()}
          icon={<Upload className="h-3 w-3" />}
        >
          Import backup
        </Button>
        <Button
          size="sm"
          variant="ghost"
          busy={busy === "gc"}
          onClick={() => void cleanAssets()}
          icon={<Trash2 className="h-3 w-3" />}
        >
          Clean orphaned images
        </Button>
      </div>

      {message ? (
        <p className="mt-3 flex items-start gap-1.5 rounded-lg border border-emerald-400/15 bg-emerald-400/5 px-2.5 py-2 text-[10px] leading-4 text-emerald-200/90">
          <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0" />
          {message}
        </p>
      ) : null}
      {error ? (
        <p className="mt-3 whitespace-pre-wrap rounded-lg border border-rose-400/15 bg-rose-500/5 px-2.5 py-2 text-[10px] leading-4 text-rose-300">
          {error}
        </p>
      ) : null}

      <div className="mt-4 border-t border-white/5 pt-3">
        <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
          Previous versions
        </p>
        {backups.length ? (
          <div className="mt-2 space-y-1">
            {backups.slice(0, 8).map((backup) => (
              <div
                key={backup.id}
                className="flex items-center gap-2 rounded-lg px-1.5 py-1 text-[10px] text-slate-500 hover:bg-white/[0.02]"
              >
                <span className="min-w-0 flex-1 truncate text-slate-400">
                  {backup.title}
                </span>
                <span className="shrink-0 font-mono text-[9px] text-slate-600">
                  {formatBytes(backup.bytes)}
                </span>
                <span className="shrink-0 text-[9px] text-slate-600">
                  {new Date(backup.updated_at).toLocaleDateString()}
                </span>
                <button
                  type="button"
                  onClick={() => void restore(backup)}
                  disabled={busy !== null}
                  className="inline-flex shrink-0 items-center gap-1 rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9px] text-slate-400 transition hover:bg-white/10 hover:text-slate-200 disabled:opacity-40"
                >
                  {busy === backup.id ? (
                    <Loader2 className="h-2.5 w-2.5 animate-spin" />
                  ) : (
                    <RotateCcw className="h-2.5 w-2.5" />
                  )}
                  Restore
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-1.5 text-[10px] leading-4 text-slate-600">
            No previous versions yet. The first edit of a saved chat creates one.
          </p>
        )}
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/5 bg-white/[0.02] px-2 py-1.5">
      <p className="text-[8px] uppercase tracking-wider text-slate-600">
        {label}
      </p>
      <p className="mt-0.5 font-mono text-[10px] text-slate-300">{value}</p>
    </div>
  );
}
