import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Loader2,
  Mic,
  ShieldCheck,
  Square,
} from "lucide-react";

import { useVoiceRecorder } from "../../hooks/useVoiceRecorder";
import { api, Events } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage, formatBytes } from "../../lib/format";
import type {
  AppSettings,
  WhisperProgress,
  WhisperStatus,
} from "../../types";
import { Button } from "../ui/Button";
import { Toggle } from "../ui/Toggle";

interface VoiceSettingsProps {
  form: AppSettings;
  update: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  onError: (message: string | null) => void;
}

const LANGUAGES: Array<[string, string]> = [
  ["auto", "Auto-detect"],
  ["en", "English"],
  ["es", "Spanish"],
  ["fr", "French"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["nl", "Dutch"],
  ["pl", "Polish"],
  ["ru", "Russian"],
  ["uk", "Ukrainian"],
  ["tr", "Turkish"],
  ["ar", "Arabic"],
  ["hi", "Hindi"],
  ["fa", "Persian"],
  ["he", "Hebrew"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
  ["zh", "Chinese"],
  ["vi", "Vietnamese"],
  ["th", "Thai"],
  ["id", "Indonesian"],
  ["sv", "Swedish"],
  ["da", "Danish"],
  ["fi", "Finnish"],
  ["no", "Norwegian"],
  ["cs", "Czech"],
  ["ro", "Romanian"],
  ["hu", "Hungarian"],
  ["el", "Greek"],
];

export function VoiceSettings({ form, update, onError }: VoiceSettingsProps) {
  const [status, setStatus] = useState<WhisperStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [progress, setProgress] = useState<WhisperProgress | null>(null);
  const [busy, setBusy] = useState<"install" | string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [micTested, setMicTested] = useState(false);
  const recorder = useVoiceRecorder();
  const testTimer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.getWhisperStatus());
      setStatusError(null);
    } catch (err) {
      setStatusError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void listen<WhisperProgress>(Events.whisperProgress, (event) => {
      setProgress(event.payload);
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

  useEffect(
    () => () => {
      if (testTimer.current !== null) window.clearTimeout(testTimer.current);
      recorder.cancel();
    },
    [recorder.cancel],
  );

  const install = async () => {
    setBusy("install");
    onError(null);
    setNotice(null);
    try {
      const result = await api.installWhisperBinary();
      setNotice(result.message);
      await refresh();
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const download = async (modelId: string) => {
    setBusy(modelId);
    onError(null);
    setNotice(null);
    try {
      const result = await api.downloadWhisperModel(modelId);
      update("whisper_model", modelId);
      setNotice(`${result.message} Save changes to select it for new recordings.`);
      await refresh();
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const cancelDownload = async () => {
    try {
      await api.cancelWhisperDownload();
    } catch (err) {
      onError(errorMessage(err));
    }
  };

  const testMicrophone = async () => {
    if (recorder.state === "recording") {
      if (testTimer.current !== null) window.clearTimeout(testTimer.current);
      recorder.cancel();
      return;
    }
    onError(null);
    setMicTested(false);
    const started = await recorder.start();
    if (!started) return;
    setMicTested(true);
    testTimer.current = window.setTimeout(() => {
      recorder.cancel();
      testTimer.current = null;
    }, 4_000);
  };

  const activeModel =
    status?.models.find((model) => model.id === form.whisper_model) ?? null;
  const downloadingModel = Boolean(
    progress?.model &&
      busy === progress.model &&
      progress.phase === "downloading",
  );
  const installProgress = busy === "install" ? progress : null;

  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="flex items-center gap-2 text-[12px] font-semibold text-white">
            <Mic className="h-3.5 w-3.5 text-acid-400" />
            Voice input · whisper.cpp
          </h3>
          <p className="mt-1 max-w-xl text-[10px] leading-4 text-slate-600">
            100% local speech-to-text. Recordings are captured as 16 kHz mono WAV
            in the webview, transcribed by a managed <span className="font-mono">whisper-cli</span>,
            and never leave this machine.
          </p>
        </div>
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[9.5px] font-medium uppercase tracking-wider",
            status?.installed
              ? "border-emerald-400/25 bg-emerald-400/5 text-emerald-300"
              : "border-amber-400/25 bg-amber-400/5 text-amber-200",
          )}
        >
          {status?.installed ? (
            <ShieldCheck className="h-3 w-3" />
          ) : (
            <AlertTriangle className="h-3 w-3" />
          )}
          {status?.installed ? status.version ?? "installed" : "not installed"}
        </span>
      </div>

      {statusError ? (
        <p className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/5 px-2.5 py-2 text-[10px] text-amber-200">
          {statusError}
        </p>
      ) : null}

      <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.02] p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-slate-200">
              {status?.message ?? "Checking Whisper…"}
            </p>
            <p className="mt-0.5 truncate font-mono text-[9px] text-slate-600">
              {status?.binary_path ?? "resolving managed bin folder…"}
            </p>
            {status?.installed ? (
              <p className="mt-0.5 truncate font-mono text-[9px] text-slate-700">
                libraries: {status.library_path}
              </p>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button
              size="sm"
              variant={status?.installed ? "secondary" : "primary"}
              busy={busy === "install"}
              onClick={() => void install()}
              icon={<Download className="h-3.5 w-3.5" />}
            >
              {status?.installed ? "Reinstall / update" : "Install whisper-cli"}
            </Button>
          </div>
        </div>

        {installProgress ? (
          <div className="mt-2">
            <div className="flex items-center justify-between gap-2 text-[9.5px] text-slate-400">
              <span className="min-w-0 truncate">{installProgress.message}</span>
              <span className="shrink-0 font-mono text-[9px] text-slate-500">
                {installProgress.percent !== null
                  ? `${Math.round(installProgress.percent)}%`
                  : installProgress.phase}
              </span>
            </div>
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-acid-400 transition-all"
                style={{ width: `${installProgress.percent ?? 12}%` }}
              />
            </div>
          </div>
        ) : null}

        {notice ? (
          <p className="mt-2 flex items-start gap-1.5 rounded-lg border border-emerald-400/20 bg-emerald-400/5 px-2.5 py-1.5 text-[10px] leading-4 text-emerald-200">
            <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0" />
            <span className="min-w-0">{notice}</span>
          </p>
        ) : null}
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[11px] font-medium text-slate-300">Speech model</p>
          <span className="text-[9px] uppercase tracking-wider text-slate-600">
            {status?.models_dir ?? "whisper-models"}
          </span>
        </div>
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          {status?.models.map((model) => {
            const selected = form.whisper_model === model.id;
            const isDownloading =
              busy === model.id && downloadingModel && progress?.model === model.id;
            return (
              <div
                key={model.id}
                className={cn(
                  "rounded-xl border p-2.5 transition",
                  selected
                    ? "border-acid-400/40 bg-acid-400/[0.05]"
                    : "border-white/10 bg-white/[0.02]",
                )}
              >
                <button
                  type="button"
                  onClick={() => update("whisper_model", model.id)}
                  className="w-full text-left"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="min-w-0">
                      <span className="block text-[11px] font-medium text-slate-200">
                        {model.label}
                      </span>
                      <span className="mt-0.5 block font-mono text-[9px] text-slate-600">
                        {model.id}
                      </span>
                    </span>
                    <span
                      className={cn(
                        "mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border",
                        selected
                          ? "border-acid-400 bg-acid-400"
                          : "border-white/20",
                      )}
                      aria-hidden
                    >
                      {selected ? (
                        <CheckCircle2 className="h-2.5 w-2.5 text-ink-950" />
                      ) : null}
                    </span>
                  </div>
                  <p className="mt-1.5 text-[9.5px] leading-4 text-slate-500">
                    {model.note}
                  </p>
                </button>
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <Badge>~{formatBytes(model.size_bytes, 0)} disk</Badge>
                  <Badge>{model.ram_mb} MB RAM</Badge>
                  <Badge>{model.multilingual ? "multilingual" : "English"}</Badge>
                  {model.installed ? (
                    <span className="ml-auto inline-flex items-center gap-1 text-[9px] text-emerald-300">
                      <CheckCircle2 className="h-3 w-3" /> installed
                    </span>
                  ) : (
                    <button
                      type="button"
                      disabled={busy !== null}
                      onClick={() => void download(model.id)}
                      className="ml-auto inline-flex items-center gap-1 rounded-md border border-acid-400/25 bg-acid-400/10 px-2 py-1 text-[9px] font-medium text-acid-200 transition hover:bg-acid-400/20 disabled:opacity-40"
                    >
                      {isDownloading ? (
                        <Loader2 className="h-2.5 w-2.5 animate-spin" />
                      ) : (
                        <Download className="h-2.5 w-2.5" />
                      )}
                      {isDownloading ? "Downloading" : "Download"}
                    </button>
                  )}
                </div>
                {isDownloading && progress ? (
                  <div className="mt-2">
                    <div className="flex items-center justify-between text-[9px] text-slate-500">
                      <span className="min-w-0 truncate">{progress.message}</span>
                      <span className="shrink-0 font-mono">
                        {progress.percent !== null
                          ? `${Math.round(progress.percent)}%`
                          : progress.phase}
                      </span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-acid-400 transition-all"
                        style={{ width: `${progress.percent ?? 5}%` }}
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            );
          }) ?? (
            <p className="rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-2 text-[10px] text-slate-600">
              Checking the managed models folder…
            </p>
          )}
        </div>
        {activeModel && !activeModel.installed ? (
          <p className="mt-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-2.5 py-2 text-[10px] text-amber-200">
            {activeModel.label} is selected but not downloaded. Download it, then
            save these settings.
          </p>
        ) : null}
        {downloadingModel ? (
          <button
            type="button"
            onClick={() => void cancelDownload()}
            className="mt-2 text-[9.5px] text-rose-300/80 underline decoration-rose-400/30 underline-offset-2 hover:text-rose-200"
          >
            Cancel download
          </button>
        ) : null}
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <label className="space-y-1.5">
          <span className="text-[10px] uppercase tracking-wider text-slate-600">
            Spoken language
          </span>
          <select
            value={form.whisper_language}
            onChange={(event) => update("whisper_language", event.target.value)}
            className="field font-mono text-[11px]"
          >
            {LANGUAGES.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
                {value === "auto" ? "" : ` · ${value}`}
              </option>
            ))}
          </select>
          <span className="block text-[9px] leading-4 text-slate-600">
            Passed to whisper-cli as <span className="font-mono">-l</span>. Auto
            costs a little extra time on short clips.
          </span>
        </label>

        <div className="space-y-1.5">
          <span className="text-[10px] uppercase tracking-wider text-slate-600">
            Microphone activation
          </span>
          <div className="grid grid-cols-2 gap-1.5">
            {(
              [
                ["toggle", "Click to toggle"],
                ["hold", "Hold to talk"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => update("voice_activation", value)}
                className={cn(
                  "rounded-lg border px-2 py-2 text-[10px] transition",
                  form.voice_activation === value
                    ? "border-acid-400/40 bg-acid-400/10 text-acid-200"
                    : "border-white/10 bg-white/[0.02] text-slate-500 hover:text-slate-300",
                )}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="block text-[9px] leading-4 text-slate-600">
            Hold mode records only while the mic button (or Space on it) is held.
            Ctrl + Space always toggles recording.
          </span>
        </div>
      </div>

      <div className="mt-4 space-y-3 border-t border-white/5 pt-4">
        <Toggle
          checked={form.voice_input_enabled}
          onChange={(value) => update("voice_input_enabled", value)}
          label="Enable voice input in the composer"
          description="Shows the mic button next to Send. Nothing is downloaded until you install whisper-cli or click a model above."
        />
        <Toggle
          checked={form.voice_auto_send}
          onChange={(value) => update("voice_auto_send", value)}
          label="Auto-send transcripts"
          description="On: a finished recording is sent as the next message. Off: it is inserted into this chat's draft so you can edit it first."
        />
        <Toggle
          checked={form.voice_strip_non_speech}
          onChange={(value) => update("voice_strip_non_speech", value)}
          label="Strip non-speech annotations"
          description="Removes Whisper tokens such as [BLANK_AUDIO], (music) and [silence] from the draft."
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] p-3">
        <Button
          size="sm"
          variant={recorder.state === "recording" ? "danger" : "secondary"}
          busy={recorder.state === "requesting"}
          onClick={() => void testMicrophone()}
          icon={
            recorder.state === "recording" ? (
              <Square className="h-3 w-3" />
            ) : (
              <Mic className="h-3.5 w-3.5" />
            )
          }
        >
          {recorder.state === "recording"
            ? "Stop test"
            : micTested
              ? "Test again"
              : "Test microphone"}
        </Button>
        <div className="flex h-5 min-w-[120px] flex-1 items-end gap-[2px]">
          {recorder.levels.slice(-32).map((value, index) => (
            <span
              key={index}
              className={cn(
                "flex-1 rounded-sm transition-[height] duration-75",
                recorder.state === "recording" ? "bg-acid-400/80" : "bg-white/10",
              )}
              style={{ height: `${Math.max(8, Math.round(value * 100))}%` }}
            />
          ))}
        </div>
        <span className="text-[9.5px] text-slate-500">
          {recorder.state === "recording"
            ? "Listening for 4 seconds — speak normally."
            : micTested
              ? "Input level captured from the system default microphone."
              : "Uses the system default input; no audio is transcribed."}
        </span>
      </div>
      {recorder.error ? (
        <p className="mt-2 rounded-lg border border-rose-400/20 bg-rose-500/5 px-2.5 py-2 text-[10px] leading-4 text-rose-300">
          {recorder.error}
        </p>
      ) : null}
    </section>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded border border-white/5 bg-white/5 px-1.5 py-0.5 font-mono text-[8.5px] text-slate-500">
      {children}
    </span>
  );
}
