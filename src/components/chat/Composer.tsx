import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  Bot,
  Check,
  Database,
  Eye,
  FolderOpen,
  Globe,
  ImagePlus,
  Loader2,
  Mic,
  Square,
  X,
} from "lucide-react";

import {
  bytesToBase64,
  decodeAudioFileToWav,
  useVoiceRecorder,
} from "../../hooks/useVoiceRecorder";
import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage, formatCount } from "../../lib/format";
import { shortcutLabel } from "../../lib/shortcuts";
import type { WhisperStatus } from "../../types";
import { Button } from "../ui/Button";
import { ProgressBar } from "../ui/ProgressBar";
import {
  ContextBreakdownPopover,
  type ContextBreakdownCategory,
  type ContextBreakdownItem,
} from "./ContextBreakdownPopover";

export interface ComposerImage {
  name: string;
  data_url: string;
  tokens: number;
}

/** The pre-flight warning shown when images are queued for a vision model
 * whose `--mmproj` projector is not loaded. */
export interface ComposerVisionWarning {
  modelName: string;
  family: string | null;
  candidateName: string | null;
  canDownload: boolean;
  busy: boolean;
  onPair: () => void;
  onDownload: () => void;
}

export interface ComposerVoiceProps {
  enabled: boolean;
  /** whisper-cli and at least one model are installed. */
  ready: boolean;
  status: WhisperStatus | null;
  activation: "toggle" | "hold" | string;
  autoSend: boolean;
  stripNonSpeech: boolean;
  onAutoSendChange: (value: boolean) => void;
  /** Insert a finished transcript into this chat's draft. */
  onTranscript: (text: string) => void;
  /** Send a finished transcript immediately. */
  onAutoSendTranscript: (text: string) => void;
  onOpenSettings: () => void;
}

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
  ready: boolean;
  contextUsed: number;
  contextMax: number;
  /** Token cost of the items the user pinned, for the breakdown warning. */
  pinnedTokens: number;
  contextCategories: ContextBreakdownCategory[];
  onToggleContextPin: (item: ContextBreakdownItem) => void;
  onToggleContextEvict: (item: ContextBreakdownItem) => void;
  onRestoreEvictedContext: () => void;
  internet: boolean;
  onInternetChange: (value: boolean) => void;
  agentMode: boolean;
  onAgentModeChange: (value: boolean) => void;
  images: ComposerImage[];
  onAddImages: (images: ComposerImage[]) => void;
  onRemoveImage: (index: number) => void;
  projectName: string | null;
  projectTokens: number;
  workspaceRoot: string;
  workspaceLocked: boolean;
  onPickProject: () => void;
  onClearProject: () => void;
  activity: string | null;
  error: string | null;
  onDismissError: () => void;
  visionWarning?: ComposerVisionWarning | null;
  /** Collections attached to this chat; 0 renders the muted Docs badge. */
  docsCount: number;
  docsLabel: string | null;
  onOpenDocs: () => void;
  /** Informational line from the Rust turn, e.g. the retrieval summary. */
  ragNotice: string | null;
  onDismissRagNotice: () => void;
  voice: ComposerVoiceProps;
}

function readImage(file: File): Promise<ComposerImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read that image."));
    reader.onload = () => {
      const dataUrl = String(reader.result);
      const image = new Image();
      image.onload = () => {
        const tokens =
          Math.ceil(image.width / 28) * Math.ceil(image.height / 28) + 1;
        resolve({ name: file.name, data_url: dataUrl, tokens });
      };
      image.onerror = () => reject(new Error("That image could not be decoded."));
      image.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  streaming,
  ready,
  contextUsed,
  contextMax,
  pinnedTokens,
  contextCategories,
  onToggleContextPin,
  onToggleContextEvict,
  onRestoreEvictedContext,
  internet,
  onInternetChange,
  agentMode,
  onAgentModeChange,
  images,
  onAddImages,
  onRemoveImage,
  projectName,
  projectTokens,
  workspaceRoot,
  workspaceLocked,
  onPickProject,
  onClearProject,
  activity,
  error,
  onDismissError,
  visionWarning,
  docsCount,
  docsLabel,
  onOpenDocs,
  ragNotice,
  onDismissRagNotice,
  voice,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [localError, setLocalError] = useState<string | null>(null);
  const [breakdownOpen, setBreakdownOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const recorder = useVoiceRecorder();
  const holdMode = voice.activation === "hold";

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "0px";
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`;
  }, [value]);

  const startVoice = useCallback(async () => {
    setLocalError(null);
    recorder.clearError();
    if (!voice.enabled) {
      setLocalError(
        "Voice input is turned off. Enable it in Settings → Voice and save the change.",
      );
      return;
    }
    if (!voice.ready) {
      setLocalError(
        "Voice input needs whisper-cli and a downloaded model. Opening Settings → Voice.",
      );
      voice.onOpenSettings();
      return;
    }
    await recorder.start();
  }, [
    recorder.clearError,
    recorder.start,
    voice.enabled,
    voice.onOpenSettings,
    voice.ready,
  ]);

  const finishVoice = useCallback(async () => {
    const audio = await recorder.finish();
    if (!audio) return;
    setVoiceBusy(true);
    setLocalError(null);
    try {
      const result = await api.transcribeAudio(
        audio.base64,
        audio.mime,
        "recording.wav",
        null,
        null,
        voice.stripNonSpeech,
      );
      const text = result.text.trim();
      if (!text) {
        setLocalError(
          "No speech was detected. Try again closer to the microphone.",
        );
        return;
      }
      if (voice.autoSend) voice.onAutoSendTranscript(text);
      else voice.onTranscript(text);
    } catch (err) {
      setLocalError(errorMessage(err));
    } finally {
      setVoiceBusy(false);
    }
  }, [
    recorder.finish,
    voice.autoSend,
    voice.onAutoSendTranscript,
    voice.onTranscript,
    voice.stripNonSpeech,
  ]);

  const transcribeAudioFiles = async (files: File[]) => {
    if (!voice.enabled) {
      setLocalError(
        "Voice input is turned off. Enable it in Settings → Voice and save the change.",
      );
      return;
    }
    if (!voice.ready) {
      setLocalError(
        "Audio attachments need whisper-cli and a downloaded model. Opening Settings → Voice.",
      );
      voice.onOpenSettings();
      return;
    }
    setVoiceBusy(true);
    setLocalError(null);
    try {
      const pieces: string[] = [];
      for (const file of files) {
        if (file.size > 200 * 1024 * 1024) {
          setLocalError(`${file.name} is larger than the 200 MB audio limit.`);
          continue;
        }
        let base64: string;
        let mime: string;
        let name: string;
        if (isWebmFile(file)) {
          // whisper.cpp does not decode WebM/Opus. WebKit's own decoder turns
          // it into PCM here, then the backend receives the same WAV the mic
          // recorder produces.
          try {
            const converted = await decodeAudioFileToWav(file);
            base64 = converted.base64;
            mime = converted.mime;
            name = `${file.name.replace(/\.[^.]+$/, "") || "recording"}.wav`;
          } catch {
            setLocalError(
              `${file.name} contains a WebM/Opus track this webview could not decode. Convert it to WAV, MP3, OGG or FLAC, or use the microphone button.`,
            );
            continue;
          }
        } else {
          const bytes = new Uint8Array(await file.arrayBuffer());
          base64 = bytesToBase64(bytes);
          mime = file.type || "application/octet-stream";
          name = file.name;
        }
        const result = await api.transcribeAudio(
          base64,
          mime,
          name,
          null,
          null,
          voice.stripNonSpeech,
        );
        const text = result.text.trim();
        if (text) pieces.push(text);
        else setLocalError(`No speech was detected in ${file.name}.`);
      }
      if (pieces.length) voice.onTranscript(pieces.join(" "));
    } catch (err) {
      setLocalError(errorMessage(err));
    } finally {
      setVoiceBusy(false);
    }
  };

  const addFiles = async (files: FileList | File[]) => {
    const list = Array.from(files);
    const audio = list.filter(isAudioFile).slice(0, 2);
    if (audio.length) void transcribeAudioFiles(audio);

    const picked = list
      .filter((file) => file.type.startsWith("image/"))
      .slice(0, 4 - images.length);
    if (!picked.length) return;
    setLocalError(null);
    try {
      const parsed = await Promise.all(picked.map(readImage));
      onAddImages([...images, ...parsed]);
    } catch (err) {
      setLocalError(errorMessage(err));
    }
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.code === "Space") {
        if (!voice.enabled) return;
        event.preventDefault();
        if (recorder.state === "recording") void finishVoice();
        else if (recorder.state === "idle" && !voiceBusy) void startVoice();
        return;
      }
      if (event.key === "Escape" && recorder.state === "recording") {
        event.preventDefault();
        recorder.cancel();
        return;
      }
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        recorder.state === "recording"
      ) {
        event.preventDefault();
        void finishVoice();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    finishVoice,
    recorder.cancel,
    recorder.state,
    startVoice,
    voice.enabled,
    voiceBusy,
  ]);

  const toggleMic = () => {
    if (recorder.state === "recording") void finishVoice();
    else void startVoice();
  };

  const contextPercent =
    contextMax > 0 ? Math.min(1, contextUsed / contextMax) : 0;
  const ringColor =
    contextPercent > 0.9
      ? "#fb7185"
      : contextPercent > 0.7
        ? "#fbbf24"
        : "rgb(var(--accent-400))";

  return (
    <div className="shrink-0 border-t border-white/5 bg-ink-950/80 px-6 py-3.5">
      <div className="mx-auto max-w-3xl">
        {error ? (
          <div className="mb-2 flex items-start gap-2 rounded-xl border border-rose-400/20 bg-rose-500/5 px-3.5 py-2.5 text-[11px] leading-5 text-rose-300">
            <span className="flex-1">{error}</span>
            <button
              type="button"
              onClick={onDismissError}
              className="text-rose-400/60 transition hover:text-rose-200"
              aria-label="Dismiss error"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : null}

        {visionWarning ? (
          <div className="mb-2 rounded-xl border border-sky-400/25 bg-sky-400/[0.06] px-3.5 py-2.5">
            <div className="flex items-start gap-2 text-[11px] leading-5 text-sky-100">
              <Eye className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-300" />
              <div className="min-w-0 flex-1">
                <p className="font-medium">
                  {visionWarning.modelName} needs a vision projector
                </p>
                <p className="text-[10.5px] leading-4 text-sky-200/70">
                  Images are queued, but llama.cpp will reject them until an
                  mmproj is bound
                  {visionWarning.family
                    ? ` for ${visionWarning.family}`
                    : ""}
                  .
                </p>
              </div>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 pl-5">
              {visionWarning.candidateName ? (
                <Button
                  size="sm"
                  variant="primary"
                  busy={visionWarning.busy}
                  onClick={visionWarning.onPair}
                  icon={<Eye className="h-3 w-3" />}
                >
                  Pair detected MMPROJ
                </Button>
              ) : null}
              {visionWarning.canDownload ? (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={visionWarning.onDownload}
                >
                  Download MMPROJ
                </Button>
              ) : null}
              {!visionWarning.candidateName && !visionWarning.canDownload ? (
                <p className="text-[10px] text-amber-200/80">
                  Download an mmproj for this family in Models, then pair it in
                  Library.
                </p>
              ) : null}
            </div>
          </div>
        ) : null}

        {localError || recorder.error ? (
          <div className="mb-2 flex items-start gap-2 rounded-xl border border-amber-400/20 bg-amber-400/5 px-3.5 py-2.5 text-[11px] leading-5 text-amber-200">
            <span className="flex-1">{localError ?? recorder.error}</span>
            <button
              type="button"
              onClick={() => {
                setLocalError(null);
                recorder.clearError();
              }}
              className="text-amber-300/60 transition hover:text-amber-100"
              aria-label="Dismiss message"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : null}

        {ragNotice ? (
          <div className="mb-2 flex items-start gap-2 rounded-xl border border-acid-400/20 bg-acid-400/[0.04] px-3 py-1.5 text-[10px] leading-5 text-acid-200/90">
            <Database className="mt-0.5 h-3 w-3 shrink-0 text-acid-400" />
            <span className="min-w-0 flex-1">{ragNotice}</span>
            <button
              type="button"
              onClick={onDismissRagNotice}
              className="text-acid-400/60 transition hover:text-acid-100"
              aria-label="Dismiss retrieval summary"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ) : null}

        {activity ? (
          <div className="mb-2 rounded-xl border border-sky-400/20 bg-sky-400/5 px-3 py-2">
            <div className="flex items-center gap-2 text-[10.5px] text-sky-200">
              <Loader2 className="h-3 w-3 animate-spin" />
              <span className="min-w-0 flex-1 truncate">{activity}</span>
              <span className="text-[9px] uppercase tracking-wider text-sky-400/60">
                web pipeline
              </span>
            </div>
            <ProgressBar value={null} tone="sky" className="mt-1.5" />
          </div>
        ) : null}

        {(projectName || images.length) ? (
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            {projectName ? (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-acid-400/20 bg-acid-400/5 px-2.5 py-1 text-[10px] text-acid-200/90">
                <FolderOpen className="h-3 w-3" />
                {projectName}
                <span className="font-mono text-[9px] text-acid-500/70">
                  {formatCount(projectTokens)} tok
                </span>
                <button
                  type="button"
                  onClick={onClearProject}
                  className="text-acid-400/60 hover:text-acid-100"
                  aria-label="Remove project context"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ) : null}
            {images.map((image, index) => (
              <span
                key={`${image.name}-${index}`}
                className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 py-1 pl-1 pr-2 text-[10px] text-slate-300"
              >
                <img
                  src={image.data_url}
                  alt={image.name}
                  className="h-5 w-5 rounded object-cover"
                />
                {image.name.slice(0, 16)}
                <span className="font-mono text-[9px] text-slate-600">
                  {image.tokens} tok
                </span>
                <button
                  type="button"
                  onClick={() => onRemoveImage(index)}
                  className="text-slate-500 hover:text-slate-200"
                  aria-label="Remove image"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        ) : null}

        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            void addFiles(event.dataTransfer.files);
          }}
          className={cn(
            "rounded-2xl border bg-ink-900 p-2 shadow-panel transition",
            dragging
              ? "border-acid-400/50 bg-acid-400/[0.04]"
              : "border-white/10 focus-within:border-acid-400/30",
          )}
        >
          {recorder.state === "recording" ||
          recorder.state === "encoding" ||
          voiceBusy ? (
            <div className="mb-1.5 flex flex-wrap items-center gap-2 rounded-xl border border-acid-400/25 bg-acid-400/[0.06] px-2.5 py-2">
              <span className="relative flex h-2 w-2 shrink-0">
                {recorder.state === "recording" ? (
                  <>
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-400/70" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-400" />
                  </>
                ) : (
                  <Loader2 className="h-2.5 w-2.5 animate-spin text-acid-300" />
                )}
              </span>
              <div className="flex h-5 min-w-[90px] flex-1 items-end gap-[2px] overflow-hidden">
                {recorder.levels.slice(-36).map((value, index) => (
                  <span
                    key={index}
                    className="flex-1 rounded-sm bg-acid-400/80 transition-[height] duration-75"
                    style={{
                      height: `${
                        recorder.state === "recording"
                          ? Math.max(8, Math.round(value * 100))
                          : 12
                      }%`,
                    }}
                  />
                ))}
              </div>
              <span className="font-mono text-[10px] tabular-nums text-acid-200">
                {voiceBusy && recorder.state !== "recording"
                  ? "transcribing…"
                  : formatClock(recorder.elapsedMs)}
              </span>
              {voice.status?.active_model ? (
                <span className="hidden font-mono text-[8.5px] text-acid-500/70 sm:inline">
                  {voice.status.active_model}
                </span>
              ) : null}
              <button
                type="button"
                onClick={() => voice.onAutoSendChange(!voice.autoSend)}
                className={cn(
                  "inline-flex h-6 items-center rounded-md border px-2 text-[9px] transition",
                  voice.autoSend
                    ? "border-acid-400/40 bg-acid-400/15 text-acid-100"
                    : "border-white/10 bg-white/5 text-slate-400 hover:text-slate-200",
                )}
                title="Choose whether a finished transcript is sent immediately or inserted into the draft"
              >
                {voice.autoSend ? "Auto-send" : "Insert to draft"}
              </button>
              {recorder.state === "recording" ? (
                <>
                  <button
                    type="button"
                    onClick={() => recorder.cancel()}
                    className="inline-flex h-6 items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 text-[9px] text-slate-400 transition hover:text-slate-200"
                    title="Cancel recording (Esc)"
                  >
                    <X className="h-3 w-3" />
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => void finishVoice()}
                    className="inline-flex h-6 items-center gap-1 rounded-md bg-acid-400 px-2 text-[9px] font-semibold text-ink-950 transition hover:bg-acid-300"
                    title="Finish and transcribe (Enter)"
                  >
                    <Check className="h-3 w-3" />
                    Finish
                  </button>
                </>
              ) : null}
            </div>
          ) : null}

          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            disabled={!ready || streaming}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              // Mod+Enter belongs to the global shortcut manager so it works
              // from anywhere; plain Enter still sends from the textarea.
              if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (ready && !streaming) onSend();
              }
            }}
            placeholder={
              ready
                ? agentMode
                  ? "Assign a task to your local agent…"
                  : internet
                    ? "Ask the model, and it will read the web first…"
                    : "Message your local model…"
                : "Start the Janas engine to begin chatting"
            }
            className="max-h-[220px] min-h-[52px] w-full resize-none bg-transparent px-3 py-2.5 text-[13.5px] leading-6 text-slate-100 outline-none placeholder:text-slate-600 disabled:cursor-not-allowed"
          />

          <div className="flex items-center justify-between gap-2 px-1 pb-0.5">
            <div className="flex items-center gap-1">
              <input
                ref={fileRef}
                type="file"
                accept="image/*,audio/wav,audio/mpeg,audio/ogg,audio/flac,audio/webm,video/webm,.wav,.mp3,.ogg,.flac,.webm"
                multiple
                className="hidden"
                onChange={(event) => {
                  if (event.target.files) void addFiles(event.target.files);
                  event.target.value = "";
                }}
              />
              <IconButton
                label={
                  agentMode
                    ? "Image attachments are not available in Agent Mode"
                    : "Attach images or audio"
                }
                onClick={() => fileRef.current?.click()}
                disabled={images.length >= 4 || agentMode}
              >
                <ImagePlus className="h-3.5 w-3.5" />
              </IconButton>
              <IconButton
                label="Attach a local project folder"
                onClick={onPickProject}
                disabled={!ready}
              >
                <FolderOpen className="h-3.5 w-3.5" />
              </IconButton>
              <button
                type="button"
                onClick={() => onAgentModeChange(!agentMode)}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-lg border px-2 text-[10px] transition",
                  agentMode
                    ? "border-acid-400/40 bg-acid-400/10 text-acid-200 shadow-glow"
                    : "border-white/10 bg-white/5 text-slate-500 hover:text-slate-300",
                )}
                aria-pressed={agentMode}
                title={`Let the local agent plan, call tools and report each step (${shortcutLabel("toggle-agent")})`}
              >
                <Bot className="h-3 w-3" />
                {agentMode ? "Agent on" : "Agent"}
              </button>
              <button
                type="button"
                onClick={() => onInternetChange(!internet)}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-lg border px-2 text-[10px] transition",
                  internet
                    ? "border-sky-400/30 bg-sky-400/10 text-sky-200"
                    : "border-white/10 bg-white/5 text-slate-500 hover:text-slate-300",
                )}
                aria-pressed={internet}
                title={`Search the web and bundle the top pages into the prompt (${shortcutLabel("toggle-internet")})`}
              >
                <Globe className="h-3 w-3" />
                {internet ? "Web on" : "Web off"}
              </button>
              <button
                type="button"
                onClick={onOpenDocs}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-lg border px-2 text-[10px] transition",
                  docsCount
                    ? "border-acid-400/40 bg-acid-400/10 text-acid-200"
                    : "border-white/10 bg-white/5 text-slate-500 hover:text-slate-300",
                )}
                aria-pressed={docsCount > 0}
                title={
                  docsCount
                    ? `Docs: ${docsLabel ?? `${docsCount} collections`}`
                    : "Attach knowledge collections to this chat"
                }
              >
                <BookOpen className="h-3 w-3" />
                Docs
                {docsCount ? (
                  <span className="rounded bg-acid-400/20 px-1 font-mono text-[8.5px] text-acid-100">
                    {docsCount}
                  </span>
                ) : null}
              </button>

            </div>

            <div className="flex items-center gap-2">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setBreakdownOpen((value) => !value)}
                  aria-expanded={breakdownOpen}
                  aria-label="Show context usage breakdown"
                  title="Show the context breakdown: pin what stays, evict what goes"
                  className="rounded-lg transition hover:bg-white/5"
                >
                  <ContextRing
                    used={contextUsed}
                    max={contextMax}
                    percent={contextPercent}
                    color={ringColor}
                  />
                </button>
                <ContextBreakdownPopover
                  open={breakdownOpen}
                  onClose={() => setBreakdownOpen(false)}
                  categories={contextCategories}
                  contextUsed={contextUsed}
                  contextMax={contextMax}
                  pinnedTokens={pinnedTokens}
                  onTogglePin={onToggleContextPin}
                  onToggleEvict={onToggleContextEvict}
                  onRestoreEvicted={onRestoreEvictedContext}
                />
              </div>

              {voice.enabled ? (
                <button
                  type="button"
                  disabled={
                    voiceBusy ||
                    recorder.state === "requesting" ||
                    recorder.state === "encoding"
                  }
                  onClick={holdMode ? undefined : toggleMic}
                  onPointerDown={
                    holdMode
                                              ? (event) => {
                            if (event.button !== 0) return;
                            event.preventDefault();
                            try {
                              event.currentTarget.setPointerCapture(
                                event.pointerId,
                              );
                            } catch {
                              // Pointer capture is best-effort.
                            }
                            if (recorder.state === "idle") void startVoice();
                          }
                      : undefined
                  }
                  onPointerUp={
                    holdMode
                      ? () => {
                          if (recorder.state === "recording")
                            void finishVoice();
                        }
                      : undefined
                  }
                  onPointerCancel={
                    holdMode ? () => recorder.cancel() : undefined
                  }
                  onKeyDown={
                    holdMode
                      ? (event) => {
                          if (event.code === "Space" && !event.repeat) {
                            event.preventDefault();
                            if (recorder.state === "idle") void startVoice();
                          }
                        }
                      : undefined
                  }
                  onKeyUp={
                    holdMode
                      ? (event) => {
                          if (event.code === "Space") {
                            event.preventDefault();
                            if (recorder.state === "recording")
                              void finishVoice();
                          }
                        }
                      : undefined
                  }
                  aria-pressed={recorder.state === "recording"}
                  aria-label={
                    voice.ready
                      ? holdMode
                        ? "Hold to talk"
                        : "Record voice input"
                      : "Set up voice input in Settings"
                  }
                  title={
                    voice.ready
                      ? holdMode
                        ? "Hold to talk · Ctrl+Space toggles"
                        : "Voice input · Ctrl+Space"
                      : "Whisper is not set up yet · open Settings → Voice"
                  }
                  className={cn(
                    "inline-flex h-8 w-8 items-center justify-center rounded-lg border transition disabled:cursor-not-allowed disabled:opacity-40",
                    recorder.state === "recording"
                      ? "border-rose-400/40 bg-rose-500/15 text-rose-200 shadow-glow"
                      : voice.ready
                        ? "border-white/10 bg-white/5 text-slate-400 hover:border-acid-400/30 hover:text-acid-200"
                        : "border-amber-400/25 bg-amber-400/5 text-amber-300/80 hover:text-amber-200",
                  )}
                >
                  {recorder.state === "requesting" ||
                  recorder.state === "encoding" ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Mic className="h-3.5 w-3.5" />
                  )}
                </button>
              ) : null}

              {streaming ? (
                <Button
                  size="sm"
                  variant="danger"
                  onClick={onStop}
                  title="Stop generating (Esc)"
                  icon={<Square className="h-3 w-3" />}
                >
                  Stop
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="primary"
                  disabled={!ready || !value.trim()}
                  onClick={onSend}
                  title={`Send (${shortcutLabel("send")})`}
                  icon={<ArrowUp className="h-3.5 w-3.5" />}
                >
                  Send
                </Button>
              )}
            </div>
          </div>
        </div>

        {agentMode ? (
          <div className="mt-1.5 flex min-w-0 items-center gap-1.5 rounded-lg border border-white/5 bg-white/[0.02] px-2 py-1 text-[9.5px] text-slate-600">
            <FolderOpen className="h-3 w-3 shrink-0 text-acid-500/70" />
            <span
              className="min-w-0 break-all font-mono text-slate-400"
              title={workspaceRoot || "No agent workspace selected"}
            >
              {workspaceRoot || "no agent workspace selected"}
            </span>
            {workspaceLocked ? (
              <span
                className="ml-auto shrink-0 rounded border border-amber-400/20 bg-amber-400/5 px-1 py-px text-[8px] uppercase tracking-wider text-amber-300/80"
                title="The agent workspace is pinned while a run or pending undo exists."
              >
                locked
              </span>
            ) : null}
          </div>
        ) : null}

        <div className="mt-1.5 flex items-center justify-between text-[9.5px] text-slate-600">
          <span>
            {agentMode ? "Enter to assign · " : "Enter to send · "}
            Shift + Enter for a new line
            {voice.enabled ? " · Ctrl + Space dictation" : ""} ·{" "}
            <span className="text-acid-500/80">
              {agentMode ? "agent workspace" : ready ? "local engine" : "engine offline"}
            </span>
          </span>
          <span className="font-mono">
            {formatCount(contextUsed)} / {formatCount(contextMax)} ctx
          </span>
        </div>
      </div>
    </div>
  );
}

function isAudioFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return (
    file.type.startsWith("audio/") ||
    [".wav", ".mp3", ".ogg", ".oga", ".flac", ".webm"].some((extension) =>
      name.endsWith(extension),
    )
  );
}

function isWebmFile(file: File): boolean {
  return (
    file.name.toLowerCase().endsWith(".webm") ||
    file.type === "audio/webm" ||
    file.type === "video/webm"
  );
}

function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function IconButton({
  children,
  label,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-500 transition hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  );
}

function ContextRing({
  used,
  max,
  percent,
  color,
}: {
  used: number;
  max: number;
  percent: number;
  color: string;
}) {
  const radius = 9;
  const circumference = 2 * Math.PI * radius;
  return (
    <div
      className="relative flex h-7 w-7 items-center justify-center"
      title={`Context ${formatCount(used)} of ${formatCount(max)} tokens (${Math.round(
        percent * 100,
      )}%)`}
    >
      <svg viewBox="0 0 24 24" className="h-7 w-7 -rotate-90">
        <circle
          cx="12"
          cy="12"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="3"
        />
        <circle
          cx="12"
          cy="12"
          r={radius}
          fill="none"
          style={{ stroke: color }}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - percent)}
          className="transition-all duration-500"
        />
      </svg>
      <span className="absolute font-mono text-[7px] text-slate-500">
        {Math.round(percent * 100)}
      </span>
    </div>
  );
}
