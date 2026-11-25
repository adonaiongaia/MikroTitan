import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../lib/api";
import { errorMessage } from "../lib/format";

/** Canonical sample rate whisper.cpp expects. */
export const VOICE_SAMPLE_RATE = 16_000;

// Global shortcuts defer Enter and Escape to the composer while a capture is
// live. A set (rather than a boolean) keeps the state correct if a recorder is
// torn down after another one starts.
const activeRecorders = new Set<symbol>();

export function isVoiceRecordingActive(): boolean {
  return activeRecorders.size > 0;
}
const LEVEL_BARS = 40;
const MIN_RECORDING_MS = 250;
// WebKitGTK opens the first enumerated input instead of the desktop default.
// On Linux, capture through the sound server, which honors @DEFAULT_SOURCE@.
const USE_NATIVE_CAPTURE =
  typeof navigator !== "undefined" && /Linux/i.test(navigator.userAgent);

export type VoiceRecorderState =
  | "idle"
  | "requesting"
  | "recording"
  | "encoding"
  | "error";

export interface RecordedVoice {
  /** Base64 WAV (no data URL prefix). */
  base64: string;
  mime: "audio/wav";
  durationMs: number;
}

interface AudioContextWindow {
  AudioContext?: typeof AudioContext;
  webkitAudioContext?: typeof AudioContext;
}

function createAudioContext(): AudioContext {
  const scope = window as unknown as AudioContextWindow;
  const Ctor = scope.AudioContext ?? scope.webkitAudioContext;
  if (!Ctor) {
    throw new Error(
      "This webview has no Web Audio support. Update WebKitGTK or use the desktop build.",
    );
  }
  return new Ctor();
}

/**
 * Linear resampling is deterministic, needs no codec, and avoids the
 * `audio/webm;codecs=opus` path that WebKitGTK's MediaRecorder would otherwise
 * force on whisper.cpp.
 */
export function resampleTo16k(
  samples: Float32Array,
  inputRate: number,
): Float32Array {
  if (!samples.length) return samples;
  if (inputRate === VOICE_SAMPLE_RATE) return samples;
  if (!Number.isFinite(inputRate) || inputRate <= 0) return samples;
  const outputLength = Math.max(
    1,
    Math.round((samples.length * VOICE_SAMPLE_RATE) / inputRate),
  );
  const step = inputRate / VOICE_SAMPLE_RATE;
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * step;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const fraction = position - left;
    output[index] =
      samples[left] * (1 - fraction) + samples[right] * fraction;
  }
  return output;
}

/** 16 kHz mono 16-bit PCM WAV, built in memory. */
export function encodeWavPcm16(samples: Float32Array): Uint8Array {
  const dataLength = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, VOICE_SAMPLE_RATE, true);
  view.setUint32(28, VOICE_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataLength, true);

  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(
      44 + index * 2,
      clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff,
      true,
    );
  }
  return new Uint8Array(buffer);
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(
      ...Array.from(bytes.subarray(index, index + chunk)),
    );
  }
  return window.btoa(binary);
}

/**
 * Decode an audio file the webview understands (notably WebM/Opus from screen
 * recorders) and convert it to the same canonical WAV the microphone emits.
 * whisper.cpp's release CLI does not decode WebM, so this conversion keeps the
 * attachment path local without adding a Rust codec sidecar.
 */
export async function decodeAudioFileToWav(file: File): Promise<RecordedVoice> {
  const context = createAudioContext();
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    const channelCount = Math.max(1, buffer.numberOfChannels);
    const mono = new Float32Array(buffer.length);
    for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
      const channel = buffer.getChannelData(channelIndex);
      for (let index = 0; index < mono.length; index += 1) {
        mono[index] += channel[index] / channelCount;
      }
    }
    const wav = encodeWavPcm16(resampleTo16k(mono, buffer.sampleRate));
    return {
      base64: bytesToBase64(wav),
      mime: "audio/wav",
      durationMs: Math.round(buffer.duration * 1000),
    };
  } finally {
    try {
      await context.close();
    } catch {
      // Closing an already-closed context is harmless.
    }
  }
}

function microphoneError(error: unknown): string {
  const name =
    error instanceof DOMException
      ? error.name
      : typeof error === "object" && error !== null && "name" in error
        ? String((error as { name?: unknown }).name ?? "")
        : "";
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return "Microphone permission was denied. Allow microphone access for Studio for Janas in your desktop privacy settings, then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone was found. Connect an input device and retry.";
    case "NotReadableError":
    case "TrackStartError":
      return "The microphone is busy or unavailable. Close the application using it and retry.";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError": {
      const constraint =
        typeof error === "object" && error !== null && "constraint" in error
          ? String((error as { constraint?: unknown }).constraint ?? "")
          : "";
      return constraint
        ? `This microphone cannot satisfy the ${constraint} audio setting. Try another input device.`
        : "This microphone does not support the requested audio settings. Try another input device.";
    }
    case "AbortError":
      return "The microphone stopped before recording began. Try again.";
    default: {
      const detail =
        error instanceof Error && error.message ? error.message : "";
      return detail
        ? `Microphone capture failed: ${detail}`
        : "Microphone capture failed. Check your input device and permissions.";
    }
  }
}

/**
 * Records canonical 16 kHz mono WAV. Linux uses the native Pulse/PipeWire
 * default source through Rust, because WebKitGTK otherwise picks the first
 * enumerated input. Other platforms retain the Web Audio path. Stopping
 * yields a ready-to-transcribe WAV plus live RMS levels for the meter.
 */
export function useVoiceRecorder() {
  const [state, setState] = useState<VoiceRecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [levels, setLevels] = useState<number[]>(() =>
    Array.from({ length: LEVEL_BARS }, () => 0),
  );
  const [elapsedMs, setElapsedMs] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sinkRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(48_000);
  const startedAtRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const levelTimerRef = useRef<number | null>(null);
  const nativeRef = useRef(false);
  const generationRef = useRef(0);
  const recordingRef = useRef(false);
  const stateRef = useRef<VoiceRecorderState>("idle");
  const recorderId = useRef(Symbol("voice-recorder"));

  useEffect(() => {
    const id = recorderId.current;
    if (state === "recording" || state === "requesting") {
      activeRecorders.add(id);
    } else {
      activeRecorders.delete(id);
    }
    return () => {
      activeRecorders.delete(id);
    };
  }, [state]);

  const updateState = useCallback((next: VoiceRecorderState) => {
    stateRef.current = next;
    setState(next);
  }, []);

  const release = useCallback(async () => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (levelTimerRef.current !== null) {
      window.clearInterval(levelTimerRef.current);
      levelTimerRef.current = null;
    }

    const processor = processorRef.current;
    processorRef.current = null;
    if (processor) {
      processor.onaudioprocess = null;
      try {
        processor.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    const source = sourceRef.current;
    sourceRef.current = null;
    if (source) {
      try {
        source.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    const sink = sinkRef.current;
    sinkRef.current = null;
    if (sink) {
      try {
        sink.disconnect();
      } catch {
        // Already disconnected.
      }
    }

    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach((track) => {
      try {
        track.stop();
      } catch {
        // A stopped track is already released.
      }
    });

    const context = contextRef.current;
    contextRef.current = null;
    if (context) {
      try {
        await context.close();
      } catch {
        // Closing an already-closed context is harmless.
      }
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
    if (stateRef.current === "error") updateState("idle");
  }, [updateState]);

  const start = useCallback(async (): Promise<boolean> => {
    if (
      recordingRef.current ||
      stateRef.current === "requesting" ||
      stateRef.current === "encoding"
    ) {
      return false;
    }
    const generation = ++generationRef.current;
    setError(null);
    setElapsedMs(0);
    setLevel(0);
    setLevels(Array.from({ length: LEVEL_BARS }, () => 0));

    if (USE_NATIVE_CAPTURE) {
      updateState("requesting");
      try {
        // Cancel is fire-and-forget in the UI. Wait out any previous session
        // before starting so a quick cancel/start pair cannot race the backend.
        await api.cancelNativeCapture().catch(() => undefined);
        await api.startNativeCapture();
        if (generation !== generationRef.current) {
          // The component unmounted or the user cancelled while IPC was in
          // flight. Do not leave a recorder running with no owner.
          await api.cancelNativeCapture().catch(() => undefined);
          updateState("idle");
          return false;
        }
        nativeRef.current = true;
        startedAtRef.current = performance.now();
        recordingRef.current = true;
        timerRef.current = window.setInterval(() => {
          setElapsedMs(performance.now() - startedAtRef.current);
        }, 100);
        levelTimerRef.current = window.setInterval(() => {
          void api
            .nativeCaptureLevel()
            .then((value) => {
              if (!nativeRef.current) return;
              const normalized = Math.max(0, Math.min(1, Number(value) || 0));
              setLevel(normalized);
              setLevels((previous) => {
                const next = previous.slice(1);
                next.push(normalized);
                return next;
              });
            })
            .catch(() => undefined);
        }, 100);
        updateState("recording");
        return true;
      } catch (err) {
        nativeRef.current = false;
        recordingRef.current = false;
        await api.cancelNativeCapture().catch(() => undefined);
        await release();
        setError(errorMessage(err));
        updateState("error");
        return false;
      }
    }

    const media = navigator.mediaDevices;
    if (!media?.getUserMedia) {
      setError(
        "Microphone capture is unavailable in this webview. Check that the app has permission and WebKitGTK is up to date.",
      );
      updateState("error");
      return false;
    }

    updateState("requesting");
    try {
      // Do not request echo cancellation, noise suppression or auto-gain.
      // WebKitGTK maps those to GStreamer's webrtcdsp/webrtcechoprobe, which
      // many Linux installs do not ship; requesting them makes getUserMedia
      // fail with OverconstrainedError even when the microphone is healthy.
      // We capture the raw stream and downmix/resample it ourselves instead.
      const stream = await media.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const context = createAudioContext();
      contextRef.current = context;
      if (context.state === "suspended") await context.resume();

      const source = context.createMediaStreamSource(stream);
      // A ScriptProcessor is intentional: it is implemented by WebKitGTK where
      // AudioWorklet module loading is inconsistent, and the input buffer is
      // already mixed to the requested single channel.
      const processor = context.createScriptProcessor(4096, 1, 1);
      const sink = context.createGain();
      sink.gain.value = 0;

      processor.onaudioprocess = (event) => {
        if (!recordingRef.current) return;
        const input = event.inputBuffer;
        const channelCount = Math.max(1, input.numberOfChannels);
        const mono = new Float32Array(input.length);
        for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
          const channel = input.getChannelData(channelIndex);
          for (let index = 0; index < mono.length; index += 1) {
            mono[index] += channel[index] / channelCount;
          }
        }
        chunksRef.current.push(mono);

        let sum = 0;
        for (let index = 0; index < mono.length; index += 1) {
          sum += mono[index] * mono[index];
        }
        const rms = Math.sqrt(sum / Math.max(1, mono.length));
        const normalized = Math.min(1, rms * 8);
        setLevel(normalized);
        setLevels((previous) => {
          const next = previous.slice(1);
          next.push(normalized);
          return next;
        });
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(context.destination);

      sourceRef.current = source;
      processorRef.current = processor;
      sinkRef.current = sink;
      sampleRateRef.current = context.sampleRate || 48_000;
      chunksRef.current = [];
      startedAtRef.current = performance.now();
      recordingRef.current = true;
      timerRef.current = window.setInterval(() => {
        setElapsedMs(performance.now() - startedAtRef.current);
      }, 100);
      updateState("recording");
      return true;
    } catch (err) {
      recordingRef.current = false;
      await release();
      setError(microphoneError(err));
      updateState("error");
      return false;
    }
  }, [release, updateState]);

  const finish = useCallback(async (): Promise<RecordedVoice | null> => {
    if (!recordingRef.current) return null;
    const elapsed = performance.now() - startedAtRef.current;
    const inputRate = sampleRateRef.current;

    if (nativeRef.current) {
      nativeRef.current = false;
      recordingRef.current = false;
      updateState("encoding");
      await release();
      try {
        const recording = await api.stopNativeCapture();
        if (recording.durationMs < MIN_RECORDING_MS) {
          setError("That recording was too short. Hold the mic a little longer.");
          updateState("error");
          setElapsedMs(0);
          setLevel(0);
          setLevels(Array.from({ length: LEVEL_BARS }, () => 0));
          return null;
        }
        updateState("idle");
        setElapsedMs(0);
        setLevel(0);
        setLevels(Array.from({ length: LEVEL_BARS }, () => 0));
        return {
          base64: recording.audioBase64,
          mime: "audio/wav",
          durationMs: recording.durationMs,
        };
      } catch (err) {
        setError(errorMessage(err));
        updateState("error");
        setElapsedMs(0);
        setLevel(0);
        setLevels(Array.from({ length: LEVEL_BARS }, () => 0));
        return null;
      }
    }

    const chunks = chunksRef.current;
    chunksRef.current = [];
    recordingRef.current = false;
    updateState("encoding");
    await release();

    const total = chunks.reduce((count, chunk) => count + chunk.length, 0);
    if (elapsed < MIN_RECORDING_MS || total === 0) {
      setError("That recording was too short. Hold the mic a little longer.");
      updateState("error");
      setElapsedMs(0);
      setLevel(0);
      return null;
    }

    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    const resampled = resampleTo16k(merged, inputRate);
    const wav = encodeWavPcm16(resampled);

    updateState("idle");
    setElapsedMs(0);
    setLevel(0);
    setLevels(Array.from({ length: LEVEL_BARS }, () => 0));
    return {
      base64: bytesToBase64(wav),
      mime: "audio/wav",
      durationMs: Math.round(elapsed),
    };
  }, [release, updateState]);

  const cancel = useCallback((): boolean => {
    const wasRecording = recordingRef.current;
    generationRef.current += 1;
    recordingRef.current = false;
    chunksRef.current = [];
    if (nativeRef.current) {
      nativeRef.current = false;
      void api.cancelNativeCapture().catch(() => undefined);
    }
    void release();
    updateState("idle");
    setElapsedMs(0);
    setLevel(0);
    setLevels(Array.from({ length: LEVEL_BARS }, () => 0));
    return wasRecording;
  }, [release, updateState]);

  useEffect(
    () => () => {
      generationRef.current += 1;
      recordingRef.current = false;
      if (nativeRef.current) {
        nativeRef.current = false;
        void api.cancelNativeCapture().catch(() => undefined);
      }
      void release();
    },
    [release],
  );

  return {
    state,
    error,
    level,
    levels,
    elapsedMs,
    start,
    finish,
    cancel,
    clearError,
  };
}
