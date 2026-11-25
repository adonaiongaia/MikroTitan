import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import { errorMessage } from "../../lib/format";
import { benchmarkApi, BenchmarkEvents } from "./lib/api";
import type {
  BenchmarkPreset,
  BenchmarkProgress,
  BenchmarkSuite,
  BenchmarkTarget,
  BenchmarkWorkload,
} from "./types";

export interface BenchmarkController {
  suites: BenchmarkSuite[];
  presets: BenchmarkPreset[];
  progress: BenchmarkProgress | null;
  suite: BenchmarkSuite | null;
  activeWorkload: BenchmarkWorkload | null;
  activeTargetCount: number;
  running: boolean;
  busy: boolean;
  error: string | null;
  copied: boolean;
  start: (
    targets: BenchmarkTarget[],
    workload: BenchmarkWorkload,
    warmup: boolean,
  ) => Promise<void>;
  cancel: () => Promise<void>;
  remove: (id: string) => Promise<void>;
  copyMarkdown: (suite: BenchmarkSuite) => Promise<void>;
  selectSuite: (suite: BenchmarkSuite | null) => void;
  clearError: () => void;
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  document.body.removeChild(area);
}

export function useBenchmark(): BenchmarkController {
  const [suites, setSuites] = useState<BenchmarkSuite[]>([]);
  const [presets, setPresets] = useState<BenchmarkPreset[]>([]);
  const [progress, setProgress] = useState<BenchmarkProgress | null>(null);
  const [suite, setSuite] = useState<BenchmarkSuite | null>(null);
  const [activeWorkload, setActiveWorkload] =
    useState<BenchmarkWorkload | null>(null);
  const [activeTargetCount, setActiveTargetCount] = useState(0);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const copiedTimer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSuites(await benchmarkApi.list());
    } catch {
      // History is optional until the backend answers; the run itself will
      // surface a real error if the app is not in a Tauri window.
    }
  }, []);

  useEffect(() => {
    let disposed = false;
    const unlisteners: UnlistenFn[] = [];

    const attach = async () => {
      const progressUnlisten = await listen<BenchmarkProgress>(
        BenchmarkEvents.progress,
        (event) => {
          setProgress(event.payload);
          setRunning(true);
        },
      );
      const finishedUnlisten = await listen<BenchmarkSuite>(
        BenchmarkEvents.finished,
        (event) => {
          setSuite(event.payload);
          setProgress(null);
          setActiveWorkload(null);
          setActiveTargetCount(0);
          setRunning(false);
          setBusy(false);
          void refresh();
        },
      );
      if (disposed) {
        progressUnlisten();
        finishedUnlisten();
      } else {
        unlisteners.push(progressUnlisten, finishedUnlisten);
      }
    };

    void attach().catch(() => undefined);
    void benchmarkApi
      .workloadPresets()
      .then(setPresets)
      .catch(() => undefined);
    void refresh();

    return () => {
      disposed = true;
      unlisteners.forEach((unlisten) => unlisten());
      if (copiedTimer.current !== null) {
        window.clearTimeout(copiedTimer.current);
      }
    };
  }, [refresh]);

  const start = useCallback(
    async (
      targets: BenchmarkTarget[],
      workload: BenchmarkWorkload,
      warmup: boolean,
    ) => {
      setBusy(true);
      setError(null);
      setSuite(null);
      setProgress(null);
      setActiveWorkload(workload);
      setActiveTargetCount(targets.length);
      try {
        await benchmarkApi.start(targets, workload, warmup);
        setRunning(true);
      } catch (err) {
        setRunning(false);
        setError(errorMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const cancel = useCallback(async () => {
    setBusy(true);
    try {
      const wasActive = await benchmarkApi.cancel();
      if (!wasActive) {
        setRunning(false);
        setProgress(null);
        setActiveWorkload(null);
        setActiveTargetCount(0);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }, []);

  const remove = useCallback(
    async (id: string) => {
      if (running) return;
      if (!window.confirm("Delete this benchmark suite from the local history?")) {
        return;
      }
      setBusy(true);
      try {
        await benchmarkApi.remove(id);
        setSuites((current) => current.filter((entry) => entry.id !== id));
        setSuite((current) => (current?.id === id ? null : current));
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [running],
  );

  const copyMarkdown = useCallback(async (target: BenchmarkSuite) => {
    try {
      const markdown = await benchmarkApi.markdown(target.id);
      await copyText(markdown);
      setCopied(true);
      if (copiedTimer.current !== null) {
        window.clearTimeout(copiedTimer.current);
      }
      copiedTimer.current = window.setTimeout(() => setCopied(false), 2_500);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  const selectSuite = useCallback((next: BenchmarkSuite | null) => {
    setSuite(next);
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    suites,
    presets,
    progress,
    suite,
    activeWorkload,
    activeTargetCount,
    running,
    busy,
    error,
    copied,
    start,
    cancel,
    remove,
    copyMarkdown,
    selectSuite,
    clearError,
  };
}
