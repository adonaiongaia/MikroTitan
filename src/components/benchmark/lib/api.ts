import { invoke } from "@tauri-apps/api/core";

import type {
  BenchmarkPreset,
  BenchmarkSuite,
  BenchmarkTarget,
  BenchmarkWorkload,
} from "../types";

export const BenchmarkEvents = {
  progress: "benchmark://progress",
  finished: "benchmark://finished",
} as const;

export const benchmarkApi = {
  workloadPresets: () =>
    invoke<BenchmarkPreset[]>("benchmark_workload_presets"),
  start: (
    targets: BenchmarkTarget[],
    workload: BenchmarkWorkload,
    warmup: boolean,
  ) => invoke<string>("start_benchmark_suite", { targets, workload, warmup }),
  cancel: () => invoke<boolean>("cancel_benchmark"),
  list: () => invoke<BenchmarkSuite[]>("list_benchmark_suites"),
  remove: (id: string) =>
    invoke<boolean>("delete_benchmark_suite", { id }),
  markdown: (suiteId: string) =>
    invoke<string>("export_benchmark_markdown", { suite_id: suiteId }),
};
