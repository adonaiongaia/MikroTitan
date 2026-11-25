export type BenchmarkEngine = "janas" | "llama";

export type BenchmarkStatus = "completed" | "failed" | "cancelled";

export type BenchmarkPhase = "switching_engine" | "warmup" | "measuring";

export interface BenchmarkTarget {
  id: string;
  label: string;
  engine: BenchmarkEngine;
  model_path: string;
  ctx_size: number;
  gpu_layers: number;
  extra_flags: string;
}

export interface BenchmarkWorkload {
  preset: string;
  label: string;
  prompt: string;
  seed: number;
  temperature: number;
  max_tokens: number;
  structured: boolean;
}

export interface BenchmarkPreset {
  id: string;
  label: string;
  description: string;
  prompt: string;
  max_tokens: number;
  structured: boolean;
}

export interface BenchmarkRunResult {
  target_id: string;
  target_label: string;
  engine: BenchmarkEngine;
  ttft_ms: number;
  prefill_tps: number;
  decode_tps: number;
  total_duration_ms: number;
  tokens_generated: number;
  peak_rss_bytes: number;
  ssd_bytes_read: number;
  peak_ssd_mbps: number;
  peak_iops: number;
  adapter_warnings: string[];
  status: BenchmarkStatus;
  error: string | null;
  engine_version: string | null;
  adapter: string | null;
}

export interface BenchmarkDevice {
  present: boolean;
  name: string;
  utilization: number | null;
  memory_used: number | null;
  memory_total: number | null;
}

export interface BenchmarkHardware {
  os: string;
  arch: string;
  cpu: string;
  cores: number;
  mem_total_bytes: number;
  gpu: BenchmarkDevice;
  npu: BenchmarkDevice;
}

export interface BenchmarkSuite {
  id: string;
  created_at_ms: number;
  workload: BenchmarkWorkload;
  warmup: boolean;
  targets: BenchmarkTarget[];
  results: BenchmarkRunResult[];
  hardware: BenchmarkHardware;
  status: BenchmarkStatus;
  error: string | null;
  duration_ms: number;
}

export interface BenchmarkProgress {
  suite_id: string;
  target_index: number;
  target_count: number;
  target_id: string;
  target_label: string;
  engine: BenchmarkEngine;
  phase: BenchmarkPhase;
  tokens_generated: number;
  live_tps: number;
  elapsed_ms: number;
  rss_bytes: number;
  read_bytes_per_sec: number;
  read_ops_per_sec: number;
  message: string;
}
