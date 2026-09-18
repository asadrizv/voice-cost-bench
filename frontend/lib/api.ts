export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export type Pipeline = "api" | "selfhosted";
export const STAGES = ["stt", "llm", "tts", "gpu", "telephony"] as const;
export type Stage = (typeof STAGES)[number];
export type Cost = Record<Stage | "total", number>;

export const LATENCY_STAGES = [
  "endpoint_detected",
  "stt_final",
  "llm_first_token",
  "llm_complete",
  "tts_first_byte",
  "end_to_end",
  "perceived_delay",
] as const;
export type LatencyStage = (typeof LATENCY_STAGES)[number];
export type Latency = Record<LatencyStage, number>;

export interface CallSummary {
  id: string;
  pipeline: Pipeline;
  persona: string;
  source: string;
  status: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number;
  turns: number;
  cost: Cost;
  cost_per_minute_usd: number;
  end_to_end_p95_ms: number;
  perceived_delay_p95_ms: number;
}

export interface Turn {
  index: number;
  user_text: string;
  agent_text: string;
  interrupted: boolean;
  usage: Record<string, number>;
  cost: Cost;
  latency: Latency | null;
}

export interface CallDetail extends CallSummary {
  usage: Record<string, number>;
  turn_list: Turn[];
}

export interface CostReport {
  call_id: string;
  pipeline: Pipeline;
  duration_seconds: number;
  total_usd: number;
  cost_per_minute_usd: number;
  projected_per_1000_minutes_usd: number;
  stages: Cost;
  units: Record<string, number>;
  rates: Record<string, number>;
  rate_card_verified_on: string;
  latency_p95_ms: { end_to_end: number; perceived_delay: number };
}

export interface PipelineSummary {
  pipeline: Pipeline;
  calls: number;
  turns: number;
  total_minutes: number;
  cost: Cost;
  cost_per_minute_usd: number;
  latency: Record<LatencyStage, { p50: number; p95: number; mean: number; count: number }>;
}

export interface LiveEvent {
  kind: "started" | "turn" | "tick" | "ended";
  call_id: string;
  pipeline: Pipeline;
  elapsed_seconds: number;
  running_cost: Record<Stage | "total", string>;
  cost_per_minute_usd: number;
  projected_per_1000_min_usd: number;
  turn_index: number | null;
  latency: Latency | null;
  turn_cost: Record<Stage | "total", string> | null;
  user_text: string;
  agent_text: string;
  interrupted: boolean;
  extra: Record<string, number>;
}

export interface TokenResponse {
  token: string;
  url: string;
  room: string;
  call_id: string;
  pipeline: Pipeline;
  persona: string;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  token: async (body: { pipeline: Pipeline; persona: string; endpointer: string }) => {
    const res = await fetch(`${API_URL}/token`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`token: ${res.status} ${await res.text()}`);
    return (await res.json()) as TokenResponse;
  },
  calls: (pipeline?: Pipeline) =>
    get<CallSummary[]>(`/calls${pipeline ? `?pipeline=${pipeline}` : ""}`),
  call: (id: string) => get<CallDetail>(`/calls/${id}`),
  cost: (id: string) => get<CostReport>(`/calls/${id}/cost`),
  compare: () => get<PipelineSummary[]>("/calls/compare"),
  config: () => get<Record<string, unknown>>("/config"),
  benchmark: (name = "benchmark.json") =>
    get<Benchmark>(`/benchmark?name=${encodeURIComponent(name)}`),
  runs: () => get<string[]>("/benchmark/runs"),
  liveUrl: (callId: string) => `${API_URL}/metrics/live/${callId}`,
};

export interface BenchmarkLevel {
  concurrency: number;
  calls_completed: number;
  calls_failed: number;
  calls_rejected: number;
  turns: number;
  call_minutes: number;
  cost_per_minute_usd: number;
  cost_per_minute_by_stage_usd: Cost;
  effective_concurrency: number;
  latency_ms: Record<LatencyStage, { p50: number; p95: number; mean: number }>;
  end_to_end_p95_ms: number;
  perceived_delay_p95_ms: number;
  within_budget: boolean;
  stt_wer: number | null;
  harness_valid: boolean;
  harness_lateness_p95_ms: number;
}

export interface Benchmark {
  provenance: Record<string, unknown> & { simulated: boolean; pipeline: string };
  budgets_ms: { end_to_end_p95_ms: number; perceived_delay_p95_ms: number };
  levels: BenchmarkLevel[];
  breaking_point: {
    concurrency: number;
    end_to_end_p95_ms: number;
    perceived_delay_p95_ms: number;
    last_within_budget: number | null;
  } | null;
  utilisation_curve: Array<Record<string, number>>;
  client_gpu_quotes: Array<{ provider: string; sku: string; region: string; hourly_usd: number }>;
  baseline?: { provenance: Record<string, unknown>; level: BenchmarkLevel | null };
}
