export function usd(value: number, digits?: number): string {
  const d = digits ?? (Math.abs(value) >= 1 ? 2 : Math.abs(value) >= 0.01 ? 4 : 5);
  return `$${value.toFixed(d)}`;
}

export function ms(value: number | null | undefined): string {
  if (value == null) return "–";
  return `${Math.round(value).toLocaleString()} ms`;
}

export function duration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function num(value: string | number | undefined): number {
  return typeof value === "number" ? value : Number(value ?? 0);
}

export const STAGE_LABEL: Record<string, string> = {
  stt: "Speech-to-text",
  llm: "LLM",
  tts: "Text-to-speech",
  gpu: "GPU share",
  telephony: "Telephony",
  endpoint_detected: "Endpoint",
  stt_final: "STT final",
  llm_first_token: "LLM first token",
  llm_complete: "LLM complete",
  tts_first_byte: "TTS first byte",
  end_to_end: "End to end",
  perceived_delay: "Perceived delay",
};

export const PIPELINE_LABEL: Record<string, string> = {
  api: "API stack",
  selfhosted: "Self-hosted GPU",
};
