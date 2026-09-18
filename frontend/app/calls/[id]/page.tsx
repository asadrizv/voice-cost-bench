"use client";

import { use, useEffect, useState } from "react";
import { api, STAGES, type CallDetail, type CostReport } from "@/lib/api";
import { API_URL } from "@/lib/api";
import { duration, ms, PIPELINE_LABEL, STAGE_LABEL, usd } from "@/lib/format";

const UNIT_FOR: Record<string, [string, string]> = {
  stt: ["stt_audio_seconds", "stt_per_minute"],
  llm: ["llm_input_tokens", "llm_input_per_1k"],
  tts: ["tts_characters", "tts_per_1k_chars"],
  gpu: ["gpu_seconds", "gpu_per_hour"],
  telephony: ["telephony_seconds", "telephony_per_minute"],
};

/** Effective divisor implied by the stored cost: the time-weighted number of calls that
 *  shared the GPU with this one. Shown so units x rate / sharing visibly equals the cost. */
function sharing(cost: CostReport): number {
  return (cost.units.gpu_seconds * cost.rates.gpu_per_hour) / 3600 / cost.stages.gpu;
}

export default function CallDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [call, setCall] = useState<CallDetail | null>(null);
  const [cost, setCost] = useState<CostReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.call(id), api.cost(id)])
      .then(([c, r]) => { setCall(c); setCost(r); })
      .catch((e) => setError(String(e)));
  }, [id]);

  if (error) return <div className="banner error">{error}</div>;
  if (!call || !cost) return <p className="muted">Loading…</p>;

  return (
    <main>
      <div style={{ margin: "16px 0 20px" }}>
        <h1>{PIPELINE_LABEL[call.pipeline]} call · {duration(call.duration_seconds)}</h1>
        <p className="secondary small" style={{ margin: 0 }}>
          {new Date(call.started_at).toLocaleString()} · {call.persona} · {call.turns} turns · rates verified {cost.rate_card_verified_on}
        </p>
      </div>
      <div className="grid cols-4">
        <div className="card"><div className="stat-label">Total</div><div className="stat-value">{usd(cost.total_usd, 4)}</div></div>
        <div className="card"><div className="stat-label">Per minute</div><div className="stat-value">{usd(cost.cost_per_minute_usd, 4)}</div></div>
        <div className="card"><div className="stat-label">E2E p95</div><div className="stat-value">{ms(cost.latency_p95_ms.end_to_end)}</div></div>
        <div className="card"><div className="stat-label">Perceived p95</div><div className="stat-value">{ms(cost.latency_p95_ms.perceived_delay)}</div></div>
      </div>

      <section className="card section">
        <div className="card-head">
          <h2>Cost attribution</h2>
          <a className="small secondary" href={`${API_URL}/calls/${id}/cost`}>GET /calls/{id}/cost</a>
        </div>
        <table>
          <thead><tr><th>Stage</th><th className="num">Units</th><th className="num">Rate</th><th className="num">Cost</th></tr></thead>
          <tbody>
            {STAGES.map((s) => {
              const [unit, rate] = UNIT_FOR[s];
              return (
                <tr key={s}>
                  <td>{STAGE_LABEL[s]}</td>
                  <td className="num">
                    {s === "llm"
                      ? `${cost.units.llm_input_tokens.toLocaleString()} in / ${cost.units.llm_output_tokens.toLocaleString()} out tokens`
                      : `${Number(cost.units[unit]).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit.split("_").pop()}`}
                    {s === "gpu" && cost.stages.gpu > 0 && ` ÷ ${sharing(cost).toFixed(1)} calls sharing`}
                  </td>
                  <td className="num">
                    {s === "llm"
                      ? `$${cost.rates.llm_input_per_1k} / $${cost.rates.llm_output_per_1k} per 1k`
                      : `$${cost.rates[rate]} ${rate.replace(/^[a-z]+_/, "").replaceAll("_", " ")}`}
                  </td>
                  <td className="num">{usd(cost.stages[s], 5)}</td>
                </tr>
              );
            })}
            <tr><td><strong>Total</strong></td><td /><td /><td className="num"><strong>{usd(cost.total_usd, 5)}</strong></td></tr>
          </tbody>
        </table>
        {call.pipeline === "selfhosted" && (
          <p className="muted small">GPU share is this call&apos;s slice of the rented card: wall seconds divided by the calls sharing it.</p>
        )}
      </section>

      <section className="card section">
        <div className="card-head"><h2>Turns</h2></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Caller</th><th>Agent</th><th className="num">Perceived</th><th className="num">E2E</th><th className="num">Cost</th></tr></thead>
            <tbody>
              {call.turn_list.map((t) => (
                <tr key={t.index}>
                  <td>{t.index}</td>
                  <td>{t.user_text || <span className="muted">(greeting)</span>}</td>
                  <td>{t.agent_text}{t.interrupted && <span className="muted"> · interrupted</span>}</td>
                  <td className="num">{ms(t.latency?.perceived_delay)}</td>
                  <td className="num">{ms(t.latency?.end_to_end)}</td>
                  <td className="num">{usd(t.cost.total, 5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
