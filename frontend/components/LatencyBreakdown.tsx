"use client";

import { LATENCY_STAGES } from "@/lib/api";
import { ms, STAGE_LABEL } from "@/lib/format";
import type { LiveTurn } from "@/lib/useLiveMetrics";
import { HBars } from "./charts";

const BUDGET: Partial<Record<string, number>> = { end_to_end: 900, perceived_delay: 1200 };

export function LatencyBreakdown({ turns }: { turns: LiveTurn[] }) {
  const measured = turns.filter((t) => t.latency);
  const latest = measured[measured.length - 1];
  return (
    <section className="card" aria-label="Latency" data-testid="latency">
      <div className="card-head">
        <h2>Latency</h2>
        <span className="muted small">
          {latest ? `turn ${latest.index}` : "waiting for the first reply"} · tick marks are the p95 budgets
        </span>
      </div>
      {latest?.latency ? (
        <HBars
          rows={LATENCY_STAGES.map((s) => ({
            key: s,
            label: STAGE_LABEL[s],
            value: latest.latency![s],
            budget: BUDGET[s],
          }))}
        />
      ) : (
        <p className="muted small">Per-stage timings appear after each turn.</p>
      )}
      {measured.length > 0 && (
        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table>
            <thead>
              <tr>
                <th>Turn</th>
                <th className="num">Endpoint</th>
                <th className="num">LLM 1st token</th>
                <th className="num">TTS 1st byte</th>
                <th className="num">End to end</th>
                <th className="num">Perceived</th>
              </tr>
            </thead>
            <tbody>
              {measured.map((t) => (
                <tr key={t.index}>
                  <td>{t.index}{t.interrupted && <span className="muted"> · cut off</span>}</td>
                  <td className="num">{ms(t.latency!.endpoint_detected)}</td>
                  <td className="num">{ms(t.latency!.llm_first_token)}</td>
                  <td className="num">{ms(t.latency!.tts_first_byte)}</td>
                  <td className="num">{ms(t.latency!.end_to_end)}</td>
                  <td className="num">{ms(t.latency!.perceived_delay)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
