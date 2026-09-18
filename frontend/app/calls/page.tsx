"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { StackedBar } from "@/components/charts";
import { api, STAGES, type CallSummary, type PipelineSummary } from "@/lib/api";
import { duration, ms, PIPELINE_LABEL, STAGE_LABEL, usd } from "@/lib/format";

const STAGE_COLOR = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

export default function HistoryPage() {
  const [calls, setCalls] = useState<CallSummary[] | null>(null);
  const [compare, setCompare] = useState<PipelineSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.calls(), api.compare()])
      .then(([c, s]) => {
        setCalls(c);
        setCompare(s);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const [a, b] = [compare.find((s) => s.pipeline === "api"), compare.find((s) => s.pipeline === "selfhosted")];
  const saving = a && b && a.cost_per_minute_usd > 0 && b.calls > 0
    ? 1 - b.cost_per_minute_usd / a.cost_per_minute_usd
    : null;

  return (
    <main>
      <div className="row" style={{ margin: "16px 0 20px" }}>
        <h1>Completed calls</h1>
        <span className="spacer" />
        {saving != null && (
          <span className="secondary small">
            Self-hosted is <strong>{Math.round(Math.abs(saving) * 100)}% {saving >= 0 ? "cheaper" : "more expensive"}</strong> per minute
            across these calls (browser calls, one at a time; see Benchmark for cost at load).
          </span>
        )}
      </div>
      {error && <div className="banner error">{error}</div>}

      <div className="grid cols-2">
        {compare.map((s) => (
          <section key={s.pipeline} className="card" data-testid={`compare-${s.pipeline}`}>
            <div className="card-head">
              <h2>{PIPELINE_LABEL[s.pipeline]}</h2>
              <span className="muted small">{s.calls} calls · {s.total_minutes.toFixed(1)} min</span>
            </div>
            <div className="grid cols-4">
              <div><div className="stat-label">Cost per minute</div><div className="stat-value">{usd(s.cost_per_minute_usd, 4)}</div></div>
              <div><div className="stat-label">End to end p95</div><div className="stat-value">{ms(s.latency.end_to_end.p95)}</div></div>
              <div><div className="stat-label">Perceived p95</div><div className="stat-value">{ms(s.latency.perceived_delay.p95)}</div></div>
            </div>
            <div style={{ marginTop: 16 }}>
              <StackedBar segments={STAGES.map((k, i) => ({
                key: k, label: STAGE_LABEL[k], value: s.cost[k], color: STAGE_COLOR[i], display: usd(s.cost[k], 4),
              }))} />
            </div>
          </section>
        ))}
      </div>

      <section className="card section">
        <div className="card-head"><h2>Calls</h2><span className="muted small">Load-test calls are excluded</span></div>
        {calls && calls.length === 0 && <p className="muted small">No calls yet. Make one from the Call page.</p>}
        {calls && calls.length > 0 && (
          <div className="table-wrap">
            <table data-testid="calls-table">
              <thead>
                <tr>
                  <th>Started</th><th>Pipeline</th><th>Persona</th><th>Status</th>
                  <th className="num">Duration</th><th className="num">Turns</th>
                  <th className="num">Cost</th><th className="num">Per minute</th>
                  <th className="num">E2E p95</th><th className="num">Perceived p95</th>
                </tr>
              </thead>
              <tbody>
                {calls.map((c) => (
                  <tr key={c.id}>
                    <td><Link href={`/calls/${c.id}`}>{new Date(c.started_at).toLocaleString()}</Link></td>
                    <td>{PIPELINE_LABEL[c.pipeline]}</td>
                    <td>{c.persona}</td>
                    <td>{c.status}</td>
                    <td className="num">{duration(c.duration_seconds)}</td>
                    <td className="num">{c.turns}</td>
                    <td className="num">{usd(c.cost.total, 4)}</td>
                    <td className="num">{usd(c.cost_per_minute_usd, 4)}</td>
                    <td className="num">{ms(c.end_to_end_p95_ms)}</td>
                    <td className="num">{ms(c.perceived_delay_p95_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
