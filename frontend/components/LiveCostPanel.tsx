"use client";

import { STAGES } from "@/lib/api";
import { duration, num, PIPELINE_LABEL, STAGE_LABEL, usd } from "@/lib/format";
import type { LiveState } from "@/lib/useLiveMetrics";
import { StackedBar } from "./charts";

const STAGE_COLOR = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

export function LiveCostPanel({ live }: { live: LiveState }) {
  const last = live.last;
  const total = num(last?.running_cost.total);
  const perMinute = last?.cost_per_minute_usd ?? 0;
  const concurrency = last?.extra?.concurrency;
  return (
    <section className="card" aria-label="Live cost" data-testid="live-cost">
      <div className="card-head">
        <h2>Cost</h2>
        <span className="muted small">
          {last ? `${PIPELINE_LABEL[last.pipeline]} · ${duration(last.elapsed_seconds)}` : "No call yet"}
          {concurrency != null && ` · ${concurrency} on GPU`}
        </span>
      </div>
      <div className="stat-label">Cost per minute</div>
      <div className="hero-value" data-testid="cost-per-minute">{usd(perMinute, 4)}</div>
      <div className="grid cols-4" style={{ marginTop: 16 }}>
        <div>
          <div className="stat-label">Running total</div>
          <div className="stat-value" data-testid="running-total">{usd(total, 4)}</div>
        </div>
        <div>
          <div className="stat-label">Per 1,000 minutes</div>
          <div className="stat-value">{usd(perMinute * 1000, 2)}</div>
        </div>
      </div>
      <div style={{ marginTop: 20 }}>
        <div className="stat-label" style={{ marginBottom: 8 }}>Where it went</div>
        <StackedBar
          segments={STAGES.map((s, i) => ({
            key: s,
            label: STAGE_LABEL[s],
            value: num(last?.running_cost[s]),
            color: STAGE_COLOR[i],
            display: usd(num(last?.running_cost[s]), 4),
          }))}
        />
      </div>
    </section>
  );
}
