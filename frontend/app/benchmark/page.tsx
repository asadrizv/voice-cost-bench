"use client";

import { useEffect, useMemo, useState } from "react";
import { LineChart } from "@/components/charts";
import { api, type Benchmark, type TelephonyQuote } from "@/lib/api";
import { ms, usd } from "@/lib/format";

export default function BenchmarkPage() {
  const [runs, setRuns] = useState<string[]>([]);
  const [name, setName] = useState("benchmark.json");
  const [data, setData] = useState<Benchmark | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .runs()
      .then((available) => {
        setRuns(available);
        if (available.length && !available.includes("benchmark.json")) setName(available[0]);
      })
      .catch(() => setRuns([]));
  }, []);
  useEffect(() => {
    setError(null);
    api.benchmark(name).then(setData).catch((e) => { setData(null); setError(String(e)); });
  }, [name]);

  const levels = data?.levels ?? [];
  const baseline = data?.baseline?.level ?? null;
  const breaking = data?.breaking_point ?? null;
  const cheapestOk = useMemo(
    () => levels.filter((l) => l.within_budget).sort((a, b) => a.cost_per_minute_usd - b.cost_per_minute_usd)[0],
    [levels],
  );
  const prov = data?.provenance;
  const carriers = data?.telephony_quotes ?? [];
  const priced = carriers.filter((q) => q.per_minute_usd != null);
  const unpriced = carriers.filter((q) => q.per_minute_usd == null);

  return (
    <main>
      <div className="row" style={{ margin: "16px 0 20px" }}>
        <div>
          <h1>Cost and latency under load</h1>
          <p className="secondary small" style={{ margin: 0 }}>
            Synthetic callers replay the same fixture conversation at each concurrency level on one GPU.
          </p>
        </div>
        <span className="spacer" />
        {runs.length > 0 && (
          <select aria-label="Run" value={name} onChange={(e) => setName(e.target.value)}>
            {runs.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        )}
      </div>

      {error && (
        <div className="banner">
          No benchmark yet. Run <code>make benchmark</code> (GPU) or <code>make benchmark-sim</code> (offline, simulated).
        </div>
      )}
      {prov?.simulated && (
        <div className="banner warn" role="note" data-testid="simulated-banner">
          <strong>Simulated run.</strong> These numbers come from a queueing model of a GPU, not a measurement.
          They exercise the harness and must not be quoted.
        </div>
      )}

      {data && (
        <>
          <div className="grid cols-4">
            <div className="card">
              <div className="stat-label">Cheapest within budget</div>
              <div className="stat-value">{cheapestOk ? usd(cheapestOk.cost_per_minute_usd, 4) : "–"}</div>
              <div className="muted small">{cheapestOk ? `at ${cheapestOk.concurrency} concurrent, p95 ${ms(cheapestOk.end_to_end_p95_ms)}` : "no level met the budget"}</div>
            </div>
            <div className="card">
              <div className="stat-label">Breaking point</div>
              <div className="stat-value">{breaking ? `${breaking.concurrency} calls` : "not reached"}</div>
              <div className="muted small">
                {breaking ? `E2E p95 ${ms(breaking.end_to_end_p95_ms)}; last within budget ${breaking.last_within_budget ?? "none"}` : `p95 stayed under ${data.budgets_ms.end_to_end_p95_ms} ms at every level tested`}
              </div>
            </div>
            <div className="card">
              <div className="stat-label">API baseline</div>
              <div className="stat-value">{baseline ? usd(baseline.cost_per_minute_usd, 4) : "–"}</div>
              <div className="muted small">{baseline ? `1 call, E2E p95 ${ms(baseline.end_to_end_p95_ms)}` : "run with --baseline to include"}</div>
            </div>
            <div className="card">
              <div className="stat-label">GPU</div>
              <div className="stat-value" style={{ fontSize: 18 }}>{String((prov?.gpu as Record<string, unknown>)?.sku ?? "–")}</div>
              <div className="muted small">
                {String((prov?.gpu as Record<string, unknown>)?.provider ?? "")} {String((prov?.gpu as Record<string, unknown>)?.region ?? "")} · ${String((prov?.gpu as Record<string, unknown>)?.hourly_usd ?? "")}/h
              </div>
            </div>
          </div>

          <div className="grid cols-2 section">
            <section className="card">
              <div className="card-head"><h2>Cost per minute vs concurrency</h2></div>
              <LineChart
                xLabel="Concurrent calls"
                formatY={(v) => usd(v, 3)}
                highlightX={breaking?.concurrency}
                series={[{ key: "selfhosted", label: "Self-hosted", color: "var(--series-2)",
                  points: levels.map((l) => ({ x: l.concurrency, y: l.cost_per_minute_usd })) }]}
                references={baseline ? [{ label: "API stack", y: baseline.cost_per_minute_usd }] : []}
              />
            </section>
            <section className="card">
              <div className="card-head"><h2>p95 latency vs concurrency</h2></div>
              <LineChart
                xLabel="Concurrent calls"
                formatY={(v) => `${Math.round(v)} ms`}
                highlightX={breaking?.concurrency}
                series={[
                  { key: "e2e", label: "End to end", color: "var(--series-1)",
                    points: levels.map((l) => ({ x: l.concurrency, y: l.end_to_end_p95_ms })) },
                  { key: "perceived", label: "Perceived", color: "var(--series-2)",
                    points: levels.map((l) => ({ x: l.concurrency, y: l.perceived_delay_p95_ms })) },
                ]}
                references={[
                  { label: "900 ms budget", y: data.budgets_ms.end_to_end_p95_ms },
                  { label: "1200 ms budget", y: data.budgets_ms.perceived_delay_p95_ms },
                ]}
              />
            </section>
          </div>

          <section className="card section">
            <div className="card-head"><h2>Levels</h2><span className="muted small">Every number is stated with its concurrency</span></div>
            <div className="table-wrap">
              <table data-testid="levels-table">
                <thead>
                  <tr>
                    <th className="num">Concurrent</th><th className="num">Calls</th><th className="num">Turns</th>
                    <th className="num">Cost / min</th><th className="num">GPU / min</th><th className="num">Telephony / min</th>
                    <th className="num">E2E p50</th><th className="num">E2E p95</th><th className="num">Perceived p95</th>
                    <th className="num">STT WER</th><th>Budget</th><th>Harness</th>
                  </tr>
                </thead>
                <tbody>
                  {levels.map((l) => (
                    <tr key={l.concurrency}>
                      <td className="num">{l.concurrency}</td>
                      <td className="num">{l.calls_completed}{l.calls_failed ? ` (+${l.calls_failed} failed)` : ""}</td>
                      <td className="num">{l.turns}</td>
                      <td className="num">{usd(l.cost_per_minute_usd, 4)}</td>
                      <td className="num">{usd(l.cost_per_minute_by_stage_usd.gpu, 4)}</td>
                      <td className="num">{usd(l.cost_per_minute_by_stage_usd.telephony, 4)}</td>
                      <td className="num">{ms(l.latency_ms.end_to_end.p50)}</td>
                      <td className="num">{ms(l.end_to_end_p95_ms)}</td>
                      <td className="num">{ms(l.perceived_delay_p95_ms)}</td>
                      <td className="num">{l.stt_wer == null ? "–" : `${(l.stt_wer * 100).toFixed(1)}%`}</td>
                      <td><span className="badge"><span className="status-dot" style={{ background: l.within_budget ? "var(--good)" : "var(--critical)" }} />{l.within_budget ? "within" : "over"}</span></td>
                      <td>{l.harness_valid ? <span className="muted small">ok</span> : <span className="badge"><span className="status-dot" style={{ background: "var(--warning)" }} />fell behind</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {priced.length > 0 && (
            <section className="card section">
              <div className="card-head">
                <h2>Cost per minute by carrier</h2>
                <span className="muted small">Only the telephony line changes between carriers</span>
              </div>
              <div className="table-wrap">
                <table data-testid="carrier-table">
                  <thead>
                    <tr>
                      <th className="num">Concurrent</th>
                      {priced.map((q) => (
                        <th key={q.carrier} className="num">
                          {q.carrier} ${q.per_minute_usd}/min{q.selected ? " (selected)" : ""}
                          {!q.verified && <> <UnverifiedBadge /></>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {levels.map((l) => (
                      <tr key={l.concurrency}>
                        <td className="num">{l.concurrency}</td>
                        {priced.map((q) => {
                          const cost = l.cost_per_minute_by_carrier_usd?.[q.carrier];
                          return <td key={q.carrier} className="num">{cost == null ? "–" : usd(cost, 4)}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ul className="muted small">
                {carriers.map((q) => <CarrierSource key={q.carrier} quote={q} />)}
              </ul>
              {unpriced.length > 0 && (
                <p className="muted small">
                  No cost column for {unpriced.map((q) => q.carrier).join(", ")}: no per-minute price is published.
                </p>
              )}
            </section>
          )}

          {data.utilisation_curve.length > 0 && (
            <section className="card section">
              <div className="card-head">
                <h2>Loaded cost at GPU utilisation</h2>
                <span className="muted small">A card rented for an hour that serves 15 minutes still costs the hour</span>
              </div>
              <div className="table-wrap">
                <table data-testid="utilisation-table">
                  <thead>
                    <tr>
                      <th className="num">Concurrent</th><th className="num">Utilisation</th>
                      <th className="num">Cost / min (benchmark GPU)</th>
                      {data.client_gpu_quotes.map((q) => <th key={q.provider} className="num">{q.provider} ${q.hourly_usd}/h</th>)}
                      {baseline && <th className="num">vs API</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {data.utilisation_curve.map((row) => (
                      <tr key={`${row.concurrency}-${row.utilisation}`}>
                        <td className="num">{row.concurrency}</td>
                        <td className="num">{Math.round(row.utilisation * 100)}%</td>
                        <td className="num">{usd(row.cost_per_minute_usd, 4)}</td>
                        {data.client_gpu_quotes.map((q) => (
                          <td key={q.provider} className="num">{usd(row[`cost_per_minute_usd_${q.provider}`], 4)}</td>
                        ))}
                        {baseline && (
                          <td className="num">{pct(1 - row.cost_per_minute_usd / baseline.cost_per_minute_usd)}</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <details>
            <summary>Provenance</summary>
            <pre>{JSON.stringify(prov, null, 2)}</pre>
          </details>
        </>
      )}
    </main>
  );
}

function UnverifiedBadge() {
  return (
    <span className="badge" title="Fetched from the carrier's site but not checked by a person: do not quote">
      <span className="status-dot" style={{ background: "var(--warning)" }} />unverified
    </span>
  );
}

function CarrierSource({ quote }: { quote: TelephonyQuote }) {
  return (
    <li>
      <a href={quote.source_url}>{quote.carrier}</a>{" "}
      {quote.verified ? `verified ${quote.checked_on}` : `unverified, fetched ${quote.checked_on}`}
      {quote.note && `: ${quote.note}`}
    </li>
  );
}

function pct(saving: number): string {
  const p = Math.round(saving * 100);
  return p >= 0 ? `${p}% cheaper` : `${-p}% dearer`;
}
