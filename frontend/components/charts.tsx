"use client";

import { useEffect, useMemo, useRef, useState } from "react";

/** Width of the element in CSS pixels, so SVG text renders at its real size instead of
 *  being scaled down with a fixed viewBox. */
function useWidth<T extends HTMLElement>(fallback = 600): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

export interface Segment {
  key: string;
  label: string;
  value: number;
  color: string;
  display: string;
}

/** One horizontal stacked bar, 2px surface gaps between segments, legend below. */
export function StackedBar({ segments, height = 20 }: { segments: Segment[]; height?: number }) {
  const [hover, setHover] = useState<{ seg: Segment; x: number } | null>(null);
  const shown = segments.filter((s) => s.value > 0);
  const total = shown.reduce((a, s) => a + s.value, 0);
  const width = 1000;
  const gap = shown.length > 1 ? 2 : 0;
  const usable = width - gap * Math.max(shown.length - 1, 0);
  let x = 0;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ height }}
        role="img" aria-label={segments.map((s) => `${s.label} ${s.display}`).join(", ")}>
        {total === 0 && <rect x={0} y={0} width={width} height={height} rx={4} fill="var(--surface-2)" />}
        {shown.map((s, i) => {
          const w = (s.value / total) * usable;
          const rect = (
            <rect key={s.key} x={x} y={0} width={Math.max(w, 1)} height={height} fill={s.color}
              rx={i === shown.length - 1 || i === 0 ? 4 : 0}
              onMouseEnter={() => setHover({ seg: s, x: (x + w / 2) / width })}
              onMouseLeave={() => setHover(null)} />
          );
          x += w + gap;
          return rect;
        })}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: `${hover.x * 100}%`, top: height + 6, transform: "translateX(-50%)" }}>
          <span className="swatch" style={{ background: hover.seg.color }} />
          {hover.seg.label}: <strong>{hover.seg.display}</strong>
          {total > 0 && <span className="muted"> · {Math.round((hover.seg.value / total) * 100)}%</span>}
        </div>
      )}
      <div className="legend" style={{ marginTop: 10 }}>
        {segments.map((s) => (
          <span key={s.key}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label} <span className="muted">{s.display}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export interface BarRow {
  key: string;
  label: string;
  value: number;
  budget?: number;
}

/** Horizontal bars (single series, so no legend), optional budget tick per row. */
export function HBars({ rows, max, unit = "ms" }: { rows: BarRow[]; max?: number; unit?: string }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const top = max ?? Math.max(1, ...rows.map((r) => Math.max(r.value, r.budget ?? 0))) * 1.1;
  const labelW = 118;
  const valueW = 72;
  const plotW = W - labelW - valueW;
  const rowH = 28;
  const barH = 14;
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH}`} width={W} height={rows.length * rowH} role="img"
        aria-label={rows.map((r) => `${r.label} ${Math.round(r.value)} ${unit}`).join(", ")}>
        {rows.map((r, i) => {
          const y = i * rowH;
          const w = Math.max(0, (r.value / top) * plotW);
          const over = r.budget != null && r.value > r.budget;
          return (
            <g key={r.key}>
              <title>{`${r.label}: ${Math.round(r.value)} ${unit}${r.budget ? ` (budget ${r.budget} ${unit})` : ""}`}</title>
              <text x={0} y={y + rowH / 2 + 4} style={{ fill: "var(--text-secondary)", fontSize: 12 }}>{r.label}</text>
              <line x1={labelW} x2={labelW} y1={y + 4} y2={y + rowH - 4} stroke="var(--axis)" />
              <path d={roundedRight(labelW, y + (rowH - barH) / 2, w, barH)}
                fill={over ? "var(--critical)" : "var(--series-1)"} />
              {r.budget != null && (
                <line x1={labelW + (r.budget / top) * plotW} x2={labelW + (r.budget / top) * plotW}
                  y1={y + 3} y2={y + rowH - 3} stroke="var(--text-secondary)" strokeWidth={2} />
              )}
              <text x={W} y={y + rowH / 2 + 4} textAnchor="end"
                style={{ fill: "var(--text-primary)", fontSize: 12, fontVariantNumeric: "tabular-nums" }}>
                {Math.round(r.value).toLocaleString()} {unit}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function roundedRight(x: number, y: number, w: number, h: number): string {
  const r = Math.min(4, w / 2, h / 2);
  if (w <= 0) return "";
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}

export interface Series {
  key: string;
  label: string;
  color: string;
  points: Array<{ x: number; y: number }>;
}

export interface Reference {
  label: string;
  y: number;
}

/** Line chart over a numeric x (concurrency). Crosshair + tooltip on hover; one y-axis. */
export function LineChart({
  series,
  references = [],
  formatY,
  xLabel,
  highlightX,
}: {
  series: Series[];
  references?: Reference[];
  formatY: (v: number) => string;
  xLabel: string;
  highlightX?: number | null;
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hoverX, setHoverX] = useState<number | null>(null);
  const H = 260, left = 56, right = 96, top = 12, bottom = 40;
  const xs = useMemo(
    () => Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b),
    [series],
  );
  const allY = [...series.flatMap((s) => s.points.map((p) => p.y)), ...references.map((r) => r.y)];
  const yMax = niceMax(Math.max(1e-9, ...allY));
  const xMin = xs[0] ?? 0, xMax = xs[xs.length - 1] ?? 1;
  const sx = (x: number) => left + ((x - xMin) / Math.max(xMax - xMin, 1)) * (W - left - right);
  const sy = (y: number) => top + (1 - y / yMax) * (H - top - bottom);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax);

  function onMove(e: React.MouseEvent) {
    const box = ref.current?.getBoundingClientRect();
    if (!box || xs.length === 0) return;
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = xs[0];
    for (const x of xs) if (Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
    setHoverX(best);
  }

  return (
    <div className="chart" ref={ref} onMouseMove={onMove} onMouseLeave={() => setHoverX(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={series.map((s) => s.label).join(", ")}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={W - right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={left - 8} y={sy(t) + 4} textAnchor="end">{formatY(t)}</text>
          </g>
        ))}
        <line x1={left} x2={W - right} y1={sy(0)} y2={sy(0)} stroke="var(--axis)" />
        {xs.map((x) => (
          <text key={x} x={sx(x)} y={H - bottom + 18} textAnchor="middle">{x}</text>
        ))}
        <text x={(left + W - right) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        {highlightX != null && (
          <line x1={sx(highlightX)} x2={sx(highlightX)} y1={top} y2={sy(0)} stroke="var(--critical)" strokeWidth={1} />
        )}
        {references.map((r) => (
          <g key={r.label}>
            <line x1={left} x2={W - right} y1={sy(r.y)} y2={sy(r.y)} stroke="var(--text-secondary)" strokeWidth={1} />
            <text x={left + 6} y={sy(r.y) - 5} style={{ fill: "var(--text-secondary)" }}>{r.label}</text>
          </g>
        ))}
        {series.map((s) => (
          <g key={s.key}>
            <path d={s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ")}
              fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            {s.points.map((p) => (
              <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={hoverX === p.x ? 5 : 4} fill={s.color}
                stroke="var(--surface-1)" strokeWidth={2} />
            ))}
            {s.points.length > 0 && (
              <text x={sx(s.points[s.points.length - 1].x) + 10} y={sy(s.points[s.points.length - 1].y) + 4}
                style={{ fill: "var(--text-primary)" }}>
                {s.label}
              </text>
            )}
          </g>
        ))}
        {hoverX != null && (
          <line x1={sx(hoverX)} x2={sx(hoverX)} y1={top} y2={sy(0)} stroke="var(--axis)" />
        )}
      </svg>
      {hoverX != null && (
        <div className="tooltip" style={{ left: `${(sx(hoverX) / W) * 100}%`, top: 8, transform: "translateX(12px)" }}>
          <div className="secondary">{xLabel}: <strong>{hoverX}</strong></div>
          {series.map((s) => {
            const p = s.points.find((q) => q.x === hoverX);
            return p ? (
              <div key={s.key}><span className="swatch" style={{ background: s.color }} />{s.label}: <strong>{formatY(p.y)}</strong></div>
            ) : null;
          })}
        </div>
      )}
      {series.length > 1 && (
        <div className="legend">
          {series.map((s) => (
            <span key={s.key}><span className="swatch" style={{ background: s.color }} />{s.label}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function niceMax(v: number): number {
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (v <= m * exp) return m * exp;
  return 10 * exp;
}
