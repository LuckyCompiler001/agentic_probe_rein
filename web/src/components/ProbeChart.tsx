"use client";

import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useQuery } from "@tanstack/react-query";
import { api, type ProbeResult } from "@/lib/api";

const STROKES = [
  "#0a0a0a",
  "#dc2626",
  "#16a34a",
  "#2563eb",
  "#ea580c",
  "#7c3aed",
  "#ca8a04",
  "#0891b2",
];

function buildSeries(results: ProbeResult[]) {
  const epochs = new Set<number>();
  results.forEach((r) => r.values.forEach((v) => epochs.add(v.epoch)));
  return [...epochs]
    .sort((a, b) => a - b)
    .map((ep) => {
      const row: Record<string, number> = { epoch: ep };
      for (const r of results) {
        const point = r.values.find((v) => v.epoch === ep);
        if (point) row[`iter_${r.n}`] = point.value;
      }
      return row;
    });
}

export function ProbeChart({ runId }: { runId: string }) {
  const q = useQuery({
    queryKey: ["probe-results", runId],
    queryFn: () => api.listProbeResults(runId),
    refetchInterval: 4000,
  });

  const results = q.data?.results ?? [];
  if (results.length === 0) {
    return (
      <section className="card p-6">
        <div className="h-section mb-2">Probe trajectory</div>
        <div className="text-sm text-fg-muted">No probe results yet.</div>
      </section>
    );
  }

  const series = buildSeries(results);
  const threshold = (() => {
    const t = results[0].threshold;
    const n = typeof t === "number" ? t : parseFloat(String(t));
    return Number.isFinite(n) ? n : null;
  })();

  const latest = results[results.length - 1];
  const metricName = latest.metric_name;

  return (
    <section className="card overflow-hidden">
      <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-border">
        <div>
          <div className="font-medium tracking-tight text-[0.95rem]">
            Probe trajectory
          </div>
          <div className="text-xs text-fg-muted mt-0.5 mono">
            {metricName}
            {threshold !== null && <> · threshold {threshold}</>}
          </div>
        </div>
      </div>
      <div className="px-5 py-4 bg-surface">
        <div className="h-[22rem]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 18, right: 28, left: 8, bottom: 12 }}>
              <CartesianGrid stroke="#e7e5e4" strokeDasharray="2 4" />
              <XAxis
                dataKey="epoch"
                stroke="#9ca3af"
                tick={{ fontFamily: "var(--font-geist-mono)", fontSize: 11, fill: "#6b7280" }}
                tickLine={{ stroke: "#9ca3af" }}
                axisLine={{ stroke: "#d4d4d4" }}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fontFamily: "var(--font-geist-mono)", fontSize: 11, fill: "#6b7280" }}
                tickLine={{ stroke: "#9ca3af" }}
                axisLine={{ stroke: "#d4d4d4" }}
                domain={["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{
                  background: "#ffffff",
                  border: "1px solid #e7e5e4",
                  borderRadius: 8,
                  fontFamily: "var(--font-geist-mono)",
                  fontSize: 12,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
                }}
                labelStyle={{ color: "#0a0a0a" }}
                formatter={(value, name) => [
                  typeof value === "number" ? value.toFixed(5) : String(value),
                  name as string,
                ]}
              />
              <Legend
                wrapperStyle={{
                  fontFamily: "var(--font-geist-sans)",
                  fontSize: 12,
                  color: "#6b7280",
                  paddingTop: 6,
                }}
              />
              {threshold !== null && (
                <ReferenceLine
                  y={threshold}
                  stroke="#dc2626"
                  strokeDasharray="6 4"
                  label={{
                    value: `threshold ${threshold}`,
                    fill: "#dc2626",
                    fontFamily: "var(--font-geist-mono)",
                    fontSize: 11,
                    position: "right",
                  }}
                />
              )}
              {results.map((r, i) => (
                <Line
                  key={r.n}
                  type="monotone"
                  dataKey={`iter_${r.n}`}
                  name={`iter ${r.n}${r.status === "PASS" ? " ✓" : ""}`}
                  stroke={STROKES[i % STROKES.length]}
                  strokeWidth={i === results.length - 1 ? 2.5 : 1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 p-4 bg-surface-2/50 border-t border-border">
        {results.map((r) => (
          <div
            key={r.n}
            className="card-flat p-3 text-xs"
          >
            <div className="flex items-center justify-between">
              <span className="text-fg-muted">iter {r.n}</span>
              <span
                className={`badge ${
                  r.status === "PASS" ? "badge-pass" : "badge-fail"
                }`}
              >
                {r.status}
              </span>
            </div>
            {r.stats && (
              <div className="mt-2 mono space-y-0.5 text-fg-muted">
                <div>μ {r.stats.mean.toFixed(4)}</div>
                <div>σ {r.stats.std.toFixed(4)}</div>
                <div>
                  Δ {r.delta !== undefined ? r.delta.toFixed(4) : "—"}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
