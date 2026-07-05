"use client";

import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Line,
  ComposedChart,
  Scatter,
} from "recharts";
import { FlipNumber } from "@/components/ui/flip-number";
import type { StoryAccuracy } from "@/lib/types";

interface AccuracyReportProps {
  data: StoryAccuracy;
}

function correctBarColor(pct: number | null | undefined): string {
  if (pct == null) return "var(--line)";
  if (pct >= 0.6) return "var(--turf)";
  if (pct >= 0.45) return "var(--amber)";
  return "var(--red)";
}

export function AccuracyReport({ data }: AccuracyReportProps) {
  const allUpsets = data.per_stage
    .flatMap((s) =>
      (s.upsets ?? []).map((u) => ({
        ...u,
        stage: s.display_long ?? s.display,
      }))
    )
    .sort((a, b) => b.confidence - a.confidence);

  const calData = data.calibration.map((b) => ({
    ...b,
    perfect: b.midpoint,
  }));

  return (
    <section className="mb-12 space-y-10">
      <div>
        <h2 className="font-display text-2xl font-black uppercase tracking-tight mb-1">
          Honest Accuracy
        </h2>
        <p className="text-xs text-[var(--chalk)] opacity-40">
          Point-in-time evaluation only — each snapshot scored against the round it predicted
        </p>
      </div>

      {/* 5d — Benchmark strip */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-[var(--ink-raised)] rounded-lg p-4 text-center">
          <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-2">
            Live tournament Brier
          </p>
          <p className="font-display text-3xl font-black">
            <FlipNumber
              value={data.overall.brier ?? 0}
              format="decimal"
              decimals={4}
            />
          </p>
        </div>
        <div className="bg-[var(--ink-raised)] rounded-lg p-4 text-center">
          <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-2">
            Historical benchmark
          </p>
          <p className="font-display text-3xl font-black">
            <FlipNumber
              value={data.overall.wc2022_benchmark_brier}
              format="decimal"
              decimals={4}
            />
          </p>
          <p className="font-mono text-[10px] opacity-30 mt-1">WC 2022 test set</p>
        </div>
        <div className="bg-[var(--ink-raised)] rounded-lg p-4 text-center">
          <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-2">
            Overall correct
          </p>
          <p className="font-display text-3xl font-black">
            <FlipNumber
              value={data.overall.correct_pct ?? 0}
              format="percent"
              decimals={1}
            />
          </p>
          <p className="font-mono text-[10px] opacity-30 mt-1">
            n = {data.overall.n_matches} matches
          </p>
        </div>
      </div>
      <p className="font-mono text-[10px] opacity-35 -mt-6">
        Lower Brier is better; 0.33 ≈ random, ~0.19 ≈ strong for international football.
      </p>

      {/* 5a — Per-stage scorecard */}
      <div>
        <h3 className="font-display font-bold text-lg mb-4">Per-stage scorecard</h3>
        <div className="space-y-2">
          {data.per_stage.map((s) => {
            const pending = s.status === "round not yet played";
            return (
              <div
                key={s.stage}
                className={`grid grid-cols-[1fr,auto,auto,120px,64px] gap-3 items-center bg-[var(--ink-raised)] rounded-lg px-4 py-3 text-sm ${
                  pending ? "opacity-40" : ""
                }`}
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-medium">{s.display_long ?? s.display}</span>
                  <span className="font-mono text-[10px] opacity-40">
                    → {s.predicts_round}
                  </span>
                </div>
                <span className="font-mono text-xs opacity-50">
                  {pending ? s.status : `n = ${s.n_matches}`}
                </span>
                {!pending && s.correct_pct != null && (
                  <>
                    <span className="font-mono text-xs">
                      {(s.correct_pct * 100).toFixed(0)}%
                    </span>
                    <div className="h-2 rounded overflow-hidden bg-[var(--ink)]">
                      <div
                        className="h-full rounded transition-all"
                        style={{
                          width: `${s.correct_pct * 100}%`,
                          backgroundColor: correctBarColor(s.correct_pct),
                        }}
                      />
                    </div>
                    <span className="font-mono text-xs opacity-60">
                      {s.brier?.toFixed(4) ?? "—"}
                    </span>
                  </>
                )}
                {pending && <span className="col-span-3" />}
              </div>
            );
          })}
        </div>
      </div>

      {/* 5b — Calibration chart */}
      {calData.length > 0 && (
        <div>
          <h3 className="font-display font-bold text-lg mb-1">Calibration</h3>
          <p className="text-[10px] font-mono opacity-35 mb-4">
            Above line = underconfident · Below line = overconfident
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={calData} margin={{ left: 0, right: 20, top: 8, bottom: 0 }}>
              <XAxis
                dataKey="midpoint"
                type="number"
                domain={[0.3, 0.85]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{
                  fontSize: 10,
                  fill: "var(--chalk)",
                  opacity: 0.4,
                  fontFamily: "var(--font-mono)",
                }}
                axisLine={{ stroke: "var(--line)" }}
                tickLine={false}
              />
              <YAxis
                dataKey="actual_rate"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{
                  fontSize: 10,
                  fill: "var(--chalk)",
                  opacity: 0.4,
                  fontFamily: "var(--font-mono)",
                }}
                axisLine={false}
                tickLine={false}
                width={40}
              />
              <ReferenceLine
                segment={[
                  { x: 0.34, y: 0.34 },
                  { x: 0.85, y: 0.85 },
                ]}
                stroke="var(--chalk)"
                strokeDasharray="4 4"
                strokeOpacity={0.3}
              />
              <Line
                type="linear"
                dataKey="perfect"
                stroke="var(--chalk)"
                strokeDasharray="4 4"
                strokeOpacity={0.2}
                dot={false}
                legendType="none"
              />
              <Scatter
                dataKey="actual_rate"
                fill="var(--turf-bright)"
              />
              <Tooltip
                contentStyle={{
                  background: "var(--ink-raised)",
                  border: "1px solid var(--line)",
                  borderRadius: 8,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                }}
                formatter={(value, _name, props) => {
                  const p = (props?.payload ?? {}) as (typeof calData)[0];
                  const v = typeof value === "number" ? value : 0;
                  return [
                    `predicted ${(p.predicted_rate * 100).toFixed(0)}%, actual ${(v * 100).toFixed(0)}% (n=${p.n})`,
                    p.bucket,
                  ];
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="mt-3 space-y-1">
            {data.calibration.map((b) => (
              <p key={b.bucket} className="font-mono text-[10px] opacity-50">
                When the model said {b.bucket}, it was right{" "}
                {(b.actual_rate * 100).toFixed(0)}% of the time (n={b.n}).
              </p>
            ))}
          </div>
        </div>
      )}

      {/* 5c — Upsets table */}
      <div>
        <h3 className="font-display font-bold text-lg mb-4 text-[var(--red)]">
          Where the model was wrong
        </h3>
        {allUpsets.length === 0 ? (
          <p className="text-sm opacity-40">No upsets recorded yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm font-mono">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-widest opacity-40 border-b" style={{ borderColor: "var(--line)" }}>
                  <th className="py-2 pr-4">Match</th>
                  <th className="py-2 pr-4">Score</th>
                  <th className="py-2 pr-4">Round</th>
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2">Actual</th>
                </tr>
              </thead>
              <tbody>
                {allUpsets.map((u, i) => (
                  <tr
                    key={i}
                    className="border-b"
                    style={{ borderColor: "var(--line)" }}
                  >
                    <td className="py-2 pr-4">
                      {u.team_a} vs {u.team_b}
                    </td>
                    <td className="py-2 pr-4">{u.score}</td>
                    <td className="py-2 pr-4 opacity-60">{u.round}</td>
                    <td className="py-2 pr-4">
                      favoured {u.model_favoured} at {(u.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2">{u.actual}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
