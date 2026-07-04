import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
  Tooltip,
} from "recharts";
import { FlipNumber } from "@/components/ui/flip-number";
import type { ModelComparison, AccuracyStats } from "@/lib/types";

const NAIVE_BASELINE = 0.235;
const TARGET_BRIER   = 0.210;

const ROUND_LABELS: Record<string, string> = {
  groups: "Group Stage",
  r32: "Round of 32",
  r16: "Round of 16",
  qf: "Quarter-finals",
  sf: "Semi-finals",
  final: "Final",
};

const MODEL_LABELS: Record<string, string> = {
  logistic_wc2018_brier:      "Logistic",
  random_forest_wc2018_brier: "Random Forest",
  xgboost_wc2018_brier:       "XGBoost",
  lightgbm_wc2018_brier:      "LightGBM",
  dixon_coles_wc2018_brier:   "Dixon-Coles",
  neural_net_wc2018_brier:    "Neural Net",
  ensemble_wc2018_brier:      "Ensemble",
};

interface LabProps {
  comparison: ModelComparison;
  accuracy: AccuracyStats;
}

export function ModelLab({ comparison, accuracy }: LabProps) {
  const ensembleBrier = comparison.ensemble_wc2022_brier;
  const naiveBrier    = comparison.naive_baseline_brier ?? NAIVE_BASELINE;

  // Build chart data
  const chartData = Object.entries(MODEL_LABELS)
    .filter(([key]) => key in comparison)
    .map(([key, label]) => ({
      label,
      brier: Number(comparison[key]),
    }))
    .sort((a, b) => a.brier - b.brier);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="font-display text-4xl font-black uppercase tracking-tight mb-2">
        Model Lab
      </h1>
      <p className="font-mono text-xs opacity-40 mb-8">
        Read-only — model performance review, not retraining.
      </p>

      {/* Live accuracy section */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 10,
          marginBottom: 24,
        }}
      >
        <div
          style={{
            background: "var(--ink-raised)",
            border: "0.5px solid var(--line)",
            borderRadius: 10,
            padding: "14px 16px",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 500,
              color: "var(--turf)",
              fontFamily: "var(--font-display)",
            }}
          >
            <FlipNumber
              value={accuracy.correct_pct * 100}
              format="decimal"
              decimals={0}
            />
            %
          </div>
          <div style={{ fontSize: 12, color: "var(--chalk)", marginTop: 4 }}>
            Correct outcomes
          </div>
          <div
            style={{
              fontSize: 11,
              color: "rgba(245,243,236,0.4)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {accuracy.correct_outcome} / {accuracy.total_played} matches
          </div>
        </div>
        <div
          style={{
            background: "var(--ink-raised)",
            border: "0.5px solid var(--line)",
            borderRadius: 10,
            padding: "14px 16px",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 500,
              color: "var(--gold)",
              fontFamily: "var(--font-display)",
            }}
          >
            {accuracy.current_brier !== null ? (
              <FlipNumber
                value={accuracy.current_brier}
                format="decimal"
                decimals={4}
              />
            ) : (
              "—"
            )}
          </div>
          <div style={{ fontSize: 12, color: "var(--chalk)", marginTop: 4 }}>
            Live Brier score
          </div>
          <div
            style={{
              fontSize: 11,
              color: "rgba(245,243,236,0.4)",
              fontFamily: "var(--font-mono)",
            }}
          >
            WC 2026 completed matches
          </div>
        </div>
        <div
          style={{
            background: "var(--ink-raised)",
            border: "0.5px solid var(--line)",
            borderRadius: 10,
            padding: "14px 16px",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 500,
              color: "var(--chalk)",
              fontFamily: "var(--font-display)",
            }}
          >
            {accuracy.total_played}
          </div>
          <div style={{ fontSize: 12, color: "var(--chalk)", marginTop: 4 }}>
            Matches evaluated
          </div>
          <div
            style={{
              fontSize: 11,
              color: "rgba(245,243,236,0.4)",
              fontFamily: "var(--font-mono)",
            }}
          >
            of 104 total matches
          </div>
        </div>
      </div>

      {Object.keys(accuracy.correct_by_round).length > 0 && (
        <div
          style={{
            background: "var(--ink-raised)",
            border: "0.5px solid var(--line)",
            borderRadius: 10,
            padding: "14px 16px",
            marginBottom: 24,
          }}
        >
          <div
            style={{
              fontSize: 11,
              color: "rgba(245,243,236,0.35)",
              fontFamily: "var(--font-mono)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              marginBottom: 12,
            }}
          >
            Accuracy by round
          </div>
          {Object.entries(accuracy.correct_by_round).map(([round, pct]) => (
            <div
              key={round}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                marginBottom: 8,
              }}
            >
              <span
                style={{
                  width: 100,
                  fontSize: 12,
                  color: "rgba(245,243,236,0.5)",
                  fontFamily: "var(--font-body)",
                }}
              >
                {ROUND_LABELS[round] ?? round}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 6,
                  background: "rgba(245,243,236,0.08)",
                  borderRadius: 3,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${pct * 100}%`,
                    height: "100%",
                    background:
                      pct >= 0.6
                        ? "var(--turf)"
                        : pct >= 0.4
                        ? "var(--amber)"
                        : "var(--red)",
                    borderRadius: 3,
                  }}
                />
              </div>
              <span
                style={{
                  width: 36,
                  textAlign: "right",
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  color: "rgba(245,243,236,0.6)",
                }}
              >
                <FlipNumber value={pct * 100} format="decimal" decimals={0} />%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Headline ensemble metric */}
      {ensembleBrier !== undefined && ensembleBrier !== null && (
        <div
          className="bg-[var(--ink-raised)] rounded-lg p-6 mb-8 flex flex-col sm:flex-row
            items-start sm:items-center gap-4"
        >
          <div>
            <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-1">
              Ensemble · WC 2022 Test Set (held-out)
            </p>
            <p className="font-display text-5xl font-black">
              <FlipNumber value={ensembleBrier} format="decimal" decimals={4} />
            </p>
          </div>
          <div
            className="sm:ml-8 pl-0 sm:pl-8 border-t sm:border-t-0 sm:border-l pt-4 sm:pt-0"
            style={{ borderColor: "var(--line)" }}
          >
            {ensembleBrier < TARGET_BRIER ? (
              <p className="font-body text-sm" style={{ color: "var(--turf-bright)" }}>
                ✓ Target met —{" "}
                <span className="font-mono">
                  {(TARGET_BRIER - ensembleBrier).toFixed(4)} below 0.210
                </span>
              </p>
            ) : (
              <p className="font-body text-sm" style={{ color: "var(--red)" }}>
                ✗ Above target —{" "}
                <span className="font-mono">
                  {(ensembleBrier - TARGET_BRIER).toFixed(4)} above 0.210
                </span>
              </p>
            )}
            <p className="font-body text-sm mt-1" style={{ color: "var(--turf-bright)" }}>
              {(naiveBrier - ensembleBrier).toFixed(4)} better than naïve baseline
            </p>
          </div>
        </div>
      )}

      {/* WC 2018 bar chart */}
      <h2
        className="font-display text-xl font-bold uppercase tracking-tight mb-4"
      >
        WC 2018 Validation — Brier by Model
      </h2>
      <div className="mb-2 flex gap-4 text-[11px] font-mono opacity-50">
        <span><span style={{ color: "var(--turf)" }}>█</span> Below naïve baseline</span>
        <span><span style={{ color: "var(--red)" }}>█</span> Above naïve baseline</span>
      </div>
      <div className="h-64 mb-8">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 48, bottom: 0, left: 90 }}
          >
            <XAxis
              type="number"
              domain={[0, 0.28]}
              tick={{ fill: "rgba(245,243,236,0.4)", fontSize: 10, fontFamily: "var(--font-mono)" }}
              axisLine={{ stroke: "var(--line)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              tick={{ fill: "rgba(245,243,236,0.7)", fontSize: 11, fontFamily: "var(--font-body)" }}
              axisLine={false}
              tickLine={false}
              width={88}
            />
            <Tooltip
              contentStyle={{
                background: "var(--ink-raised)",
                border: "1px solid var(--line)",
                borderRadius: "6px",
                color: "var(--chalk)",
                fontFamily: "var(--font-mono)",
                fontSize: "12px",
              }}
              formatter={(v) => [typeof v === "number" ? v.toFixed(4) : String(v), "Brier score"]}
            />
            <ReferenceLine
              x={naiveBrier}
              stroke="rgba(245,243,236,0.3)"
              strokeDasharray="4 3"
              label={{
                value: `Naïve ${naiveBrier}`,
                fill: "rgba(245,243,236,0.4)",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                position: "insideTopRight",
              }}
            />
            <ReferenceLine
              x={TARGET_BRIER}
              stroke="var(--turf)"
              strokeDasharray="4 3"
              label={{
                value: `Target ${TARGET_BRIER}`,
                fill: "var(--turf-bright)",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                position: "insideTopLeft",
              }}
            />
            <Bar dataKey="brier" radius={[0, 3, 3, 0]}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.label}
                  fill={entry.brier < naiveBrier ? "var(--turf)" : "var(--red)"}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Brier explainer */}
      <BrierExplainer naiveBrier={naiveBrier} />

      {/* Full comparison table */}
      <h2
        className="font-display text-xl font-bold uppercase tracking-tight mt-8 mb-4"
      >
        Full Comparison
      </h2>
      <FullTable comparison={comparison} naiveBrier={naiveBrier} />

      {comparison.trained_at && (
        <p className="font-mono text-[10px] opacity-30 mt-4">
          Models trained: {comparison.trained_at}
        </p>
      )}
    </div>
  );
}

// ── Brier explainer ───────────────────────────────────────────────────────────
function BrierExplainer({ naiveBrier }: { naiveBrier: number }) {
  return (
    <div
      className="bg-[var(--ink-raised)] rounded-lg p-5 border"
      style={{ borderColor: "var(--line)" }}
    >
      <h3 className="font-display text-base font-bold uppercase tracking-tight mb-3">
        What is the Brier Score?
      </h3>
      <p className="font-body text-sm opacity-70 leading-relaxed mb-3">
        The Brier score measures how accurate predicted probabilities are — lower
        is better. A model that always says "40% home win / 30% draw / 30% away win"
        scores around <span className="font-mono">{naiveBrier}</span>, regardless
        of the actual result. That is the{" "}
        <em>naïve baseline</em> — the floor any useful model must beat.
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
        {[
          { label: "Perfect",  value: "0.000", color: "var(--turf-bright)" },
          { label: "Target",   value: "0.210", color: "var(--turf)" },
          { label: "Naïve",    value: naiveBrier.toFixed(3), color: "var(--amber)" },
          { label: "Random",   value: "0.444", color: "var(--red)" },
        ].map(({ label, value, color }) => (
          <div key={label} className="flex flex-col gap-0.5">
            <span className="opacity-40 uppercase tracking-widest text-[9px]">{label}</span>
            <span className="text-base font-bold" style={{ color }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Full comparison table ─────────────────────────────────────────────────────
function FullTable({
  comparison,
  naiveBrier,
}: {
  comparison: ModelComparison;
  naiveBrier: number;
}) {
  const rows = Object.entries(comparison)
    .filter(([k, v]) => k.endsWith("_brier") && typeof v === "number")
    .sort(([, a], [, b]) => (a as number) - (b as number));

  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-xs">
        <thead>
          <tr className="opacity-40" style={{ borderBottom: "1px solid var(--line)" }}>
            <th className="py-2 text-left">Metric</th>
            <th className="py-2 text-right">Brier Score</th>
            <th className="py-2 text-right">vs Naïve</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, val]) => {
            const v = val as number;
            const delta = v - naiveBrier;
            return (
              <tr key={key} className="border-b" style={{ borderColor: "var(--line)" }}>
                <td className="py-1.5 opacity-70">
                  {key.replace(/_brier$/, "").replace(/_/g, " ")}
                </td>
                <td
                  className="py-1.5 text-right font-bold"
                  style={{ color: v < TARGET_BRIER ? "var(--turf-bright)" : "inherit" }}
                >
                  {v.toFixed(4)}
                </td>
                <td
                  className="py-1.5 text-right"
                  style={{ color: delta < 0 ? "var(--turf-bright)" : "var(--red)" }}
                >
                  {delta > 0 ? "+" : ""}
                  {delta.toFixed(4)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
