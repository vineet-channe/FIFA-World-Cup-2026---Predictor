"use client";

import { useEffect, useState } from "react";
import { getMatchPredictionHistory } from "@/lib/api";
import { MatchPredictionHistory } from "@/lib/types";
import { FlipNumber } from "@/components/ui/flip-number";

const STAGE_LABELS: Record<string, string> = {
  pre_tournament: "Pre-tournament",
  post_group_stage: "After Groups",
  post_r32: "After R32",
  post_r16: "After R16",
  post_qf: "After QF",
  post_sf: "After SF",
  current: "Current",
};

function MiniBar({ pa, pd, pb }: { pa: number; pd: number; pb: number }) {
  return (
    <div
      style={{
        display: "flex",
        height: 4,
        borderRadius: 2,
        overflow: "hidden",
        gap: 1,
      }}
    >
      <div style={{ width: `${pa * 100}%`, background: "var(--turf)" }} />
      <div
        style={{ width: `${pd * 100}%`, background: "rgba(245,243,236,0.15)" }}
      />
      <div style={{ width: `${pb * 100}%`, background: "var(--amber)" }} />
    </div>
  );
}

export function PredictionHistoryStrip({
  teamA,
  teamB,
}: {
  teamA: string;
  teamB: string;
}) {
  const [data, setData] = useState<MatchPredictionHistory | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!teamA || !teamB || teamA === teamB) return;
    setLoading(true);
    getMatchPredictionHistory(teamA, teamB)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [teamA, teamB]);

  if (loading) {
    return (
      <p
        style={{
          fontSize: 12,
          color: "rgba(245,243,236,0.3)",
          fontFamily: "var(--font-mono)",
        }}
      >
        Loading prediction history...
      </p>
    );
  }
  if (!data || !data.predictions.length) return null;

  return (
    <div style={{ marginTop: 24 }}>
      <div
        style={{
          fontSize: 10,
          color: "rgba(245,243,236,0.35)",
          fontFamily: "var(--font-mono)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 12,
        }}
      >
        How the prediction evolved
      </div>

      {data.played && (
        <div
          style={{
            padding: "10px 16px",
            background: "var(--ink-raised)",
            border: "1px solid var(--turf)",
            borderRadius: 8,
            marginBottom: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div
            style={{
              fontSize: 11,
              color: "var(--turf)",
              fontFamily: "var(--font-mono)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Actual result
          </div>
          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 22,
              fontWeight: 700,
              color: "var(--chalk)",
              letterSpacing: "0.05em",
            }}
          >
            {data.actual_score_a} — {data.actual_score_b}
          </div>
          <div
            style={{
              fontSize: 11,
              color: "var(--chalk)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {data.actual_winner === teamA
              ? `${teamA} win`
              : data.actual_winner === teamB
              ? `${teamB} win`
              : "Draw"}
          </div>
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${Math.min(data.predictions.length, 3)}, 1fr)`,
          gap: 8,
        }}
      >
        {data.predictions.map((pred, i) => {
          let modelResult: "correct" | "wrong" | null = null;
          if (data.played) {
            const favA =
              pred.p_team_a_win >= pred.p_draw &&
              pred.p_team_a_win >= pred.p_team_b_win;
            const favB =
              pred.p_team_b_win >= pred.p_draw &&
              pred.p_team_b_win >= pred.p_team_a_win;
            const favD =
              pred.p_draw > pred.p_team_a_win && pred.p_draw > pred.p_team_b_win;
            const sa = data.actual_score_a ?? 0;
            const sb = data.actual_score_b ?? 0;
            const actualDraw = sa === sb;
            const actualA = sa > sb;
            modelResult =
              (favA && actualA) ||
              (favB && !actualA && !actualDraw) ||
              (favD && actualDraw)
                ? "correct"
                : "wrong";
          }

          return (
            <div
              key={i}
              style={{
                background: "var(--ink-raised)",
                border: "0.5px solid var(--line)",
                borderRadius: 8,
                padding: "10px 12px",
                borderTop:
                  modelResult === "correct"
                    ? "2px solid var(--turf)"
                    : modelResult === "wrong"
                    ? "2px solid var(--red)"
                    : "2px solid transparent",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: "rgba(245,243,236,0.4)",
                  fontFamily: "var(--font-mono)",
                  marginBottom: 8,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span>{STAGE_LABELS[pred.stage] ?? pred.stage}</span>
                {modelResult && (
                  <span
                    style={{
                      color:
                        modelResult === "correct" ? "var(--turf)" : "var(--red)",
                    }}
                  >
                    {modelResult === "correct" ? "✓" : "✗"}
                  </span>
                )}
              </div>

              <MiniBar
                pa={pred.p_team_a_win}
                pd={pred.p_draw}
                pb={pred.p_team_b_win}
              />

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginTop: 6,
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  color: "rgba(245,243,236,0.7)",
                }}
              >
                <span style={{ color: "var(--turf)" }}>
                  <FlipNumber
                    value={pred.p_team_a_win * 100}
                    format="decimal"
                    decimals={0}
                  />
                  %
                </span>
                <span style={{ color: "rgba(245,243,236,0.3)" }}>
                  <FlipNumber value={pred.p_draw * 100} format="decimal" decimals={0} />
                  %
                </span>
                <span style={{ color: "var(--amber)" }}>
                  <FlipNumber
                    value={pred.p_team_b_win * 100}
                    format="decimal"
                    decimals={0}
                  />
                  %
                </span>
              </div>

              <div
                style={{
                  textAlign: "center",
                  marginTop: 6,
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  color: "rgba(245,243,236,0.25)",
                }}
              >
                {pred.expected_score_a.toFixed(1)} — {pred.expected_score_b.toFixed(1)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
