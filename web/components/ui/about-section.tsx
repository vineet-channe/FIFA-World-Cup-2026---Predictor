"use client";
import { useState } from "react";

export function AboutSection() {
  const [expanded, setExpanded] = useState(false);

  return (
    <div style={{ marginTop: 32, borderTop: "0.5px solid var(--line)", paddingTop: 20 }}>
      <div style={{ fontSize: 13, color: "rgba(245,243,236,0.7)", lineHeight: 1.7,
                    fontFamily: "var(--font-body)", maxWidth: 640 }}>
        This dashboard predicts FIFA World Cup 2026 outcomes using a machine
        learning pipeline trained on 18 years of international football data.
        Seven models — including XGBoost, LightGBM, and a Poisson scoreline
        model — are combined into a stacking ensemble, which scored 0.19 Brier
        on the 2022 World Cup as an unseen test set. As the 2026 tournament
        progresses, the model retrains on real results after every matchday
        and re-simulates the remaining bracket 10,000 times to update every
        team&apos;s odds.
      </div>

      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          marginTop: 10, fontSize: 12, color: "var(--turf)", background: "none",
          border: "none", cursor: "pointer", padding: 0, fontFamily: "var(--font-body)",
        }}
      >
        {expanded ? "Show less" : "For the technically curious →"}
      </button>

      {expanded && (
        <div style={{ marginTop: 12, fontSize: 12, color: "rgba(245,243,236,0.5)",
                      lineHeight: 1.7, fontFamily: "var(--font-mono)", maxWidth: 640 }}>
          21 tree-model features and 15 linear-model features per match,
          covering Elo rating, FIFA ranking points, squad market value, recent
          form (last 10 matches), head-to-head history, and World Cup
          experience. Training uses a strict temporal split — pre-2018 for
          training, WC 2018 for validation, WC 2022 touched exactly once as
          the final test set — to avoid any lookahead bias. The Story page
          evaluates every prediction only against results that came after it
          was made; nothing is scored with hindsight. See the Lab tab for
          full model comparison and live accuracy tracking.
        </div>
      )}
    </div>
  );
}
