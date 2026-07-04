"use client";

import type { ReactNode } from "react";
import { RoundData, MatchResultCard } from "@/lib/types";
import { getFlagClass, getKitColor } from "@/lib/flags";
import { FlipNumber } from "@/components/ui/flip-number";

// ── helpers ────────────────────────────────────────────────────────────────

function labelledDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "";
  try {
    return new Date(dateStr).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
    });
  } catch {
    return "";
  }
}

function predictedOutcomeText(match: MatchResultCard): string {
  const { p_team_a_win, p_draw, p_team_b_win, team_a, team_b } = match;
  if (
    p_team_a_win == null ||
    p_draw == null ||
    p_team_b_win == null
  ) {
    return "Unknown";
  }
  if (p_team_a_win >= p_draw && p_team_a_win >= p_team_b_win) {
    return `${team_a} win`;
  }
  if (p_team_b_win >= p_draw && p_team_b_win >= p_team_a_win) {
    return `${team_b} win`;
  }
  return "Draw";
}

function leadProbability(match: MatchResultCard): number {
  const { p_team_a_win, p_draw, p_team_b_win } = match;
  if (
    p_team_a_win == null ||
    p_draw == null ||
    p_team_b_win == null
  ) {
    return 0;
  }
  return Math.max(p_team_a_win, p_draw, p_team_b_win);
}

// ── shared sub-components ──────────────────────────────────────────────────

function SectionLabel({ text, right }: { text: string; right?: ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 6,
      }}
    >
      <span
        style={{
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          fontWeight: 500,
          color: "rgba(245,243,236,0.35)",
          textTransform: "uppercase",
          letterSpacing: "0.10em",
        }}
      >
        {text}
      </span>
      {right}
    </div>
  );
}

function Divider() {
  return (
    <div
      style={{ height: "0.5px", background: "var(--line)", margin: "10px 0" }}
    />
  );
}

function ProbabilityBar({
  pa,
  pd,
  pb,
}: {
  pa: number;
  pd: number;
  pb: number;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          height: 6,
          borderRadius: 3,
          overflow: "hidden",
          gap: 1,
          marginBottom: 6,
        }}
      >
        <div
          style={{
            width: `${pa * 100}%`,
            background: "var(--turf)",
            transition: "width 0.3s ease",
            minWidth: pa > 0 ? 2 : 0,
          }}
        />
        <div
          style={{
            width: `${pd * 100}%`,
            background: "rgba(245,243,236,0.18)",
            minWidth: pd > 0 ? 2 : 0,
          }}
        />
        <div
          style={{
            width: `${pb * 100}%`,
            background: "var(--amber)",
            transition: "width 0.3s ease",
            minWidth: pb > 0 ? 2 : 0,
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "rgba(245,243,236,0.45)",
        }}
      >
        <span style={{ color: "var(--turf)" }}>
          <FlipNumber value={pa * 100} format="decimal" decimals={0} />%
        </span>
        <span>{(pd * 100).toFixed(0)}% draw</span>
        <span style={{ color: "var(--amber)" }}>
          <FlipNumber value={pb * 100} format="decimal" decimals={0} />%
        </span>
      </div>
    </div>
  );
}

// ── main component ─────────────────────────────────────────────────────────

function MatchCard({ match }: { match: MatchResultCard }) {
  const completed = match.status === "completed";
  const { team_a, team_b } = match;

  if (completed) {
    const resultLabel =
      match.actual_score_a !== null
        ? `${match.actual_score_a} — ${match.actual_score_b}`
        : "? — ?";

    const actualWinner = match.actual_winner;
    const actualDraw = match.actual_score_a === match.actual_score_b;
    const hasForecast = match.has_pre_match_prediction !== false &&
      match.p_team_a_win != null &&
      match.p_draw != null &&
      match.p_team_b_win != null;
    const modelPredicted = predictedOutcomeText(match);
    const correct = match.model_correct;

    return (
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
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 12,
          }}
        >
          <span
            style={{
              fontSize: 9,
              fontFamily: "var(--font-mono)",
              color: "rgba(245,243,236,0.25)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            {match.round}
          </span>
          <span
            style={{
              fontSize: 9,
              fontFamily: "var(--font-mono)",
              color: "rgba(245,243,236,0.25)",
            }}
          >
            {labelledDate(match.match_date)}
          </span>
        </div>

        <SectionLabel text="Result" />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 4,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              flex: 1,
            }}
          >
            <span
              style={{
                width: 3,
                height: 18,
                borderRadius: 2,
                background: getKitColor(team_a),
                display: "inline-block",
                flexShrink: 0,
              }}
            />
            {getFlagClass(team_a) && <i className={getFlagClass(team_a)} />}
            <span
              style={{
                fontSize: 13,
                fontWeight: 500,
                color:
                  actualWinner === team_a
                    ? "var(--chalk)"
                    : "rgba(245,243,236,0.5)",
                fontFamily: "var(--font-body)",
              }}
            >
              {team_a}
            </span>
          </div>

          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 26,
              fontWeight: 700,
              color: "var(--chalk)",
              letterSpacing: "0.04em",
              padding: "0 12px",
              flexShrink: 0,
            }}
          >
            {resultLabel}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              flex: 1,
              justifyContent: "flex-end",
            }}
          >
            <span
              style={{
                fontSize: 13,
                fontWeight: 500,
                color:
                  actualWinner === team_b
                    ? "var(--chalk)"
                    : "rgba(245,243,236,0.5)",
                fontFamily: "var(--font-body)",
              }}
            >
              {team_b}
            </span>
            {getFlagClass(team_b) && <i className={getFlagClass(team_b)} />}
            <span
              style={{
                width: 3,
                height: 18,
                borderRadius: 2,
                background: getKitColor(team_b),
                display: "inline-block",
                flexShrink: 0,
              }}
            />
          </div>
        </div>

        <div style={{ textAlign: "center", marginBottom: 10 }}>
          <span
            style={{
              fontSize: 10,
              fontFamily: "var(--font-mono)",
              color: "rgba(245,243,236,0.3)",
            }}
          >
            {actualDraw ? "Match drawn" : `${actualWinner} won`}
          </span>
        </div>

        <Divider />

        {hasForecast ? (
          <>
            <SectionLabel
              text="Model predicted"
              right={
                correct !== null ? (
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      fontFamily: "var(--font-mono)",
                      color: correct ? "var(--turf)" : "var(--red)",
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                    }}
                  >
                    {correct ? "✓ Correct" : "✗ Wrong"}
                  </span>
                ) : null
              }
            />

            <div
              style={{
                marginBottom: 8,
                fontSize: 12,
                fontFamily: "var(--font-body)",
                color: "rgba(245,243,236,0.65)",
              }}
            >
              {modelPredicted}
              <span
                style={{
                  marginLeft: 6,
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                  color: "rgba(245,243,236,0.35)",
                }}
              >
                ({(leadProbability(match) * 100).toFixed(0)}% confidence)
              </span>
            </div>

            <ProbabilityBar
              pa={match.p_team_a_win!}
              pd={match.p_draw!}
              pb={match.p_team_b_win!}
            />
          </>
        ) : (
          <p
            style={{
              margin: 0,
              fontSize: 11,
              fontFamily: "var(--font-mono)",
              color: "rgba(245,243,236,0.35)",
              lineHeight: 1.5,
            }}
          >
            Pre-match forecast not saved for this match. The pipeline only
            stores predictions while a fixture is still upcoming.
          </p>
        )}
      </div>
    );
  }

  return (
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
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <span
          style={{
            fontSize: 9,
            fontFamily: "var(--font-mono)",
            color: "rgba(245,243,236,0.25)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}
        >
          {match.round}
        </span>
        <span
          style={{
            fontSize: 9,
            fontFamily: "var(--font-mono)",
            color: "rgba(245,243,236,0.25)",
          }}
        >
          {labelledDate(match.match_date)}
        </span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 14,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            flex: 1,
          }}
        >
          <span
            style={{
              width: 3,
              height: 18,
              borderRadius: 2,
              background: getKitColor(team_a),
              display: "inline-block",
              flexShrink: 0,
            }}
          />
          {getFlagClass(team_a) && <i className={getFlagClass(team_a)} />}
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: "var(--chalk)",
              fontFamily: "var(--font-body)",
            }}
          >
            {team_a}
          </span>
        </div>

        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "rgba(245,243,236,0.2)",
            padding: "0 12px",
            letterSpacing: "0.12em",
          }}
        >
          VS
        </span>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            flex: 1,
            justifyContent: "flex-end",
          }}
        >
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: "var(--chalk)",
              fontFamily: "var(--font-body)",
            }}
          >
            {team_b}
          </span>
          {getFlagClass(team_b) && <i className={getFlagClass(team_b)} />}
          <span
            style={{
              width: 3,
              height: 18,
              borderRadius: 2,
              background: getKitColor(team_b),
              display: "inline-block",
              flexShrink: 0,
            }}
          />
        </div>
      </div>

      <SectionLabel text="Model predicts" />

      <div
        style={{
          marginBottom: 8,
          fontSize: 12,
          fontFamily: "var(--font-body)",
          color: "rgba(245,243,236,0.65)",
        }}
      >
        {predictedOutcomeText(match)}
        <span
          style={{
            marginLeft: 6,
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            color: "rgba(245,243,236,0.35)",
          }}
        >
          ({(leadProbability(match) * 100).toFixed(0)}% likely)
        </span>
      </div>

      <ProbabilityBar
        pa={match.p_team_a_win}
        pd={match.p_draw}
        pb={match.p_team_b_win}
      />

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          marginTop: 10,
          paddingTop: 10,
          borderTop: "0.5px solid var(--line)",
        }}
      >
        <span
          style={{
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            color: "rgba(245,243,236,0.3)",
          }}
        >
          Predicted score
        </span>
        <span
          style={{
            fontSize: 14,
            fontWeight: 500,
            fontFamily: "var(--font-display)",
            color: "rgba(245,243,236,0.55)",
            letterSpacing: "0.05em",
          }}
        >
          {Math.round(match.expected_score_a)} —{" "}
          {Math.round(match.expected_score_b)}
        </span>
        <span
          style={{
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            color: "rgba(245,243,236,0.2)",
          }}
        >
          ({match.expected_score_a.toFixed(1)} —{" "}
          {match.expected_score_b.toFixed(1)} avg)
        </span>
      </div>
    </div>
  );
}

export function KnockoutRound({ round }: { round: RoundData }) {
  const cols =
    round.round_key === "r32" ? 2 : round.round_key === "r16" ? 2 : 1;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${cols}, 1fr)`,
        gap: 12,
      }}
    >
      {round.matches.map((m) => (
        <MatchCard key={m.match_id} match={m} />
      ))}
    </div>
  );
}
