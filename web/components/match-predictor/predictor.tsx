"use client";

import { useState, useEffect, useCallback } from "react";
import { FlipNumber } from "@/components/ui/flip-number";
import { PredictionHistoryStrip } from "@/components/match-predictor/prediction-history-strip";
import { getFlagClass, getKitColor } from "@/lib/flags";
import { predictMatchup, getH2H } from "@/lib/api";
import type { PredictResponse, TeamSummary, H2HMatch } from "@/lib/types";

interface PredictorProps {
  teams: TeamSummary[];
}

const DEBOUNCE_MS = 400;

/**
 * Integer scoreline for the scoreboard display.
 * Uses floor (Poisson mode) by default; when both sides floor to the same
 * value but xG differs, breaks the tie with ceil on the higher-xG side so
 * the big digits match win probability (e.g. 1.35 vs 1.09 → 2–1, not 1–1).
 */
function displayScoreline(xgA: number, xgB: number): [number, number] {
  let a = Math.floor(xgA);
  let b = Math.floor(xgB);

  if (a === b && Math.abs(xgA - xgB) > 0.01) {
    if (xgA > xgB) {
      a = Math.ceil(xgA);
      b = Math.floor(xgB);
    } else {
      a = Math.floor(xgA);
      b = Math.ceil(xgB);
    }
  }

  return [Math.max(0, a), Math.max(0, b)];
}

export function Predictor({ teams }: PredictorProps) {
  const teamNames = teams.map((t) => t.name).sort();
  const [teamA, setTeamA] = useState(teamNames[3] ?? "Brazil");   // Brazil
  const [teamB, setTeamB] = useState(teamNames[1] ?? "Argentina");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [h2h, setH2H] = useState<H2HMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sameTeam = teamA === teamB;

  const fetchPrediction = useCallback(async () => {
    if (sameTeam) return;
    setLoading(true);
    setError(null);
    try {
      const [pred, h2hData] = await Promise.all([
        predictMatchup(teamA, teamB),
        getH2H(teamA, teamB),
      ]);
      setPrediction(pred);
      setH2H(h2hData);
    } catch (e) {
      setError("Prediction failed — is the API running?");
    } finally {
      setLoading(false);
    }
  }, [teamA, teamB, sameTeam]);

  // Debounce fetch on team change
  useEffect(() => {
    const t = setTimeout(fetchPrediction, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [fetchPrediction]);

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <h1 className="font-display text-4xl font-black uppercase tracking-tight mb-8">
        Match Predictor
      </h1>

      {/* Squad-card face-off */}
      <div className="grid grid-cols-[1fr,auto,1fr] items-center gap-4 mb-8">
        <TeamCard
          teamName={teamA}
          teams={teams}
          selected={teamA}
          onSelect={setTeamA}
          side="a"
        />
        <span className="font-display text-2xl font-black text-[var(--chalk)] opacity-40">
          VS
        </span>
        <TeamCard
          teamName={teamB}
          teams={teams}
          selected={teamB}
          onSelect={setTeamB}
          side="b"
        />
      </div>

      {sameTeam && (
        <p className="text-center text-sm text-[var(--amber)] mb-6">
          Pick two different teams to compare.
        </p>
      )}

      {/* Results area */}
      {loading && !prediction && <PredictionSkeleton />}
      {error && (
        <p className="text-center text-sm text-[var(--red)] mb-6">{error}</p>
      )}

      {prediction && !sameTeam && (
        <>
          <WDLBar
            teamA={teamA}
            teamB={teamB}
            prediction={prediction}
            loading={loading}
          />
          <Scoreline
            prediction={prediction}
            teamA={teamA}
            teamB={teamB}
          />
          <PredictionHistoryStrip teamA={teamA} teamB={teamB} />
          {prediction.top_features.length > 0 && (
            <KeyFeatures features={prediction.top_features} teamA={teamA} teamB={teamB} />
          )}
          {h2h.length > 0 && <H2HTable matches={h2h} teamA={teamA} teamB={teamB} />}
        </>
      )}
    </div>
  );
}

// ── Team selector card ────────────────────────────────────────────────────────
function TeamCard({
  teamName,
  teams,
  selected,
  onSelect,
  side,
}: {
  teamName: string;
  teams: TeamSummary[];
  selected: string;
  onSelect: (t: string) => void;
  side: "a" | "b";
}) {
  const team = teams.find((t) => t.name === teamName);
  const flagClass = getFlagClass(teamName);
  const kitColor = getKitColor(teamName);

  return (
    <div
      className="bg-[var(--ink-raised)] rounded-lg p-4 flex flex-col gap-3"
      style={{ borderTop: `3px solid ${kitColor}` }}
    >
      <div className="flex items-center gap-2">
        {flagClass ? (
          <span className={`${flagClass} w-8 h-5 rounded flex-shrink-0`} />
        ) : (
          <span className="text-xl">⚽</span>
        )}
        <span className="font-display font-bold text-lg truncate">{teamName}</span>
      </div>
      {team && (
        <p className="font-mono text-xs opacity-50">
          Elo {Math.round(team.elo)} · {team.fifa_points.toFixed(0)} pts
        </p>
      )}
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full bg-[var(--ink)] text-[var(--chalk)] border rounded px-2 py-1.5
          text-xs font-body focus-visible:outline-none focus-visible:ring-2
          focus-visible:ring-[var(--turf-bright)] transition-colors"
        style={{ borderColor: "var(--line)" }}
        aria-label={`Select team ${side.toUpperCase()}`}
      >
        {teams.map((t) => (
          <option key={t.name} value={t.name}>
            {t.name}
          </option>
        ))}
      </select>
    </div>
  );
}

// ── W/D/L segmented bar ───────────────────────────────────────────────────────
function WDLBar({
  teamA,
  teamB,
  prediction,
  loading,
}: {
  teamA: string;
  teamB: string;
  prediction: PredictResponse;
  loading: boolean;
}) {
  const { p_team_a_win, p_draw, p_team_b_win } = prediction;

  return (
    <div className={`mb-6 ${loading ? "opacity-50" : ""} transition-opacity`}>
      {/* Bar */}
      <div className="flex h-8 rounded overflow-hidden gap-px">
        <div
          className="flex items-center justify-start pl-2 transition-all duration-500"
          style={{ width: `${p_team_a_win * 100}%`, backgroundColor: "var(--turf)" }}
        >
          {p_team_a_win > 0.15 && (
            <span className="font-display font-bold text-xs text-[var(--chalk)]">▲</span>
          )}
        </div>
        <div
          className="flex items-center justify-center transition-all duration-500"
          style={{ width: `${p_draw * 100}%`, backgroundColor: "var(--amber)" }}
        >
          {p_draw > 0.12 && (
            <span className="font-display font-bold text-xs text-[var(--chalk)]">—</span>
          )}
        </div>
        <div
          className="flex items-center justify-end pr-2 transition-all duration-500"
          style={{ width: `${p_team_b_win * 100}%`, backgroundColor: "var(--red)" }}
        >
          {p_team_b_win > 0.15 && (
            <span className="font-display font-bold text-xs text-[var(--chalk)]">▼</span>
          )}
        </div>
      </div>

      {/* Labels */}
      <div className="flex justify-between mt-2 text-sm">
        <div className="text-left">
          <span className="font-display font-bold">
            <FlipNumber value={p_team_a_win} format="percent" decimals={1} />
          </span>
          <span className="font-body text-xs opacity-50 ml-1">▲ {teamA} win</span>
        </div>
        <div className="text-center">
          <span className="font-display font-bold text-[var(--amber)]">
            <FlipNumber value={p_draw} format="percent" decimals={1} />
          </span>
          <span className="font-body text-xs opacity-50 ml-1">— Draw</span>
        </div>
        <div className="text-right">
          <span className="font-display font-bold">
            <FlipNumber value={p_team_b_win} format="percent" decimals={1} />
          </span>
          <span className="font-body text-xs opacity-50 ml-1">▼ {teamB} win</span>
        </div>
      </div>
    </div>
  );
}

// ── Predicted scoreline ───────────────────────────────────────────────────────
function Scoreline({
  prediction,
  teamA,
  teamB,
}: {
  prediction: PredictResponse;
  teamA: string;
  teamB: string;
}) {
  const { expected_score_a, expected_score_b } = prediction;
  const [displayA, displayB] = displayScoreline(
    expected_score_a,
    expected_score_b
  );

  return (
    <div
      className="flex flex-col items-center py-8 mb-6"
      style={{ borderTop: "1px solid var(--line)", borderBottom: "1px solid var(--line)" }}
    >
      <span className="font-mono text-xs uppercase tracking-widest opacity-40 mb-3">
        Predicted Score
      </span>
      <div className="flex items-center gap-6">
        <span className="font-display text-5xl font-black" style={{ fontFeatureSettings: '"tnum"' }}>
          {displayA}
        </span>
        <span className="font-display text-3xl font-black opacity-30">—</span>
        <span className="font-display text-5xl font-black" style={{ fontFeatureSettings: '"tnum"' }}>
          {displayB}
        </span>
      </div>
      <div className="flex gap-8 mt-2 text-xs font-mono opacity-40">
        <span>{teamA}</span>
        <span>{teamB}</span>
      </div>
      {/* Raw xG underneath so the user sees the underlying model output */}
      <p className="font-mono text-[11px] opacity-30 mt-3">
        xG &nbsp;{expected_score_a.toFixed(2)} — {expected_score_b.toFixed(2)}
      </p>
    </div>
  );
}

// ── Key features ─────────────────────────────────────────────────────────────
function KeyFeatures({
  features,
  teamA,
  teamB,
}: {
  features: PredictResponse["top_features"];
  teamA: string;
  teamB: string;
}) {
  return (
    <div className="mb-6">
      <h2
        className="font-display text-sm font-bold uppercase tracking-widest opacity-50 mb-3"
      >
        Key Factors
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {features.map((f) => {
          const favorLabel = f.favors === "a" ? teamA : f.favors === "b" ? teamB : "Neutral";
          const favorColor =
            f.favors === "a" ? "var(--turf)" : f.favors === "b" ? "var(--red)" : "var(--amber)";
          return (
            <div
              key={f.name}
              className="bg-[var(--ink-raised)] rounded p-3 flex items-center justify-between"
            >
              <div>
                <p className="font-body text-xs font-medium">{f.name}</p>
                <p className="font-mono text-[10px] mt-0.5" style={{ color: favorColor }}>
                  favours {favorLabel}
                </p>
              </div>
              <span className="font-mono text-sm font-medium">
                {f.value > 0 ? "+" : ""}{f.value.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── H2H table ────────────────────────────────────────────────────────────────
function H2HTable({
  matches,
  teamA,
  teamB,
}: {
  matches: H2HMatch[];
  teamA: string;
  teamB: string;
}) {
  return (
    <div>
      <h2
        className="font-display text-sm font-bold uppercase tracking-widest opacity-50 mb-3"
      >
        Head to Head
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr
              className="opacity-40"
              style={{ borderBottom: "1px solid var(--line)" }}
            >
              <th className="py-2 text-left">Date</th>
              <th className="py-2 text-left">Competition</th>
              <th className="py-2 text-right">{teamA}</th>
              <th className="py-2 text-right">{teamB}</th>
            </tr>
          </thead>
          <tbody>
            {matches.map((m, i) => {
              const aWon = m.score_a > m.score_b;
              const bWon = m.score_b > m.score_a;
              return (
                <tr
                  key={i}
                  className="border-b"
                  style={{ borderColor: "var(--line)" }}
                >
                  <td className="py-1.5 opacity-50">{m.date}</td>
                  <td className="py-1.5 opacity-50 truncate max-w-[120px]">
                    {m.competition}
                  </td>
                  <td
                    className="py-1.5 text-right font-bold"
                    style={{ color: aWon ? "var(--turf-bright)" : "inherit" }}
                  >
                    {m.score_a}
                  </td>
                  <td
                    className="py-1.5 text-right font-bold"
                    style={{ color: bWon ? "var(--turf-bright)" : "inherit" }}
                  >
                    {m.score_b}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Loading skeleton ──────────────────────────────────────────────────────────
function PredictionSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-8 rounded bg-[var(--ink-raised)]" />
      <div className="h-24 rounded bg-[var(--ink-raised)]" />
    </div>
  );
}
