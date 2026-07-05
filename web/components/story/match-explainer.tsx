"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { FlipNumber } from "@/components/ui/flip-number";
import { getMatchExplanation } from "@/lib/api";
import { getFlagClass, getKitColor } from "@/lib/flags";
import type { MatchExplanation, StageMeta, TeamSummary } from "@/lib/types";

interface MatchExplainerProps {
  teams: TeamSummary[];
  stages: StageMeta[];
}

function favouriteOf(pred: {
  p_team_a_win: number;
  p_draw: number;
  p_team_b_win: number;
}): { side: "a" | "b" | "draw"; conf: number } {
  const { p_team_a_win: pa, p_draw: pd, p_team_b_win: pb } = pred;
  if (pa >= pd && pa >= pb) return { side: "a", conf: pa };
  if (pb >= pd && pb >= pa) return { side: "b", conf: pb };
  return { side: "draw", conf: pd };
}

function TeamSelectorCard({
  label,
  value,
  teams,
  onChange,
}: {
  label: string;
  value: string;
  teams: TeamSummary[];
  onChange: (v: string) => void;
}) {
  const kitColor = getKitColor(value);
  const flagClass = getFlagClass(value);

  return (
    <div
      className="bg-[var(--ink-raised)] rounded-lg p-4 flex-1"
      style={{ borderTop: `3px solid ${kitColor}` }}
    >
      <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-2">
        {label}
      </p>
      <div className="flex items-center gap-2 mb-3">
        {flagClass ? (
          <span className={`${flagClass} w-8 h-5 rounded flex-shrink-0`} />
        ) : (
          <span>⚽</span>
        )}
        <span className="font-display font-bold text-lg">{value}</span>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-[var(--ink)] text-[var(--chalk)] border rounded px-2 py-1.5 text-xs"
        style={{ borderColor: "var(--line)" }}
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

function WDLBar({
  teamA,
  teamB,
  pred,
}: {
  teamA: string;
  teamB: string;
  pred: NonNullable<MatchExplanation["recorded_prediction"]>;
}) {
  const { p_team_a_win, p_draw, p_team_b_win, expected_score_a, expected_score_b } = pred;

  return (
    <div className="mb-4">
      <div className="flex h-8 rounded overflow-hidden gap-px">
        <div
          className="flex items-center justify-start pl-2"
          style={{ width: `${p_team_a_win * 100}%`, backgroundColor: "var(--turf)" }}
        />
        <div
          className="flex items-center justify-center"
          style={{ width: `${p_draw * 100}%`, backgroundColor: "var(--amber)" }}
        />
        <div
          className="flex items-center justify-end pr-2"
          style={{ width: `${p_team_b_win * 100}%`, backgroundColor: "var(--red)" }}
        />
      </div>
      <div className="flex justify-between mt-2 text-sm">
        <span className="font-display font-bold">
          <FlipNumber value={p_team_a_win} format="percent" decimals={1} /> ▲ {teamA}
        </span>
        <span className="font-display font-bold text-[var(--amber)]">
          <FlipNumber value={p_draw} format="percent" decimals={1} /> Draw
        </span>
        <span className="font-display font-bold">
          <FlipNumber value={p_team_b_win} format="percent" decimals={1} /> ▼ {teamB}
        </span>
      </div>
      <p className="font-mono text-xs opacity-40 mt-2 text-center">
        Predicted score {expected_score_a.toFixed(1)} – {expected_score_b.toFixed(1)}
      </p>
    </div>
  );
}

function FeatureChart({
  features,
  teamA,
  teamB,
  summary,
}: {
  features: MatchExplanation["features"];
  teamA: string;
  teamB: string;
  summary: string;
}) {
  const maxNorm = Math.max(...features.map((f) => f.normalised_magnitude ?? 0), 0.01);

  return (
    <div>
      <p className="font-body text-sm mb-4 opacity-70">{summary}</p>

      <div className="flex justify-between text-[10px] font-mono uppercase tracking-widest opacity-40 mb-2 px-1">
        <span>favours {teamB}</span>
        <span>favours {teamA}</span>
      </div>

      <div className="space-y-2">
        {features.map((f) => {
          const norm = f.normalised_magnitude ?? 0;
          const pct = (norm / maxNorm) * 50;
          const favB = f.favours === "b";
          const favA = f.favours === "a";
          const label =
            f.feature === "goals_conceded_diff"
              ? `${f.display} (lower = better)`
              : f.display;

          return (
            <div key={f.feature} className="grid grid-cols-[140px,1fr,56px] gap-2 items-center">
              <span className="text-xs truncate" title={f.description}>
                {label}
              </span>
              <div className="relative h-5 bg-[var(--ink)] rounded overflow-hidden">
                <div
                  className="absolute top-0 bottom-0 left-1/2 w-px"
                  style={{ backgroundColor: "var(--line)" }}
                />
                {favB && (
                  <div
                    className="absolute top-0 bottom-0 rounded-l"
                    style={{
                      right: "50%",
                      width: `${pct}%`,
                      backgroundColor: "var(--amber)",
                      opacity: 0.85,
                    }}
                  />
                )}
                {favA && (
                  <div
                    className="absolute top-0 bottom-0 rounded-r"
                    style={{
                      left: "50%",
                      width: `${pct}%`,
                      backgroundColor: "var(--turf)",
                      opacity: 0.85,
                    }}
                  />
                )}
              </div>
              <span className="font-mono text-[10px] text-right">
                {f.value >= 0 ? "+" : ""}
                {f.value.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function MatchExplainer({ teams, stages }: MatchExplainerProps) {
  const defaultStage = stages[stages.length - 1]?.stage ?? "post_r32";

  const [teamA, setTeamA] = useState("Norway");
  const [teamB, setTeamB] = useState("Brazil");
  const [stage, setStage] = useState(defaultStage);
  const [data, setData] = useState<MatchExplanation | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async () => {
    if (teamA === teamB) return;
    setLoading(true);
    try {
      const result = await getMatchExplanation(teamA, teamB, stage);
      setData(result);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [teamA, teamB, stage]);

  useEffect(() => {
    const t = setTimeout(fetchData, 300);
    return () => clearTimeout(t);
  }, [fetchData]);

  const verdict = useMemo(() => {
    if (!data?.recorded_prediction || !data.actual_result) return null;
    const fav = favouriteOf(data.recorded_prediction);
    const { score_a, score_b } = data.actual_result;
    let actualSide: "a" | "b" | "draw" = "draw";
    if (score_a > score_b) actualSide = "a";
    else if (score_a < score_b) actualSide = "b";
    const correct = fav.side === actualSide;
    const favTeam =
      fav.side === "a" ? teamA : fav.side === "b" ? teamB : "Draw";
    return { correct, favTeam, conf: fav.conf };
  }, [data, teamA, teamB]);

  return (
    <section className="mb-12">
      <h2 className="font-display text-2xl font-black uppercase tracking-tight mb-1">
        Match Explainer
      </h2>
      <p className="text-xs text-[var(--chalk)] opacity-40 mb-6">
        Feature-level attribution — deterministic, point-in-time
      </p>

      <div className="flex flex-col lg:flex-row gap-4 mb-6">
        <TeamSelectorCard label="Team A" value={teamA} teams={teams} onChange={setTeamA} />
        <TeamSelectorCard label="Team B" value={teamB} teams={teams} onChange={setTeamB} />
        <div className="bg-[var(--ink-raised)] rounded-lg p-4 lg:w-56">
          <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-2">
            Snapshot stage
          </p>
          <select
            value={stage}
            onChange={(e) => setStage(e.target.value)}
            className="w-full bg-[var(--ink)] text-[var(--chalk)] border rounded px-2 py-1.5 text-xs"
            style={{ borderColor: "var(--line)" }}
          >
            {stages.map((s) => (
              <option key={s.stage} value={s.stage}>
                {s.display_long ?? s.display}
              </option>
            ))}
          </select>
        </div>
      </div>

      {teamA === teamB && (
        <p className="text-sm text-[var(--amber)] mb-4">Pick two different teams.</p>
      )}

      {loading && <p className="text-sm opacity-40">Building feature vector…</p>}

      {data && !loading && teamA !== teamB && (
        <div className="bg-[var(--ink-raised)] rounded-lg p-6 space-y-6">
          <div>
            <h3 className="font-display font-bold text-lg mb-4">Prediction vs Reality</h3>

            {data.recorded_prediction && (
              <WDLBar teamA={teamA} teamB={teamB} pred={data.recorded_prediction} />
            )}

            {data.actual_result && (
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-display text-4xl font-black">
                  {data.actual_result.score_a} – {data.actual_result.score_b}
                </span>
                <span className="font-mono text-xs opacity-40">
                  {data.actual_result.round}
                </span>
                {verdict && (
                  <span
                    className={`px-2 py-1 rounded text-xs font-mono ${
                      verdict.correct
                        ? "bg-[var(--turf)]/30 text-[var(--turf-bright)]"
                        : "bg-[var(--red)]/30 text-[var(--red)]"
                    }`}
                  >
                    {verdict.correct ? "✓" : "✗"} Model favoured {verdict.favTeam} at{" "}
                    {(verdict.conf * 100).toFixed(1)}% — {verdict.correct ? "correct" : "wrong"}
                  </span>
                )}
              </div>
            )}

            {!data.recorded_prediction && (
              <p className="text-sm opacity-40">No recorded prediction at this stage.</p>
            )}
          </div>

          {data.features.length > 0 && (
            <div>
              <h3 className="font-display font-bold text-lg mb-4">Feature Attribution</h3>

              {data.context && (
                <div
                  className="flex flex-wrap gap-4 mb-4 text-xs opacity-50 font-mono"
                >
                  {data.context.team_a_elo != null && (
                    <span>
                      {teamA} Elo:{" "}
                      <strong className="text-[var(--chalk)]">{data.context.team_a_elo}</strong>
                    </span>
                  )}
                  {data.context.team_b_elo != null && (
                    <span>
                      {teamB} Elo:{" "}
                      <strong className="text-[var(--chalk)]">{data.context.team_b_elo}</strong>
                    </span>
                  )}
                  {data.context.h2h_meetings != null && (
                    <span>
                      H2H meetings:{" "}
                      <strong className="text-[var(--chalk)]">{data.context.h2h_meetings}</strong>
                    </span>
                  )}
                </div>
              )}

              <FeatureChart
                features={data.features}
                teamA={teamA}
                teamB={teamB}
                summary={data.summary}
              />
            </div>
          )}

          <p className="font-mono text-[10px] opacity-35 leading-relaxed">{data.notes}</p>
        </div>
      )}
    </section>
  );
}
