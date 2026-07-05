"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";
import { TeamSelector } from "@/components/story/team-selector";
import { getKitColor } from "@/lib/flags";
import type { StageMeta, StoryEvolution, TeamStagePoint } from "@/lib/types";

interface OddsEvolutionChartProps {
  data: StoryEvolution;
}

function buildChartData(
  stages: StageMeta[],
  teamsData: Record<string, TeamStagePoint[]>,
  selectedTeams: string[]
) {
  return stages.map((stageMeta) => {
    const row: Record<string, string | number> = {
      stage: stageMeta.display_short,
    };
    for (const team of selectedTeams) {
      const points = teamsData[team] || [];
      const point = points.find((p) => p.stage === stageMeta.stage);
      row[team] = point ? +(point.p_champion * 100).toFixed(2) : 0;
    }
    return row;
  });
}

function firstEliminationPoint(points: TeamStagePoint[]) {
  const idx = points.findIndex((p) => p.eliminated);
  return idx >= 0 ? points[idx] : null;
}

function computeCaption(data: StoryEvolution): string {
  const { stages, teams } = data;
  if (stages.length < 2) return "";

  const firstStage = stages[0].stage;
  const lastStage = stages[stages.length - 1].stage;

  let bestRiser = { team: "", before: 0, after: 0, delta: 0 };
  for (const [team, points] of Object.entries(teams)) {
    const p0 = points.find((p) => p.stage === firstStage);
    const p1 = points.find((p) => p.stage === lastStage);
    if (!p0 || !p1) continue;
    const delta = p1.p_champion - p0.p_champion;
    if (delta > bestRiser.delta) {
      bestRiser = { team, before: p0.p_champion, after: p1.p_champion, delta };
    }
  }

  let surpriseExit = { team: "", round: "", peak: 0 };
  for (const [team, points] of Object.entries(teams)) {
    const elimPoint = points.find((p) => p.eliminated);
    if (!elimPoint) continue;
    const peak = Math.max(...points.map((p) => p.p_champion));
    if (peak > surpriseExit.peak) {
      surpriseExit = {
        team,
        round: elimPoint.eliminated_in ?? "Unknown",
        peak,
      };
    }
  }

  const riserPart = bestRiser.team
    ? `Biggest riser so far: ${bestRiser.team} (${(bestRiser.before * 100).toFixed(1)}% → ${(bestRiser.after * 100).toFixed(1)}%).`
    : "";

  const exitPart = surpriseExit.team
    ? ` Biggest surprise exit: ${surpriseExit.team}, eliminated in ${surpriseExit.round} after peaking at ${(surpriseExit.peak * 100).toFixed(1)}%.`
    : "";

  return (riserPart + exitPart).trim();
}

export function OddsEvolutionChart({ data }: OddsEvolutionChartProps) {
  const allTeams = Object.keys(data.teams).sort();

  const top8 = useMemo(() => {
    const peaks = allTeams.map((team) => ({
      team,
      peak: Math.max(...(data.teams[team]?.map((p) => p.p_champion) ?? [0])),
    }));
    return peaks.sort((a, b) => b.peak - a.peak).slice(0, 8).map((t) => t.team);
  }, [allTeams, data.teams]);

  const [selectedTeams, setSelectedTeams] = useState<string[]>(top8);
  const [hoverTeam, setHoverTeam] = useState<string | null>(null);

  const chartData = useMemo(
    () => buildChartData(data.stages, data.teams, selectedTeams),
    [data.stages, data.teams, selectedTeams]
  );

  const leader = useMemo(() => {
    const lastStage = data.stages[data.stages.length - 1]?.stage;
    if (!lastStage) return "";
    let best = { team: "", p: 0 };
    for (const team of allTeams) {
      const pt = data.teams[team]?.find((p) => p.stage === lastStage);
      if (pt && pt.p_champion > best.p) {
        best = { team, p: pt.p_champion };
      }
    }
    return best.team;
  }, [allTeams, data.teams, data.stages]);

  const caption = computeCaption(data);

  return (
    <section className="mb-12">
      <h2 className="font-display text-2xl font-black uppercase tracking-tight mb-1">
        Championship Odds Evolution
      </h2>
      <p className="text-xs text-[var(--chalk)] opacity-40 mb-6">
        One line per team — championship probability at each milestone
      </p>

      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={chartData} margin={{ left: 0, right: 20, top: 8, bottom: 0 }}>
          <XAxis
            dataKey="stage"
            tick={{
              fontSize: 10,
              fill: "var(--chalk)",
              opacity: 0.5,
              fontFamily: "var(--font-body)",
            }}
            axisLine={{ stroke: "var(--line)" }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
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
          <Tooltip
            contentStyle={{
              background: "var(--ink-raised)",
              border: "1px solid var(--line)",
              borderRadius: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
            formatter={(value, name) => [
              `${typeof value === "number" ? value.toFixed(1) : value}%`,
              String(name),
            ]}
            labelFormatter={(label) => String(label)}
          />
          {selectedTeams.map((team) => {
            const isLeader = team === leader;
            const dimmed = hoverTeam !== null && hoverTeam !== team;
            return (
              <Line
                key={team}
                type="monotone"
                dataKey={team}
                name={team}
                stroke={isLeader ? "var(--gold)" : getKitColor(team)}
                strokeWidth={isLeader ? 2.5 : 1.5}
                dot={{ r: 3, fill: isLeader ? "var(--gold)" : getKitColor(team) }}
                connectNulls={false}
                strokeOpacity={dimmed ? 0.25 : 1}
                onMouseEnter={() => setHoverTeam(team)}
                onMouseLeave={() => setHoverTeam(null)}
              />
            );
          })}
          {selectedTeams.map((team) => {
            const elimPoint = firstEliminationPoint(data.teams[team] || []);
            if (!elimPoint) return null;
            const stageObj = data.stages.find((s) => s.stage === elimPoint.stage);
            if (!stageObj) return null;
            return (
              <ReferenceDot
                key={`elim-${team}`}
                x={stageObj.display_short}
                y={0}
                r={5}
                fill="var(--red)"
                stroke="var(--ink)"
                strokeWidth={1.5}
                label={{
                  value: "✕",
                  position: "bottom",
                  fill: "var(--red)",
                  fontSize: 10,
                }}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>

      {caption && (
        <p className="font-body text-sm opacity-60 mt-4">{caption}</p>
      )}

      <div className="mt-6">
        <TeamSelector
          allTeams={allTeams}
          selected={selectedTeams}
          onChange={setSelectedTeams}
        />
      </div>
    </section>
  );
}
