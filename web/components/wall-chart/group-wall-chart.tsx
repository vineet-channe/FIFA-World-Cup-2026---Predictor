"use client";

import { getFlagClass, getKitColor } from "@/lib/flags";
import { FlipNumber } from "@/components/ui/flip-number";
import type { MatchResultCard } from "@/lib/types";

function groupLetterFromMatch(match: MatchResultCard): string {
  if (match.group) return match.group;
  const prefix = "Group Stage - ";
  if (match.round.startsWith(prefix)) {
    return match.round.slice(prefix.length).trim();
  }
  return "A";
}

/** Group wall chart driven by tournament bracket match cards. */
export function GroupWallChart({ matches }: { matches: MatchResultCard[] }) {
  const byGroup = new Map<string, MatchResultCard[]>();
  for (const m of matches) {
    const letter = groupLetterFromMatch(m);
    const list = byGroup.get(letter) ?? [];
    list.push(m);
    byGroup.set(letter, list);
  }

  const letters = [...byGroup.keys()].sort();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-[var(--line)]">
      {letters.map((letter) => {
        const groupMatches = byGroup.get(letter) ?? [];
        const teams = [
          ...new Set(groupMatches.flatMap((m) => [m.team_a, m.team_b])),
        ].sort();

        return (
          <div key={letter} className="bg-[var(--ink-raised)] p-4">
            <div
              className="flex items-center justify-between mb-3 pb-2"
              style={{ borderBottom: "1px solid var(--line)" }}
            >
              <span className="font-display text-xl font-black uppercase tracking-widest">
                Group {letter}
              </span>
              <span className="font-mono text-xs opacity-40">
                {groupMatches.filter((m) => m.status === "completed").length}/
                {groupMatches.length}
              </span>
            </div>

            <div className="space-y-1 mb-4">
              {teams.map((team) => (
                <div
                  key={team}
                  className="flex items-center gap-2 text-sm"
                  style={{
                    borderLeft: `3px solid ${getKitColor(team)}`,
                    paddingLeft: "8px",
                  }}
                >
                  {getFlagClass(team) ? (
                    <span
                      className={`${getFlagClass(team)} w-5 h-3.5 rounded-sm flex-shrink-0`}
                    />
                  ) : (
                    <span className="text-xs">⚽</span>
                  )}
                  <span className="font-body flex-1 truncate text-xs">{team}</span>
                </div>
              ))}
            </div>

            <div className="space-y-2">
              {groupMatches.map((match) => (
                <BracketMatchRow key={match.match_id} match={match} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function BracketMatchRow({ match }: { match: MatchResultCard }) {
  const completed = match.status === "completed";

  return (
    <div className="text-[11px] font-mono">
      <div className="flex items-center justify-between opacity-50 mb-0.5">
        <span className="truncate max-w-[40%]">{match.team_a}</span>
        <span className="px-1 text-[10px]">
          {completed
            ? `${match.actual_score_a}–${match.actual_score_b}`
            : match.match_date?.slice(5) ?? "—"}
        </span>
        <span className="truncate max-w-[40%] text-right">{match.team_b}</span>
      </div>
      <div className="flex h-2 rounded overflow-hidden gap-px">
        <div
          className="rounded-l"
          style={{
            width: `${match.p_team_a_win * 100}%`,
            backgroundColor: "var(--turf)",
          }}
        />
        <div
          style={{
            width: `${match.p_draw * 100}%`,
            backgroundColor: "var(--amber)",
          }}
        />
        <div
          className="rounded-r"
          style={{
            width: `${match.p_team_b_win * 100}%`,
            backgroundColor: "var(--red)",
          }}
        />
      </div>
      <div className="flex justify-between mt-0.5 opacity-50">
        <span>
          <FlipNumber
            value={match.p_team_a_win * 100}
            format="decimal"
            decimals={0}
          />
          %
        </span>
        <span>
          {Math.round(match.expected_score_a)}–{Math.round(match.expected_score_b)}
        </span>
        <span>
          <FlipNumber
            value={match.p_team_b_win * 100}
            format="decimal"
            decimals={0}
          />
          %
        </span>
      </div>
    </div>
  );
}
