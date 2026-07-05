"use client";

import { useState } from "react";
import { TournamentBracket } from "@/lib/types";
import { GroupWallChart } from "@/components/wall-chart/group-wall-chart";
import { KnockoutRound } from "@/components/tournament/knockout-round";
import { RoundPending } from "@/components/ui/round-pending";

const ALL_ROUNDS: { round_key: string; round_name: string }[] = [
  { round_key: "groups", round_name: "Group Stage" },
  { round_key: "r32", round_name: "Round of 32" },
  { round_key: "r16", round_name: "Round of 16" },
  { round_key: "qf", round_name: "Quarter-finals" },
  { round_key: "sf", round_name: "Semi-finals" },
  { round_key: "3rd", round_name: "3rd Place" },
  { round_key: "final", round_name: "Final" },
];

export function TournamentTabs({ bracket }: { bracket: TournamentBracket }) {
  const roundMap = new Map(bracket.rounds.map((r) => [r.round_key, r]));
  const [active, setActive] = useState(
    bracket.rounds[bracket.rounds.length - 1]?.round_key ?? "groups"
  );
  const activeRound = roundMap.get(active);

  if (!bracket.rounds.length) {
    return (
      <p
        style={{
          fontSize: 14,
          color: "rgba(245,243,236,0.4)",
          fontFamily: "var(--font-mono)",
        }}
      >
        No tournament data yet
      </p>
    );
  }

  return (
    <div>
      <div
        style={{
          borderBottom: "0.5px solid var(--line)",
          display: "flex",
          gap: 0,
          overflowX: "auto",
          marginBottom: 24,
        }}
      >
        {ALL_ROUNDS.map((def) => {
          const round = roundMap.get(def.round_key);
          const hasData = round && round.total > 0;
          return (
            <button
              key={def.round_key}
              type="button"
              onClick={() => setActive(def.round_key)}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                background: "none",
                border: "none",
                borderBottom:
                  active === def.round_key
                    ? "2px solid var(--turf)"
                    : "2px solid transparent",
                color:
                  active === def.round_key
                    ? "var(--chalk)"
                    : hasData
                    ? "rgba(245,243,236,0.45)"
                    : "rgba(245,243,236,0.25)",
                cursor: "pointer",
                whiteSpace: "nowrap",
                fontFamily: "var(--font-body)",
                fontWeight: active === def.round_key ? 500 : 400,
                marginBottom: -1,
              }}
            >
              {def.round_name}
              {hasData && (
                <span
                  style={{
                    marginLeft: 6,
                    fontSize: 10,
                    fontFamily: "var(--font-mono)",
                    color:
                      round.completed === round.total
                        ? "var(--turf)"
                        : "rgba(245,243,236,0.3)",
                  }}
                >
                  {round.completed}/{round.total}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {!activeRound || activeRound.total === 0 ? (
        <RoundPending
          roundName={ALL_ROUNDS.find((r) => r.round_key === active)?.round_name ?? active}
        />
      ) : activeRound.round_key === "groups" ? (
        <GroupWallChart matches={activeRound.matches} />
      ) : (
        <KnockoutRound round={activeRound} />
      )}
    </div>
  );
}
