"use client";

import { useState } from "react";
import { TournamentBracket } from "@/lib/types";
import { GroupWallChart } from "@/components/wall-chart/group-wall-chart";
import { KnockoutRound } from "@/components/tournament/knockout-round";

export function TournamentTabs({ bracket }: { bracket: TournamentBracket }) {
  const [active, setActive] = useState(bracket.rounds[0]?.round_key ?? "groups");
  const activeRound = bracket.rounds.find((r) => r.round_key === active);

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
        {bracket.rounds.map((round) => (
          <button
            key={round.round_key}
            type="button"
            onClick={() => setActive(round.round_key)}
            style={{
              padding: "8px 16px",
              fontSize: 13,
              background: "none",
              border: "none",
              borderBottom:
                active === round.round_key
                  ? "2px solid var(--turf)"
                  : "2px solid transparent",
              color:
                active === round.round_key
                  ? "var(--chalk)"
                  : "rgba(245,243,236,0.45)",
              cursor: "pointer",
              whiteSpace: "nowrap",
              fontFamily: "var(--font-body)",
              fontWeight: active === round.round_key ? 500 : 400,
              marginBottom: -1,
            }}
          >
            {round.round_name}
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
          </button>
        ))}
      </div>

      {activeRound?.round_key === "groups" ? (
        <GroupWallChart matches={activeRound.matches} />
      ) : activeRound ? (
        <KnockoutRound round={activeRound} />
      ) : null}
    </div>
  );
}
