import { getGroups } from "@/lib/api";
import { getFlagClass, getKitColor } from "@/lib/flags";
import type { GroupData, MatchPrediction } from "@/lib/types";

export { GroupWallChart } from "@/components/wall-chart/group-wall-chart";

export async function GroupGrid() {
  const groups = await getGroups();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="font-display text-4xl font-black uppercase tracking-tight mb-8">
        Group Stage
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-[var(--line)]">
        {groups.map((group) => (
          <GroupPanel key={group.letter} group={group} />
        ))}
      </div>
    </div>
  );
}

function GroupPanel({ group }: { group: GroupData }) {
  return (
    <div className="bg-[var(--ink-raised)] p-4">
      {/* Group header */}
      <div
        className="flex items-center justify-between mb-3 pb-2"
        style={{ borderBottom: "1px solid var(--line)" }}
      >
        <span className="font-display text-xl font-black uppercase tracking-widest">
          Group {group.letter}
        </span>
        <span className="font-mono text-xs opacity-40">Advanced</span>
      </div>

      {/* Team rows */}
      <div className="space-y-1 mb-4">
        {group.teams
          .slice()
          .sort(
            (a, b) =>
              (group.advance_probs[b] ?? 0) - (group.advance_probs[a] ?? 0)
          )
          .map((team) => {
            const adv = group.advance_probs[team] ?? 0;
            const kitColor = getKitColor(team);
            const flagClass = getFlagClass(team);
            return (
              <div
                key={team}
                className="flex items-center gap-2 text-sm"
                style={{ borderLeft: `3px solid ${kitColor}`, paddingLeft: "8px" }}
              >
                {flagClass ? (
                  <span className={`${flagClass} w-5 h-3.5 rounded-sm flex-shrink-0`} />
                ) : (
                  <span className="text-xs">⚽</span>
                )}
                <span className="font-body flex-1 truncate text-xs">{team}</span>
                {/* Advance probability bar */}
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <div className="w-16 h-1.5 rounded-full bg-[var(--ink)] overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${adv * 100}%`,
                        backgroundColor:
                          adv > 0.6
                            ? "var(--turf-bright)"
                            : adv > 0.4
                            ? "var(--amber)"
                            : "var(--red)",
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs w-10 text-right opacity-70">
                    {(adv * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            );
          })}
      </div>

      {/* Group matches */}
      {group.matches.length > 0 && (
        <>
          <div
            className="text-[10px] font-mono uppercase tracking-widest opacity-30 mb-2 pt-2"
            style={{ borderTop: "1px solid var(--line)" }}
          >
            Predictions & Results
          </div>
          <div className="space-y-2">
            {group.matches.map((match) => (
              <MatchRow key={match.match_id} match={match} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function MatchRow({ match }: { match: MatchPrediction }) {
  const { team_a, team_b, p_team_a_win, p_draw, p_team_b_win,
          expected_score_a, expected_score_b, match_date } = match;

  return (
    <div className="text-[11px] font-mono">
      <div className="flex items-center justify-between opacity-50 mb-0.5">
        <span className="truncate max-w-[40%]">{team_a}</span>
        <span className="px-1 text-[10px]">{match_date?.slice(5)}</span>
        <span className="truncate max-w-[40%] text-right">{team_b}</span>
      </div>
      {/* W/D/L stacked bar */}
      <div className="flex h-2 rounded overflow-hidden gap-px">
        <div
          className="rounded-l"
          style={{ width: `${p_team_a_win * 100}%`, backgroundColor: "var(--turf)" }}
        />
        <div
          style={{ width: `${p_draw * 100}%`, backgroundColor: "var(--amber)" }}
        />
        <div
          className="rounded-r"
          style={{ width: `${p_team_b_win * 100}%`, backgroundColor: "var(--red)" }}
        />
      </div>
      {/* Score and probabilities */}
      <div className="flex justify-between mt-0.5 opacity-50">
        <span>{(p_team_a_win * 100).toFixed(0)}%</span>
        <span>
          {Math.round(expected_score_a)}–{Math.round(expected_score_b)}
        </span>
        <span>{(p_team_b_win * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}
