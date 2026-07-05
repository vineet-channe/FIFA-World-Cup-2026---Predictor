import { getTeams, getGroups, getMeta } from "@/lib/api";
import { getKitColor, getFlagClass } from "@/lib/flags";
import { FlipNumber } from "@/components/ui/flip-number";
import { AboutSection } from "@/components/ui/about-section";
import type { TeamSummary } from "@/lib/types";

interface TeamProbRecord {
  p_champion?: number;
  p_final?: number;
  p_semi?: number;
  p_quarter?: number;
  p_r16?: number;
  p_r32?: number;
  p_advance_groups?: number;
  eliminated_in?: string;
}

// We need championship probs — fetch from simulation endpoint
async function getChampionshipData() {
  const [teams, groups, meta] = await Promise.all([
    getTeams(),
    getGroups(),
    getMeta(),
  ]);

  const simRes = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/simulation`,
    { cache: "no-store" }
  );
  const sim = await simRes.json();
  const teamProbs: Record<string, TeamProbRecord> = sim.team_probabilities ?? {};

  return { teams, teamProbs, meta };
}

export async function ChampionshipLeaderboard() {
  const { teams, teamProbs, meta } = await getChampionshipData();

  // Sort by p_champion desc
  const sorted = [...teams].sort((a, b) => {
    const pa = teamProbs[a.name]?.p_champion ?? 0;
    const pb = teamProbs[b.name]?.p_champion ?? 0;
    return pb - pa;
  });

  const activeTeams = sorted.filter(
    (t) => (teamProbs[t.name]?.p_champion ?? 0) > 0
  );
  const eliminatedTeams = sorted
    .filter((t) => (teamProbs[t.name]?.p_champion ?? 0) === 0)
    .sort(
      (a, b) =>
        (teamProbs[b.name]?.p_r32 ?? 0) - (teamProbs[a.name]?.p_r32 ?? 0)
    );

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-8">
        <h1 className="font-display text-4xl md:text-5xl font-black uppercase tracking-tight text-[var(--chalk)]">
          Championship Odds
        </h1>
        <span className="font-mono text-sm text-[var(--chalk)] opacity-50">
          {meta.n_simulations.toLocaleString()} simulations
        </span>
      </div>

      {/* Meta bar */}
      <div
        className="flex items-center gap-4 mb-6 pb-4 text-xs font-mono text-[var(--chalk)] opacity-40"
        style={{ borderBottom: "1px solid var(--line)" }}
      >
        <span>Model: {meta.model_version}</span>
        {meta.ensemble_brier_wc2022 && (
          <span>Brier (WC 2022): {meta.ensemble_brier_wc2022.toFixed(4)}</span>
        )}
        <span className="ml-auto">Last run: {meta.last_simulation.slice(0, 10)}</span>
      </div>

      {/* Leaderboard rows */}
      <div className="space-y-px">
        {activeTeams.map((team, idx) => {
          const probs = teamProbs[team.name] ?? {};
          const pChamp = probs.p_champion ?? 0;
          const isFirst = idx === 0;

          return (
            <LeaderboardRow
              key={team.name}
              rank={idx + 1}
              team={team}
              probs={probs}
              pChamp={pChamp}
              isFirst={isFirst}
            />
          );
        })}
      </div>

      {eliminatedTeams.length > 0 && (
        <div
          style={{
            marginTop: 24,
            borderTop: "0.5px solid var(--line)",
            paddingTop: 16,
          }}
        >
          <div
            style={{
              fontSize: 10,
              color: "rgba(245,243,236,0.3)",
              fontFamily: "var(--font-mono)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 10,
            }}
          >
            Eliminated · {eliminatedTeams.length} teams
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {eliminatedTeams.map((team) => {
              const probs = teamProbs[team.name] ?? {};
              const kit = getKitColor(team.name);
              return (
                <span
                  key={team.name}
                  style={{
                    fontSize: 11,
                    padding: "3px 10px",
                    borderRadius: 100,
                    background: "var(--ink-raised)",
                    color: "rgba(245,243,236,0.3)",
                    fontFamily: "var(--font-body)",
                    borderLeft: `2px solid ${kit}`,
                  }}
                >
                  {getFlagClass(team.name) && (
                    <i
                      className={getFlagClass(team.name)}
                      style={{ marginRight: 4 }}
                    />
                  )}
                  {team.name}
                  {probs.eliminated_in && (
                    <span style={{ marginLeft: 4, fontSize: 9, opacity: 0.6 }}>
                      ({probs.eliminated_in})
                    </span>
                  )}
                </span>
              );
            })}
          </div>
        </div>
      )}

      <AboutSection />

      {/* Full data table */}
      <FullTable sorted={sorted} teamProbs={teamProbs} />
    </div>
  );
}

// ── Row component (client-side hover for expansion) ──────────────────────────
function LeaderboardRow({
  rank,
  team,
  probs,
  pChamp,
  isFirst,
}: {
  rank: number;
  team: TeamSummary;
  probs: TeamProbRecord;
  pChamp: number;
  isFirst: boolean;
}) {
  const barMax = 0.25; // 25% = full bar width (clamp so bars spread nicely)
  const barPct = Math.min((pChamp / barMax) * 100, 100);
  const kitColor = getKitColor(team.name);
  const flagClass = getFlagClass(team.name);

  return (
    <details className="group" style={{ "--kit": kitColor } as React.CSSProperties}>
      <summary
        className="flex items-center gap-3 px-3 py-2.5 rounded cursor-pointer
          hover:bg-[var(--ink-raised)] transition-colors duration-120 list-none"
        style={{ borderLeft: `3px solid ${kitColor}` }}
      >
        {/* Jersey-number rank badge */}
        <span
          className="font-display font-black text-sm w-8 h-8 rounded flex items-center justify-center flex-shrink-0"
          style={{
            backgroundColor: isFirst ? "var(--gold)" : "var(--turf)",
            color: isFirst ? "var(--ink)" : "var(--chalk)",
          }}
        >
          {String(rank).padStart(2, "0")}
        </span>

        {/* Flag */}
        {flagClass ? (
          <span className={`${flagClass} w-6 h-4 rounded-sm flex-shrink-0`} />
        ) : (
          <span className="w-6 text-center text-sm flex-shrink-0">⚽</span>
        )}

        {/* Team name */}
        <span className="font-body font-medium text-sm flex-1 truncate">
          {team.name}
        </span>

        {/* Probability bar */}
        <div className="hidden sm:flex items-center gap-3 flex-1 max-w-xs">
          <div className="flex-1 h-2 rounded-full bg-[var(--ink-raised)] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${barPct}%`,
                background: `linear-gradient(90deg, var(--turf) 0%, var(--turf-bright) 100%)`,
              }}
            />
          </div>
        </div>

        {/* Percentage */}
        <span
          className="font-display font-bold text-base w-16 text-right flex-shrink-0"
          style={{ color: isFirst ? "var(--gold)" : "var(--chalk)" }}
        >
          <FlipNumber value={pChamp} format="percent" decimals={1} />
        </span>
      </summary>

      {/* Expanded detail */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-4 py-3 ml-11
        text-xs font-mono text-[var(--chalk)] opacity-70"
        style={{ borderLeft: `3px solid ${kitColor}`, background: "var(--ink-raised)" }}>
        {[
          ["Final",   probs.p_final],
          ["Semi",    probs.p_semi],
          ["QF",      probs.p_quarter],
          ["Groups",  probs.p_advance_groups],
        ].map(([label, val]) => (
          <div key={String(label)} className="flex flex-col gap-0.5">
            <span className="opacity-50 uppercase tracking-wider text-[10px]">{label}</span>
            <span className="text-sm font-medium">
              <FlipNumber value={Number(val) ?? 0} format="percent" decimals={1} />
            </span>
          </div>
        ))}
      </div>
    </details>
  );
}

// ── Sortable full table ───────────────────────────────────────────────────────
function FullTable({
  sorted,
  teamProbs,
}: {
  sorted: TeamSummary[];
  teamProbs: Record<string, TeamProbRecord>;
}) {
  return (
    <div className="mt-12">
      <h2
        className="font-display text-xl font-bold uppercase tracking-tight mb-4 pb-3"
        style={{ borderBottom: "1px solid var(--line)" }}
      >
        All 48 Teams
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-xs text-[var(--chalk)]">
          <thead>
            <tr className="opacity-40" style={{ borderBottom: "1px solid var(--line)" }}>
              <th className="py-2 text-left w-8">#</th>
              <th className="py-2 text-left">Team</th>
              <th className="py-2 text-right">Champion</th>
              <th className="py-2 text-right">Final</th>
              <th className="py-2 text-right">Semi</th>
              <th className="py-2 text-right">QF</th>
              <th className="py-2 text-right">Groups</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((team, idx) => {
              const p = teamProbs[team.name] ?? {};
              const kitColor = getKitColor(team.name);
              return (
                <tr
                  key={team.name}
                  className="border-b hover:bg-[var(--ink-raised)] transition-colors"
                  style={{ borderColor: "var(--line)" }}
                >
                  <td className="py-1.5 opacity-40">{idx + 1}</td>
                  <td className="py-1.5">
                    <span
                      className="inline-block w-1 h-3.5 mr-2 rounded-sm align-middle"
                      style={{ backgroundColor: kitColor }}
                    />
                    {team.name}
                  </td>
                  <td className="py-1.5 text-right">{pct(p.p_champion)}</td>
                  <td className="py-1.5 text-right">{pct(p.p_final)}</td>
                  <td className="py-1.5 text-right">{pct(p.p_semi)}</td>
                  <td className="py-1.5 text-right">{pct(p.p_quarter)}</td>
                  <td className="py-1.5 text-right">{pct(p.p_advance_groups)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function pct(v?: number): string {
  if (v === undefined || v === null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}
