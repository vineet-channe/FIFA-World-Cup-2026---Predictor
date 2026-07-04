import { getTournamentBracket } from "@/lib/api";
import { TournamentTabs } from "@/components/tournament/tournament-tabs";

export const dynamic = "force-dynamic";

export default async function TournamentPage() {
  let bracket;
  try {
    bracket = await getTournamentBracket();
  } catch {
    return (
      <main className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-display font-bold text-chalk mb-1">Tournament</h1>
        <p className="text-sm text-chalk/40 font-mono">No tournament data yet</p>
      </main>
    );
  }

  const played = bracket.rounds.reduce((a, r) => a + r.completed, 0);
  const total = bracket.rounds.reduce((a, r) => a + r.total, 0);

  return (
    <main className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-display font-bold text-[var(--chalk)] mb-1">
        Tournament
      </h1>
      <p className="text-sm text-[var(--chalk)] opacity-40 mb-6 font-mono">
        {bracket.current_round} · {played} of {total || 104} matches played
      </p>
      <TournamentTabs bracket={bracket} />
    </main>
  );
}
