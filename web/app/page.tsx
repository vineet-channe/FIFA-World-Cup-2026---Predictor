import { ChampionshipLeaderboard } from "@/components/leaderboard/championship-leaderboard";
import { Suspense } from "react";

export const dynamic = "force-dynamic";

export default function FavouritesPage() {
  return (
    <Suspense fallback={<LoadingSkeleton />}>
      <ChampionshipLeaderboard />
    </Suspense>
  );
}

function LoadingSkeleton() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-3">
      <div className="h-12 w-64 rounded bg-[var(--ink-raised)] animate-pulse" />
      {Array.from({ length: 10 }).map((_, i) => (
        <div key={i} className="h-12 rounded bg-[var(--ink-raised)] animate-pulse" />
      ))}
    </div>
  );
}
