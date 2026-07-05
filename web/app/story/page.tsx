import { StageTimeline } from "@/components/story/stage-timeline";
import { OddsEvolutionChart } from "@/components/story/odds-evolution-chart";
import { MoversSection } from "@/components/story/movers-section";
import { MatchExplainer } from "@/components/story/match-explainer";
import { AccuracyReport } from "@/components/story/accuracy-report";
import { getStoryEvolution, getStoryAccuracy, getTeams } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function StoryPage() {
  const [evolution, accuracy, teams] = await Promise.all([
    getStoryEvolution(),
    getStoryAccuracy(),
    getTeams(),
  ]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <h1 className="font-display text-4xl font-black uppercase tracking-tight mb-3">
        The Story
      </h1>
      <p className="font-body text-sm opacity-60 mb-8 max-w-2xl leading-relaxed">
        Every number on this page was predicted before the matches were played.
        Each stage&apos;s forecasts are evaluated only against results that came after them —
        no hindsight. Where a snapshot was reconstructed after a data fix, it is labelled.
      </p>

      <StageTimeline stages={evolution.stages} />
      <OddsEvolutionChart data={evolution} />
      <MoversSection stages={evolution.stages} />
      <MatchExplainer teams={teams} stages={evolution.stages} />
      <AccuracyReport data={accuracy} />
    </div>
  );
}
