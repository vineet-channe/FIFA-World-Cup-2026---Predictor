import { FlipNumber } from "@/components/ui/flip-number";
import type { StageMeta } from "@/lib/types";

interface StageTimelineProps {
  stages: StageMeta[];
}

function formatDate(iso: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
    });
  } catch {
    return iso.slice(0, 10);
  }
}

export function StageTimeline({ stages }: StageTimelineProps) {
  if (!stages.length) return null;

  return (
    <section className="mb-10 overflow-x-auto">
      <div className="flex items-start min-w-max px-1">
        {stages.map((stage, i) => {
          const isLatest = i === stages.length - 1;
          return (
            <div key={stage.stage} className="flex items-start">
              <div className="flex flex-col items-center w-44 px-2">
                <div
                  className="w-3 h-3 rounded-full mb-3 flex-shrink-0"
                  style={{
                    backgroundColor: isLatest ? "var(--turf)" : "var(--line)",
                    boxShadow: isLatest ? "0 0 8px var(--turf-bright)" : undefined,
                  }}
                />
                <p className="font-body text-xs font-medium text-center leading-tight">
                  {stage.display_long}
                </p>
                <p className="font-mono text-[10px] opacity-40 mt-1">
                  {formatDate(stage.saved_at)}
                </p>
                <p className="font-display text-3xl font-black mt-2 text-[var(--turf-bright)]">
                  <FlipNumber value={stage.teams_alive} format="integer" />
                </p>
                <p className="font-body text-[10px] opacity-40 uppercase tracking-widest">
                  teams alive
                </p>
                <p className="font-mono text-[10px] opacity-30 mt-1">
                  {stage.matches_completed} matches done
                </p>
              </div>
              {i < stages.length - 1 && (
                <div
                  className="w-12 h-px mt-1.5 flex-shrink-0 self-start"
                  style={{ backgroundColor: "var(--line)" }}
                />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
