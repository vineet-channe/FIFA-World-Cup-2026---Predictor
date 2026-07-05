"use client";

import { useEffect, useMemo, useState } from "react";
import { FlipNumber } from "@/components/ui/flip-number";
import { getStoryMovers } from "@/lib/api";
import { getFlagEmoji } from "@/lib/flags";
import type { MoverCard, StageMeta, StoryMovers } from "@/lib/types";
import { clsx } from "clsx";

const PAIR_LABELS: Record<string, string> = {
  "pre_tournament→post_group_stage": "Pre-Tournament → Entering R32",
  "post_group_stage→post_r32": "Entering R32 → Entering R16",
  "post_r32→post_r16": "Entering R16 → Entering QF",
  "post_r16→post_qf": "Entering QF → Entering SF",
  "post_qf→post_sf": "Entering SF → Entering Final",
};

interface MoversSectionProps {
  stages: StageMeta[];
}

function matchVerdict(result: "W" | "D" | "L"): string {
  if (result === "W") return "Beat";
  if (result === "D") return "Drew with";
  return "Lost to";
}

function WhyBlock({ mover, accent }: { mover: MoverCard; accent: "rise" | "fall" }) {
  const { record, elo_change, eliminated_in } = mover;
  return (
    <div className="mt-3 pt-3 border-t text-xs space-y-1" style={{ borderColor: "var(--line)" }}>
      <p className="font-mono uppercase tracking-widest opacity-40 mb-2">
        WHY (computed from results):
      </p>
      <p>
        • Round record: {record.wins}W-{record.draws}D-{record.losses}L,{" "}
        {record.gf} GF / {record.ga} GA
      </p>
      {record.matches.map((m, i) => (
        <p key={i}>
          • {matchVerdict(m.result)} {m.opponent} {m.score.replace("-", "–")} ({m.round})
        </p>
      ))}
      {elo_change !== null && (
        <p>
          • Elo {elo_change >= 0 ? "+" : "−"}
          {Math.abs(elo_change).toFixed(1)} over this period
        </p>
      )}
      {accent === "fall" && eliminated_in && (
        <p className="text-[var(--red)]">• Eliminated in {eliminated_in}</p>
      )}
    </div>
  );
}

function MoverCardView({
  mover,
  accent,
}: {
  mover: MoverCard;
  accent: "rise" | "fall";
}) {
  const color = accent === "rise" ? "var(--turf-bright)" : "var(--red)";
  const arrow = accent === "rise" ? "▲" : "▼";

  return (
    <div
      className="bg-[var(--ink-raised)] rounded-lg p-4"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="font-display font-bold text-base">
          {getFlagEmoji(mover.team)} {mover.team}
        </span>
        <span className="font-mono text-sm whitespace-nowrap">
          <FlipNumber value={mover.before} format="percent" decimals={1} />
          {" → "}
          <FlipNumber value={mover.after} format="percent" decimals={1} />
        </span>
      </div>
      <p className="font-mono text-sm mt-1" style={{ color }}>
        {arrow} {mover.delta >= 0 ? "+" : ""}
        {(mover.delta * 100).toFixed(1)}pp
      </p>
      <WhyBlock mover={mover} accent={accent} />
    </div>
  );
}

export function MoversSection({ stages }: MoversSectionProps) {
  const pairs = useMemo(() => {
    const out: { from: string; to: string; label: string }[] = [];
    for (let i = 0; i < stages.length - 1; i++) {
      const from = stages[i].stage;
      const to = stages[i + 1].stage;
      const key = `${from}→${to}`;
      out.push({
        from,
        to,
        label:
          PAIR_LABELS[key] ??
          `${stages[i].display_short} → ${stages[i + 1].display_short}`,
      });
    }
    return out;
  }, [stages]);

  const [pairIdx, setPairIdx] = useState(Math.max(0, pairs.length - 1));
  const [movers, setMovers] = useState<StoryMovers | null>(null);
  const [loading, setLoading] = useState(false);

  const currentPair = pairs[pairIdx];

  useEffect(() => {
    if (!currentPair) return;
    setLoading(true);
    getStoryMovers(currentPair.from, currentPair.to)
      .then(setMovers)
      .catch(() => setMovers(null))
      .finally(() => setLoading(false));
  }, [currentPair]);

  if (!pairs.length) return null;

  return (
    <section className="mb-12">
      <h2 className="font-display text-2xl font-black uppercase tracking-tight mb-1">
        Biggest Movers
      </h2>
      <p className="text-xs text-[var(--chalk)] opacity-40 mb-4">
        Championship probability shifts with computed reasoning from intervening results
      </p>

      <div
        className="flex flex-wrap gap-1 mb-6 p-1 rounded-lg bg-[var(--ink-raised)]"
        style={{ border: "1px solid var(--line)" }}
      >
        {pairs.map((pair, i) => (
          <button
            key={pair.label}
            type="button"
            onClick={() => setPairIdx(i)}
            className={clsx(
              "px-3 py-1.5 rounded text-xs font-medium transition-colors",
              i === pairIdx
                ? "bg-[var(--turf)] text-[var(--chalk)]"
                : "opacity-50 hover:opacity-80"
            )}
          >
            {pair.label}
          </button>
        ))}
      </div>

      {loading && (
        <p className="text-sm opacity-40">Loading movers…</p>
      )}

      {movers && !loading && (
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-display font-bold text-lg mb-3 text-[var(--turf-bright)]">
              Risers
            </h3>
            <div className="space-y-3">
              {movers.risers.map((m) => (
                <MoverCardView key={m.team} mover={m} accent="rise" />
              ))}
              {!movers.risers.length && (
                <p className="text-sm opacity-40">No risers in this window.</p>
              )}
            </div>
          </div>
          <div>
            <h3 className="font-display font-bold text-lg mb-3 text-[var(--red)]">
              Fallers
            </h3>
            <div className="space-y-3">
              {movers.fallers.map((m) => (
                <MoverCardView key={m.team} mover={m} accent="fall" />
              ))}
              {!movers.fallers.length && (
                <p className="text-sm opacity-40">No fallers in this window.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
