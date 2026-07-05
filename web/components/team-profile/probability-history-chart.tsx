"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { getProbabilityHistory } from "@/lib/api";

const STAGE_LABELS: Record<string, string> = {
  pre_tournament: "Pre-tournament",
  post_group_stage: "After Groups",
  post_r32: "After R32",
  post_r16: "After R16",
  post_qf: "After QF",
  post_sf: "After SF",
};

export function ProbabilityHistoryChart({ team }: { team: string }) {
  const [data, setData] = useState<
    {
      stage: string;
      champion: number;
      final: number;
      eliminated: boolean;
      eliminated_in?: string;
    }[]
  >([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getProbabilityHistory(team)
      .then((res) =>
        setData(
          res.history.map((h) => ({
            stage: STAGE_LABELS[h.stage] || h.stage,
            champion: +(h.p_champion * 100).toFixed(1),
            final: +(h.p_final * 100).toFixed(1),
            eliminated: h.eliminated,
            eliminated_in: h.eliminated_in,
          }))
        )
      )
      .finally(() => setLoading(false));
  }, [team]);

  if (loading) {
    return (
      <p className="text-chalk/40 text-sm">Loading prediction history...</p>
    );
  }
  if (!data.length) {
    return (
      <p className="text-chalk/40 text-sm">No prediction history saved yet.</p>
    );
  }

  const elimPoint = data.find((d) => d.eliminated);

  return (
    <div>
      <p className="text-xs text-chalk/40 uppercase tracking-widest mb-3 font-body">
        Championship probability — tournament evolution
      </p>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ left: 0, right: 20, top: 8, bottom: 0 }}>
          <XAxis
            dataKey="stage"
            tick={{
              fontSize: 10,
              fill: "var(--chalk)",
              opacity: 0.4,
              fontFamily: "var(--font-body)",
            }}
            axisLine={{ stroke: "var(--line)" }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{
              fontSize: 10,
              fill: "var(--chalk)",
              opacity: 0.4,
              fontFamily: "var(--font-mono)",
            }}
            axisLine={false}
            tickLine={false}
            width={34}
          />
          <Tooltip
            contentStyle={{
              background: "var(--ink-raised)",
              border: "1px solid var(--line)",
              borderRadius: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              color: "var(--chalk)",
            }}
            formatter={(v) => [`${typeof v === "number" ? v : 0}%`]}
          />
          {elimPoint && (
            <ReferenceLine
              x={elimPoint.stage}
              stroke="var(--red)"
              strokeDasharray="4 2"
              label={{
                value: `Eliminated (${elimPoint.eliminated_in})`,
                fill: "var(--red)",
                fontSize: 10,
              }}
            />
          )}
          <Line
            type="monotone"
            dataKey="champion"
            stroke="var(--gold)"
            strokeWidth={2}
            dot={{ fill: "var(--gold)", r: 4, strokeWidth: 0 }}
            activeDot={{ r: 6, strokeWidth: 0 }}
          />
          <Line
            type="monotone"
            dataKey="final"
            stroke="var(--turf)"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-2">
        <span className="flex items-center gap-1.5 text-xs text-chalk/40">
          <span className="w-4 h-0.5 bg-gold inline-block" /> Champion %
        </span>
        <span className="flex items-center gap-1.5 text-xs text-chalk/40">
          <span className="w-4 h-0.5 bg-turf inline-block opacity-60" /> Final %
        </span>
      </div>
    </div>
  );
}
