"use client";

import { useState, useEffect } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { FlipNumber } from "@/components/ui/flip-number";
import { getTeam } from "@/lib/api";
import { ProbabilityHistoryChart } from "@/components/team-profile/probability-history-chart";
import { getFlagClass, getKitColor } from "@/lib/flags";
import type { TeamDetail, TeamSummary } from "@/lib/types";

interface ProfileProps {
  teams: TeamSummary[];
}

export function TeamProfile({ teams }: ProfileProps) {
  const teamNames = teams.map((t) => t.name).sort();
  const [selected, setSelected] = useState(teamNames[0] ?? "Brazil");
  const [detail, setDetail] = useState<TeamDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getTeam(selected)
      .then((d) => { if (!cancelled) { setDetail(d); setLoading(false); } })
      .catch(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selected]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="font-display text-4xl font-black uppercase tracking-tight mb-6">
        Team Profile
      </h1>

      {/* Selector */}
      <select
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
        className="mb-8 bg-[var(--ink-raised)] text-[var(--chalk)] border rounded px-3 py-2
          text-sm font-body focus-visible:outline-none focus-visible:ring-2
          focus-visible:ring-[var(--turf-bright)]"
        style={{ borderColor: "var(--line)" }}
        aria-label="Select team"
      >
        {teams.map((t) => (
          <option key={t.name} value={t.name}>
            {t.name}
          </option>
        ))}
      </select>

      {loading && <ProfileSkeleton />}

      {detail && !loading && (
        <ProfileBody detail={detail} teamName={selected} />
      )}
    </div>
  );
}

// ── Main profile body ─────────────────────────────────────────────────────────
function ProfileBody({ detail, teamName }: { detail: TeamDetail; teamName: string }) {
  const { summary, probabilities, radar_stats, form_timeline, tournament_path } = detail;
  const flagClass = getFlagClass(teamName);
  const kitColor = getKitColor(teamName);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div
        className="flex items-center gap-4 pb-4"
        style={{ borderBottom: "1px solid var(--line)" }}
      >
        {flagClass ? (
          <span className={`${flagClass} w-10 h-7 rounded flex-shrink-0`} />
        ) : (
          <span className="text-3xl">⚽</span>
        )}
        <h2
          className="font-display text-3xl font-black uppercase"
          style={{ borderLeft: `4px solid ${kitColor}`, paddingLeft: "12px" }}
        >
          {teamName}
        </h2>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Elo Rating",     value: Math.round(summary.elo).toString() },
          { label: "FIFA Points",    value: summary.fifa_points.toFixed(0) },
          { label: "Squad Value",    value: `€${(summary.squad_value_eur / 1e6).toFixed(0)}M` },
          { label: "WC Appearances", value: summary.wc_appearances.toString() },
        ].map(({ label, value }) => (
          <div
            key={label}
            className="bg-[var(--ink-raised)] rounded p-3"
          >
            <p className="font-mono text-[10px] uppercase tracking-widest opacity-40 mb-1">
              {label}
            </p>
            <p className="font-display text-2xl font-bold">{value}</p>
          </div>
        ))}
      </div>

      {/* Radar + Funnel row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3
            className="font-display text-sm font-bold uppercase tracking-widest opacity-50 mb-4"
          >
            Strength Radar
          </h3>
          <TeamRadar stats={radar_stats} />
        </div>
        <div>
          <h3
            className="font-display text-sm font-bold uppercase tracking-widest opacity-50 mb-4"
          >
            Tournament Path
          </h3>
          <TournamentFunnel path={tournament_path} />
        </div>
      </div>

      <div>
        <ProbabilityHistoryChart team={teamName} />
      </div>

      {/* Form timeline */}
      <div>
        <h3
          className="font-display text-sm font-bold uppercase tracking-widest opacity-50 mb-3"
        >
          Recent Form
        </h3>
        <div className="flex flex-wrap gap-2">
          {form_timeline.map((m, i) => {
            const bg =
              m.result === "W"
                ? "var(--turf)"
                : m.result === "D"
                ? "var(--amber)"
                : "var(--red)";
            const icon = m.result === "W" ? "▲" : m.result === "D" ? "—" : "▼";
            return (
              <div
                key={i}
                title={`${m.date} vs ${m.opponent} (${m.score}) · ${m.competition}`}
                className="flex flex-col items-center rounded px-2 py-1.5 min-w-[48px] cursor-default"
                style={{ backgroundColor: bg }}
              >
                <span className="font-display font-black text-base leading-none text-[var(--chalk)]">
                  {icon}
                </span>
                <span className="font-mono text-[9px] text-[var(--chalk)] opacity-80 mt-0.5">
                  {m.score}
                </span>
                <span className="font-mono text-[8px] text-[var(--chalk)] opacity-60 truncate max-w-[48px]">
                  {m.opponent.slice(0, 5)}
                </span>
              </div>
            );
          })}
          {form_timeline.length === 0 && (
            <p className="font-mono text-xs opacity-40">No recent competitive matches found.</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Radar chart ───────────────────────────────────────────────────────────────
function TeamRadar({ stats }: { stats: TeamDetail["radar_stats"] }) {
  const data = [
    { axis: "Elo",        value: stats.elo_strength },
    { axis: "Form",       value: stats.recent_form },
    { axis: "H2H",        value: stats.h2h_dominance },
    { axis: "Squad",      value: stats.squad_value },
    { axis: "WC Exp.",    value: stats.tournament_experience },
    { axis: "Advance %",  value: stats.advance_probability },
  ];

  return (
    <ResponsiveContainer width="100%" height={240}>
      <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke="rgba(245,243,236,0.12)" />
        <PolarAngleAxis
          dataKey="axis"
          tick={{ fill: "rgba(245,243,236,0.6)", fontSize: 11, fontFamily: "var(--font-body)" }}
        />
        <Radar
          name="Team"
          dataKey="value"
          stroke="var(--gold)"
          fill="var(--turf)"
          fillOpacity={0.25}
          strokeWidth={2}
        />
        <Tooltip
          contentStyle={{
            background: "var(--ink-raised)",
            border: "1px solid var(--line)",
            borderRadius: "6px",
            color: "var(--chalk)",
            fontFamily: "var(--font-mono)",
            fontSize: "12px",
          }}
          formatter={(v) => [typeof v === "number" ? `${v.toFixed(1)}/100` : String(v), ""]}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

// ── Tournament funnel (custom stepped bars) ───────────────────────────────────
function TournamentFunnel({ path }: { path: TeamDetail["tournament_path"] }) {
  const stages: [string, number, boolean][] = [
    ["Advance Groups", path.advance_groups, false],
    ["Round of 16",    path.r16,            false],
    ["Quarter-final",  path.quarter,        false],
    ["Semi-final",     path.semi,           false],
    ["Final",          path.final,          false],
    ["Champion",       path.champion,       true],
  ];

  return (
    <div className="space-y-2">
      {stages.map(([label, prob, isChamp]) => (
        <div key={label} className="flex items-center gap-3">
          <span
            className="font-mono text-[10px] w-24 flex-shrink-0 text-right opacity-50"
          >
            {label}
          </span>
          <div className="flex-1 h-5 bg-[var(--ink)] rounded overflow-hidden">
            <div
              className="h-full rounded flex items-center justify-end pr-2 transition-all duration-700"
              style={{
                width: `${Math.max(prob * 100, 1)}%`,
                backgroundColor: isChamp ? "var(--gold)" : "var(--turf)",
              }}
            />
          </div>
          <span
            className="font-display font-bold text-sm w-12 flex-shrink-0"
            style={{ color: isChamp ? "var(--gold)" : "var(--chalk)" }}
          >
            <FlipNumber value={prob} format="percent" decimals={1} />
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
function ProfileSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-12 w-48 rounded bg-[var(--ink-raised)]" />
      <div className="grid grid-cols-4 gap-3">
        {[1,2,3,4].map(i => <div key={i} className="h-20 rounded bg-[var(--ink-raised)]" />)}
      </div>
      <div className="h-64 rounded bg-[var(--ink-raised)]" />
    </div>
  );
}
