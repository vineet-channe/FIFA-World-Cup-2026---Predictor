"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import { getFlagClass, getKitColor } from "@/lib/flags";

interface TeamSelectorProps {
  allTeams: string[];
  selected: string[];
  onChange: (teams: string[]) => void;
}

export function TeamSelector({ allTeams, selected, onChange }: TeamSelectorProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const filtered = useMemo(
    () => allTeams.filter((t) => t.toLowerCase().includes(query.toLowerCase())),
    [allTeams, query]
  );

  function toggle(team: string) {
    if (selected.includes(team)) {
      onChange(selected.filter((t) => t !== team));
    } else {
      onChange([...selected, team]);
    }
  }

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
        {selected.map((team) => (
          <span
            key={team}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              fontSize: 11,
              padding: "3px 8px 3px 4px",
              borderRadius: 100,
              background: "var(--ink-raised)",
              border: "0.5px solid var(--line)",
              borderLeft: `3px solid ${getKitColor(team)}`,
              color: "var(--chalk)",
              fontFamily: "var(--font-body)",
            }}
          >
            <i className={getFlagClass(team)} />
            {team}
            <button
              type="button"
              onClick={() => toggle(team)}
              style={{
                background: "none",
                border: "none",
                color: "rgba(245,243,236,0.4)",
                cursor: "pointer",
                fontSize: 12,
                lineHeight: 1,
                padding: 0,
              }}
              aria-label={`Remove ${team}`}
            >
              ✕
            </button>
          </span>
        ))}
      </div>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setOpen(true)}
        placeholder="Add a team to compare..."
        style={{
          width: "100%",
          fontSize: 13,
          padding: "8px 12px",
          borderRadius: 8,
          background: "var(--ink-raised)",
          border: "0.5px solid var(--line)",
          color: "var(--chalk)",
          fontFamily: "var(--font-body)",
        }}
      />

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 4px)",
            left: 0,
            right: 0,
            maxHeight: 260,
            overflowY: "auto",
            zIndex: 20,
            background: "var(--ink-raised)",
            border: "0.5px solid var(--line)",
            borderRadius: 8,
            boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
          }}
        >
          {filtered.length === 0 && (
            <div
              style={{
                padding: "10px 12px",
                fontSize: 12,
                color: "rgba(245,243,236,0.3)",
              }}
            >
              No teams match &quot;{query}&quot;
            </div>
          )}
          {filtered.map((team) => {
            const isSelected = selected.includes(team);
            return (
              <div
                key={team}
                role="button"
                tabIndex={0}
                onClick={() => toggle(team)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") toggle(team);
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "8px 12px",
                  cursor: "pointer",
                  fontSize: 13,
                  color: isSelected ? "var(--turf)" : "var(--chalk)",
                  background: isSelected ? "rgba(31,111,74,0.12)" : "transparent",
                }}
              >
                <span
                  style={{
                    width: 3,
                    height: 14,
                    borderRadius: 2,
                    flexShrink: 0,
                    background: getKitColor(team),
                  }}
                />
                <i className={getFlagClass(team)} />
                {team}
                {isSelected && (
                  <span style={{ marginLeft: "auto", fontSize: 11 }}>✓</span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
