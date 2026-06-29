"""Standalone audit script — checks team name consistency across all datasets.

Loads processed parquets plus raw schedule and transfermarkt data, then
verifies canonical-name compatibility and WC 2026 coverage.

Usage:
    python scripts/02_audit_names.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from config.team_names import (
    WC2026_CANONICAL_TEAMS,
    WC2026_GROUPS,
    audit_mismatches,
    normalise,
)


def _schedule_real_teams(schedule: list[dict]) -> set[str]:
    """Return distinct team names from group-stage fixtures (excludes TBD/bracket)."""
    teams: set[str] = set()
    for match in schedule:
        if match.get("round") != "Group Stage":
            continue
        for col in ("team_a", "team_b"):
            name = match.get(col, "")
            if name and name != "TBD":
                teams.add(name)
    return teams


def main() -> None:
    processed = settings.DATA_DIR / "processed"
    raw = settings.DATA_DIR / "raw"
    issues = 0

    # ── Load datasets ──────────────────────────────────────────────────────
    def _load(name: str) -> pd.DataFrame | None:
        path = processed / name
        if not path.exists():
            print(f"[WARN] {path} not found — run 01_collect_all_data.py first")
            return None
        df = pd.read_parquet(path)
        print(f"Loaded {name}: {len(df):,} rows")
        return df

    matches  = _load("matches_clean.parquet")
    elo      = _load("elo_clean.parquet")
    rankings = _load("rankings_clean.parquet")

    if matches is None or elo is None or rankings is None:
        print("\nOne or more required files missing — cannot run full audit.")
        return

    print()

    # ── Cross-source audit (historical datasets) ───────────────────────────
    mismatches = audit_mismatches(matches, rankings, elo)

    # ── WC 2026 nations coverage (processed) ───────────────────────────────
    print("\n=== WC 2026 nations coverage (processed) ===")
    elo_teams = set(elo["team"].dropna().unique())
    missing_from_elo = [t for t in WC2026_CANONICAL_TEAMS if t not in elo_teams]
    if missing_from_elo:
        issues += len(missing_from_elo)
        print(f"WC 2026 teams missing from elo_clean ({len(missing_from_elo)}):")
        for t in missing_from_elo:
            print(f"  {t!r}")
    else:
        print("All 48 WC 2026 nations present in elo_clean ✓")

    ranking_teams = set(rankings["country_full"].dropna().unique())
    missing_from_rankings = [t for t in WC2026_CANONICAL_TEAMS if t not in ranking_teams]
    if missing_from_rankings:
        issues += len(missing_from_rankings)
        print(f"\nWC 2026 teams missing from rankings_clean ({len(missing_from_rankings)}):")
        for t in missing_from_rankings:
            print(f"  {t!r}")
    else:
        print("All 48 WC 2026 nations present in rankings_clean ✓")

    canonical = set(WC2026_CANONICAL_TEAMS)

    # ── Transfermarkt (raw) ────────────────────────────────────────────────
    print("\n=== Transfermarkt vs canonical ===")
    tm_path = raw / "transfermarkt" / "squad_values.parquet"
    if tm_path.exists():
        tm = pd.read_parquet(tm_path)
        tm_teams = set(tm["team"].dropna().unique())
        print(f"Loaded squad_values.parquet: {len(tm):,} rows")

        missing_tm = sorted(canonical - tm_teams)
        extra_tm = sorted(tm_teams - canonical)
        norm_changes = {t: normalise(t) for t in tm_teams if normalise(t) != t}

        if missing_tm:
            issues += len(missing_tm)
            print(f"WC 2026 teams missing from transfermarkt: {missing_tm}")
        else:
            print("All 48 WC 2026 nations present in transfermarkt ✓")

        if extra_tm:
            issues += len(extra_tm)
            print(f"Non-WC teams in transfermarkt: {extra_tm}")

        if norm_changes:
            issues += len(norm_changes)
            print(f"Names that would change under normalise(): {norm_changes}")
        else:
            print("All transfermarkt team names already canonical ✓")
    else:
        issues += 1
        print(f"[WARN] {tm_path} not found")

    # ── Schedule (raw) ─────────────────────────────────────────────────────
    print("\n=== Schedule vs canonical ===")
    sched_path = raw / "schedule" / "wc2026_schedule.json"
    if sched_path.exists():
        with open(sched_path, encoding="utf-8") as f:
            schedule = json.load(f)
        sched_teams = _schedule_real_teams(schedule)
        print(f"Loaded wc2026_schedule.json: {len(schedule)} matches, "
              f"{len(sched_teams)} group-stage teams")

        not_canonical = sorted(sched_teams - canonical)
        missing_sched = sorted(canonical - sched_teams)
        norm_changes = {t: normalise(t) for t in sched_teams if normalise(t) != t}

        if not_canonical:
            issues += len(not_canonical)
            print(f"Schedule teams not in WC2026_CANONICAL_TEAMS: {not_canonical}")
        else:
            print("All schedule group-stage teams are canonical ✓")

        if missing_sched:
            issues += len(missing_sched)
            print(f"WC 2026 teams missing from schedule: {missing_sched}")
        else:
            print("All 48 WC 2026 nations present in schedule ✓")

        if norm_changes:
            issues += len(norm_changes)
            print(f"Names that would change under normalise(): {norm_changes}")
        else:
            print("All schedule team names already canonical ✓")

        # Group composition check
        from collections import defaultdict
        sched_groups: dict[str, set[str]] = defaultdict(set)
        for match in schedule:
            group = match.get("group")
            if not group or match.get("round") != "Group Stage":
                continue
            sched_groups[group].add(match["team_a"])
            sched_groups[group].add(match["team_b"])

        group_issues = 0
        for group, teams in WC2026_GROUPS.items():
            if set(teams) != sched_groups.get(group, set()):
                group_issues += 1
                print(f"Group {group} mismatch — config: {teams}, schedule: "
                      f"{sorted(sched_groups.get(group, set()))}")

        if group_issues:
            issues += group_issues
        else:
            print("All 12 groups match WC2026_GROUPS ✓")
    else:
        issues += 1
        print(f"[WARN] {sched_path} not found")

    # ── Cross-dataset join check (schedule → all sources) ──────────────────
    print("\n=== Join compatibility (schedule → all datasets) ===")
    if sched_path.exists() and tm_path.exists():
        match_teams = (
            set(matches["home_team"].dropna().unique())
            | set(matches["away_team"].dropna().unique())
        )
        sources = {
            "elo_clean": elo_teams,
            "rankings_clean": ranking_teams,
            "matches_clean": match_teams,
            "transfermarkt": tm_teams,
        }
        join_ok = True
        for label, teams in sources.items():
            missing = sorted(sched_teams - teams)
            if missing:
                join_ok = False
                issues += len(missing)
                print(f"Schedule teams missing from {label}: {missing}")
            else:
                print(f"Schedule → {label}: all 48 teams found ✓")

        if join_ok:
            print("Schedule is join-compatible with all datasets ✓")

    # ── Final verdict ──────────────────────────────────────────────────────
    historical_issues = len(mismatches["not_in_rankings"]) + len(mismatches["not_in_elo"])
    if issues == 0 and historical_issues == 0:
        print("\n✅ No mismatches found — all team names are consistent.")
    else:
        wc_issues = issues
        total = historical_issues + wc_issues
        print(f"\n⚠️  {total} WC 2026 / join issues found "
              f"({historical_issues} historical cross-source, {wc_issues} WC 2026 specific)")
        if wc_issues:
            print("   Fix schedule/transfermarkt IDs or aliases in config/team_names.py")
        if historical_issues:
            print("   Historical results contain non-FIFA teams — expected, not blocking Phase 2")


if __name__ == "__main__":
    main()
