"""
Rebuild the post_group_stage snapshot from real API data.

The existing post_group_stage.json was generated from fixture_fallback
synthetic data before football-data.org was wired up. It has two problems:
  1. Group results are synthetic (England and Argentina shown as eliminated)
  2. p_champion = 0 for all teams (simulation counter bug)

This script fixes it:
  - Pulls real group stage results from football-data.org (72 matches)
  - Resets R32 bracket to all unplayed (simulates end-of-groups state)
  - Runs 10,000 Monte Carlo simulations from that boundary
  - Overwrites the broken snapshot with correct data

The snapshot is marked "reconstructed: true" — honest about being rebuilt
after the fact rather than captured live. The model used is the current
LightGBM (trained on ~90 real matches) not a point-in-time model, which
means there is minor future leakage in the probabilities. For a history
display chart this is acceptable — far better than broken synthetic data.

Usage:
    source worldcup/bin/activate
    python scripts/rebuild_post_group_stage.py --dry-run   # verify first
    python scripts/rebuild_post_group_stage.py              # run and save
    python scripts/rebuild_post_group_stage.py --n 5000    # faster test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.retraining.ingestion import get_all_fixtures, get_standings, get_bracket
from src.retraining.live_simulation import build_tournament_state, run_live_simulation
from src.retraining.pipeline import LiveRetrainPipeline
from src.retraining.snapshots import save_snapshot


def main(dry_run: bool = False, n_sim: int = 10_000) -> None:

    # ── 1. Pull real fixtures ────────────────────────────────────────────────
    logger.info("Pulling all WC 2026 fixtures from football-data.org...")
    all_fixtures = get_all_fixtures()

    group_fixtures = [
        f for f in all_fixtures
        if f.get("round", "").startswith("Group Stage") and f["status"] == "FT"
    ]
    logger.info(f"Completed group stage fixtures: {len(group_fixtures)} (expect 72)")

    if len(group_fixtures) < 72:
        logger.error(
            f"Only {len(group_fixtures)} group stage fixtures found. "
            "Check FOOTBALL_DATA_API_KEY and that football-data.org is returning "
            "WC 2026 group stage data."
        )
        sys.exit(1)

    # ── 2. Pull real standings and R32 bracket ───────────────────────────────
    logger.info("Pulling real group standings...")
    standings = get_standings()
    logger.info(f"Standings for {len(standings)} groups (expect 12)")

    logger.info("Pulling R32 bracket...")
    bracket_raw = get_bracket()
    logger.info(f"R32 bracket entries: {len(bracket_raw)} (expect 16)")

    # ── 3. Reset R32 to unplayed — critical step ─────────────────────────────
    # This is what makes this a group-stage snapshot, not a mid-R32 snapshot.
    # Every R32 match is forced to NS (not started) so the simulation treats
    # R32 as entirely upcoming and produces proper end-of-groups odds.
    bracket_unplayed = [
        {
            **m,
            "status":      "NS",
            "home_score":  None,
            "away_score":  None,
            "home_winner": None,
            "away_winner": None,
        }
        for m in bracket_raw
    ]

    completed_r32 = sum(1 for m in bracket_raw if m["status"] == "FT")
    logger.info(
        f"R32 bracket: {completed_r32} already completed in reality, "
        f"all {len(bracket_unplayed)} reset to NS for this snapshot"
    )

    # ── 4. Build actual_results from group stage only ────────────────────────
    actual_at_groups = {
        str(f["fixture_id"]): {
            "team_a":  f["home_team"],
            "team_b":  f["away_team"],
            "score_a": f["home_score"],
            "score_b": f["away_score"],
            "round":   f["round"],
            "played":  True,
            "winner": (
                f["home_team"] if f.get("home_winner")
                else f["away_team"] if f.get("away_winner")
                else None
            ),
        }
        for f in group_fixtures
        if f["home_score"] is not None
    }
    logger.info(
        f"actual_results: {len(actual_at_groups)} group matches "
        f"(no R32 results included)"
    )

    # ── 5. Build tournament state ─────────────────────────────────────────────
    # Pass group_fixtures as the fixture list so build_tournament_state
    # correctly determines group_stage_complete=True.
    # Pass bracket_unplayed so no R32 teams are marked as eliminated.
    state = build_tournament_state(group_fixtures, standings, bracket_unplayed)
    state["current_round"] = "Group Stage"

    r32_teams = {t for m in state["r32_matches"] for t in (m["team_a"], m["team_b"])}
    tbd_slots = sum(1 for t in r32_teams if t in ("TBD", ""))
    logger.info(
        f"Tournament state: "
        f"group_stage_complete={state['group_stage_complete']} | "
        f"r32_matches={len(state['r32_matches'])} | "
        f"eliminated_teams={len(state['eliminated_teams'])} | "
        f"TBD bracket slots={tbd_slots}"
    )

    if tbd_slots > 4:
        logger.warning(
            f"{tbd_slots} R32 bracket slots are TBD — standings may be incomplete. "
            "Continuing, but check the group standings API response."
        )

    if dry_run:
        logger.info("")
        logger.info("DRY RUN — stopping before simulation. Re-run without --dry-run to save.")
        logger.info(f"Would simulate {n_sim:,} runs from group-stage boundary:")
        logger.info(f"  Group matches in actual_results: {len(actual_at_groups)}")
        logger.info(f"  R32 matches (all unplayed):      {len(bracket_unplayed)}")
        logger.info(f"  Group standings loaded:          {len(standings)} groups")
        logger.info(f"  TBD bracket slots:               {tbd_slots}")
        return

    # ── 6. Load models ───────────────────────────────────────────────────────
    logger.info("Loading models (ensemble + LightGBM + Dixon-Coles)...")
    pipeline = LiveRetrainPipeline()

    elo_df = pd.read_parquet(settings.DATA_DIR / "processed" / "elo_clean.parquet")
    elo_df["date"] = pd.to_datetime(elo_df["date"])

    # Blend weight: 72 real group matches / 100, capped at 0.75
    # Use 72 (group stage count) — honest for this snapshot's moment in time
    blend_weight = min(len(actual_at_groups) / 100, 0.75)
    logger.info(
        f"Blend weight: {blend_weight:.0%} LightGBM / {1 - blend_weight:.0%} ensemble"
    )

    # ── 7. Simulate from group-stage boundary ────────────────────────────────
    logger.info(
        f"Running {n_sim:,} simulations from group-stage boundary..."
    )

    group_output = run_live_simulation(
        state,
        pipeline.ensemble,
        pipeline.dc_model,
        elo_df,
        n_sim=n_sim,
        actual_results=actual_at_groups,
        lgbm_model=pipeline.current_lgbm,
        lgbm_blend_weight=blend_weight,
    )

    # ── 8. Verify before saving ───────────────────────────────────────────────
    tp = group_output.get("team_probabilities", {})
    champ_sum = sum(v.get("p_champion", 0) for v in tp.values())
    alive = sum(1 for v in tp.values() if v.get("p_champion", 0) > 0)

    logger.info(
        f"Simulation complete: "
        f"sum(p_champion)={champ_sum:.4f} | "
        f"alive teams (p_champion>0)={alive}"
    )

    # Hard stops — if either fails, something is broken in the simulation
    if abs(champ_sum - 1.0) > 0.02:
        logger.error(
            f"sum(p_champion) = {champ_sum:.4f}, expected ~1.0. "
            "Simulation has a champion-counter bug. NOT saving snapshot."
        )
        sys.exit(1)

    if alive < 16:
        logger.error(
            f"Only {alive} teams have p_champion > 0, expected at least 16. "
            "NOT saving snapshot."
        )
        sys.exit(1)

    # ── 9. Add reconstruction metadata and save ───────────────────────────────
    group_output.setdefault("metadata", {})
    group_output["metadata"]["reconstructed"] = True
    group_output["metadata"]["reconstruction_note"] = (
        "Rebuilt from real group-stage API data. "
        "Original snapshot was synthetic (fixture_fallback). "
        "Model: current LightGBM (minor future leakage — acceptable for display). "
        "Simulation state: 72 real group results, R32 unplayed."
    )
    group_output["metadata"]["tournament_phase"] = "Group Stage"

    path = save_snapshot(
        stage="post_group_stage",
        simulation=group_output,
        model_version=Path(str(pipeline.current_lgbm_path)).name,
        overwrite=True,         # ← replaces the broken synthetic snapshot
    )

    # ── 10. Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("post_group_stage snapshot rebuilt successfully")
    logger.info(f"Saved: {path}")
    logger.info("")
    logger.info("Top 10 championship odds (post group stage):")
    for team, v in sorted(tp.items(), key=lambda x: -x[1].get("p_champion", 0))[:10]:
        logger.info(f"  {team:<25} {v['p_champion']:.1%}")
    logger.info("")
    logger.info("Verify the fix:")
    logger.info("  python scripts/verify_snapshots.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild post_group_stage snapshot from real API data"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check data without running simulation or saving"
    )
    parser.add_argument(
        "--n", type=int, default=10_000,
        help="Number of Monte Carlo simulations (default 10,000)"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, n_sim=args.n)