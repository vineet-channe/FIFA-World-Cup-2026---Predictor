"""
Manually save a named milestone snapshot from the CURRENT live simulation
state. Use this if you suspect auto-save has not triggered for a stage that
should already qualify (e.g. all 8 Round of 16 matches are showing as played
in tournament_simulation.json but data/predictions/snapshots/post_r16.json
does not exist).

This does NOT re-run the pipeline or re-simulate anything — it archives
whatever is currently in tournament_simulation.json under the given stage
label. Run 'make pipeline' first if you want the live data refreshed before
saving.

Usage:
    python scripts/save_snapshot.py --stage post_r16
    python scripts/save_snapshot.py --stage post_r16 --overwrite
    python scripts/save_snapshot.py --check   # show progress toward all thresholds, save nothing
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.retraining.snapshots import (
    save_snapshot, STAGE_LABELS, STAGE_COMPLETION, _count_completed,
)


def load_live() -> dict:
    path = settings.DATA_DIR / "predictions" / "tournament_simulation.json"
    with open(path) as f:
        return json.load(f)


def show_progress() -> None:
    """Print how many real completed matches exist per round vs the
    threshold required to auto-save each milestone. Saves nothing."""
    sim = load_live()
    actual = sim.get("actual_results", {})

    print("Auto-save progress toward each milestone:")
    print("-" * 55)
    for stage, cfg in STAGE_COMPLETION.items():
        n = _count_completed(actual, cfg["round_prefix"])
        required = cfg["required"]
        existing_path = settings.DATA_DIR / "predictions" / "snapshots" / f"{stage}.json"
        already_saved = existing_path.exists()
        status = "SAVED" if already_saved else ("READY — should auto-save on next pipeline run" if n >= required else "waiting")
        print(f"  {stage:<20} {n:>3} / {required:<3} completed  [{cfg['round_prefix']}]   {status}")
    print("-" * 55)
    print("If a stage shows 'READY' but is not SAVED after the next 'make pipeline' run,")
    print("the round-name string chain likely has a mismatch — check ingestion.py's")
    print("STAGE_MAP output against what actually appears in tournament_simulation.json's")
    print("actual_results[...]['round'] field for that round.")


def main(stage: str | None, overwrite: bool, check: bool) -> None:
    if check:
        show_progress()
        return

    if stage is None:
        print("Provide --stage <name> or use --check to see progress.")
        print(f"Valid stages: {STAGE_LABELS}")
        sys.exit(1)

    if stage not in STAGE_LABELS:
        print(f"Unknown stage '{stage}'. Valid stages: {STAGE_LABELS}")
        sys.exit(1)

    sim = load_live()
    actual = sim.get("actual_results", {})

    # Sanity check before saving — warn if the threshold for this stage
    # doesn't actually look met yet, but don't block (user may have a reason)
    if stage in STAGE_COMPLETION:
        cfg = STAGE_COMPLETION[stage]
        n = _count_completed(actual, cfg["round_prefix"])
        if n < cfg["required"]:
            print(
                f"WARNING: only {n}/{cfg['required']} '{cfg['round_prefix']}' "
                f"matches are marked played in the current live simulation. "
                f"Saving anyway since you asked explicitly."
            )

    path = save_snapshot(
        stage=stage,
        simulation=sim,
        model_version=sim.get("metadata", {}).get("model", "unknown"),
        overwrite=overwrite,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Show auto-save progress toward every milestone, save nothing")
    args = parser.parse_args()
    main(args.stage, args.overwrite, args.check)
