"""
Saves labelled, immutable copies of tournament predictions at each major
tournament milestone and archives every pipeline run to the history directory.
"""
import json
import pandas as pd
from pathlib import Path
from loguru import logger
from config.settings import settings

SNAPSHOT_DIR = settings.DATA_DIR / "predictions" / "snapshots"
HISTORY_DIR = settings.DATA_DIR / "predictions" / "history"
LIVE_PATH = settings.DATA_DIR / "predictions" / "tournament_simulation.json"

STAGE_LABELS = [
    "pre_tournament",
    "post_group_stage",
    "post_r32",
    "post_r16",
    "post_qf",
    "post_sf",
]

STAGE_COMPLETION = {
    "post_group_stage": {"round_prefix": "Group Stage", "required": 72},
    "post_r32": {"round_prefix": "Round of 32", "required": 16},
    "post_r16": {"round_prefix": "Round of 16", "required": 8},
    "post_qf": {"round_prefix": "Quarter-finals", "required": 4},
    "post_sf": {"round_prefix": "Semi-finals", "required": 2},
}


def _ensure_dirs():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _count_completed(actual_results: dict, round_prefix: str) -> int:
    return sum(
        1
        for v in actual_results.values()
        if v.get("round", "").startswith(round_prefix) and v.get("played", False)
    )


def _snapshot_meta(simulation: dict, stage: str, model_version: str) -> dict:
    actual = simulation.get("actual_results", {})
    preds = simulation.get("match_predictions", {})
    teams_remaining = [
        t
        for t, v in simulation.get("team_probabilities", {}).items()
        if v.get("p_champion", 0) > 0
    ]
    completed_by_round: dict[str, int] = {}
    for v in actual.values():
        r = v.get("round", "Unknown")
        completed_by_round[r] = completed_by_round.get(r, 0) + 1
    return {
        "stage": stage,
        "saved_at": pd.Timestamp.now().isoformat(),
        "model_version": model_version,
        "n_simulations": simulation.get("metadata", {}).get("n_simulations", 0),
        "data_as_of": simulation.get("metadata", {}).get("data_as_of", ""),
        "matches_completed": len(actual),
        "matches_remaining": sum(1 for v in preds.values() if v.get("status") == "predicted"),
        "teams_remaining": len(teams_remaining),
        "completed_by_round": completed_by_round,
    }


def save_snapshot(
    stage: str,
    simulation: dict | None = None,
    model_version: str = "ensemble_v1",
    overwrite: bool = False,
) -> Path:
    """Save a named milestone snapshot. Will not overwrite unless overwrite=True."""
    _ensure_dirs()
    assert stage in STAGE_LABELS, f"Unknown stage: {stage}"
    dest = SNAPSHOT_DIR / f"{stage}.json"
    if dest.exists() and not overwrite:
        logger.info(f"Snapshot already exists: {dest} — skipping")
        return dest
    if simulation is None:
        with open(LIVE_PATH) as f:
            simulation = json.load(f)
    meta = _snapshot_meta(simulation, stage, model_version)
    with open(dest, "w") as f:
        json.dump({"snapshot_metadata": meta, **simulation}, f, indent=2)
    logger.info(
        f"Snapshot saved: {dest} | stage={stage} | teams_remaining={meta['teams_remaining']}"
    )
    return dest


def save_history_entry(simulation: dict, model_version: str = "ensemble_v1") -> Path:
    """Save a timestamped archive entry for every pipeline run."""
    _ensure_dirs()
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    stage = detect_current_stage(simulation.get("actual_results", {}))
    dest = HISTORY_DIR / f"{ts}_{stage}.json"
    meta = _snapshot_meta(simulation, stage, model_version)
    with open(dest, "w") as f:
        json.dump({"snapshot_metadata": meta, **simulation}, f, indent=2)
    logger.info(f"History entry saved: {dest}")
    return dest


def detect_current_stage(actual_results: dict) -> str:
    """Detect the furthest completed stage based on actual results."""
    for stage, cfg in reversed(list(STAGE_COMPLETION.items())):
        if _count_completed(actual_results, cfg["round_prefix"]) >= cfg["required"]:
            return stage
    return "pre_tournament"


def should_save_named_snapshot(simulation: dict, existing_stages: set[str]) -> str | None:
    """Return the highest newly crossed milestone stage label, else None."""
    actual = simulation.get("actual_results", {})
    crossed: str | None = None
    for stage, cfg in STAGE_COMPLETION.items():
        if stage in existing_stages:
            continue
        if _count_completed(actual, cfg["round_prefix"]) >= cfg["required"]:
            crossed = stage
    return crossed


def list_snapshots() -> list[dict]:
    """Return metadata for all named snapshots + history entries, chronologically."""
    _ensure_dirs()
    result = []
    for stage in STAGE_LABELS:
        path = SNAPSHOT_DIR / f"{stage}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            result.append(
                {
                    "type": "milestone",
                    "stage": stage,
                    "path": str(path),
                    **data.get("snapshot_metadata", {}),
                }
            )
    for path in sorted(HISTORY_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        result.append(
            {"type": "history", "path": str(path), **data.get("snapshot_metadata", {})}
        )
    return sorted(result, key=lambda x: x.get("saved_at", ""))


def load_snapshot(stage: str) -> dict:
    path = SNAPSHOT_DIR / f"{stage}.json"
    if not path.exists():
        raise FileNotFoundError(f"No snapshot for stage: {stage}")
    with open(path) as f:
        return json.load(f)


def get_probability_history(team: str) -> list[dict]:
    """Championship + knockout odds for one team across all milestone snapshots."""
    result = []
    for snap in list_snapshots():
        if snap["type"] != "milestone":
            continue
        try:
            with open(snap["path"]) as f:
                data = json.load(f)
            tp = data.get("team_probabilities", {}).get(team)
            if tp:
                result.append(
                    {
                        "stage": snap["stage"],
                        "saved_at": snap.get("saved_at"),
                        "p_champion": tp.get("p_champion", 0),
                        "p_final": tp.get("p_final", 0),
                        "p_semi": tp.get("p_semi", 0),
                        "p_quarter": tp.get("p_quarter", 0),
                        "eliminated": tp.get("p_champion", 0) == 0,
                        "eliminated_in": tp.get("eliminated_in"),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not read {snap['path']}: {e}")
    return result
