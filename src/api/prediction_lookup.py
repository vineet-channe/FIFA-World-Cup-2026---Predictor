"""
Recover pre-match model probabilities for completed fixtures.

Live pipeline runs overwrite completed matches with scores only. This module
rebuilds an index from archived history + milestone snapshots so the API can
show what the model actually predicted before each match was played.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings

PREDICTION_FIELDS = (
    "p_team_a_win",
    "p_draw",
    "p_team_b_win",
    "expected_score_a",
    "expected_score_b",
)

_index_by_id: dict[str, dict] | None = None
_index_by_pair: dict[tuple[str, str], dict] | None = None


def _extract_prediction(mp: dict) -> dict | None:
    if mp.get("p_team_a_win") is None:
        return None
    return {k: mp[k] for k in PREDICTION_FIELDS if k in mp and mp[k] is not None}


def _scan_sources() -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    by_id: dict[str, dict] = {}
    by_pair: dict[tuple[str, str], dict] = {}

    sources: list[Path] = []
    history_dir = settings.DATA_DIR / "predictions" / "history"
    snapshot_dir = settings.DATA_DIR / "predictions" / "snapshots"

    if snapshot_dir.exists():
        sources.extend(sorted(snapshot_dir.glob("*.json")))
    if history_dir.exists():
        sources.extend(sorted(history_dir.glob("*simulation.json")))

    for path in sources:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for mid, mp in data.get("match_predictions", {}).items():
            pred = _extract_prediction(mp)
            if not pred:
                continue

            ta = mp.get("team_a", "")
            tb = mp.get("team_b", "")
            if not ta or not tb:
                continue

            # Later archives overwrite earlier ones — keep the newest forecast.
            by_id[str(mid)] = pred
            by_pair[tuple(sorted((ta, tb)))] = pred

    return by_id, by_pair


def _ensure_index() -> None:
    global _index_by_id, _index_by_pair
    if _index_by_id is None:
        _index_by_id, _index_by_pair = _scan_sources()


def invalidate_prediction_index() -> None:
    global _index_by_id, _index_by_pair
    _index_by_id = None
    _index_by_pair = None


def lookup_pre_match_prediction(
    match_id: str,
    team_a: str,
    team_b: str,
) -> dict | None:
    """Return saved pre-match probabilities for a fixture, if archived."""
    _ensure_index()
    assert _index_by_id is not None and _index_by_pair is not None

    pred = _index_by_id.get(str(match_id))
    if pred:
        return pred

    return _index_by_pair.get(tuple(sorted((team_a, team_b))))


def enrich_completed_match(
    match_id: str,
    team_a: str,
    team_b: str,
    entry: dict,
) -> dict:
    """
    Attach archived pre-match probabilities to a completed match row when the
    live simulation JSON no longer carries them.
    """
    if entry.get("p_team_a_win") is not None:
        entry["has_pre_match_prediction"] = True
        return entry

    archived = lookup_pre_match_prediction(match_id, team_a, team_b)
    if archived:
        entry.update(archived)
        entry["has_pre_match_prediction"] = True
    else:
        entry["has_pre_match_prediction"] = False

    return entry
