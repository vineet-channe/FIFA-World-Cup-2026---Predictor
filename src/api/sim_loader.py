"""Load tournament_simulation.json with automatic reload when the file changes."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings

_sim_cache: dict | None = None
_sim_mtime: float | None = None

SIM_PATH = settings.DATA_DIR / "predictions" / "tournament_simulation.json"


def load_simulation(*, required: bool = True) -> dict:
    """
    Return tournament_simulation.json, reloading from disk when the file
    is updated (e.g. after `make pipeline`).
    """
    global _sim_cache, _sim_mtime

    if not SIM_PATH.exists():
        if required:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=503,
                detail="Simulation results not found. Run Phase 4 first.",
            )
        return {}

    mtime = SIM_PATH.stat().st_mtime
    if _sim_cache is None or _sim_mtime != mtime:
        with open(SIM_PATH) as f:
            _sim_cache = json.load(f)
        _sim_mtime = mtime
        from src.api.prediction_lookup import invalidate_prediction_index

        invalidate_prediction_index()

    return _sim_cache


def invalidate_simulation_cache() -> None:
    """Force the next load to re-read tournament_simulation.json from disk."""
    global _sim_cache, _sim_mtime
    _sim_cache = None
    _sim_mtime = None
