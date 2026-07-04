"""Model router — model comparison / performance data."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from config.settings import settings

router = APIRouter()

_comp_cache: dict | None = None


def _load_comparison() -> dict:
    global _comp_cache
    if _comp_cache is None:
        p = settings.DATA_DIR / "processed" / "model_comparison.json"
        if not p.exists():
            raise HTTPException(
                status_code=503,
                detail="model_comparison.json not found. Run Phase 3 first.",
            )
        with open(p) as f:
            _comp_cache = json.load(f)
    return _comp_cache


@router.get("/model-comparison")
def get_model_comparison() -> dict:
    """Full passthrough of model_comparison.json."""
    return _load_comparison()
