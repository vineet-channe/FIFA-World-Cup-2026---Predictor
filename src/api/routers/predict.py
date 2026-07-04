"""Predict router — live matchup prediction and H2H history."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from config.settings import settings
from config.team_names import WC2026_CANONICAL_TEAMS
from src.api.schemas import H2HMatch, PredictRequest, PredictResponse
from src.api.sim_loader import load_simulation

router = APIRouter()

_matches_df:  pd.DataFrame | None = None


def _load_matches() -> pd.DataFrame:
    global _matches_df
    if _matches_df is None:
        _matches_df = pd.read_parquet(settings.DATA_DIR / "processed" / "matches_clean.parquet")
        _matches_df["date"] = pd.to_datetime(_matches_df["date"])
    return _matches_df


def _load_sim() -> dict:
    return load_simulation(required=False)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict outcome probabilities for any pair of WC 2026 teams.

    First checks the pre-computed simulation cache; falls back to live
    ensemble inference for matchups not in the cache.
    """
    team_a, team_b = req.team_a, req.team_b

    if team_a not in WC2026_CANONICAL_TEAMS:
        raise HTTPException(status_code=400, detail=f"Unknown team: {team_a}")
    if team_b not in WC2026_CANONICAL_TEAMS:
        raise HTTPException(status_code=400, detail=f"Unknown team: {team_b}")
    if team_a == team_b:
        raise HTTPException(status_code=400, detail="team_a and team_b must differ")

    # Check pre-computed cache first
    cached = _find_cached(team_a, team_b)
    if cached:
        return cached

    # Live inference
    try:
        from src.api.predict_service import predict_matchup
        return predict_matchup(team_a, team_b)
    except Exception as exc:
        logger.error(f"Live prediction failed for {team_a} vs {team_b}: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@router.get("/h2h", response_model=list[H2HMatch])
def h2h(
    team_a: str = Query(..., description="First team canonical name"),
    team_b: str = Query(..., description="Second team canonical name"),
) -> list[H2HMatch]:
    """Return the last 5 head-to-head meetings between two teams."""
    m = _load_matches()
    mask = (
        ((m["home_team"] == team_a) & (m["away_team"] == team_b))
        | ((m["home_team"] == team_b) & (m["away_team"] == team_a))
    )
    h2h = m[mask].sort_values("date", ascending=False).head(5)

    results = []
    for _, row in h2h.iterrows():
        is_normal = row["home_team"] == team_a
        hs  = _safe_int(row.get("home_score"))
        as_ = _safe_int(row.get("away_score"))
        score_a = hs if is_normal else as_
        score_b = as_ if is_normal else hs
        results.append(
            H2HMatch(
                date=str(row["date"])[:10],
                competition=str(row.get("tournament", "")),
                score_a=score_a,
                score_b=score_b,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(v: object) -> int:
    """Convert a value that may be float, NaN, or None to int safely."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return 0 if (f != f) else int(f)  # f != f is True only for NaN
    except (TypeError, ValueError):
        return 0


def _find_cached(team_a: str, team_b: str) -> PredictResponse | None:
    """Return a PredictResponse from the simulation cache, or None."""
    try:
        sim = _load_sim()
        match_preds = sim.get("match_predictions", {})
        for mp in match_preds.values():
            if mp.get("status") == "completed":
                continue
            if "p_team_a_win" not in mp:
                continue
            pa, pb = mp.get("team_a"), mp.get("team_b")
            if pa == team_a and pb == team_b:
                return _mp_to_response(mp, flipped=False)
            if pa == team_b and pb == team_a:
                return _mp_to_response(mp, flipped=True)
    except Exception:
        pass
    return None


def _mp_to_response(mp: dict, flipped: bool) -> PredictResponse:
    p_a = mp.get("p_team_a_win", 0.333)
    p_d = mp.get("p_draw", 0.333)
    p_b = mp.get("p_team_b_win", 0.333)
    sc_a = mp.get("expected_score_a", 1.0)
    sc_b = mp.get("expected_score_b", 1.0)
    if flipped:
        p_a, p_b = p_b, p_a
        sc_a, sc_b = sc_b, sc_a
    return PredictResponse(
        p_team_a_win=round(p_a, 4),
        p_draw=round(p_d, 4),
        p_team_b_win=round(p_b, 4),
        expected_score_a=round(sc_a, 2),
        expected_score_b=round(sc_b, 2),
        top_features=[],
    )
