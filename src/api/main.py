"""WC 2026 Predictor — FastAPI backend entry point.

Run with:
    uvicorn src.api.main:app --reload --port 8000
or:
    make api
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run via uvicorn from the project root
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routers import model, predict, simulation, story, teams

app = FastAPI(
    title="WC 2026 Predictor API",
    description="Read-only ML prediction API for the FIFA World Cup 2026.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(teams.router,      prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(predict.router,    prefix="/api")
app.include_router(model.router,      prefix="/api")
app.include_router(story.router,      prefix="/api")


def _find_latest_lgbm() -> Path | None:
    """Return the most recently modified LightGBM model file, or None."""
    model_dir = _ROOT / "models"
    live_models = sorted(
        model_dir.glob("lightgbm_live_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if live_models:
        return live_models[0]
    base = model_dir / "lightgbm_v1.pkl"
    return base if base.exists() else None


def _compute_blend_weight() -> tuple[int, float]:
    """Count WC 2026 matches in the feature matrix and compute blend weight."""
    try:
        import pandas as pd
        fm_path = _ROOT / "data" / "processed" / "feature_matrix.parquet"
        if not fm_path.exists():
            return 0, 0.0
        fm = pd.read_parquet(fm_path, columns=["tournament", "match_date"])
        fm["match_date"] = pd.to_datetime(fm["match_date"])
        n = int(
            (
                fm["tournament"].str.contains("FIFA World Cup", na=False)
                & (fm["match_date"].dt.year == 2026)
            ).sum()
        )
        return n, min(n / 100, 0.75)
    except Exception as exc:
        logger.warning(f"Could not compute blend weight: {exc}")
        return 0, 0.0


@app.on_event("startup")
def load_models() -> None:
    """Load the stacking ensemble and Dixon-Coles model once at startup."""
    from src.models.dixon_coles import DixonColesModel
    from src.models.ensemble import load_ensemble

    app.state.ensemble = None
    app.state.dc_model = None

    ensemble_path = _ROOT / "models" / "ensemble_v1.pkl"
    dc_path = _ROOT / "models" / "dixon_coles_v1.json"

    if ensemble_path.exists():
        try:
            app.state.ensemble = load_ensemble(str(ensemble_path))
            logger.info("Ensemble model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load ensemble: {exc}")
    else:
        logger.warning(f"Ensemble model not found at {ensemble_path}. Live prediction disabled.")

    if dc_path.exists():
        try:
            app.state.dc_model = DixonColesModel.load(str(dc_path))
            logger.info("Dixon-Coles model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load Dixon-Coles: {exc}")
    else:
        logger.warning(f"Dixon-Coles model not found at {dc_path}. Live scorelines disabled.")

    # LightGBM — retrained after each matchday
    app.state.lgbm              = None
    app.state.lgbm_blend_weight = 0.0
    app.state.lgbm_n_wc_matches = 0

    lgbm_path = _find_latest_lgbm()
    if lgbm_path and lgbm_path.exists():
        try:
            import joblib
            app.state.lgbm = joblib.load(str(lgbm_path))
            n_wc, weight   = _compute_blend_weight()
            app.state.lgbm_n_wc_matches = n_wc
            app.state.lgbm_blend_weight = weight
            logger.info(
                f"LightGBM loaded: {lgbm_path.name} | "
                f"WC 2026 matches: {n_wc} | blend: {weight:.0%}"
            )
        except Exception as exc:
            logger.error(f"Failed to load LightGBM: {exc} — ensemble only")
    else:
        logger.info("No LightGBM model found — predictions use ensemble only")


@app.post("/api/reload-models")
def reload_models_endpoint() -> dict:
    """
    Reload all ML models from disk and clear the prediction cache.
    Called automatically by the pipeline after each successful retrain.
    """
    from src.api.predict_service import predict_matchup
    load_models()
    predict_matchup.cache_clear()
    logger.info("Models reloaded and prediction cache cleared")
    return {
        "status":              "reloaded",
        "lgbm_blend_weight":   app.state.lgbm_blend_weight,
        "lgbm_n_wc_matches":   app.state.lgbm_n_wc_matches,
        "lgbm_loaded":         app.state.lgbm is not None,
    }


@app.get("/api/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}
