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

from src.api.routers import model, predict, simulation, teams

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


@app.get("/api/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}
