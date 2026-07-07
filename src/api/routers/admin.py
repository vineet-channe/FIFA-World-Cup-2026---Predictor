"""
Admin endpoints for manually triggering the live retrain pipeline.
Protected by a shared secret token so it cannot be triggered by the public.
"""
import os

from fastapi import APIRouter, HTTPException, Header
from loguru import logger

router = APIRouter()


def _check_admin_token(x_admin_token: str | None) -> None:
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected:
        # No token configured — refuse all admin requests rather than
        # silently allowing them. This forces the token to be set explicitly
        # in production before this endpoint can be used at all.
        raise HTTPException(503, "Admin endpoint not configured (ADMIN_TOKEN unset)")
    if x_admin_token != expected:
        raise HTTPException(403, "Invalid admin token")


@router.post("/admin/run-pipeline")
def run_pipeline_now(x_admin_token: str | None = Header(default=None)):
    """
    Manually trigger a full pipeline run (ingest + retrain + re-simulate).
    Requires the X-Admin-Token header to match the ADMIN_TOKEN env var.

    Usage:
        curl -X POST https://your-backend-url/api/admin/run-pipeline \
             -H "X-Admin-Token: your_secret_token"
    """
    _check_admin_token(x_admin_token)
    try:
        from src.retraining.pipeline import LiveRetrainPipeline
        pipeline = LiveRetrainPipeline()
        output = pipeline.run()
        n_results = len(output.get("actual_results", {}))
        logger.info(f"Manual pipeline run complete via /api/admin/run-pipeline — {n_results} results")
        return {
            "status": "ok",
            "matches_ingested": n_results,
            "current_round": output.get("metadata", {}).get("tournament_phase"),
        }
    except Exception as exc:
        logger.error(f"Manual pipeline run failed: {exc}")
        raise HTTPException(500, f"Pipeline run failed: {exc}")


@router.get("/admin/status")
def admin_status(x_admin_token: str | None = Header(default=None)):
    """Quick health/status check for the admin, no side effects."""
    _check_admin_token(x_admin_token)
    from src.retraining.run_status import get_run_status
    return get_run_status()
