"""
Admin endpoints for manually triggering the live retrain pipeline.
Protected by a shared secret token so it cannot be triggered by the public.
"""
import io
import os
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
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


@router.post("/admin/upload-seed")
async def upload_seed_data(
    file: UploadFile = File(...),
    target: str = "data",  # "data" or "models" — which volume subfolder to extract into
    x_admin_token: str | None = Header(default=None),
):
    """
    One-time (or occasional) seeding endpoint: accepts a zip file over HTTPS
    and extracts it into the Railway volume at DATA_DIR or MODEL_DIR.

    This exists specifically as a reliable alternative to `railway volume
    browse` / `railway ssh`, which have a known auth bug on macOS. A plain
    HTTPS upload to your own already-running API has no such dependency.

    Usage:
        curl -X POST https://your-railway-url/api/admin/upload-seed \
             -H "X-Admin-Token: your_token" \
             -F "file=@data.zip" \
             -F "target=data"

        curl -X POST https://your-railway-url/api/admin/upload-seed \
             -H "X-Admin-Token: your_token" \
             -F "file=@models.zip" \
             -F "target=models"
    """
    _check_admin_token(x_admin_token)

    if target not in ("data", "models"):
        raise HTTPException(400, "target must be 'data' or 'models'")

    dest_dir = Path(os.getenv("DATA_DIR", "data")) if target == "data" \
        else Path(os.getenv("MODEL_DIR", "models"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    max_size = 500 * 1024 * 1024  # 500MB safety cap on a single upload
    if len(content) > max_size:
        raise HTTPException(413, f"File too large: {len(content) / 1024 / 1024:.0f}MB (max 500MB)")

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Uploaded file is not a valid zip archive")

    logger.info(f"Uploaded and extracted {len(names)} files to {dest_dir} via /admin/upload-seed")
    return {
        "status": "ok",
        "extracted_to": str(dest_dir),
        "files_extracted": len(names),
        "sample_files": names[:10],
    }
