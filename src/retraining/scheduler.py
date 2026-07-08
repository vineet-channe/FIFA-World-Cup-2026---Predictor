"""
Background scheduler — triggers pipeline.run() once daily at 10:00 IST.

Every run is wrapped so a failure is logged and recorded to run_status.json
instead of silently vanishing into APScheduler's default exception handling.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from loguru import logger

from src.retraining.pipeline import LiveRetrainPipeline
from src.retraining.run_status import (
    record_run_start, record_run_success, record_run_failure,
)


def _guarded_run(n_sim: int) -> None:
    """
    Reuses models already loaded in the FastAPI process's app.state instead
    of loading a second copy — this is what prevents the pipeline run from
    roughly doubling model memory usage at its peak, which previously
    caused an OOM kill on Railway.
    """
    record_run_start()
    try:
        from src.api.main import app as _app
        ensemble = getattr(_app.state, "ensemble", None)
        dc_model = getattr(_app.state, "dc_model", None)
        lgbm     = getattr(_app.state, "lgbm", None)

        pipeline = LiveRetrainPipeline(
            ensemble=ensemble, dc_model=dc_model, lgbm=lgbm
        )
        output = pipeline.run(n_sim=n_sim)
        n_matches = len(output.get("actual_results", {}))
        record_run_success(n_matches_ingested=n_matches)
        logger.info(f"Scheduled pipeline run succeeded — {n_matches} matches ingested")
    except Exception as exc:
        logger.error(f"Scheduled pipeline run FAILED: {exc}")
        record_run_failure(str(exc))
        # Do not re-raise — a failed scheduled run should not crash the
        # scheduler process; the next scheduled run will try again.


def _on_job_error(event):
    logger.error(f"APScheduler job '{event.job_id}' raised: {event.exception}")


def build_scheduler(n_sim: int = 10_000) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)

    scheduler.add_job(
        func=_guarded_run,
        trigger="cron",
        hour=10, minute=0,
        id="daily_update",
        args=[n_sim],
        misfire_grace_time=1800,
    )

    return scheduler
