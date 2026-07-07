"""
Background scheduler — triggers pipeline.run() after each daily match window.
WC 2026 matches kick off at 17:00, 20:00, and 23:00 UTC.
Two daily runs catch all windows.

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
    """Build the pipeline lazily so models aren't held in RAM between runs."""
    record_run_start()
    try:
        pipeline = LiveRetrainPipeline()
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
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)

    scheduler.add_job(
        func=_guarded_run,
        trigger="cron",
        hour=23, minute=30,
        id="evening_update",
        args=[n_sim],
        misfire_grace_time=1800,
    )

    scheduler.add_job(
        func=_guarded_run,
        trigger="cron",
        hour=2, minute=30,
        id="morning_update",
        args=[n_sim],
        misfire_grace_time=1800,
    )

    return scheduler
