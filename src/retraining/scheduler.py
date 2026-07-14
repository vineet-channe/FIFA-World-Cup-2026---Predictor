"""
Background scheduler — triggers pipeline.run() once daily at 10:00 IST.

Every run is wrapped so a failure is logged and recorded to run_status.json
instead of silently vanishing into APScheduler's default exception handling.
"""
# from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
# from apscheduler.triggers.date import DateTrigger
from loguru import logger

from src.retraining.pipeline import LiveRetrainPipeline
from src.retraining.run_status import (
    record_run_start, record_run_success, record_run_failure,
)

# # Fixed kickoff times for all remaining WC 2026 matches, converted to UTC,
# # with the pipeline scheduled to fire 4 hours after each kickoff — enough
# # buffer for a match (including extra time and penalties) to finish and
# # for football-data.org to reflect the final confirmed result.
# #
# # Source: IST kickoff times as of 2026-07-11, converted UTC = IST - 5:30.
# REMAINING_MATCH_SCHEDULE = [
#     # (fire_time_utc, descriptive_id, human-readable note)
#     (datetime(2026, 7, 12, 1, 0, 0, tzinfo=timezone.utc),
#      "qf_norway_england", "QF: Norway vs England (kickoff Sun 12 Jul 02:30 IST)"),
#     (datetime(2026, 7, 12, 5, 0, 0, tzinfo=timezone.utc),
#      "qf_argentina_switzerland", "QF: Argentina vs Switzerland (kickoff Sun 12 Jul 06:30 IST)"),
#     (datetime(2026, 7, 14, 23, 0, 0, tzinfo=timezone.utc),
#      "sf_france_spain", "SF: France vs Spain (kickoff Wed 15 Jul 00:30 IST)"),
#     (datetime(2026, 7, 15, 23, 0, 0, tzinfo=timezone.utc),
#      "sf_tbd", "SF: TBD vs TBD (kickoff Thu 16 Jul 00:30 IST)"),
#     (datetime(2026, 7, 19, 1, 0, 0, tzinfo=timezone.utc),
#      "third_place_tbd", "3rd Place: TBD vs TBD (kickoff Sun 19 Jul 02:30 IST)"),
#     (datetime(2026, 7, 19, 23, 0, 0, tzinfo=timezone.utc),
#      "final_tbd", "Final: TBD vs TBD (kickoff Mon 20 Jul 00:30 IST)"),
# ]


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


# def build_match_schedule_scheduler(n_sim: int = 10_000) -> BackgroundScheduler:
#     """
#     Builds a scheduler with one-time triggers, each firing 4 hours after a
#     remaining WC 2026 match's real kickoff time. Unlike build_scheduler()
#     (recurring twice-daily cron), this fires exactly once per match and
#     then naturally has nothing left to do once the Final has passed —
#     no end-of-tournament handling needed.
#     """
#     scheduler = BackgroundScheduler(timezone="UTC")
#     scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)
#
#     now = datetime.now(timezone.utc)
#     scheduled_count = 0
#
#     for fire_time, job_id, note in REMAINING_MATCH_SCHEDULE:
#         if fire_time <= now:
#             logger.info(f"Skipping already-past match trigger: {job_id} ({note})")
#             continue
#
#         scheduler.add_job(
#             func=_guarded_run,
#             trigger=DateTrigger(run_date=fire_time),
#             id=job_id,
#             args=[n_sim],
#             misfire_grace_time=3600,  # 1 hour grace if the process was briefly down
#         )
#         logger.info(f"Scheduled: {job_id} → fires {fire_time.isoformat()} UTC | {note}")
#         scheduled_count += 1
#
#     logger.info(f"Match-based scheduler built: {scheduled_count} upcoming triggers scheduled")
#     return scheduler
