"""
Background scheduler — triggers pipeline.run() after each daily match window.
WC 2026 matches kick off at 17:00, 20:00, and 23:00 UTC.
Two daily runs catch all windows.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from src.retraining.pipeline import LiveRetrainPipeline


def build_scheduler(n_sim: int = 10_000) -> BackgroundScheduler:
    pipeline = LiveRetrainPipeline()
    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        func=pipeline.run,
        trigger="cron",
        hour=23,
        minute=30,
        id="evening_update",
        kwargs={"n_sim": n_sim},
        misfire_grace_time=1800,
    )

    scheduler.add_job(
        func=pipeline.run,
        trigger="cron",
        hour=2,
        minute=30,
        id="morning_update",
        kwargs={"n_sim": n_sim},
        misfire_grace_time=1800,
    )

    return scheduler
