"""
Tracks the outcome of every pipeline run (scheduled or manual) so the API
can report whether the system is healthy or has been silently failing.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

from config.settings import settings

STATUS_PATH = settings.DATA_DIR / "predictions" / "run_status.json"


def record_run_start() -> None:
    _write({"status": "running", "started_at": _now()})


def record_run_success(n_matches_ingested: int = 0) -> None:
    _write({
        "status": "ok",
        "started_at": _read().get("started_at"),
        "finished_at": _now(),
        "n_matches_ingested": n_matches_ingested,
    })


def record_run_failure(error: str) -> None:
    _write({
        "status": "failed",
        "started_at": _read().get("started_at"),
        "finished_at": _now(),
        "error": error[:500],
    })


# Pipeline runs normally complete in 2-4 minutes. Any "running" status
# older than this is almost certainly a crashed/killed process that never
# got the chance to call record_run_failure() — not a run still in progress.
MAX_EXPECTED_RUNTIME_MINUTES = 15


def get_run_status() -> dict:
    """Returns the last recorded run outcome. Never raises.

    A "running" status older than MAX_EXPECTED_RUNTIME_MINUTES is reported
    as "failed" instead — this handles processes killed mid-run (e.g. by
    an OOM kill) that never reached record_run_failure().
    """
    default = {"status": "unknown", "finished_at": None, "error": None}
    try:
        data = {**default, **_read()}
    except Exception:
        return default

    if data.get("status") == "running":
        started_at = data.get("started_at")
        if started_at:
            try:
                started = datetime.fromisoformat(started_at)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                age_minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60
                if age_minutes > MAX_EXPECTED_RUNTIME_MINUTES:
                    return {
                        **data,
                        "status": "failed",
                        "error": (
                            f"Run started {age_minutes:.0f} min ago and never "
                            f"completed — likely killed mid-run (e.g. OOM). "
                            f"Original started_at: {started_at}"
                        ),
                    }
            except Exception:
                pass

    return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read() -> dict:
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            return json.load(f)
    return {}


def _write(data: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w") as f:
        json.dump(data, f, indent=2)
