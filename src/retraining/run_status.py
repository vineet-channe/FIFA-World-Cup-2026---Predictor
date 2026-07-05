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


def get_run_status() -> dict:
    """Returns the last recorded run outcome. Never raises."""
    default = {"status": "unknown", "finished_at": None, "error": None}
    try:
        return {**default, **_read()}
    except Exception:
        return default


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
