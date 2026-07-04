"""
Start the background scheduler. Keep this process running for the
remainder of the tournament — it auto-triggers after each match window.

Usage:
    python scripts/07_start_scheduler.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retraining.scheduler import build_scheduler

scheduler = build_scheduler()
scheduler.start()
print("Scheduler running. Triggers at 23:30 UTC and 02:30 UTC daily.")
print("Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    scheduler.shutdown()
    print("Scheduler stopped.")
