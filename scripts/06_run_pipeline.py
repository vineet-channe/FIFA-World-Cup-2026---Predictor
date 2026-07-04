"""
Manually trigger the live retraining pipeline. Use after a matchday
if the scheduler missed a run or you want to force an immediate update.

Usage:
    python scripts/06_run_pipeline.py
    python scripts/06_run_pipeline.py --n 5000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retraining.pipeline import LiveRetrainPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10_000)
args = parser.parse_args()

pipeline = LiveRetrainPipeline()
output = pipeline.run(n_sim=args.n)
print("Pipeline complete. tournament_simulation.json updated.")
