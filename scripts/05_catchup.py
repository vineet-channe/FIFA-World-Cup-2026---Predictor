"""
Run ONCE immediately to catch up with all completed WC 2026 results.
Rebuilds tournament_simulation.json from the real group stage outcomes
and any completed R32 results.

Usage:
    python scripts/05_catchup.py
    python scripts/05_catchup.py --n 5000   # faster for testing
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retraining.pipeline import LiveRetrainPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10_000)
args = parser.parse_args()

print("=== WC 2026 Catch-up ===")
print("Pulling all completed results from API-Football...")

pipeline = LiveRetrainPipeline()
output = pipeline.run(n_sim=args.n)

actual = output.get("actual_results", {})
preds = output.get("match_predictions", {})
remaining = [v for v in preds.values() if v.get("status") == "predicted"]

print(f"\n✅ Completed matches ingested: {len(actual)}")
print(f"✅ Remaining matches to predict: {len(remaining)}")
print(
    f"✅ Group stage standings saved: "
    f"{len(output.get('group_stage_standings', {})) == 12}"
)

print("\nTop 10 championship probabilities:")
tp = output["team_probabilities"]
for team, p in sorted(tp.items(), key=lambda x: -x[1]["p_champion"])[:10]:
    elim = p.get("eliminated_in")
    tag = f"  [OUT: {elim}]" if elim else ""
    print(f"  {team:20s}  {p['p_champion']:.1%}{tag}")

print("\ntournament_simulation.json updated.")
