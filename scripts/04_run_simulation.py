"""
Run the WC 2026 Monte Carlo simulation.

Usage:
  python scripts/04_run_simulation.py              # 10,000 sims (default)
  python scripts/04_run_simulation.py --n 1000     # quick test run
  python scripts/04_run_simulation.py --n 100000   # high-precision run
  python scripts/04_run_simulation.py --n 100 --seed 0   # smoke test
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation import run_simulation

parser = argparse.ArgumentParser(description="WC 2026 Monte Carlo simulation")
parser.add_argument("--n",    type=int, default=10_000, help="Number of simulations")
parser.add_argument("--seed", type=int, default=42,     help="Random seed")
args = parser.parse_args()

results = run_simulation(n_sim=args.n, random_seed=args.seed)

# ── Championship table ──────────────────────────────────────────────────────
probs  = results["team_probabilities"]
ranked = sorted(probs.items(), key=lambda x: x[1]["p_champion"], reverse=True)

print("\n" + "=" * 65)
print(f"WC 2026 CHAMPIONSHIP PROBABILITIES  ({args.n:,} simulations)")
print("=" * 65)
print(f"{'Team':<25} {'Champion':>9} {'Final':>8} {'Semi':>7} {'QF':>7} {'Groups':>8}")
print("-" * 65)
for team, p in ranked[:16]:
    print(
        f"{team:<25} {p['p_champion']:>8.1%} {p['p_final']:>7.1%} "
        f"{p['p_semi']:>6.1%} {p['p_quarter']:>6.1%} {p['p_advance_groups']:>7.1%}"
    )
print("=" * 65)

# Summary stats
total_champ = sum(p["p_champion"] for p in probs.values())
print(f"\nSum of all p_champion: {total_champ:.4f}  (should be ~1.0)")
print(f"Output saved to: data/predictions/tournament_simulation.json")
