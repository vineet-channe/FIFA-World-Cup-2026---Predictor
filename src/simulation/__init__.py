"""Phase 4 — Tournament Simulation Engine.

Exports the public API for Monte Carlo simulation of WC 2026.
"""

from .monte_carlo import run_simulation, load_simulation_results
from .group_stage import simulate_group, resolve_tiebreaker
from .knockout import simulate_knockout_round
from .penalties import simulate_penalties
from .scoreline import sample_scoreline

__all__ = [
    "run_simulation",
    "load_simulation_results",
    "simulate_group",
    "resolve_tiebreaker",
    "simulate_knockout_round",
    "simulate_penalties",
    "sample_scoreline",
]
