"""Scoreline sampling via Dixon-Coles lambdas + Poisson rejection sampling.

The ensemble decides *who wins*; this module decides *by how much*.
Dixon-Coles lambdas set the scale; rejection sampling ensures the sampled
scoreline is consistent with the pre-determined outcome.
"""

from __future__ import annotations

import numpy as np

# Global average expected-goals rates for international football.
# Used as fallback when a team pair is not in dc_cache.
_DEFAULT_LAMBDA_A = 1.3
_DEFAULT_LAMBDA_B = 1.1


def get_lambdas(
    team_a: str,
    team_b: str,
    dc_cache: dict,
) -> tuple[float, float]:
    """Look up pre-computed Dixon-Coles lambdas from the cache.

    Args:
        team_a:   Name of the first team.
        team_b:   Name of the second team.
        dc_cache: Dict mapping (team_a, team_b) → (lambda_a, lambda_b).

    Returns:
        Tuple ``(lambda_a, lambda_b)`` as floats.  Falls back to global
        international-football averages (1.3, 1.1) if the pair is missing.
    """
    pair = (team_a, team_b)
    if pair in dc_cache:
        return dc_cache[pair]
    # Log-level fallback — should only fire for TBD teams
    return (_DEFAULT_LAMBDA_A, _DEFAULT_LAMBDA_B)


def sample_scoreline(
    outcome: int,
    lam_a: float,
    lam_b: float,
    max_attempts: int = 200,
) -> tuple[int, int]:
    """Sample a scoreline from Poisson distributions consistent with *outcome*.

    Uses rejection sampling: draw scores from Poisson(lam_a) and
    Poisson(lam_b) until the resulting scoreline matches the desired outcome.
    Keeps a safety cap to avoid any infinite-loop risk.

    Args:
        outcome:      2 = team_a wins, 1 = draw, 0 = team_b wins.
        lam_a:        Expected goals for team_a (from Dixon-Coles).
        lam_b:        Expected goals for team_b (from Dixon-Coles).
        max_attempts: Maximum rejection-sampling iterations before fallback.

    Returns:
        ``(score_a, score_b)`` as non-negative integers consistent with outcome.
    """
    # Guard against degenerate lambdas
    lam_a = max(lam_a, 0.05)
    lam_b = max(lam_b, 0.05)

    for _ in range(max_attempts):
        s_a = np.random.poisson(lam_a)
        s_b = np.random.poisson(lam_b)

        if outcome == 2 and s_a > s_b:
            return (s_a, s_b)
        if outcome == 1 and s_a == s_b:
            return (s_a, s_b)
        if outcome == 0 and s_b > s_a:
            return (s_a, s_b)

    # Extremely rare fallback — return the minimal consistent scoreline
    if outcome == 2:
        return (1, 0)
    if outcome == 1:
        return (0, 0)
    return (0, 1)
