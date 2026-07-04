"""Penalty shootout model for knockout-round draws.

Historical WC shootout data shows ~50/50 outcomes with a marginal
Elo-based edge for the stronger team.  Win probability is compressed
to [0.45, 0.55] to reflect the near-randomness of shootouts.
"""

from __future__ import annotations

import numpy as np

# Default Elo used when a team has no Elo entry in the cache.
_DEFAULT_ELO = 1700.0


def simulate_penalties(
    team_a: str,
    team_b: str,
    elo_ratings: dict[str, float],
) -> str:
    """Simulate a penalty shootout and return the winning team name.

    Args:
        team_a:      Name of the first team (would be "home" in 90-min context).
        team_b:      Name of the second team.
        elo_ratings: Dict mapping team name → Elo rating as of tournament start.

    Returns:
        The name of the winning team (either *team_a* or *team_b*).
    """
    elo_a = elo_ratings.get(team_a, _DEFAULT_ELO)
    elo_b = elo_ratings.get(team_b, _DEFAULT_ELO)
    elo_diff = elo_a - elo_b

    # Standard Elo expected-score formula
    raw_p_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 2000.0))

    # Compress to [0.45, 0.55] — shootouts are close to 50/50
    p_a_wins = 0.45 + 0.10 * (raw_p_a - 0.5)

    return team_a if np.random.random() < p_a_wins else team_b
