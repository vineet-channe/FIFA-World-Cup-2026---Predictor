"""Knockout-stage simulation: R32 through Final.

Bracket construction follows FIFA's announced WC 2026 format.
All knockout matches that finish level at 90 min go to penalty shootout.
"""

from __future__ import annotations

import numpy as np

from .penalties import simulate_penalties
from .scoreline import get_lambdas, sample_scoreline

# ---------------------------------------------------------------------------
# R32 bracket template (FIFA WC 2026 announced pairing)
# ---------------------------------------------------------------------------
# Slot codes:  W_X = winner of group X,  RU_X = runner-up of group X
# TP_N  = N-th best third-place team (ranked 1–8 by the best-third logic)
#
# FIFA has defined the bracket so that group winners and runners-up from
# different confederation zones are separated.  The 16 matches below
# use the official WC 2026 bracket released by FIFA (placeholder — update
# when the official draw document is confirmed).
# The last 4 slots involve the best 8 third-place teams; their exact
# assignment depends on which groups they come from; we use a fixed slot
# order here (update once FIFA publishes the official matrix).

R32_BRACKET_TEMPLATE: list[tuple[str, str]] = [
    ("W_A",  "RU_B"),
    ("W_C",  "RU_D"),
    ("W_E",  "RU_F"),
    ("W_G",  "RU_H"),
    ("W_I",  "RU_J"),
    ("W_K",  "RU_L"),
    ("W_B",  "RU_A"),
    ("W_D",  "RU_C"),
    ("W_F",  "RU_E"),
    ("W_H",  "RU_G"),
    ("W_J",  "RU_I"),
    ("W_L",  "RU_K"),
    # 4 slots for best 8 third-place teams (ranked by the best-third logic)
    ("TP_1", "TP_2"),
    ("TP_3", "TP_4"),
    ("TP_5", "TP_6"),
    ("TP_7", "TP_8"),
]


def build_r32_bracket(
    group_standings: dict[str, list[str]],
    best_thirds: list[str],
) -> list[tuple[str, str]]:
    """Construct the 16 R32 matchups from group standings and best-thirds.

    Args:
        group_standings: Dict group_letter → [1st, 2nd, 3rd, 4th].
        best_thirds:     Ordered list of 8 best third-place team names
                         (best first).

    Returns:
        List of 16 ``(team_a, team_b)`` tuples for R32.
    """
    # Build slot → team mapping
    slot_map: dict[str, str] = {}
    for group, standings in group_standings.items():
        slot_map[f"W_{group}"]  = standings[0]
        slot_map[f"RU_{group}"] = standings[1]

    for i, team in enumerate(best_thirds[:8], start=1):
        slot_map[f"TP_{i}"] = team

    pairs: list[tuple[str, str]] = []
    for slot_a, slot_b in R32_BRACKET_TEMPLATE:
        team_a = slot_map.get(slot_a)
        team_b = slot_map.get(slot_b)
        if team_a and team_b:
            pairs.append((team_a, team_b))

    return pairs


# ---------------------------------------------------------------------------
# Single knockout match
# ---------------------------------------------------------------------------

def simulate_knockout_match(
    team_a: str,
    team_b: str,
    prob_cache: dict,
    dc_cache: dict,
    elo_ratings: dict,
) -> dict:
    """Simulate one knockout match (no draws allowed).

    Outcome sampling flow:
      1. Ensemble probability → outcome (0/1/2).
      2. Dixon-Coles lambdas → rejection-sampled scoreline.
      3. If outcome == draw (1) → penalty shootout decides the winner.

    Args:
        team_a:      First team name.
        team_b:      Second team name.
        prob_cache:  Pre-computed ensemble probabilities.
        dc_cache:    Pre-computed Dixon-Coles lambdas.
        elo_ratings: Team → Elo rating dict.

    Returns:
        Dict with keys: team_a, team_b, score_a, score_b, winner, penalties.
    """
    proba = _get_proba(team_a, team_b, prob_cache)
    outcome = int(np.random.choice([0, 1, 2], p=proba))

    lam_a, lam_b = get_lambdas(team_a, team_b, dc_cache)
    score_a, score_b = sample_scoreline(outcome, lam_a, lam_b)

    went_to_penalties = False

    if outcome == 1:
        # Draw after 90 min → penalty shootout
        winner = simulate_penalties(team_a, team_b, elo_ratings)
        went_to_penalties = True
    elif outcome == 2:
        winner = team_a
    else:
        winner = team_b

    return {
        "team_a":     team_a,
        "team_b":     team_b,
        "score_a":    score_a,
        "score_b":    score_b,
        "winner":     winner,
        "penalties":  went_to_penalties,
    }


# ---------------------------------------------------------------------------
# Full knockout round
# ---------------------------------------------------------------------------

def simulate_knockout_round(
    pairs: list[tuple[str, str]],
    prob_cache: dict,
    dc_cache: dict,
    elo_ratings: dict,
) -> dict:
    """Simulate a complete knockout round (e.g. all 16 R32 matches).

    Args:
        pairs:       List of (team_a, team_b) matchups for this round.
        prob_cache:  Pre-computed ensemble probabilities.
        dc_cache:    Pre-computed Dixon-Coles lambdas.
        elo_ratings: Team → Elo rating dict.

    Returns:
        Dict with:
          ``"winners"``  — ordered list of winning teams.
          ``"results"``  — list of match-result dicts from simulate_knockout_match().
    """
    winners: list[str] = []
    results: list[dict] = []

    for team_a, team_b in pairs:
        match = simulate_knockout_match(team_a, team_b, prob_cache, dc_cache, elo_ratings)
        winners.append(match["winner"])
        results.append(match)

    return {"winners": winners, "results": results}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_proba(
    team_a: str,
    team_b: str,
    prob_cache: dict,
) -> np.ndarray:
    """Return cached ensemble probabilities, falling back to 1/3 each."""
    pair = (team_a, team_b)
    if pair in prob_cache:
        return prob_cache[pair]
    return np.array([1 / 3, 1 / 3, 1 / 3])
