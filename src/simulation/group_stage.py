"""Group-stage simulation and FIFA tiebreaker logic.

Simulates all 6 matches within a 4-team group, builds the full standings
table, and applies the official 8-criterion FIFA tiebreaker to rank teams.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .scoreline import get_lambdas, sample_scoreline


# ---------------------------------------------------------------------------
# Core group simulator
# ---------------------------------------------------------------------------

def simulate_group(
    teams: list[str],
    prob_cache: dict,
    dc_cache: dict,
) -> dict:
    """Simulate all 6 matches in a 4-team group.

    Args:
        teams:      List of exactly 4 canonical team names.
        prob_cache: Pre-computed probability cache: (team_a, team_b) →
                    np.array shape (3,) as [P(team_b_wins), P(draw), P(team_a_wins)].
        dc_cache:   Pre-computed Dixon-Coles cache: (team_a, team_b) →
                    (lambda_a, lambda_b).

    Returns:
        Dict with keys:
          ``"matches"`` — list of 6 match-result dicts.
          ``"table"``   — per-team standings dict with points, gd, gf, ga, h2h.
    """
    assert len(teams) == 4, f"Expected 4 teams, got {len(teams)}"

    # Initialise standings
    table: dict[str, dict] = {
        t: {
            "points": 0,
            "gd": 0,
            "gf": 0,
            "ga": 0,
            "h2h": {other: {"points": 0, "gd": 0, "gf": 0}
                    for other in teams if other != t},
        }
        for t in teams
    }

    matches = []
    for team_a, team_b in itertools.combinations(teams, 2):
        # --- Outcome from ensemble cache ---
        proba = _get_proba(team_a, team_b, prob_cache)
        outcome = int(np.random.choice([0, 1, 2], p=proba))

        # --- Scoreline from Dixon-Coles + rejection sampling ---
        lam_a, lam_b = get_lambdas(team_a, team_b, dc_cache)
        score_a, score_b = sample_scoreline(outcome, lam_a, lam_b)

        winner: str | None = (
            team_a if score_a > score_b else (team_b if score_b > score_a else None)
        )

        # --- Update overall standings ---
        pts_a = 3 if score_a > score_b else (1 if score_a == score_b else 0)
        pts_b = 3 if score_b > score_a else (1 if score_a == score_b else 0)

        table[team_a]["points"] += pts_a
        table[team_a]["gf"] += score_a
        table[team_a]["ga"] += score_b
        table[team_a]["gd"] += score_a - score_b

        table[team_b]["points"] += pts_b
        table[team_b]["gf"] += score_b
        table[team_b]["ga"] += score_a
        table[team_b]["gd"] += score_b - score_a

        # --- Update head-to-head records ---
        table[team_a]["h2h"][team_b]["points"] += pts_a
        table[team_a]["h2h"][team_b]["gd"]     += score_a - score_b
        table[team_a]["h2h"][team_b]["gf"]     += score_a

        table[team_b]["h2h"][team_a]["points"] += pts_b
        table[team_b]["h2h"][team_a]["gd"]     += score_b - score_a
        table[team_b]["h2h"][team_a]["gf"]     += score_b

        matches.append({
            "team_a": team_a,
            "team_b": team_b,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
        })

    return {"matches": matches, "table": table}


# ---------------------------------------------------------------------------
# FIFA 8-criterion tiebreaker
# ---------------------------------------------------------------------------

def resolve_tiebreaker(
    tied_teams: list[str],
    table: dict,
    elo_ratings: dict,
) -> list[str]:
    """Apply the FIFA 8-criterion tiebreaker to a set of equally-ranked teams.

    Criteria applied in order:
      1. Points in all group matches
      2. Goal difference in all group matches
      3. Goals scored in all group matches
      4. Points in head-to-head matches among the tied teams
      5. Goal difference in head-to-head matches among the tied teams
      6. Goals scored in head-to-head matches among the tied teams
      7. Disciplinary points — not simulated, skipped
      8. Elo rating as proxy for FIFA Confederation ranking (last resort)

    Args:
        tied_teams:  List of teams tied on the same criterion.
        table:       Full group standings dict from simulate_group().
        elo_ratings: Dict team → float Elo rating.

    Returns:
        *tied_teams* sorted from best to worst.
    """
    if len(tied_teams) <= 1:
        return list(tied_teams)

    def _sort_key_overall(team: str) -> tuple:
        t = table[team]
        return (t["points"], t["gd"], t["gf"])

    def _h2h_stats(team: str, opponents: list[str]) -> tuple:
        """Aggregate H2H stats vs a specific list of opponents."""
        pts = sum(table[team]["h2h"][opp]["points"] for opp in opponents)
        gd  = sum(table[team]["h2h"][opp]["gd"]     for opp in opponents)
        gf  = sum(table[team]["h2h"][opp]["gf"]     for opp in opponents)
        return (pts, gd, gf)

    def _resolve_subset(subset: list[str]) -> list[str]:
        """Recursively resolve a subset using all 8 criteria."""
        if len(subset) <= 1:
            return subset

        opponents = [t for t in subset]

        # Criterion 1–3: overall group stats (already equal for top-level ties,
        # but may differ when resolving a sub-tie after H2H)
        sorted_by_overall = sorted(
            subset,
            key=lambda t: (table[t]["points"], table[t]["gd"], table[t]["gf"]),
            reverse=True,
        )

        # Criterion 4–6: head-to-head among tied teams
        h2h_stats = {t: _h2h_stats(t, [x for x in opponents if x != t])
                     for t in subset}
        sorted_by_h2h = sorted(
            subset,
            key=lambda t: h2h_stats[t],
            reverse=True,
        )

        # Criterion 8: Elo
        sorted_by_elo = sorted(
            subset,
            key=lambda t: elo_ratings.get(t, 1500.0),
            reverse=True,
        )

        # Apply criteria in order, splitting into resolved groups
        result: list[str] = []
        remaining = list(subset)

        # Step 1–3: sort by overall stats (group them if equal)
        groups_overall = _group_equal(remaining, lambda t: (table[t]["points"], table[t]["gd"], table[t]["gf"]))

        for group in groups_overall:
            if len(group) == 1:
                result.extend(group)
            else:
                # Step 4–6: sort by H2H within the tied group
                h2h_sub = {t: _h2h_stats(t, [x for x in group if x != t]) for t in group}
                groups_h2h = _group_equal(group, lambda t: h2h_sub[t])
                for sub in groups_h2h:
                    if len(sub) == 1:
                        result.extend(sub)
                    else:
                        # Step 8: Elo tiebreaker
                        result.extend(sorted(sub, key=lambda t: elo_ratings.get(t, 1500.0), reverse=True))

        return result

    return _resolve_subset(tied_teams)


def _group_equal(teams: list[str], key_fn) -> list[list[str]]:
    """Group teams that are equal under key_fn into contiguous sub-lists,
    preserving the descending sort order."""
    if not teams:
        return []
    sorted_teams = sorted(teams, key=key_fn, reverse=True)
    groups: list[list[str]] = []
    current_group = [sorted_teams[0]]
    current_key = key_fn(sorted_teams[0])
    for t in sorted_teams[1:]:
        k = key_fn(t)
        if k == current_key:
            current_group.append(t)
        else:
            groups.append(current_group)
            current_group = [t]
            current_key = k
    groups.append(current_group)
    return groups


# ---------------------------------------------------------------------------
# Group standings builder
# ---------------------------------------------------------------------------

def get_group_standings(
    group_result: dict,
    elo_ratings: dict,
) -> list[str]:
    """Return [1st, 2nd, 3rd, 4th] from a simulate_group() result.

    Applies resolve_tiebreaker() wherever teams share the same rank.

    Args:
        group_result: Dict returned by simulate_group().
        elo_ratings:  Dict team → Elo rating.

    Returns:
        List of 4 team names ordered 1st → 4th.
    """
    table = group_result["table"]
    teams = list(table.keys())

    # Initial sort by points → gd → gf
    sorted_teams = sorted(
        teams,
        key=lambda t: (table[t]["points"], table[t]["gd"], table[t]["gf"]),
        reverse=True,
    )

    # Detect and resolve ties
    result: list[str] = []
    i = 0
    while i < len(sorted_teams):
        # Find the extent of this tie group
        j = i + 1
        while j < len(sorted_teams) and (
            table[sorted_teams[j]]["points"] == table[sorted_teams[i]]["points"]
            and table[sorted_teams[j]]["gd"]     == table[sorted_teams[i]]["gd"]
            and table[sorted_teams[j]]["gf"]     == table[sorted_teams[i]]["gf"]
        ):
            j += 1

        tied = sorted_teams[i:j]
        if len(tied) > 1:
            tied = resolve_tiebreaker(tied, table, elo_ratings)
        result.extend(tied)
        i = j

    return result


# ---------------------------------------------------------------------------
# Best third-place selection
# ---------------------------------------------------------------------------

def select_best_third_place(
    third_place_teams: list[dict],
    elo_ratings: dict,
) -> list[str]:
    """Pick the 8 best third-place teams to advance to R32.

    Ranking criteria (in order):
      1. Points
      2. Goal difference
      3. Goals scored
      4. Elo rating

    Args:
        third_place_teams: List of dicts with keys:
            ``team``, ``points``, ``gd``, ``gf``, ``elo``.
        elo_ratings: Dict team → Elo (used as fallback if not in team dict).

    Returns:
        List of 8 team name strings (best 8 of the 12 third-place finishers).
    """
    ranked = sorted(
        third_place_teams,
        key=lambda d: (
            d["points"],
            d["gd"],
            d["gf"],
            d.get("elo", elo_ratings.get(d["team"], 1500.0)),
        ),
        reverse=True,
    )
    return [d["team"] for d in ranked[:8]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_proba(
    team_a: str,
    team_b: str,
    prob_cache: dict,
) -> np.ndarray:
    """Look up outcome probabilities from cache, falling back to 1/3 each."""
    pair = (team_a, team_b)
    if pair in prob_cache:
        return prob_cache[pair]
    # Fallback — equal probabilities (should not happen for known WC teams)
    return np.array([1 / 3, 1 / 3, 1 / 3])
