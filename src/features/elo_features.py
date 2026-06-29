"""Elo-based features for the WC 2026 predictor feature matrix."""

import pandas as pd
import numpy as np
from loguru import logger

TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup":            1.5,
    "World Cup qualifier":       1.5,
    "Copa America":              1.2,
    "UEFA Euro":                 1.2,
    "African Cup of Nations":    1.2,
    "AFCON":                     1.2,
    "Gold Cup":                  1.1,
    "CONCACAF Gold Cup":         1.1,
    "CONCACAF Nations League":   1.1,
    "UEFA Nations League":       1.1,
    "Friendly":                  0.5,
    "Friendlies":                0.5,
}


def get_elo_on_date(
    elo_df: pd.DataFrame,
    team: str,
    cut_date: pd.Timestamp,
) -> float:
    """Return the most recent Elo rating for *team* strictly before *cut_date*.

    Args:
        elo_df:   DataFrame with columns ``team``, ``date`` (datetime), ``elo_rating``.
        team:     Canonical team name.
        cut_date: Exclusive upper bound — returns the last row where date < cut_date.

    Returns:
        Elo rating as float, or 1500.0 if no record exists before cut_date.
    """
    team_elo = elo_df[(elo_df["team"] == team) & (elo_df["date"] < cut_date)]
    if team_elo.empty:
        logger.debug(f"No Elo data for {team!r} before {cut_date.date()} — defaulting to 1500")
        return 1500.0
    return float(team_elo.sort_values("date")["elo_rating"].iloc[-1])


def get_elo_trajectory(
    elo_df: pd.DataFrame,
    team: str,
    cut_date: pd.Timestamp,
    window_days: int = 90,
) -> float:
    """Compute Elo momentum — rating change over the last *window_days* before *cut_date*.

    Args:
        elo_df:      DataFrame with columns ``team``, ``date``, ``elo_rating``.
        team:        Canonical team name.
        cut_date:    Exclusive upper bound for the window.
        window_days: Length of the look-back window in days (default 90).

    Returns:
        Float delta (positive = improving), or 0.0 if fewer than 2 data points found.
    """
    cutoff = cut_date - pd.Timedelta(days=window_days)
    window = elo_df[
        (elo_df["team"] == team)
        & (elo_df["date"] >= cutoff)
        & (elo_df["date"] < cut_date)
    ].sort_values("date")

    if len(window) < 2:
        return 0.0
    return float(window["elo_rating"].iloc[-1] - window["elo_rating"].iloc[0])


def compute_elo_features(match_row: pd.Series, elo_df: pd.DataFrame) -> dict:
    """Compute all Elo-based features for a single match.

    Reads ``home_team`` / ``away_team`` and ``date`` from *match_row*.
    Uses strict ``date < cut_date`` filtering to prevent data leakage.

    Args:
        match_row: A pandas Series with at least ``home_team``, ``away_team``, ``date``.
        elo_df:    Cleaned Elo DataFrame (columns: ``team``, ``date``, ``elo_rating``).

    Returns:
        dict with keys: elo_a, elo_b, elo_diff, elo_trajectory_diff.
    """
    team_a = match_row["home_team"]
    team_b = match_row["away_team"]
    cut_date = pd.Timestamp(match_row["date"])

    elo_a = get_elo_on_date(elo_df, team_a, cut_date)
    elo_b = get_elo_on_date(elo_df, team_b, cut_date)
    traj_a = get_elo_trajectory(elo_df, team_a, cut_date)
    traj_b = get_elo_trajectory(elo_df, team_b, cut_date)

    return {
        "elo_a":               elo_a,
        "elo_b":               elo_b,
        "elo_diff":            elo_a - elo_b,
        "elo_trajectory_diff": traj_a - traj_b,
    }


if __name__ == "__main__":
    elo = pd.read_parquet("data/processed/elo_clean.parquet")
    elo["date"] = pd.to_datetime(elo["date"])

    test = pd.Series({"home_team": "Brazil", "away_team": "Argentina", "date": "2022-12-18"})
    result = compute_elo_features(test, elo)
    print(result)
    print("Expected: elo_diff close to 0, both Elos above 2000")
