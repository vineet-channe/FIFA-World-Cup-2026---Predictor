"""Recent form features — last N competitive matches for each team."""

import pandas as pd
import numpy as np
from loguru import logger

COMPETITIVE_TOURNAMENTS: list[str] = [
    "FIFA World Cup",
    "World Cup qualifier",
    "Copa America",
    "UEFA Euro",
    "African Cup of Nations",
    "AFCON",
    "Gold Cup",
    "CONCACAF Gold Cup",
    "CONCACAF Nations League",
    "UEFA Nations League",
    "AFC Asian Cup",
    "OFC Nations Cup",
    "Confederations Cup",
]

_MIN_MATCHES = 3  # minimum matches required to return non-NaN form


def get_recent_matches(
    matches_df: pd.DataFrame,
    team: str,
    cut_date: pd.Timestamp,
    n: int = 10,
    competitive_only: bool = True,
) -> pd.DataFrame:
    """Retrieve the last *n* matches for *team* strictly before *cut_date*.

    Args:
        matches_df:       Full match history DataFrame with columns
                          ``home_team``, ``away_team``, ``date``, ``is_competitive``.
        team:             Canonical team name.
        cut_date:         Exclusive upper bound — date < cut_date only.
        n:                Maximum number of recent matches to return.
        competitive_only: When True, exclude friendlies (``is_competitive == False``).

    Returns:
        DataFrame of up to *n* matches sorted ascending by date (most recent last).
    """
    team_matches = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team))
        & (matches_df["date"] < cut_date)
    ].copy()

    if competitive_only:
        team_matches = team_matches[team_matches["is_competitive"]]

    return team_matches.sort_values("date").tail(n)


def compute_form_features(
    matches_df: pd.DataFrame,
    team: str,
    cut_date: pd.Timestamp,
    n: int = 10,
) -> dict:
    """Compute recent form statistics for *team* before *cut_date*.

    All statistics cover the last *n* competitive matches.
    Returns NaN values (not 0.0) when fewer than ``_MIN_MATCHES`` records exist
    — insufficient data is semantically different from bad performance.

    Args:
        matches_df: Full match history DataFrame.
        team:       Canonical team name.
        cut_date:   Exclusive upper bound.
        n:          Window size (default 10 competitive matches).

    Returns:
        dict with keys: ppg, goals_scored_pg, goals_conceded_pg,
        clean_sheet_pct, win_pct, neutral_win_pct, n_matches.
    """
    nan_result = {
        "ppg":               np.nan,
        "goals_scored_pg":   np.nan,
        "goals_conceded_pg": np.nan,
        "clean_sheet_pct":   np.nan,
        "win_pct":           np.nan,
        "neutral_win_pct":   np.nan,
        "n_matches":         0,
    }

    df = get_recent_matches(matches_df, team, cut_date, n=n, competitive_only=True)

    if len(df) < _MIN_MATCHES:
        logger.debug(f"Insufficient form data for {team!r} before {cut_date.date()} ({len(df)} matches)")
        return nan_result

    is_home = df["home_team"] == team
    goals_for = np.where(is_home, df["home_score"], df["away_score"]).astype(float)
    goals_ag  = np.where(is_home, df["away_score"], df["home_score"]).astype(float)

    wins   = (goals_for > goals_ag).astype(int)
    draws  = (goals_for == goals_ag).astype(int)
    points = wins * 3 + draws

    # Neutral-venue or away games only (WC proxy — team is never the "home" side)
    neutral_mask = df["neutral"].values | (~is_home.values)
    neutral_df   = df[neutral_mask]

    if len(neutral_df) > 0:
        n_is_home = neutral_df["home_team"] == team
        n_gf = np.where(n_is_home, neutral_df["home_score"], neutral_df["away_score"]).astype(float)
        n_ga = np.where(n_is_home, neutral_df["away_score"], neutral_df["home_score"]).astype(float)
        neutral_win_pct = float((n_gf > n_ga).mean())
    else:
        neutral_win_pct = float(wins.mean())

    return {
        "ppg":               float(points.mean()),
        "goals_scored_pg":   float(goals_for.mean()),
        "goals_conceded_pg": float(goals_ag.mean()),
        "clean_sheet_pct":   float((goals_ag == 0).mean()),
        "win_pct":           float(wins.mean()),
        "neutral_win_pct":   neutral_win_pct,
        "n_matches":         len(df),
    }


if __name__ == "__main__":
    import pandas as pd

    matches = pd.read_parquet("data/processed/matches_clean.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    cut = pd.Timestamp("2022-12-18")
    print("Brazil form:", compute_form_features(matches, "Brazil", cut))
    print("Argentina form:", compute_form_features(matches, "Argentina", cut))
