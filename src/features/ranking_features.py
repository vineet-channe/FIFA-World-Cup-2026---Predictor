"""FIFA ranking point features for the WC 2026 predictor."""

import pandas as pd
from loguru import logger

_FALLBACK_POINTS = 1000.0  # Median-ish FIFA points for unknown teams


def get_ranking_on_date(
    rankings_df: pd.DataFrame,
    team: str,
    cut_date: pd.Timestamp,
) -> float:
    """Return the most recent FIFA ``total_points`` for *team* strictly before *cut_date*.

    Uses ``country_full`` as the team identifier column.

    Args:
        rankings_df: DataFrame with columns ``country_full``, ``total_points``, ``rank_date``.
        team:        Canonical team name matching ``country_full``.
        cut_date:    Exclusive upper bound — only rows where rank_date < cut_date.

    Returns:
        FIFA total_points as float, or 1000.0 if no record found.
    """
    team_rows = rankings_df[
        (rankings_df["country_full"] == team)
        & (rankings_df["rank_date"] < cut_date)
    ]
    if team_rows.empty:
        logger.debug(f"No FIFA ranking for {team!r} before {cut_date.date()} — defaulting to {_FALLBACK_POINTS}")
        return _FALLBACK_POINTS
    return float(team_rows.sort_values("rank_date")["total_points"].iloc[-1])


def compute_ranking_features(
    match_row: pd.Series,
    rankings_df: pd.DataFrame,
) -> dict:
    """Compute FIFA ranking point features for a single match.

    Reads ``home_team`` / ``away_team`` and ``date`` from *match_row*.
    Uses strict ``date < cut_date`` filtering to prevent data leakage.

    Args:
        match_row:   A pandas Series with at least ``home_team``, ``away_team``, ``date``.
        rankings_df: Cleaned rankings DataFrame.

    Returns:
        dict with keys: fifa_pts_a, fifa_pts_b, fifa_pts_diff.
    """
    team_a = match_row["home_team"]
    team_b = match_row["away_team"]
    cut_date = pd.Timestamp(match_row["date"])

    pts_a = get_ranking_on_date(rankings_df, team_a, cut_date)
    pts_b = get_ranking_on_date(rankings_df, team_b, cut_date)

    return {
        "fifa_pts_a":    pts_a,
        "fifa_pts_b":    pts_b,
        "fifa_pts_diff": pts_a - pts_b,
    }


if __name__ == "__main__":
    import pandas as pd

    rankings = pd.read_parquet("data/processed/rankings_clean.parquet")
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])

    test = pd.Series({"home_team": "Brazil", "away_team": "Argentina", "date": "2022-12-18"})
    result = compute_ranking_features(test, rankings)
    print(result)
