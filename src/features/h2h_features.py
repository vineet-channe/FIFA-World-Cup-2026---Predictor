"""Head-to-head historical features for the WC 2026 predictor."""

import pandas as pd
import numpy as np
from loguru import logger


def compute_h2h_features(
    matches_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    cut_date: pd.Timestamp,
    years_back: int = 10,
) -> dict:
    """Compute head-to-head statistics between *team_a* and *team_b*.

    All results are from *team_a*'s perspective. Only meetings within the
    last *years_back* years and strictly before *cut_date* are considered.

    Args:
        matches_df: Full match history DataFrame with columns
                    ``home_team``, ``away_team``, ``date``, ``home_score``,
                    ``away_score``, ``neutral``.
        team_a:     Canonical name of the primary team.
        team_b:     Canonical name of the opposing team.
        cut_date:   Exclusive upper bound — date < cut_date only.
        years_back: Look-back window in years (default 10).

    Returns:
        dict with keys: h2h_win_rate, h2h_goal_diff_avg,
        h2h_neutral_win_rate, h2h_n_matches.
        When no meetings exist returns neutral defaults (0.5 win rate, 0 goal diff).
    """
    cutoff = cut_date - pd.DateOffset(years=years_back)

    h2h = matches_df[
        (
            ((matches_df["home_team"] == team_a) & (matches_df["away_team"] == team_b))
            | ((matches_df["home_team"] == team_b) & (matches_df["away_team"] == team_a))
        )
        & (matches_df["date"] >= cutoff)
        & (matches_df["date"] < cut_date)
    ].copy()

    if h2h.empty:
        return {
            "h2h_win_rate":         0.5,
            "h2h_goal_diff_avg":    0.0,
            "h2h_neutral_win_rate": 0.5,
            "h2h_n_matches":        0,
        }

    a_home = h2h["home_team"] == team_a
    goals_a = np.where(a_home, h2h["home_score"], h2h["away_score"]).astype(float)
    goals_b = np.where(a_home, h2h["away_score"], h2h["home_score"]).astype(float)

    a_wins = int((goals_a > goals_b).sum())
    n = len(h2h)

    # Neutral-venue H2H win rate
    neutral = h2h[h2h["neutral"]]
    if len(neutral) > 0:
        na_home = neutral["home_team"] == team_a
        n_ga = np.where(na_home, neutral["home_score"], neutral["away_score"]).astype(float)
        n_gb = np.where(na_home, neutral["away_score"], neutral["home_score"]).astype(float)
        neutral_win_rate = float((n_ga > n_gb).mean())
    else:
        neutral_win_rate = float(a_wins / n) if n > 0 else 0.5

    return {
        "h2h_win_rate":         float(a_wins / n),
        "h2h_goal_diff_avg":    float((goals_a - goals_b).mean()),
        "h2h_neutral_win_rate": neutral_win_rate,
        "h2h_n_matches":        n,
    }


if __name__ == "__main__":
    import pandas as pd

    matches = pd.read_parquet("data/processed/matches_clean.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    cut = pd.Timestamp("2022-12-18")
    result = compute_h2h_features(matches, "Brazil", "Argentina", cut)
    print("Brazil vs Argentina H2H:", result)
