"""Tournament context features: rest days, neutral venue, WC experience."""

import pandas as pd
import numpy as np
from loguru import logger

ROUND_IMPORTANCE: dict[str, float] = {
    "FIFA World Cup":          2.0,  # refined per-round during tournament
    "World Cup qualifier":     1.0,
    "Copa America":            1.2,
    "UEFA Euro":               1.2,
    "African Cup of Nations":  1.2,
    "AFCON":                   1.2,
    "Gold Cup":                1.1,
    "CONCACAF Gold Cup":       1.1,
    "CONCACAF Nations League": 1.0,
    "UEFA Nations League":     1.0,
    "Friendly":                0.5,
}

_DEFAULT_REST_DAYS = 4.0  # used when rest days are unknown


def compute_context_features(
    match_row: pd.Series,
    rest_days_a: float = _DEFAULT_REST_DAYS,
    rest_days_b: float = _DEFAULT_REST_DAYS,
) -> dict:
    """Compute match context features: rest days differential, neutral flag, round importance.

    For historical training data the actual rest days are typically unknown,
    so both sides default to 4.0 — this is an intentional approximation.

    Args:
        match_row:    A pandas Series with at least ``tournament`` and ``neutral``.
        rest_days_a:  Days since team_a's previous match (default 4.0).
        rest_days_b:  Days since team_b's previous match (default 4.0).

    Returns:
        dict with keys: rest_days_diff, is_neutral, round_importance.
    """
    tournament = match_row.get("tournament", "")
    is_neutral = int(bool(match_row.get("neutral", True)))
    importance = ROUND_IMPORTANCE.get(tournament, 1.0)

    return {
        "rest_days_diff":  float(rest_days_a - rest_days_b),
        "is_neutral":      is_neutral,
        "round_importance": importance,
    }


def compute_wc_experience(
    matches_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    cut_date: pd.Timestamp,
) -> dict:
    """Compute World Cup appearances and average stage reached for both teams.

    Stage is approximated by counting matches played in each WC edition:
        3 → Group stage (1), 4 → R16/R32 (2), 5 → QF (3),
        6 → SF (4), 7 → Runner-up or 3rd (5), 7+won_final → Champion (6)

    Averages over the last 3 WC editions before cut_date.

    Args:
        matches_df: Full match history DataFrame.
        team_a:     Canonical name of the primary team.
        team_b:     Canonical name of the opposing team.
        cut_date:   Exclusive upper bound — only WC editions that ended before this date.

    Returns:
        dict with keys: wc_appearances_a, wc_appearances_b, wc_appearances_diff,
        avg_wc_finish_a, avg_wc_finish_b, avg_wc_finish_diff.
    """
    wc = matches_df[
        (matches_df["tournament"] == "FIFA World Cup")
        & (matches_df["date"] < cut_date)
    ].copy()

    def _team_wc_stats(team: str) -> tuple[int, float]:
        """Return (total_appearances, avg_finish_last_3) for team."""
        team_wc = wc[
            (wc["home_team"] == team) | (wc["away_team"] == team)
        ].copy()

        if team_wc.empty:
            return 0, np.nan

        # Group by WC edition (year)
        team_wc["year"] = team_wc["date"].dt.year
        editions = sorted(team_wc["year"].unique())
        appearances = len(editions)

        finish_by_edition = []
        for yr in editions:
            ed = team_wc[team_wc["year"] == yr]
            n_matches = len(ed)
            # Approximate stage by match count
            if n_matches <= 3:
                finish = 1
            elif n_matches == 4:
                finish = 2
            elif n_matches == 5:
                finish = 3
            elif n_matches == 6:
                finish = 4
            else:  # 7+
                # Check if team won the final (scored more in last match of that edition)
                last_match = ed.sort_values("date").iloc[-1]
                is_home_in_final = last_match["home_team"] == team
                if is_home_in_final:
                    won = last_match["home_score"] > last_match["away_score"]
                else:
                    won = last_match["away_score"] > last_match["home_score"]
                finish = 6 if won else 5
            finish_by_edition.append(finish)

        # Average over last 3 editions
        last_3 = finish_by_edition[-3:]
        avg_finish = float(np.mean(last_3)) if last_3 else np.nan
        return appearances, avg_finish

    try:
        apps_a, avg_finish_a = _team_wc_stats(team_a)
        apps_b, avg_finish_b = _team_wc_stats(team_b)
    except Exception as exc:
        logger.warning(f"WC experience computation failed for {team_a} vs {team_b}: {exc}")
        apps_a = apps_b = 0
        avg_finish_a = avg_finish_b = np.nan

    if np.isnan(avg_finish_a) or np.isnan(avg_finish_b):
        avg_finish_diff = np.nan
    else:
        avg_finish_diff = avg_finish_a - avg_finish_b

    return {
        "wc_appearances_a":    apps_a,
        "wc_appearances_b":    apps_b,
        "wc_appearances_diff": float(apps_a - apps_b),
        "avg_wc_finish_a":     avg_finish_a,
        "avg_wc_finish_b":     avg_finish_b,
        "avg_wc_finish_diff":  avg_finish_diff,
    }


if __name__ == "__main__":
    import pandas as pd

    matches = pd.read_parquet("data/processed/matches_clean.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    cut = pd.Timestamp("2022-12-18")

    test_row = pd.Series({"tournament": "FIFA World Cup", "neutral": True})
    print("Context features:", compute_context_features(test_row))

    print("\nBrazil vs Bolivia WC experience:")
    print(compute_wc_experience(matches, "Brazil", "Bolivia", cut))
