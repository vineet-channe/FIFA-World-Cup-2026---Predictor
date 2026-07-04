"""
Updates Elo ratings using actual WC 2026 match results.
Appends new rows to elo_clean.parquet so the feature pipeline sees current ratings.
"""
import pandas as pd
from loguru import logger
from config.settings import settings

ELO_PATH = settings.DATA_DIR / "processed" / "elo_clean.parquet"
K_FACTOR_WC = 40


def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def _get_outcome(score_a: int, score_b: int) -> float:
    if score_a > score_b:
        return 1.0
    if score_a < score_b:
        return 0.0
    return 0.5


def update_elo_with_results(
    fixtures: list[dict],
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute Elo updates for each completed WC 2026 match and append new rows.
    Processes matches in chronological order.
    """
    if elo_df is None:
        elo_df = pd.read_parquet(ELO_PATH)
    elo_df["date"] = pd.to_datetime(elo_df["date"]).dt.tz_localize(None)

    completed = sorted(
        [
            f
            for f in fixtures
            if f["status"] == "FT"
            and f["home_score"] is not None
            and f["away_score"] is not None
        ],
        key=lambda x: x["date"],
    )

    if "source" in elo_df.columns:
        wc_actual = elo_df[elo_df["source"] == "wc2026_actual"].copy()
    else:
        wc_actual = elo_df.iloc[0:0]
    processed_dates = set(
        pd.to_datetime(wc_actual["date"]).dt.normalize()
        if len(wc_actual)
        else []
    )

    def _already_processed(fix: dict) -> bool:
        match_date = pd.Timestamp(fix["date"]).tz_localize(None).normalize()
        if match_date not in processed_dates:
            return False
        teams_on_date = set(
            wc_actual.loc[
                pd.to_datetime(wc_actual["date"]).dt.normalize() == match_date, "team"
            ]
        )
        return fix["home_team"] in teams_on_date and fix["away_team"] in teams_on_date

    completed = [f for f in completed if not _already_processed(f)]

    new_rows = []
    current_elo: dict[str, float] = (
        elo_df.sort_values("date").groupby("team")["elo_rating"].last().to_dict()
    )

    for fix in completed:
        team_a = fix["home_team"]
        team_b = fix["away_team"]
        score_a = int(fix["home_score"])
        score_b = int(fix["away_score"])
        match_date = pd.Timestamp(fix["date"]).tz_localize(None).normalize()

        elo_a = current_elo.get(team_a, 1500.0)
        elo_b = current_elo.get(team_b, 1500.0)

        p_a = _elo_win_prob(elo_a, elo_b)
        outcome = _get_outcome(score_a, score_b)

        new_elo_a = elo_a + K_FACTOR_WC * (outcome - p_a)
        new_elo_b = elo_b + K_FACTOR_WC * ((1 - outcome) - (1 - p_a))

        current_elo[team_a] = new_elo_a
        current_elo[team_b] = new_elo_b

        new_rows.extend(
            [
                {
                    "team": team_a,
                    "elo_rating": round(new_elo_a, 2),
                    "date": match_date,
                    "source": "wc2026_actual",
                },
                {
                    "team": team_b,
                    "elo_rating": round(new_elo_b, 2),
                    "date": match_date,
                    "source": "wc2026_actual",
                },
            ]
        )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated = pd.concat([elo_df, new_df], ignore_index=True)
        updated.to_parquet(ELO_PATH, index=False, engine="pyarrow")
        logger.info(
            f"Elo updated: {len(new_rows)} new rows for {len(completed)} matches"
        )
        return updated

    logger.info("No new Elo updates (no completed fixtures)")
    return elo_df
