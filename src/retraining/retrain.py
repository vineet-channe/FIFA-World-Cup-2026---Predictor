"""
Fast LightGBM retrain using actual WC 2026 match results.
Completes in under 90 seconds.
"""
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from src.features.pipeline import FEATURE_COLS_TREES, build_feature_matrix
from src.models.metrics import brier_score_multi
from config.settings import settings

FM_PATH = settings.DATA_DIR / "processed" / "feature_matrix.parquet"
MATCHES_PATH = settings.DATA_DIR / "processed" / "matches_clean.parquet"
MODEL_DIR = settings.MODEL_DIR

WEIGHTS = {
    "wc2026_actual": 3.0,
    "wc_historical": 1.5,
    "default": 1.0,
}


def _get_sample_weights(fm: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(fm))
    wc2026_mask = (fm["tournament"].str.contains("FIFA World Cup", na=False)) & (
        fm["match_date"].dt.year == 2026
    )
    wc_hist_mask = (fm["tournament"].str.contains("FIFA World Cup", na=False)) & (
        fm["match_date"].dt.year < 2026
    )
    weights[wc2026_mask] = WEIGHTS["wc2026_actual"]
    weights[wc_hist_mask] = WEIGHTS["wc_historical"]
    return weights


def build_extended_feature_matrix(
    new_fixtures: list[dict],
    elo_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    squad_df: pd.DataFrame,
    fbref_shooting_df: pd.DataFrame,
    fbref_keeper_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append feature rows for new WC 2026 matches to the existing feature matrix.
    Uses the same build_feature_matrix function from Phase 2.
    Only processes matches not already in the feature matrix.
    """
    existing = pd.read_parquet(FM_PATH)
    existing["match_date"] = pd.to_datetime(existing["match_date"])
    existing_dates_teams = set(
        zip(
            existing["match_date"].dt.strftime("%Y-%m-%d"),
            existing["team_a"],
            existing["team_b"],
        )
    )

    to_add = [
        f
        for f in new_fixtures
        if f["status"] == "FT"
        and f["home_score"] is not None
        and (
            pd.Timestamp(f["date"]).strftime("%Y-%m-%d"),
            f["home_team"],
            f["away_team"],
        )
        not in existing_dates_teams
    ]

    if not to_add:
        logger.info("No new matches to add to feature matrix")
        return existing

    matches_df = pd.read_parquet(MATCHES_PATH)
    matches_df["date"] = pd.to_datetime(matches_df["date"]).dt.tz_localize(None)

    existing_match_keys = set(
        zip(
            matches_df["date"].dt.strftime("%Y-%m-%d"),
            matches_df["home_team"],
            matches_df["away_team"],
        )
    )

    new_rows = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(f["date"]).tz_localize(None).normalize(),
                "home_team": f["home_team"],
                "away_team": f["away_team"],
                "home_score": f["home_score"],
                "away_score": f["away_score"],
                "tournament": "FIFA World Cup",
                "neutral": True,
                "is_competitive": True,
            }
            for f in to_add
            if (
                pd.Timestamp(f["date"]).strftime("%Y-%m-%d"),
                f["home_team"],
                f["away_team"],
            )
            not in existing_match_keys
        ]
    )
    if new_rows.empty and len(to_add) == 0:
        logger.info("No new matches to add to feature matrix")
        return existing

    if not new_rows.empty:
        matches_df = pd.concat([matches_df, new_rows], ignore_index=True)
        matches_df = matches_df.sort_values("date").reset_index(drop=True)
        matches_df.to_parquet(MATCHES_PATH, index=False, engine="pyarrow")

    built = build_feature_matrix(
        matches_df=matches_df,
        elo_df=elo_df,
        rankings_df=rankings_df,
        squad_df=squad_df,
        fbref_shooting_df=fbref_shooting_df,
        fbref_keeper_df=fbref_keeper_df,
        start_year=2026,
    )
    built["match_date"] = pd.to_datetime(built["match_date"])
    new_keys = {
        (pd.Timestamp(f["date"]).strftime("%Y-%m-%d"), f["home_team"], f["away_team"])
        for f in to_add
    }
    new_features = built[
        built.apply(
            lambda r: (
                r["match_date"].strftime("%Y-%m-%d"),
                r["team_a"],
                r["team_b"],
            )
            in new_keys,
            axis=1,
        )
    ].copy()

    extended = pd.concat([existing, new_features], ignore_index=True)
    extended = extended.sort_values("match_date").reset_index(drop=True)
    extended.to_parquet(FM_PATH, index=False, engine="pyarrow")
    logger.info(
        f"Feature matrix extended: +{len(new_features)} rows (total {len(extended):,})"
    )
    return extended


def retrain_lightgbm(feature_matrix: pd.DataFrame) -> tuple[object, object]:
    """
    Retrain LightGBM from scratch on the extended feature matrix.
    WC 2026 matches are weighted 3x. Completes in under 90 seconds.

    WC 2022 rows (the original test set) are EXCLUDED from training.
    """
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from src.models.split import get_tscv

    train_mask = ~(
        (feature_matrix["match_date"] >= "2022-11-20")
        & (feature_matrix["match_date"] <= "2022-12-18")
    )
    train_df = feature_matrix[train_mask].copy()

    X_train = train_df[FEATURE_COLS_TREES].fillna(
        train_df[FEATURE_COLS_TREES].median()
    )
    y_train = train_df["outcome"].values
    w_train = _get_sample_weights(train_df)

    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    calibrated = CalibratedClassifierCV(lgbm, method="isotonic", cv=get_tscv(5))
    calibrated.fit(X_train, y_train, sample_weight=w_train)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = MODEL_DIR / f"lightgbm_live_{ts}.pkl"
    joblib.dump(calibrated, path)
    logger.info(f"LightGBM retrained and saved: {path}")
    return calibrated, path


def validate_on_recent_wc(
    model, feature_matrix: pd.DataFrame, n: int = 10
) -> float | None:
    """Evaluate Brier score on the most recent n WC 2026 matches."""
    wc2026 = feature_matrix[
        (feature_matrix["tournament"].str.contains("FIFA World Cup", na=False))
        & (feature_matrix["match_date"].dt.year == 2026)
    ].sort_values("match_date").tail(n)

    if len(wc2026) < 5:
        logger.info("Fewer than 5 WC 2026 matches — skipping Brier validation")
        return None

    X = wc2026[FEATURE_COLS_TREES].fillna(wc2026[FEATURE_COLS_TREES].median())
    y = wc2026["outcome"].values
    proba = model.predict_proba(X)
    brier = brier_score_multi(y, proba)
    logger.info(f"Brier on last {len(wc2026)} WC 2026 matches: {brier:.4f}")
    return brier
