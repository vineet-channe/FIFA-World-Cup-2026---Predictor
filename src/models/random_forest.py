"""Random Forest baseline model.

Uses FEATURE_COLS_TREES (21 features).  Calibrated with isotonic regression
via CalibratedClassifierCV over TimeSeriesSplit(5).

Expected WC 2018 Brier: 0.200 – 0.215.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split, get_tscv

MODEL_PATH = Path("models/random_forest_v1.pkl")


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CalibratedClassifierCV:
    """Train a calibrated Random Forest classifier.

    Args:
        X_train: Training features array.
        y_train: Training labels (0 / 1 / 2).

    Returns:
        Fitted CalibratedClassifierCV wrapping a RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=get_tscv(5))
    logger.info("Training Random Forest (calibrated, 500 trees)...")
    calibrated.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return calibrated


def get_feature_importances(
    calibrated: CalibratedClassifierCV,
    feat_cols: list[str],
) -> pd.Series:
    """Extract feature importances from the underlying RF (via calibrated wrapper).

    CalibratedClassifierCV trains one estimator per fold internally.
    We average importances across all folds' base estimators.
    """
    importances_list = []
    for cc in calibrated.calibrated_classifiers_:
        inner_rf = cc.estimator
        importances_list.append(inner_rf.feature_importances_)
    avg_importances = np.mean(importances_list, axis=0)
    return pd.Series(avg_importances, index=feat_cols).sort_values(ascending=False)


def save_random_forest(model: CalibratedClassifierCV, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved RF model → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_random_forest(path: Path = MODEL_PATH) -> CalibratedClassifierCV:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    mlflow.set_experiment("wc2026_phase3")

    X_train, y_train, X_val, y_val, _, _, feat_cols = get_split("trees")

    with mlflow.start_run(run_name="random_forest_v1"):
        model = train_random_forest(X_train, y_train)

        proba_val = model.predict_proba(X_val)
        result = evaluate_model("Random Forest", y_val, proba_val, "val")

        mlflow.log_params({
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "calibration": "isotonic",
        })
        mlflow.log_metric("brier_val", result["brier"])
        mlflow.log_metric("log_loss_val", result["log_loss"])

        importances = get_feature_importances(model, feat_cols)
        print("\nTop-10 feature importances:")
        print(importances.head(10).to_string())

    save_random_forest(model)
