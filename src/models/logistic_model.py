"""Logistic regression baseline model.

Uses FEATURE_COLS_LINEAR (15 features).  Calibrated with Platt scaling
(sigmoid) via CalibratedClassifierCV over TimeSeriesSplit(5).

Expected WC 2018 Brier: 0.205 – 0.220.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split, get_tscv

MODEL_PATH = Path("models/logistic_v1.pkl")


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CalibratedClassifierCV:
    """Train a calibrated logistic regression classifier.

    Pipeline: StandardScaler → LogisticRegression.
    Wrapped in CalibratedClassifierCV(method='sigmoid', cv=TimeSeriesSplit(5)).

    Args:
        X_train: Training features array.
        y_train: Training labels (0 / 1 / 2).

    Returns:
        Fitted CalibratedClassifierCV.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=get_tscv(5))
    logger.info("Training Logistic Regression (calibrated)...")
    calibrated.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return calibrated


def save_logistic(model: CalibratedClassifierCV, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved logistic model → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_logistic(path: Path = MODEL_PATH) -> CalibratedClassifierCV:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    mlflow.set_experiment("wc2026_phase3")

    X_train, y_train, X_val, y_val, _, _, feat_cols = get_split("linear")

    # Naive baseline reference
    naive_proba = np.tile([0.30, 0.25, 0.45], (len(y_val), 1))
    naive_brier = brier_score_multi(y_val, naive_proba)
    print(f"Naive baseline Brier (WC 2018): {naive_brier:.4f}")

    with mlflow.start_run(run_name="logistic_v1"):
        model = train_logistic(X_train, y_train)

        proba_val = model.predict_proba(X_val)
        result = evaluate_model("Logistic Regression", y_val, proba_val, "val")

        mlflow.log_params({"C": 1.0, "solver": "lbfgs", "calibration": "sigmoid"})
        mlflow.log_metric("brier_val", result["brier"])
        mlflow.log_metric("log_loss_val", result["log_loss"])
        mlflow.log_metric("accuracy_val", result["accuracy"])

    save_logistic(model)

    if result["brier"] > naive_brier:
        print(f"WARNING: Logistic ({result['brier']:.4f}) did not beat naive ({naive_brier:.4f})")
    else:
        improvement = naive_brier - result["brier"]
        print(f"Beat naive by {improvement:.4f} ({improvement/naive_brier:.1%})")
