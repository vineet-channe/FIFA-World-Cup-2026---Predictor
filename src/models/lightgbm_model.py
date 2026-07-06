"""LightGBM classifier — fast retrain variant.

Uses FEATURE_COLS_TREES (21 features).  Two functions:
  - train_lightgbm: HPO (50 trials) + calibration.  Must complete in < 60 s.
  - retrain_lightgbm: speed-over-calibration retrain for Phase 6 live pipeline.

Expected WC 2018 Brier: 0.190 – 0.207.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split, get_tscv

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_PATH = Path("models/lightgbm_v1.pkl")

FAST_RETRAIN_PARAMS: dict = {
    "num_leaves":        31,
    "learning_rate":     0.05,
    "n_estimators":      500,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.5,
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}


def _lgb_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 20, 60),
        "learning_rate":     trial.suggest_float("lr", 0.01, 0.15, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 200, 700),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("col_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("alpha", 0.0, 1.0),
        "reg_lambda":        trial.suggest_float("lambda_", 0.0, 2.0),
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "verbose":           -1,
        "n_jobs":            -1,
        "random_state":      42,
    }
    clf = lgb.LGBMClassifier(**params)
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=get_tscv(5),
        scoring="neg_log_loss",
        n_jobs=1,
    )
    return float(scores.mean())


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> CalibratedClassifierCV:
    """HPO (50 trials) + calibrated LightGBM.

    Times the full train and prints elapsed — must stay under 60 s.

    Returns:
        Fitted CalibratedClassifierCV.
    """
    t0 = time.time()
    logger.info(f"LightGBM HPO: running {n_trials} Optuna trials...")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _lgb_objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    bp = study.best_params
    best_params = {
        "num_leaves":        bp["num_leaves"],
        "learning_rate":     bp["lr"],
        "n_estimators":      bp["n_estimators"],
        "min_child_samples": bp["min_child_samples"],
        "subsample":         bp["subsample"],
        "colsample_bytree":  bp["col_bytree"],
        "reg_alpha":         bp["alpha"],
        "reg_lambda":        bp["lambda_"],
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "verbose":           -1,
        "n_jobs":            -1,
        "random_state":      42,
    }

    best_clf = lgb.LGBMClassifier(**best_params)
    calibrated = CalibratedClassifierCV(best_clf, method="isotonic", cv=get_tscv(5))
    logger.info("Training LightGBM with best params + calibration...")
    calibrated.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(f"LightGBM total time: {elapsed:.1f}s")
    if elapsed > 60:
        logger.warning(f"LightGBM took {elapsed:.0f}s — target is < 60 s")

    return calibrated, study.best_params


def retrain_lightgbm(
    X_all: np.ndarray,
    y_all: np.ndarray,
    wc_indices: list[int],
    weight_mult: float = 2.0,
) -> lgb.LGBMClassifier:
    """Fast live-retrain for Phase 6 (speed over calibration).

    Applies ``weight_mult`` sample weight to WC match rows.
    Returns a fitted (NOT calibrated) LGBMClassifier for speed.

    Args:
        X_all:      Full feature matrix (train + WC matches so far).
        y_all:      Full label array.
        wc_indices: Row indices of WC 2026 matches (to upweight).
        weight_mult: Sample weight multiplier for WC matches (default 2.0).

    Returns:
        Fitted lgb.LGBMClassifier.
    """
    weights = np.ones(len(y_all))
    weights[wc_indices] = weight_mult

    clf = lgb.LGBMClassifier(**FAST_RETRAIN_PARAMS)
    clf.fit(X_all, y_all, sample_weight=weights)
    return clf


def save_lightgbm(model: CalibratedClassifierCV, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved LightGBM model → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_lightgbm(path: Path = MODEL_PATH) -> CalibratedClassifierCV:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    mlflow.set_experiment("wc2026_phase3")

    X_train, y_train, X_val, y_val, _, _, feat_cols = get_split("trees")

    with mlflow.start_run(run_name="lightgbm_v1"):
        model, best_params = train_lightgbm(X_train, y_train, X_val, y_val, n_trials=50)

        proba_val = model.predict_proba(X_val)
        result = evaluate_model("LightGBM", y_val, proba_val, "val")

        mlflow.log_params(best_params)
        mlflow.log_metric("brier_val", result["brier"])
        mlflow.log_metric("log_loss_val", result["log_loss"])

    save_lightgbm(model)
