"""XGBoost classifier with Optuna HPO.

Uses FEATURE_COLS_TREES (21 features).  HPO runs 100 Optuna trials using
TimeSeriesSplit(5) cross-validation on the training set, then calibrates
with isotonic regression.

Expected WC 2018 Brier: 0.190 – 0.205.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import numpy as np
import optuna
import xgboost as xgb
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split, get_tscv

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_PATH = Path("models/xgboost_v1.pkl")


def _objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("lr", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("col_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_cw", 1, 8),
        "reg_alpha":        trial.suggest_float("alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("lambda_", 0.5, 2.0),
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
    }
    clf = xgb.XGBClassifier(**params)
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=get_tscv(5),
        scoring="neg_log_loss",
        n_jobs=1,
    )
    return float(scores.mean())


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
) -> CalibratedClassifierCV:
    """Run Optuna HPO then train a calibrated XGBoost classifier.

    Args:
        X_train:  Training features.
        y_train:  Training labels.
        X_val:    Validation features (WC 2018) — used only for final reporting.
        y_val:    Validation labels.
        n_trials: Number of Optuna trials (default 100).

    Returns:
        Fitted CalibratedClassifierCV.
    """
    logger.info(f"XGBoost HPO: running {n_trials} Optuna trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_val = -study.best_value
    logger.info(f"Best CV log-loss: {best_val:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Re-map param names to XGBClassifier kwargs
    bp = study.best_params
    best_params = {
        "n_estimators":     bp["n_estimators"],
        "max_depth":        bp["max_depth"],
        "learning_rate":    bp["lr"],
        "subsample":        bp["subsample"],
        "colsample_bytree": bp["col_bytree"],
        "min_child_weight": bp["min_cw"],
        "reg_alpha":        bp["alpha"],
        "reg_lambda":       bp["lambda_"],
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    best_clf = xgb.XGBClassifier(**best_params)
    calibrated = CalibratedClassifierCV(best_clf, method="isotonic", cv=get_tscv(5))
    logger.info("Training XGBoost with best params + calibration...")
    calibrated.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return calibrated, study.best_params


def save_xgboost(model: CalibratedClassifierCV, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved XGBoost model → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_xgboost(path: Path = MODEL_PATH) -> CalibratedClassifierCV:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    mlflow.set_experiment("wc2026_phase3")

    X_train, y_train, X_val, y_val, _, _, feat_cols = get_split("trees")

    with mlflow.start_run(run_name="xgboost_v1"):
        model, best_params = train_xgboost(X_train, y_train, X_val, y_val, n_trials=100)

        proba_val = model.predict_proba(X_val)
        result = evaluate_model("XGBoost", y_val, proba_val, "val")

        mlflow.log_params(best_params)
        mlflow.log_metric("brier_val", result["brier"])
        mlflow.log_metric("log_loss_val", result["log_loss"])

    save_xgboost(model)
