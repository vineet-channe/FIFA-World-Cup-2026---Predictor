"""Manual stacking ensemble.

Base models: XGBoost, LightGBM, Random Forest, MLP (FEATURE_COLS_TREES).
             Logistic Regression (FEATURE_COLS_LINEAR) — handled separately.
Meta-learner: LogisticRegression(C=0.5, multinomial).

Manual stacking (not sklearn StackingClassifier) because base models use
different feature sets (trees vs linear).

Expected WC 2018 Brier: 0.185 – 0.200.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.features.pipeline import FEATURE_COLS_LINEAR, FEATURE_COLS_TREES
from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split

ENSEMBLE_PATH = Path("models/ensemble_v1.pkl")


# ---------------------------------------------------------------------------
# Out-of-fold prediction generator
# ---------------------------------------------------------------------------

def generate_oof_predictions(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate out-of-fold (OOF) predictions for one base model.

    CRITICAL: uses TimeSeriesSplit — data is sorted by date, never shuffled.

    Args:
        model:    Unfitted estimator with .fit() and .predict_proba().
        X_train:  Training features array.
        y_train:  Training labels array.
        n_splits: Number of CV folds.

    Returns:
        Array of shape (len(X_train), 3) with OOF probability predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros((len(X_train), 3))

    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_arr)):
        logger.debug(f"  OOF fold {fold + 1}/{n_splits}  "
                     f"train={len(tr_idx):,}  val={len(val_idx):,}")
        try:
            import copy
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_arr[tr_idx], y_arr[tr_idx])
            oof_preds[val_idx] = fold_model.predict_proba(X_arr[val_idx])
        except Exception as exc:
            logger.warning(f"  Fold {fold + 1} failed: {exc} — using uniform proba")
            oof_preds[val_idx] = np.full((len(val_idx), 3), 1 / 3)

    return oof_preds


# ---------------------------------------------------------------------------
# Ensemble builder
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    X_train_trees:  pd.DataFrame,
    y_train:        np.ndarray,
    X_train_linear: pd.DataFrame,
    X_val_trees:    pd.DataFrame,
    y_val:          np.ndarray,
    X_val_linear:   pd.DataFrame,
) -> dict:
    """Build and fit the stacking ensemble.

    Loads base models from disk, generates OOF predictions on X_train,
    stacks them, and fits a LogisticRegression meta-learner.

    Args:
        X_train_trees:   Train features for tree models (21 cols).
        y_train:         Train labels.
        X_train_linear:  Train features for logistic model (15 cols).
        X_val_trees:     Validation features for tree models.
        y_val:           Validation labels.
        X_val_linear:    Validation features for logistic model.

    Returns:
        Dict with keys ``base_models``, ``meta_learner``,
        ``feat_cols_trees``, ``feat_cols_linear``.
    """
    # ---- Load base models ----
    from src.models.logistic_model import load_logistic
    from src.models.random_forest import load_random_forest
    from src.models.xgboost_model import load_xgboost
    from src.models.lightgbm_model import load_lightgbm
    from src.models.neural_network import load_neural_network

    logger.info("Loading base models from disk...")
    logistic_model = load_logistic()
    rf_model       = load_random_forest()
    xgb_model      = load_xgboost()
    lgb_model      = load_lightgbm()
    nn_model       = load_neural_network()

    base_models = [
        ("logistic",       logistic_model, "linear"),
        ("random_forest",  rf_model,       "trees"),
        ("xgboost",        xgb_model,      "trees"),
        ("lightgbm",       lgb_model,      "trees"),
        ("neural_network", nn_model,       "trees"),
    ]

    Xtr_t = np.asarray(X_train_trees,  dtype=float)
    Xtr_l = np.asarray(X_train_linear, dtype=float)

    # ---- Generate OOF predictions for each base model ----
    oof_parts: list[np.ndarray] = []
    for name, model, fset in base_models:
        logger.info(f"Generating OOF predictions for {name} ({fset} features)...")
        X_oof = Xtr_l if fset == "linear" else Xtr_t
        oof = generate_oof_predictions(model, X_oof, y_train, n_splits=5)
        oof_parts.append(oof)
        logger.info(f"  {name} OOF mean proba: {oof.mean(axis=0)}")

    meta_X_train = np.hstack(oof_parts)  # shape: (n_train, n_models × 3)

    # ---- Fit meta-learner ----
    logger.info("Fitting meta-learner (LogisticRegression)...")
    meta_learner = LogisticRegression(
        C=0.5,
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    meta_learner.fit(meta_X_train, y_train)

    ensemble = {
        "base_models":     [(name, model, fset) for name, model, fset in base_models],
        "meta_learner":    meta_learner,
        "feat_cols_trees": FEATURE_COLS_TREES,
        "feat_cols_linear": FEATURE_COLS_LINEAR,
    }

    # Quick validation report
    proba_val = predict_ensemble(ensemble, X_val_trees, X_val_linear)
    result = evaluate_model("Stacking Ensemble", y_val, proba_val, "val")
    logger.info(f"Ensemble WC 2018 Brier: {result['brier']:.4f}")

    return ensemble


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_ensemble(
    ensemble: dict,
    X_trees:  pd.DataFrame | np.ndarray,
    X_linear: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Generate ensemble predictions.

    Args:
        ensemble: Dict returned by build_stacking_ensemble().
        X_trees:  Features for tree/NN models (21 cols).
        X_linear: Features for logistic model (15 cols).

    Returns:
        Probability array of shape (n, 3).
    """
    Xt = np.asarray(X_trees,  dtype=float)
    Xl = np.asarray(X_linear, dtype=float)

    base_probas: list[np.ndarray] = []
    for name, model, fset in ensemble["base_models"]:
        X = Xl if fset == "linear" else Xt
        try:
            proba = model.predict_proba(X)
        except Exception as exc:
            logger.warning(f"Base model '{name}' predict_proba failed: {exc} — using uniform")
            proba = np.full((len(X), 3), 1 / 3)
        base_probas.append(proba)

    meta_X = np.hstack(base_probas)
    return ensemble["meta_learner"].predict_proba(meta_X)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ensemble(ensemble: dict, path: Path = ENSEMBLE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_ensemble(path: Path = ENSEMBLE_PATH) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mlflow.set_experiment("wc2026_phase3")

    X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, _ = get_split("trees")
    X_train_l, _,       X_val_l, _,     X_test_l, _,       _ = get_split("linear")

    with mlflow.start_run(run_name="ensemble_v1"):
        ensemble = build_stacking_ensemble(
            X_train_t, y_train, X_train_l,
            X_val_t,   y_val,   X_val_l,
        )

        proba_val = predict_ensemble(ensemble, X_val_t, X_val_l)
        result = evaluate_model("Stacking Ensemble", y_val, proba_val, "val")

        mlflow.log_metric("brier_val", result["brier"])
        mlflow.log_metric("log_loss_val", result["log_loss"])

    save_ensemble(ensemble)

    print(f"\nStacking Ensemble — WC 2018 Brier: {result['brier']:.4f}")
