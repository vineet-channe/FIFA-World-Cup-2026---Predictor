"""Phase 3 master training script.

Trains all models in sequence, evaluates each on WC 2018 (validation),
and runs ONE final evaluation of the stacking ensemble on WC 2022 (test set).

Usage:
    python scripts/03_train_all_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

# --- Project root on sys.path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.pipeline import FEATURE_COLS_LINEAR, FEATURE_COLS_TREES
from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split


def _banner(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _step(n: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'─' * 60}")


# ===========================================================================
# Step 0 — Guard rails
# ===========================================================================
_step(0, "Verify feature lists")

assert len(FEATURE_COLS_TREES)  == 21, f"Expected 21, got {len(FEATURE_COLS_TREES)}"
assert len(FEATURE_COLS_LINEAR) == 15, f"Expected 15, got {len(FEATURE_COLS_LINEAR)}"
assert "avg_wc_finish_diff" not in FEATURE_COLS_LINEAR
print(f"Feature lists OK — Trees: {len(FEATURE_COLS_TREES)} | Linear: {len(FEATURE_COLS_LINEAR)}")

# ===========================================================================
# Step 1 — Load splits
# ===========================================================================
_step(1, "Load feature matrix and compute splits")

X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, _ = get_split("trees")
X_train_l, _,       X_val_l, _,     X_test_l, _,       _ = get_split("linear")

# ===========================================================================
# Step 2 — Naive baseline
# ===========================================================================
_step(2, "Compute naive baseline Brier on WC 2018")

naive_proba = np.tile([0.30, 0.25, 0.45], (len(y_val), 1))
naive_brier = brier_score_multi(y_val, naive_proba)
print(f"Naive baseline Brier (WC 2018): {naive_brier:.4f}")

results: dict[str, float] = {"naive_baseline_brier": naive_brier}

mlflow.set_experiment("wc2026_phase3")

# ===========================================================================
# Step 3 — Logistic Regression
# ===========================================================================
_step(3, "Train Logistic Regression")

from src.models.logistic_model import train_logistic, save_logistic

with mlflow.start_run(run_name="logistic_v1"):
    logistic_model = train_logistic(X_train_l, y_train)
    proba_lr = logistic_model.predict_proba(X_val_l)
    r_lr = evaluate_model("Logistic Regression", y_val, proba_lr, "val")
    mlflow.log_params({"C": 1.0, "solver": "lbfgs", "calibration": "sigmoid"})
    mlflow.log_metric("brier_val", r_lr["brier"])

save_logistic(logistic_model)
results["logistic_wc2018_brier"] = r_lr["brier"]

# ===========================================================================
# Step 4 — Random Forest
# ===========================================================================
_step(4, "Train Random Forest")

from src.models.random_forest import train_random_forest, get_feature_importances, save_random_forest

with mlflow.start_run(run_name="random_forest_v1"):
    rf_model = train_random_forest(X_train_t, y_train)
    proba_rf = rf_model.predict_proba(X_val_t)
    r_rf = evaluate_model("Random Forest", y_val, proba_rf, "val")
    mlflow.log_params({"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 5})
    mlflow.log_metric("brier_val", r_rf["brier"])
    importances = get_feature_importances(rf_model, FEATURE_COLS_TREES)
    print("\nTop-10 RF feature importances:")
    print(importances.head(10).to_string())

save_random_forest(rf_model)
results["random_forest_wc2018_brier"] = r_rf["brier"]

# ===========================================================================
# Step 5 — XGBoost
# ===========================================================================
_step(5, "Train XGBoost (Optuna 100 trials)")

from src.models.xgboost_model import train_xgboost, save_xgboost

with mlflow.start_run(run_name="xgboost_v1"):
    xgb_model, xgb_best_params = train_xgboost(X_train_t, y_train, X_val_t, y_val, n_trials=100)
    proba_xgb = xgb_model.predict_proba(X_val_t)
    r_xgb = evaluate_model("XGBoost", y_val, proba_xgb, "val")
    mlflow.log_params(xgb_best_params)
    mlflow.log_metric("brier_val", r_xgb["brier"])

save_xgboost(xgb_model)
results["xgboost_wc2018_brier"] = r_xgb["brier"]

# ===========================================================================
# Step 6 — LightGBM
# ===========================================================================
_step(6, "Train LightGBM (Optuna 50 trials)")

from src.models.lightgbm_model import train_lightgbm, save_lightgbm

with mlflow.start_run(run_name="lightgbm_v1"):
    lgb_model, lgb_best_params = train_lightgbm(X_train_t, y_train, X_val_t, y_val, n_trials=50)
    proba_lgb = lgb_model.predict_proba(X_val_t)
    r_lgb = evaluate_model("LightGBM", y_val, proba_lgb, "val")
    mlflow.log_params(lgb_best_params)
    mlflow.log_metric("brier_val", r_lgb["brier"])

save_lightgbm(lgb_model)
results["lightgbm_wc2018_brier"] = r_lgb["brier"]

# ===========================================================================
# Step 7 — Dixon-Coles
# ===========================================================================
_step(7, "Train Dixon-Coles (Poisson scoreline model)")

from src.models.dixon_coles import DixonColesModel

matches_df = pd.read_parquet("data/processed/matches_clean.parquet")

with mlflow.start_run(run_name="dixon_coles_v1"):
    dc_model = DixonColesModel().fit(matches_df)
    dc_brier = dc_model.evaluate_on_wc2018(matches_df)
    print(f"Dixon-Coles WC 2018 Brier: {dc_brier:.4f}")
    mlflow.log_metric("brier_val", dc_brier)
    mlflow.log_params({
        "home_adv": round(dc_model.home_adv_, 4),
        "rho":      round(dc_model.rho_, 4),
    })

dc_model.save()
results["dixon_coles_wc2018_brier"] = dc_brier

# ===========================================================================
# Step 8 — Neural Network
# ===========================================================================
_step(8, "Train Neural Network (MLP)")

from src.models.neural_network import train_neural_network, save_neural_network

with mlflow.start_run(run_name="neural_network_v1"):
    nn_model = train_neural_network(
        np.asarray(X_train_t), y_train,
        np.asarray(X_val_t),   y_val,
    )
    proba_nn = nn_model.predict_proba(np.asarray(X_val_t))
    r_nn = evaluate_model("Neural Network (MLP)", y_val, proba_nn, "val")
    mlflow.log_params({
        "hidden_dims": "[128, 64, 32]",
        "dropout": 0.3,
        "lr": 0.001,
        "epochs": 150,
        "patience": 15,
    })
    mlflow.log_metric("brier_val", r_nn["brier"])

save_neural_network(nn_model)
results["neural_net_wc2018_brier"] = r_nn["brier"]

# ===========================================================================
# Step 9 — Stacking Ensemble (WC 2018 evaluation)
# ===========================================================================
_step(9, "Build stacking ensemble")

from src.models.ensemble import (
    build_stacking_ensemble,
    predict_ensemble,
    save_ensemble,
    load_ensemble,
)

with mlflow.start_run(run_name="ensemble_v1"):
    ensemble = build_stacking_ensemble(
        X_train_t, y_train, X_train_l,
        X_val_t,   y_val,   X_val_l,
    )
    proba_ens_val = predict_ensemble(ensemble, X_val_t, X_val_l)
    r_ens_val = evaluate_model("Stacking Ensemble", y_val, proba_ens_val, "val")
    mlflow.log_metric("brier_val", r_ens_val["brier"])

save_ensemble(ensemble)
results["ensemble_wc2018_brier"] = r_ens_val["brier"]

# ===========================================================================
# Step 10 — FINAL TEST SET EVALUATION (WC 2022 — one time only)
# ===========================================================================
_step(10, "FINAL — Evaluate stacking ensemble on WC 2022 (test set)")

ensemble_loaded = load_ensemble()
proba_ens_test  = predict_ensemble(ensemble_loaded, X_test_t, X_test_l)
r_ens_test      = evaluate_model("Stacking Ensemble", y_test, proba_ens_test, "test (WC 2022)")

results["ensemble_wc2022_brier"] = r_ens_test["brier"]

with mlflow.start_run(run_name="ensemble_v1_test"):
    mlflow.log_metric("brier_test_wc2022", r_ens_test["brier"])
    mlflow.set_tag("final_evaluation", "true")

# ===========================================================================
# Step 11 — Comparison table
# ===========================================================================
_step(11, "Print comparison table")

_banner("MODEL TRAINING COMPLETE — Phase 3 Summary")

header = f"{'Model':<25} {'WC 2018 Brier':>14} {'vs Naive':>10} {'Status':>8}"
sep    = "─" * 62
print(header)
print(sep)

model_rows = [
    ("Naive baseline",     naive_brier,                      None),
    ("Logistic Regression", results["logistic_wc2018_brier"], naive_brier),
    ("Random Forest",       results["random_forest_wc2018_brier"], naive_brier),
    ("XGBoost",             results["xgboost_wc2018_brier"],  naive_brier),
    ("LightGBM",            results["lightgbm_wc2018_brier"], naive_brier),
    ("Dixon-Coles",         results["dixon_coles_wc2018_brier"], naive_brier),
    ("Neural Network",      results["neural_net_wc2018_brier"], naive_brier),
    ("Stacking Ensemble",   results["ensemble_wc2018_brier"], naive_brier),
]

for name, brier, ref in model_rows:
    if ref is None:
        diff_str = "—"
        status   = "reference"
    else:
        diff = brier - ref
        diff_str = f"{diff:+.4f}"
        status = "✅" if brier < ref else "❌"
    print(f"{name:<25} {brier:>14.4f} {diff_str:>10} {status:>8}")

print(sep)
ens_brier22 = results["ensemble_wc2022_brier"]
target_pass = "PASS ✅" if ens_brier22 < 0.210 else "FAIL ❌"
print(f"\nFINAL — Ensemble WC 2022 (test set): {ens_brier22:.4f}  [{target_pass}]")
print(f"Target: < 0.210 on WC 2022")
print("═" * 62)

# ===========================================================================
# Step 12 — Save model_comparison.json
# ===========================================================================
_step(12, "Save model_comparison.json")

results["phase3_complete"] = True
results["trained_at"]      = datetime.now(timezone.utc).isoformat()

out_path = Path("data/processed/model_comparison.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved → {out_path}")

logger.info("Phase 3 complete.")
