"""Resume Phase 3 from Step 7 (Dixon-Coles) — all prior models already saved.

Run after 03_train_all_models.py was interrupted mid-way (after LightGBM saved).
Steps 3-6 are already complete.  This script runs Steps 7-12 only.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.pipeline import FEATURE_COLS_LINEAR, FEATURE_COLS_TREES
from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split


def _step(n: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'─' * 60}")


mlflow.set_experiment("wc2026_phase3")

# Load splits
X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, _ = get_split("trees")
X_train_l, _,       X_val_l, _,     X_test_l, _,       _ = get_split("linear")

naive_proba = np.tile([0.30, 0.25, 0.45], (len(y_val), 1))
naive_brier = brier_score_multi(y_val, naive_proba)
print(f"Naive baseline Brier (WC 2018): {naive_brier:.4f}")

results: dict[str, float] = {"naive_baseline_brier": naive_brier}

# Load previously computed brier scores from saved models
from src.models.logistic_model import load_logistic
from src.models.random_forest import load_random_forest
from src.models.xgboost_model import load_xgboost
from src.models.lightgbm_model import load_lightgbm

lr_m  = load_logistic()
rf_m  = load_random_forest()
xgb_m = load_xgboost()
lgb_m = load_lightgbm()

results["logistic_wc2018_brier"]      = brier_score_multi(y_val, lr_m.predict_proba(X_val_l))
results["random_forest_wc2018_brier"] = brier_score_multi(y_val, rf_m.predict_proba(X_val_t))
results["xgboost_wc2018_brier"]       = brier_score_multi(y_val, xgb_m.predict_proba(X_val_t))
results["lightgbm_wc2018_brier"]      = brier_score_multi(y_val, lgb_m.predict_proba(X_val_t))

print(f"Logistic    WC2018 Brier: {results['logistic_wc2018_brier']:.4f}")
print(f"RF          WC2018 Brier: {results['random_forest_wc2018_brier']:.4f}")
print(f"XGBoost     WC2018 Brier: {results['xgboost_wc2018_brier']:.4f}")
print(f"LightGBM    WC2018 Brier: {results['lightgbm_wc2018_brier']:.4f}")

# ===========================================================================
# Step 7 — Dixon-Coles (vectorised)
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

# Sanity check
print(f"\nBrazil vs France: {dc_model.predict('Brazil', 'France')}")
print(f"England vs Germany: {dc_model.predict('England', 'Germany')}")

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
    mlflow.log_params({"hidden_dims": "[128, 64, 32]", "dropout": 0.3, "lr": 0.001})
    mlflow.log_metric("brier_val", r_nn["brier"])

save_neural_network(nn_model)
results["neural_net_wc2018_brier"] = r_nn["brier"]

# ===========================================================================
# Step 9 — Stacking Ensemble
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
# Step 10 — FINAL TEST: WC 2022
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
print(f"\n{'═' * 62}")
print("  MODEL TRAINING COMPLETE — Phase 3 Summary")
print(f"{'═' * 62}")

header = f"{'Model':<25} {'WC 2018 Brier':>14} {'vs Naive':>10} {'Status':>8}"
print(header)
print("─" * 62)

rows = [
    ("Naive baseline",      naive_brier,                            None),
    ("Logistic Regression", results["logistic_wc2018_brier"],      naive_brier),
    ("Random Forest",       results["random_forest_wc2018_brier"], naive_brier),
    ("XGBoost",             results["xgboost_wc2018_brier"],       naive_brier),
    ("LightGBM",            results["lightgbm_wc2018_brier"],      naive_brier),
    ("Dixon-Coles",         results["dixon_coles_wc2018_brier"],   naive_brier),
    ("Neural Network",      results["neural_net_wc2018_brier"],    naive_brier),
    ("Stacking Ensemble",   results["ensemble_wc2018_brier"],      naive_brier),
]

for name, brier, ref in rows:
    if ref is None:
        diff_str, status = "—", "reference"
    else:
        diff = brier - ref
        diff_str = f"{diff:+.4f}"
        status = "✅" if brier < ref else "❌"
    print(f"{name:<25} {brier:>14.4f} {diff_str:>10} {status:>8}")

print("─" * 62)
ens22 = results["ensemble_wc2022_brier"]
flag  = "PASS ✅" if ens22 < 0.210 else "FAIL ❌"
print(f"\nFINAL — Ensemble WC 2022 (test set): {ens22:.4f}  [{flag}]")
print(f"Target: < 0.210 on WC 2022")
print("═" * 62)

# ===========================================================================
# Step 12 — Save results
# ===========================================================================
results["phase3_complete"] = True
results["trained_at"]      = datetime.now(timezone.utc).isoformat()

out_path = Path("data/processed/model_comparison.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")
logger.info("Phase 3 complete.")
