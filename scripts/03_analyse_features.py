"""Phase 2.5 — Feature Analysis

Systematically analyses the 23 feature columns in the feature matrix and
produces two definitive feature lists:
  • FEATURE_COLS_TREES  — for XGBoost, LightGBM, RF, Neural Network
  • FEATURE_COLS_LINEAR — for Logistic Regression (baseline + meta-learner)

Writes results to:
  data/processed/feature_analysis.json
  src/features/pipeline.py (updates FEATURE_COLS_TREES / FEATURE_COLS_LINEAR)

Usage:
    python scripts/03_analyse_features.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

FM_PATH       = Path("data/processed/feature_matrix.parquet")
OUT_PATH      = Path("data/processed/feature_analysis.json")
PIPELINE_PATH = Path("src/features/pipeline.py")

# ─────────────────────────────────────────────────────────────────────────────
# Load and prepare
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading feature matrix...")
fm = pd.read_parquet(FM_PATH)
fm = fm.sort_values("match_date").reset_index(drop=True)

META    = ["match_date", "team_a", "team_b", "tournament", "is_neutral"]
TARGETS = ["outcome", "goals_a", "goals_b"]
ALL_FEATURES = [c for c in fm.columns if c not in META + TARGETS]

# Internal NaN-fill (median) — used ONLY for sklearn analyses, never persisted.
# Columns that are 100% NaN have no median; fill those with 0 as a safe neutral value.
col_medians = fm[ALL_FEATURES].median()
col_medians = col_medians.fillna(0)  # 100%-NaN columns get 0 as placeholder
fm_filled = fm[ALL_FEATURES].fillna(col_medians)
y = fm["outcome"]

logger.info(f"Feature matrix: {len(fm):,} rows × {len(ALL_FEATURES)} feature columns")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 — NaN Audit
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== ANALYSIS 1: NaN Audit ===")
print(f"{'Feature':<30} {'NaN count':>12} {'NaN %':>8}    Decision")
print("-" * 70)

nan_decisions: dict[str, str] = {}

null_counts = fm[ALL_FEATURES].isnull().sum()
null_pcts   = (null_counts / len(fm) * 100)

for feat in sorted(ALL_FEATURES, key=lambda f: null_pcts[f], reverse=True):
    n_nan = int(null_counts[feat])
    pct   = float(null_pcts[feat])
    if pct > 90:
        decision = "DROP_ALL"
    elif pct >= 50:
        decision = "TREES_ONLY"
    elif pct >= 10:
        decision = "KEEP_BOTH"
    else:
        decision = "KEEP_BOTH"
    nan_decisions[feat] = decision
    flag = "  ← DROP" if decision == "DROP_ALL" else ("  ← trees only" if decision == "TREES_ONLY" else "")
    print(f"{feat:<30} {n_nan:>12,} {pct:>7.1f}%    {decision}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Variance and Near-Zero Check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== ANALYSIS 2: Variance Check ===")
print(f"{'Feature':<30} {'Std Dev':>10} {'IQR':>10}    Flag")
print("-" * 70)

variance_flags: dict[str, str] = {}

for feat in ALL_FEATURES:
    std_val = float(fm_filled[feat].std())
    q75 = float(fm_filled[feat].quantile(0.75))
    q25 = float(fm_filled[feat].quantile(0.25))
    iqr = q75 - q25

    if feat == "rest_days_diff":
        flag = "LOW_VARIANCE_TRAINING_ONLY"
        note = "  ← all default to 4 days in training"
    elif std_val < 0.01:
        flag = "NEAR_ZERO_VARIANCE"
        note = "  ⚠️  near-constant"
    elif std_val < 0.5 and iqr < 0.5:
        flag = "LOW_VARIANCE"
        note = "  ⚠️  low variance"
    else:
        flag = "OK"
        note = ""

    variance_flags[feat] = flag
    print(f"{feat:<30} {std_val:>10.4f} {iqr:>10.4f}    {flag}{note}")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 — Pairwise Pearson Correlation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== ANALYSIS 3: High Correlation Pairs (|r| > 0.70) ===")
print(f"{'Feature A':<28} {'Feature B':<28} {'r':>7}    Linear action")
print("-" * 80)

corr_matrix = fm_filled[ALL_FEATURES].corr()
high_corr_pairs: list[dict] = []
drop_for_linear_set: set[str] = set()

# Collect all pairs with |r| > 0.70
seen_pairs: set[tuple] = set()
pair_list = []
for i, fa in enumerate(ALL_FEATURES):
    for j, fb in enumerate(ALL_FEATURES):
        if i >= j:
            continue
        r = corr_matrix.loc[fa, fb]
        if abs(r) > 0.70:
            pair_list.append((fa, fb, r))

pair_list.sort(key=lambda x: abs(x[2]), reverse=True)

for fa, fb, r in pair_list:
    # Determine drop rule for linear models
    # Mathematical subsets: elo_a / elo_b are components of elo_diff
    math_transforms = {
        ("elo_a",  "elo_diff"):            "elo_a",
        ("elo_b",  "elo_diff"):            "elo_b",
        ("market_value_ratio", "squad_log_value_diff"): "market_value_ratio",
        ("win_pct_diff",       "ppg_diff"):              "win_pct_diff",
        ("elo_a",  "elo_b"):               None,   # both subsumed by elo_diff, handle via VIF
    }
    drop_candidate = None
    action = "Keep both (different constructs)"

    for (a, b), drop in math_transforms.items():
        if (fa == a and fb == b) or (fa == b and fb == a):
            if drop:
                drop_candidate = drop
                action = f"Drop {drop} (mathematical component of the other)"
            break
    else:
        # High corr but not a known math transform — flag for VIF
        if abs(r) > 0.85:
            action = "Flag for VIF check"

    if drop_candidate:
        drop_for_linear_set.add(drop_candidate)

    high_corr_pairs.append({
        "feature_a": fa,
        "feature_b": fb,
        "r": round(float(r), 4),
        "drop_for_linear": drop_candidate,
        "action": action,
    })

    drop_str = f"  → drop {drop_candidate}" if drop_candidate else ""
    print(f"{fa:<28} {fb:<28} {r:>7.3f}    {action}{drop_str}")

if not pair_list:
    print("  (no pairs above threshold)")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4 — Random Forest Importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== ANALYSIS 4: Feature Importance ===")
logger.info("Training Random Forest on pre-2018 data...")

train_mask = fm["match_date"] < "2018-06-14"
val_mask   = (fm["match_date"] >= "2018-06-14") & (fm["match_date"] <= "2018-07-15")

X_train_filled = fm_filled.loc[train_mask, ALL_FEATURES]
y_train        = y[train_mask]
X_val_filled   = fm_filled.loc[val_mask, ALL_FEATURES]
y_val          = y[val_mask]

logger.info(f"Train: {train_mask.sum():,} | Val (WC2018): {val_mask.sum()}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_filled, y_train)
impurity_imp = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)

logger.info("Computing permutation importance on WC 2018 validation set (n_repeats=30)...")
perm = permutation_importance(
    rf, X_val_filled, y_val,
    n_repeats=30, random_state=42, n_jobs=-1,
)
perm_imp = pd.Series(perm.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)

# Rank by impurity for display
impurity_rank = {feat: i+1 for i, feat in enumerate(impurity_imp.index)}
perm_rank     = {feat: i+1 for i, feat in enumerate(perm_imp.index)}

print(f"{'Rank':<6} {'Feature':<28} {'RF Impurity':>13} {'Permutation':>13}")
print("-" * 65)
low_importance_features: set[str] = set()
for rank, feat in enumerate(impurity_imp.index, 1):
    imp_val  = float(impurity_imp[feat])
    perm_val = float(perm_imp[feat])
    flag = "  ⚠️  LOW" if perm_val < 0.001 else ""
    if perm_val < 0.001:
        low_importance_features.add(feat)
    print(f"{rank:<6} {feat:<28} {imp_val:>13.5f} {perm_val:>13.5f}{flag}")

rf_importance   = {k: round(float(v), 6) for k, v in impurity_imp.items()}
perm_importance = {k: round(float(v), 6) for k, v in perm_imp.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Analysis 5 — Mutual Information
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== ANALYSIS 5: Mutual Information ===")
logger.info("Computing mutual information...")

mi_scores = mutual_info_classif(
    X_train_filled, y_train,
    discrete_features=False,
    random_state=42,
)
mi = pd.Series(mi_scores, index=ALL_FEATURES).sort_values(ascending=False)
mi_rank      = {feat: i+1 for i, feat in enumerate(mi.index)}
pearson_corr = fm_filled[ALL_FEATURES].corrwith(y.astype(float)).abs()
pearson_rank = {feat: i+1 for i, feat in enumerate(pearson_corr.sort_values(ascending=False).index)}

print(f"{'Feature':<30} {'MI Score':>10} {'MI Rank':>9} {'|Pearson| Rank':>15} {'Delta':>8}")
print("-" * 75)
for feat in mi.index:
    mi_val  = float(mi[feat])
    mi_r    = mi_rank[feat]
    pear_r  = pearson_rank[feat]
    delta   = pear_r - mi_r    # positive = MI rank better (lower number) than Pearson
    delta_s = f"+{delta}" if delta > 0 else str(delta)
    note = "  ← non-linear signal" if delta > 4 else ""
    print(f"{feat:<30} {mi_val:>10.5f} {mi_r:>9} {pear_r:>15} {delta_s:>8}{note}")

mi_scores_dict = {k: round(float(v), 6) for k, v in mi.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Analysis 6 — VIF for Linear Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== ANALYSIS 6: VIF (Variance Inflation Factor) ===")
logger.info("Computing VIF on linear feature candidates...")

# Start with surviving features minus DROP_ALL and already-flagged linear drops
linear_candidates = [
    f for f in ALL_FEATURES
    if nan_decisions.get(f) != "DROP_ALL"
    and f not in drop_for_linear_set
]

def compute_vif(feature_list: list[str]) -> pd.DataFrame:
    """Compute VIF for a list of features using the NaN-filled training data."""
    # Use full dataset for VIF (need all variation represented)
    X = fm_filled[feature_list].copy()
    # Add small jitter to handle constant columns (avoids singular matrix)
    X = X + np.random.default_rng(42).normal(0, 1e-8, X.shape)
    vif_vals = []
    for i in range(len(feature_list)):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.inf
        vif_vals.append(v)
    return pd.DataFrame({"feature": feature_list, "VIF": vif_vals}).sort_values("VIF", ascending=False)

# Iteratively drop highest-VIF feature until all < 10
MAX_VIF = 10.0
iteration = 0
vif_drop_log: list[str] = []

while True:
    vif_df = compute_vif(linear_candidates)
    max_vif_row = vif_df.iloc[0]
    if max_vif_row["VIF"] <= MAX_VIF:
        break
    feat_to_drop = max_vif_row["feature"]
    logger.info(f"  VIF iteration {iteration+1}: dropping {feat_to_drop!r} (VIF={max_vif_row['VIF']:.2f})")
    drop_for_linear_set.add(feat_to_drop)
    vif_drop_log.append(feat_to_drop)
    linear_candidates = [f for f in linear_candidates if f != feat_to_drop]
    iteration += 1
    if iteration > 20:
        logger.warning("VIF iteration limit reached")
        break

# Recompute final VIF for display
vif_df = compute_vif(linear_candidates)
print(f"{'Feature':<30} {'VIF':>8}    Status")
print("-" * 55)
for _, row in vif_df.iterrows():
    v = row["VIF"]
    feat = row["feature"]
    if v > 10:
        status = "⚠️  HIGH — should have been dropped"
    elif v > 5:
        status = "⚠️  MODERATE (5–10)"
    else:
        status = "✅ OK (< 5)"
    print(f"{feat:<30} {v:>8.2f}    {status}")

vif_scores = vif_df.set_index("feature")["VIF"].round(3).to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# Final Feature Lists
# ─────────────────────────────────────────────────────────────────────────────

# Trees: all features except DROP_ALL
FEATURE_COLS_TREES: list[str] = [
    f for f in ALL_FEATURES
    if nan_decisions.get(f) != "DROP_ALL"
]

# Linear: trees set minus collinear / VIF-dropped features
FEATURE_COLS_LINEAR: list[str] = [
    f for f in FEATURE_COLS_TREES
    if f not in drop_for_linear_set
]

# Build drop_reasons dict for transparency
drop_reasons: dict[str, str] = {}

for feat in ALL_FEATURES:
    if nan_decisions.get(feat) == "DROP_ALL":
        pct = float(null_pcts[feat])
        drop_reasons[feat] = f"DROP_ALL: {pct:.1f}% NaN — no usable signal"

for feat in drop_for_linear_set:
    if feat not in drop_reasons:
        # Check if it came from math-transform rule
        math_reason = None
        for pair in high_corr_pairs:
            if pair["drop_for_linear"] == feat:
                other = pair["feature_b"] if pair["feature_a"] == feat else pair["feature_a"]
                math_reason = (
                    f"LINEAR_ONLY_DROP: mathematical component of {other!r} "
                    f"(r={pair['r']:.3f}) — subsumed by diff feature"
                )
                break
        if feat in vif_drop_log and not math_reason:
            drop_reasons[feat] = f"LINEAR_ONLY_DROP: VIF iteration removal (multicollinearity)"
        elif math_reason:
            drop_reasons[feat] = math_reason
        else:
            drop_reasons[feat] = "LINEAR_ONLY_DROP: high correlation with another retained feature"

# ─────────────────────────────────────────────────────────────────────────────
# Print Final Summary
# ─────────────────────────────────────────────────────────────────────────────
sep = "═" * 70
print(f"\n{sep}")
print("FINAL FEATURE LISTS")
print(sep)

print(f"\nFEATURE_COLS_TREES  ({len(FEATURE_COLS_TREES)} features)")
print("Used by: XGBoost, LightGBM, RF, Neural Network")
dropped_all = [f for f in ALL_FEATURES if nan_decisions.get(f) == "DROP_ALL"]
if dropped_all:
    print("  Dropped from ALL_FEATURES:")
    for feat in dropped_all:
        print(f"    ✗ {feat:<28}  {drop_reasons.get(feat, '')}")
print("  Retained:")
for feat in FEATURE_COLS_TREES:
    note = ""
    if nan_decisions.get(feat) == "TREES_ONLY":
        note = "  (TREES_ONLY — sparse NaN)"
    if variance_flags.get(feat) == "LOW_VARIANCE_TRAINING_ONLY":
        note = "  (low variance in training; informative in live tournament)"
    print(f"    ✓ {feat:<28}{note}")

print(f"\nFEATURE_COLS_LINEAR  ({len(FEATURE_COLS_LINEAR)} features)")
print("Used by: Logistic Regression (baseline + stacking meta-learner)")
linear_dropped = [f for f in FEATURE_COLS_TREES if f in drop_for_linear_set]
if linear_dropped:
    print("  Also dropped vs TREES:")
    for feat in linear_dropped:
        print(f"    ✗ {feat:<28}  {drop_reasons.get(feat, '')}")
print("  Retained:")
for feat in FEATURE_COLS_LINEAR:
    note = ""
    if nan_decisions.get(feat) == "TREES_ONLY":
        note = "  (median imputed for linear models)"
    print(f"    ✓ {feat:<28}{note}")

print(f"\nNotes on special features:")
for feat in FEATURE_COLS_TREES:
    vf = variance_flags.get(feat, "OK")
    nd = nan_decisions.get(feat, "KEEP_BOTH")
    if vf == "LOW_VARIANCE_TRAINING_ONLY":
        print(f"  ~ {feat:<28}: low variance in training data; informative in live tournament only")
    if nd == "TREES_ONLY" and feat in FEATURE_COLS_LINEAR:
        print(f"  ~ {feat:<28}: TREES_ONLY NaN band — median-imputed for linear models")
print(sep)

# ─────────────────────────────────────────────────────────────────────────────
# Update src/features/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
logger.info(f"Updating {PIPELINE_PATH}...")

pipeline_src = PIPELINE_PATH.read_text()

trees_lines  = ",\n    ".join(f'"{f}"' for f in FEATURE_COLS_TREES)
linear_lines = ",\n    ".join(f'"{f}"' for f in FEATURE_COLS_LINEAR)

new_constants = f'''
# ---------------------------------------------------------------------------
# Definitive feature lists — generated by scripts/03_analyse_features.py
# ---------------------------------------------------------------------------

# All features with usable signal (>= 1 non-NaN row).
# XGBoost, LightGBM, RF, Neural Network — handle NaN natively.
FEATURE_COLS_TREES: list[str] = [
    {trees_lines}
]

# Reduced set for logistic regression — multicollinear and redundant features
# removed based on Pearson |r| > 0.85 and iterative VIF pruning (threshold 10).
# NaN values in this set are median-imputed at train time.
FEATURE_COLS_LINEAR: list[str] = [
    {linear_lines}
]

# Backwards-compatible alias imported by existing pipeline / Phase 3 code.
FEATURE_COLS = FEATURE_COLS_TREES

'''

# Locate the existing FEATURE_COLS block and replace it
import re
# Pattern: from "# ----" comment block through "assert len(FEATURE_COLS)" line
pattern = re.compile(
    r"# -{10,}.*?Column definitions.*?# -{10,}\n.*?assert len\(FEATURE_COLS\).*?\n",
    re.DOTALL,
)

if pattern.search(pipeline_src):
    new_src = pattern.sub(new_constants.lstrip("\n"), pipeline_src)
else:
    # Fallback: replace just the FEATURE_COLS list and assert
    old_block_pattern = re.compile(
        r"# -{10,}.*?Column definitions.*?# -{10,}\n(.*?)assert len\(FEATURE_COLS\).*?\n",
        re.DOTALL,
    )
    if old_block_pattern.search(pipeline_src):
        new_src = old_block_pattern.sub(new_constants.lstrip("\n"), pipeline_src)
    else:
        # Last resort: append before build_feature_matrix definition
        insert_marker = "\ndef _build_team_elo_index"
        new_src = pipeline_src.replace(insert_marker, new_constants + insert_marker)

PIPELINE_PATH.write_text(new_src)
logger.info(f"pipeline.py updated with {len(FEATURE_COLS_TREES)} tree features and {len(FEATURE_COLS_LINEAR)} linear features")

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON Summary
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "total_features_before": len(ALL_FEATURES),
    "feature_cols_trees":  FEATURE_COLS_TREES,
    "feature_cols_linear": FEATURE_COLS_LINEAR,
    "nan_decisions":       nan_decisions,
    "variance_flags":      variance_flags,
    "high_corr_pairs":     high_corr_pairs,
    "rf_importance":       rf_importance,
    "perm_importance":     perm_importance,
    "mi_scores":           mi_scores_dict,
    "vif_scores":          {k: (round(v, 3) if not np.isinf(v) else 999) for k, v in vif_scores.items()},
    "drop_reasons":        drop_reasons,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAnalysis saved to {OUT_PATH}")
print(f"pipeline.py updated: FEATURE_COLS_TREES ({len(FEATURE_COLS_TREES)}) + FEATURE_COLS_LINEAR ({len(FEATURE_COLS_LINEAR)})")
print("\n✅ Phase 2.5 complete — run verification checks before proceeding to Phase 3.")
