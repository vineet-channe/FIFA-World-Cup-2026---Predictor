"""Train / validation / test split for the WC 2026 prediction system.

Temporal boundaries (immutable):
  Train : match_date <= 2018-06-13   (everything before WC 2018)
  Val   : 2018-06-14 – 2018-07-15   (WC 2018 — 64 matches)
  Test  : 2022-11-20 – 2022-12-18   (WC 2022 — 64 matches, touched once)

WC 2022 is the held-out test set.  Do NOT evaluate any individual model on it —
only the final stacking ensemble gets one shot at the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features.pipeline import (
    FEATURE_COLS_LINEAR,
    FEATURE_COLS_TREES,
    load_feature_matrix,
)

# ---------------------------------------------------------------------------
# Temporal boundaries — do not change these
# ---------------------------------------------------------------------------
TRAIN_END  = "2018-06-13"   # last day before WC 2018 group stage
VAL_START  = "2018-06-14"   # WC 2018 group stage starts
VAL_END    = "2018-07-15"   # WC 2018 Final
TEST_START = "2022-11-20"   # WC 2022 group stage starts
TEST_END   = "2022-12-18"   # WC 2022 Final


def get_split(
    feature_set: str = "trees",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    """Load the feature matrix and return temporal train / val / test splits.

    Args:
        feature_set: ``"trees"`` (21 features) or ``"linear"`` (15 features).

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test, feat_cols)``

    NaN handling:
        Medians are computed on X_train only and applied to all three splits.
        This prevents any leakage of future distribution information.
    """
    fm, _ = load_feature_matrix()
    fm = fm.sort_values("match_date").reset_index(drop=True)
    fm["match_date"] = pd.to_datetime(fm["match_date"])

    feat_cols: list[str] = FEATURE_COLS_TREES if feature_set == "trees" else FEATURE_COLS_LINEAR

    train_mask = fm["match_date"] <= TRAIN_END
    val_mask   = (fm["match_date"] >= VAL_START) & (fm["match_date"] <= VAL_END)
    test_mask  = (fm["match_date"] >= TEST_START) & (fm["match_date"] <= TEST_END)

    X_train = fm.loc[train_mask, feat_cols].copy()
    y_train = fm.loc[train_mask, "outcome"].to_numpy()

    X_val   = fm.loc[val_mask,   feat_cols].copy()
    y_val   = fm.loc[val_mask,   "outcome"].to_numpy()

    X_test  = fm.loc[test_mask,  feat_cols].copy()
    y_test  = fm.loc[test_mask,  "outcome"].to_numpy()

    # Median imputation — fit on training set only
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val   = X_val.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)

    print(
        f"[{feature_set}] Train: {len(X_train):,} | "
        f"Val (WC2018): {len(X_val)} | Test (WC2022): {len(X_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols


def get_tscv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return the project-standard cross-validation splitter.

    TimeSeriesSplit is the ONLY CV strategy allowed in this project.
    Never use KFold or StratifiedKFold — they would shuffle temporal data.
    """
    return TimeSeriesSplit(n_splits=n_splits)


if __name__ == "__main__":
    for fs in ("trees", "linear"):
        X_tr, y_tr, X_v, y_v, X_te, y_te, cols = get_split(fs)
        print(
            f"  X_train: {X_tr.shape}  X_val: {X_v.shape}  X_test: {X_te.shape}"
        )
