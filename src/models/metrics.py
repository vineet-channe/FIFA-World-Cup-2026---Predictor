"""Evaluation metrics for multi-class match outcome prediction.

The primary metric is ``brier_score_multi`` — the average binary Brier score
across all three outcome classes (away win / draw / home win).

Lower Brier is better. Perfect = 0.0. Random ≈ 0.44. Naive baseline ≈ 0.235.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def brier_score_multi(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multi-class Brier score: average of per-class binary Brier scores.

    Args:
        y_true:  1-D array of class labels (0 = away win, 1 = draw, 2 = home win).
        y_proba: 2-D array of shape (n, 3).  Column order must be [P(away), P(draw), P(home)].

    Returns:
        Scalar float. Lower is better.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return float(np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in [0, 1, 2]
    ]))


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    split_name: str = "val",
) -> dict:
    """Compute and print Brier, log-loss, and accuracy for one model on one split.

    Args:
        model_name: Human-readable label (e.g. ``"XGBoost"``).
        y_true:     1-D array of true labels.
        y_proba:    2-D array of shape (n, 3).
        split_name: Label printed in the output line (``"val"`` or ``"test"``).

    Returns:
        Dict with keys ``model``, ``split``, ``brier``, ``log_loss``, ``accuracy``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    bs = brier_score_multi(y_true, y_proba)
    ll = log_loss(y_true, y_proba)
    preds = y_proba.argmax(axis=1)
    acc = float((preds == y_true).mean())

    print(
        f"{model_name:30s} [{split_name}] "
        f"Brier: {bs:.4f} | Log-loss: {ll:.4f} | Acc: {acc:.3f}"
    )
    return {
        "model":    model_name,
        "split":    split_name,
        "brier":    bs,
        "log_loss": ll,
        "accuracy": acc,
    }
