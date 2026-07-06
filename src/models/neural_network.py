"""MLP neural network for match outcome prediction (PyTorch).

Uses FEATURE_COLS_TREES (21 features).  The SklearnMLP wrapper exposes
.fit() / .predict_proba() so it integrates with the stacking ensemble.

Expected WC 2018 Brier: 0.198 – 0.212.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.metrics import brier_score_multi, evaluate_model
from src.models.split import get_split

MODEL_PATH = Path("models/neural_network_v1.pkl")


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class MatchMLP(nn.Module):
    """Multi-layer perceptron for 3-class match outcome prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits — softmax applied externally


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class SklearnMLP:
    """sklearn-compatible wrapper around MatchMLP.

    Stores a fitted StandardScaler internally so predict_proba() handles
    scaling consistently without the caller needing to remember.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 0.001,
        epochs: int = 150,
        batch_size: int = 64,
        patience: int = 15,
    ) -> None:
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience

        self.scaler_: StandardScaler | None = None
        self.model_:  MatchMLP | None = None
        self.device_: torch.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnMLP":
        """Train the MLP with early stopping on an internal (time-ordered) val set.

        Args:
            X: Feature array (NaN-free — caller must impute before calling fit).
            y: Label array (0 / 1 / 2).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # Internal val split: last 10% of data (time-ordered, not shuffled)
        val_n = max(1, int(len(X) * 0.10))
        X_tr, X_iv = X[:-val_n], X[-val_n:]
        y_tr, y_iv = y[:-val_n], y[-val_n:]

        # Scale using training portion only
        self.scaler_ = StandardScaler()
        X_tr = self.scaler_.fit_transform(X_tr)
        X_iv = self.scaler_.transform(X_iv)

        tr_ds = TensorDataset(
            torch.from_numpy(X_tr).to(self.device_),
            torch.from_numpy(y_tr).to(self.device_),
        )
        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=False)

        X_iv_t = torch.from_numpy(X_iv).to(self.device_)
        y_iv_t = torch.from_numpy(y_iv).to(self.device_)

        self.model_ = MatchMLP(self.input_dim, self.hidden_dims, self.dropout).to(self.device_)
        optimizer   = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion   = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        no_improve    = 0
        best_state    = None

        for epoch in range(self.epochs):
            self.model_.train()
            for X_batch, y_batch in tr_loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            # Validation loss for early stopping
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_iv_t)
                val_loss   = criterion(val_logits, y_iv_t).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                no_improve    = 0
                best_state    = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= self.patience:
                logger.debug(f"Early stopping at epoch {epoch + 1} (val_loss={best_val_loss:.4f})")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array of shape (n, 3).

        Applies the stored StandardScaler before forward pass.
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Model not fitted — call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        X_t = torch.from_numpy(X_scaled).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    # sklearn compatibility shim
    def get_params(self, deep: bool = True) -> dict:
        return {
            "input_dim":   self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout":     self.dropout,
            "lr":          self.lr,
            "epochs":      self.epochs,
            "batch_size":  self.batch_size,
            "patience":    self.patience,
        }

    def set_params(self, **params) -> "SklearnMLP":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    input_dim: int | None = None,
) -> SklearnMLP:
    """Train the MLP and evaluate on WC 2018 validation set.

    Args:
        X_train:   Training features (NaN-filled by split.py).
        y_train:   Training labels.
        X_val:     Validation features (WC 2018).
        y_val:     Validation labels.
        input_dim: Feature dimension (inferred from X_train if None).

    Returns:
        Fitted SklearnMLP.
    """
    if input_dim is None:
        input_dim = X_train.shape[1]

    mlp = SklearnMLP(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        lr=0.001,
        epochs=150,
        batch_size=64,
        patience=15,
    )

    logger.info("Training MLP neural network...")
    mlp.fit(np.asarray(X_train), np.asarray(y_train))
    logger.info("MLP training complete.")

    proba_val = mlp.predict_proba(np.asarray(X_val))
    brier = brier_score_multi(np.asarray(y_val), proba_val)
    logger.info(f"MLP WC 2018 Brier: {brier:.4f}")

    return mlp


def save_neural_network(model: SklearnMLP, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved NN model → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_neural_network(path: Path = MODEL_PATH) -> SklearnMLP:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    mlflow.set_experiment("wc2026_phase3")

    X_train, y_train, X_val, y_val, _, _, feat_cols = get_split("trees")

    with mlflow.start_run(run_name="neural_network_v1"):
        model = train_neural_network(
            np.asarray(X_train), y_train,
            np.asarray(X_val),   y_val,
        )

        proba_val = model.predict_proba(np.asarray(X_val))
        result = evaluate_model("Neural Network (MLP)", y_val, proba_val, "val")

        mlflow.log_params({
            "hidden_dims": "[128, 64, 32]",
            "dropout":     0.3,
            "lr":          0.001,
            "epochs":      150,
            "patience":    15,
        })
        mlflow.log_metric("brier_val", result["brier"])
        mlflow.log_metric("log_loss_val", result["log_loss"])

    save_neural_network(model)
