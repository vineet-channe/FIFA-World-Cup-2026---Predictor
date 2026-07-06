"""
One-time conversion: move the neural network's tensors and its stored
device attribute from MPS to CPU, then re-save both neural_network_v1.pkl
and ensemble_v1.pkl so they load correctly on ANY machine — including
Linux/Railway, which has no MPS backend at all.

Run this ONCE, locally, on this Mac (where MPS is available and these
files currently load without any issue). After this runs, both files use
plain, boring, fully portable pickle.load() everywhere, forever — no
special loading code needed anywhere in the codebase.

Usage:
    python scripts/fix_mps_models.py
"""
import sys
from pathlib import Path
import pickle
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NN_PATH = Path("models/neural_network_v1.pkl")
ENSEMBLE_PATH = Path("models/ensemble_v1.pkl")


def fix_sklearn_mlp(mlp) -> bool:
    """
    Move an SklearnMLP instance's model AND its stored device_ attribute
    to CPU, in place. Returns True if anything was changed.
    """
    changed = False

    if getattr(mlp, "model_", None) is not None:
        current_device = next(mlp.model_.parameters()).device
        if current_device.type != "cpu":
            mlp.model_ = mlp.model_.cpu()
            changed = True

    if getattr(mlp, "device_", None) is not None and mlp.device_.type != "cpu":
        mlp.device_ = torch.device("cpu")
        changed = True

    return changed


def main() -> None:
    # --- Fix standalone neural_network_v1.pkl ---
    print(f"Loading {NN_PATH} (loads normally here — MPS is available)...")
    with open(NN_PATH, "rb") as f:
        nn_model = pickle.load(f)

    if fix_sklearn_mlp(nn_model):
        with open(NN_PATH, "wb") as f:
            pickle.dump(nn_model, f)
        print(f"  Fixed and re-saved {NN_PATH} — now CPU-only.")
    else:
        print(f"  {NN_PATH} was already CPU-only, no change needed.")

    # --- Fix ensemble_v1.pkl (embeds a second copy of the NN model) ---
    print(f"Loading {ENSEMBLE_PATH} (loads normally here — MPS is available)...")
    with open(ENSEMBLE_PATH, "rb") as f:
        ensemble = pickle.load(f)

    any_changed = False
    for name, model, fset in ensemble["base_models"]:
        if name == "neural_network":
            if fix_sklearn_mlp(model):
                any_changed = True
                print(f"  Fixed embedded '{name}' model inside the ensemble.")

    if any_changed:
        with open(ENSEMBLE_PATH, "wb") as f:
            pickle.dump(ensemble, f)
        print(f"  Re-saved {ENSEMBLE_PATH} — now fully CPU-portable.")
    else:
        print(f"  {ENSEMBLE_PATH} was already CPU-only, no change needed.")

    # --- Verify: reload both and confirm no MPS tensors or device_ remain ---
    print("\nVerifying...")

    with open(NN_PATH, "rb") as f:
        check_nn = pickle.load(f)
    nn_param_device = next(check_nn.model_.parameters()).device
    print(f"  neural_network_v1.pkl model parameter device: {nn_param_device}  (expect cpu)")
    print(f"  neural_network_v1.pkl device_ attribute:       {check_nn.device_}  (expect cpu)")
    assert nn_param_device.type == "cpu", "Model parameters still not CPU!"
    assert check_nn.device_.type == "cpu", "device_ attribute still not CPU!"

    with open(ENSEMBLE_PATH, "rb") as f:
        check_ens = pickle.load(f)
    for name, model, fset in check_ens["base_models"]:
        if name == "neural_network":
            ens_param_device = next(model.model_.parameters()).device
            print(f"  ensemble's embedded NN parameter device: {ens_param_device}  (expect cpu)")
            print(f"  ensemble's embedded NN device_ attribute: {model.device_}  (expect cpu)")
            assert ens_param_device.type == "cpu", "Embedded model parameters still not CPU!"
            assert model.device_.type == "cpu", "Embedded device_ attribute still not CPU!"

    # --- End-to-end sanity check: run an actual prediction ---
    print("\nRunning an end-to-end prediction sanity check...")
    import numpy as np
    for name, model, fset in check_ens["base_models"]:
        if name == "neural_network":
            n_features = check_ens["feat_cols_trees"].__len__() if hasattr(check_ens["feat_cols_trees"], "__len__") else 21
            dummy_input = np.random.randn(2, n_features).astype(np.float32)
            proba = model.predict_proba(dummy_input)
            print(f"  Dummy prediction shape: {proba.shape}  (expect (2, 3))")
            assert proba.shape == (2, 3)

    print("\nBoth files are now CPU-portable. Safe to deploy to Linux/Railway.")
    print("No changes needed to load_ensemble() or load_neural_network() —")
    print("plain pickle.load() works correctly on these files from now on.")


if __name__ == "__main__":
    main()
