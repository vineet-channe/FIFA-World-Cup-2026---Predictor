"""Transfermarkt squad value features for the WC 2026 predictor."""

import numpy as np
import pandas as pd
from loguru import logger

_FALLBACK_VALUE = 1e8  # €100M — neutral fallback when team not found


def compute_squad_features(
    squad_df: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> dict:
    """Compute squad market value features for a matchup.

    Squad values are static (single Transfermarkt snapshot). They do not vary
    by date — the 2025 value is used uniformly across the historical matrix.
    Log-transforms are applied to handle the heavy right-skew of market values.

    Args:
        squad_df: DataFrame with columns ``team``, ``total_market_value_eur``.
        team_a:   Canonical name of the primary team.
        team_b:   Canonical name of the opposing team.

    Returns:
        dict with keys: squad_log_value_a, squad_log_value_b,
        squad_log_value_diff, market_value_ratio.
    """
    row_a = squad_df[squad_df["team"] == team_a]
    row_b = squad_df[squad_df["team"] == team_b]

    if row_a.empty:
        logger.warning(f"Squad value not found for {team_a!r} — using €100M fallback")
        val_a = _FALLBACK_VALUE
    else:
        val_a = float(row_a["total_market_value_eur"].iloc[0])
        if val_a <= 0:
            val_a = _FALLBACK_VALUE

    if row_b.empty:
        logger.warning(f"Squad value not found for {team_b!r} — using €100M fallback")
        val_b = _FALLBACK_VALUE
    else:
        val_b = float(row_b["total_market_value_eur"].iloc[0])
        if val_b <= 0:
            val_b = _FALLBACK_VALUE

    log_val_a = np.log1p(val_a)
    log_val_b = np.log1p(val_b)

    return {
        "squad_log_value_a":    log_val_a,
        "squad_log_value_b":    log_val_b,
        "squad_log_value_diff": log_val_a - log_val_b,
        "market_value_ratio":   val_a / val_b if val_b > 0 else 1.0,
    }


if __name__ == "__main__":
    squad = pd.read_parquet("data/raw/transfermarkt/squad_values.parquet")
    print(squad.head(10))
    result = compute_squad_features(squad, "France", "Brazil")
    print("France vs Brazil squad features:", result)
