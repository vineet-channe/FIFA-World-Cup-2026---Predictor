"""Phase 2 runner — build the feature matrix from all cleaned data sources.

Usage:
    python scripts/02_build_features.py

Expected runtime: 20–40 minutes (15,000+ matches × per-team historical lookups).
Output: data/processed/feature_matrix.parquet
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from src.features.pipeline import FEATURE_COLS, build_feature_matrix
from src.features.tactical_features import load_fbref_data, load_fbref_keeper_data

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load all source DataFrames                                        #
    # ------------------------------------------------------------------ #
    logger.info("Loading source data...")

    matches = pd.read_parquet("data/processed/matches_clean.parquet")
    matches["date"] = pd.to_datetime(matches["date"])
    logger.info(f"Matches: {len(matches):,} rows")

    elo = pd.read_parquet("data/processed/elo_clean.parquet")
    elo["date"] = pd.to_datetime(elo["date"])
    logger.info(f"Elo: {len(elo):,} rows, {elo['team'].nunique()} teams")

    rankings = pd.read_parquet("data/processed/rankings_clean.parquet")
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    logger.info(f"Rankings: {len(rankings):,} rows")

    squad = pd.read_parquet("data/raw/transfermarkt/squad_values.parquet")
    logger.info(f"Squad values: {len(squad)} teams")

    # ------------------------------------------------------------------ #
    # 2. Load FBref data                                                   #
    # ------------------------------------------------------------------ #
    logger.info("Loading FBref data...")
    fbref_dir = Path("data/raw/fbref")
    fbref_shooting = load_fbref_data(fbref_dir)
    fbref_keeper   = load_fbref_keeper_data(fbref_dir)
    logger.info(f"FBref shooting: {len(fbref_shooting):,} rows, {fbref_shooting['team'].nunique()} teams")
    logger.info(f"FBref keeper:   {len(fbref_keeper):,} rows")

    # ------------------------------------------------------------------ #
    # 3. Build feature matrix                                              #
    # ------------------------------------------------------------------ #
    fm = build_feature_matrix(
        matches_df=matches,
        elo_df=elo,
        rankings_df=rankings,
        squad_df=squad,
        fbref_shooting_df=fbref_shooting,
        fbref_keeper_df=fbref_keeper,
        start_year=2000,
    )

    # ------------------------------------------------------------------ #
    # 4. Per-column NaN summary                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("NaN counts per feature column:")
    null_counts = fm[FEATURE_COLS].isnull().sum()
    if null_counts.any():
        print(null_counts[null_counts > 0].sort_values(ascending=False).to_string())
    else:
        print("  (none)")

    # ------------------------------------------------------------------ #
    # 5. Leakage check                                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Leakage check (threshold: |corr| > 0.65):")
    found_leak = False
    for col in FEATURE_COLS:
        if fm[col].isnull().all():
            continue
        corr = fm[col].corr(fm["outcome"])
        if abs(corr) > 0.65:
            print(f"  ⚠️  LEAKAGE WARNING: {col} correlation = {corr:.4f}")
            found_leak = True
    if not found_leak:
        print("  ✅ Leakage check passed — no suspicious correlations")

    # ------------------------------------------------------------------ #
    # 6. Feature strength check                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Top 10 features by |correlation| with outcome:")
    corrs = fm[FEATURE_COLS].corrwith(fm["outcome"]).abs().sort_values(ascending=False)
    print(corrs.head(10).round(4).to_string())
    if corrs.index[0] != "elo_diff" and "elo_diff" not in corrs.head(3).index:
        print("  ⚠️  WARNING: elo_diff is not in top-3 — check Elo date filtering")

    print("\n" + "=" * 60)
    print(f"✅ Phase 2 complete. Feature matrix saved.")
    print(f"   Rows:    {len(fm):,}")
    print(f"   Columns: {len(fm.columns)}")
    print(f"   Path:    data/processed/feature_matrix.parquet")


if __name__ == "__main__":
    main()
