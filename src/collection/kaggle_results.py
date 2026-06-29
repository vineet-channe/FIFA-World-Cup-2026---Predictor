"""Download and clean the Kaggle international football results dataset (DS-01)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a script from anywhere in the repo
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings

__all__ = ["download_results"]

_KAGGLE_DATASET = "martj42/international-football-results"

_EXPECTED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
    "is_competitive",
]


def download_results(output_dir: str | None = None) -> pd.DataFrame:
    """Download Kaggle international results, add is_competitive flag, save parquet.

    Saves:
        data/raw/results/results.csv      — original Kaggle CSV
        data/raw/results/results.parquet  — cleaned, typed parquet

    Returns the cleaned DataFrame.
    Raises RuntimeError on Kaggle auth failure (401).
    """
    import kaggle  # imported lazily so the module can be imported without kaggle installed

    out = Path(output_dir) if output_dir else settings.DATA_DIR / "raw" / "results"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Authenticating with Kaggle API...")
    try:
        kaggle.api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed — check ~/.kaggle/kaggle.json exists "
            "and contains your username and key"
        ) from exc

    logger.info(f"Downloading dataset '{_KAGGLE_DATASET}'...")
    try:
        kaggle.api.dataset_download_files(
            _KAGGLE_DATASET, path=str(out), unzip=True, quiet=False
        )
    except Exception as exc:
        error_str = str(exc)
        if "401" in error_str or "Unauthorized" in error_str.lower():
            raise RuntimeError(
                "Kaggle authentication failed — check ~/.kaggle/kaggle.json exists "
                "and contains your username and key"
            ) from exc
        if "403" in error_str or "Forbidden" in error_str.lower():
            raise RuntimeError(
                "Kaggle 403 Forbidden — you need to accept the dataset terms first.\n"
                "  1. Go to: https://www.kaggle.com/datasets/martj42/international-football-results\n"
                "  2. Click 'Download' and accept the terms if prompted\n"
                "  3. Then re-run this script"
            ) from exc
        raise

    csv_path = out / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} after Kaggle download — check dataset name."
        )

    logger.info(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # ── Type coercion ──────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.array(df["home_score"], dtype="Int64")
    df["away_score"] = pd.array(df["away_score"], dtype="Int64")
    df["neutral"] = df["neutral"].astype(bool)

    # ── Derived column ─────────────────────────────────────────────────────
    df["is_competitive"] = ~df["tournament"].str.contains(
        r"\bFriendly\b", case=False, na=False, regex=True
    )

    # ── Column order ───────────────────────────────────────────────────────
    available = [c for c in _EXPECTED_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in _EXPECTED_COLUMNS]
    df = df[available + extra]

    parquet_path = out / "results.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(
        f"Saved {len(df):,} matches → {parquet_path}  "
        f"(date range: {df['date'].min().date()} → {df['date'].max().date()})"
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = download_results()
    print(f"\nRows:        {len(df):,}")
    print(f"Date range:  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Competitive: {df['is_competitive'].sum():,}")
    print(f"Columns:     {df.columns.tolist()}")
