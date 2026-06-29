"""Download FIFA World Rankings history from Kaggle (DS-02)."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings

__all__ = ["download_rankings"]

_KAGGLE_DATASET = "cashncarry/fifaworldranking"


def download_rankings(output_dir: str | None = None) -> pd.DataFrame:
    """Download FIFA rankings from Kaggle, normalise columns, save parquet.

    Saves:
        data/raw/rankings/rankings.parquet

    Returns the cleaned DataFrame.
    Raises RuntimeError on Kaggle auth failure.
    """
    import kaggle

    out = Path(output_dir) if output_dir else settings.DATA_DIR / "raw" / "rankings"
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
                "  1. Go to: https://www.kaggle.com/datasets/cashncarry/fifaworldranking\n"
                "  2. Click 'Download' and accept the terms if prompted\n"
                "  3. Then re-run this script"
            ) from exc
        raise

    # The dataset may ship with varying filenames — pick the first CSV found
    csvs = sorted(glob.glob(str(out / "*.csv")))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {out} after Kaggle download."
        )
    logger.info(f"Downloaded files: {[Path(p).name for p in csvs]}")

    df = pd.read_csv(csvs[0])

    # ── Normalise column names ─────────────────────────────────────────────
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # ── Parse rank_date ────────────────────────────────────────────────────
    if "rank_date" in df.columns:
        df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    else:
        # Try to find a date-like column
        date_cols = [c for c in df.columns if "date" in c]
        if date_cols:
            df.rename(columns={date_cols[0]: "rank_date"}, inplace=True)
            df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")

    # ── Coerce numeric columns ─────────────────────────────────────────────
    for col in ["rank", "total_points", "rank_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parquet_path = out / "rankings.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(
        f"Saved {len(df):,} ranking records → {parquet_path}  "
        f"(latest date: {df['rank_date'].max().date() if 'rank_date' in df.columns else 'unknown'})"
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = download_rankings()
    print(f"\nRows:         {len(df):,}")
    print(f"Columns:      {df.columns.tolist()}")

    if "rank_date" in df.columns:
        latest_date = df["rank_date"].max()
        print(f"Latest date:  {latest_date.date()}")

        team_col = next(
            (c for c in ["country_full", "country", "team"] if c in df.columns), None
        )
        if team_col:
            rank_col = next(
                (c for c in ["rank", "rank_position"] if c in df.columns), None
            )
            latest = df[df["rank_date"] == latest_date]
            if rank_col:
                top5 = (
                    latest.sort_values(rank_col)
                    .head(5)[[team_col, rank_col, "total_points"]]
                    if "total_points" in df.columns
                    else latest.sort_values(rank_col).head(5)[[team_col, rank_col]]
                )
                print(f"\nTop-5 ranked teams ({latest_date.date()}):")
                print(top5.to_string(index=False))
