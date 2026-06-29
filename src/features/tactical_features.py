"""FBref-based tactical features: xG differential and SoT%.

Important: NaN is returned whenever data is unavailable — never fill with 0.
XGBoost and LightGBM handle NaN natively. A 0 would falsely signal equal teams.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_MIN_ROWS = 3  # minimum FBref rows required to return a non-NaN feature

_FBREF_DIR = Path("data/raw/fbref")


def load_fbref_data(fbref_dir: Path | None = None) -> pd.DataFrame:
    """Load all FBref shooting data into a single DataFrame.

    Loads the combined parquet (``fbref_all_combined.parquet`` or
    ``fbref_shooting_combined.parquet``) and supplements with any individual
    ``*_shooting.csv`` files not already present.  Deduplicates on
    (team, date, comp).

    Args:
        fbref_dir: Directory containing FBref files. Defaults to ``data/raw/fbref/``.

    Returns:
        Combined DataFrame with at least columns ``team``, ``date``, ``sotpct``.
    """
    if fbref_dir is None:
        fbref_dir = _FBREF_DIR

    dfs: list[pd.DataFrame] = []

    # Try both combined-parquet naming conventions
    for combined_name in ("fbref_all_combined.parquet", "fbref_shooting_combined.parquet"):
        combined_path = fbref_dir / combined_name
        if combined_path.exists():
            try:
                raw = pd.read_parquet(combined_path)
                # Filter to shooting table rows if the parquet mixes tables
                if "table" in raw.columns:
                    shooting = raw[raw["table"] == "shooting"].copy()
                else:
                    shooting = raw.copy()
                dfs.append(shooting)
                logger.debug(f"Loaded {len(shooting)} shooting rows from {combined_name}")
            except Exception as exc:
                logger.warning(f"Failed to load {combined_name}: {exc}")

    # Supplement with individual shooting CSVs
    csv_files = sorted(glob.glob(str(fbref_dir / "*_shooting.csv")))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as exc:
            logger.warning(f"Failed to load {csv_path}: {exc}")

    if not dfs:
        logger.warning("No FBref shooting data found — tactical features will be NaN")
        return pd.DataFrame(columns=["team", "date", "sotpct"])

    combined = pd.concat(dfs, ignore_index=True)

    # Normalise date column
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])

    # Deduplicate
    dedup_cols = ["team", "date"]
    if "comp" in combined.columns:
        dedup_cols.append("comp")
    elif "comp_id" in combined.columns:
        dedup_cols.append("comp_id")
    combined = combined.drop_duplicates(subset=dedup_cols)

    logger.info(f"FBref shooting data: {len(combined):,} rows, {combined['team'].nunique()} teams")
    return combined.reset_index(drop=True)


def load_fbref_keeper_data(fbref_dir: Path | None = None) -> pd.DataFrame:
    """Load all FBref goalkeeper data into a single DataFrame.

    Looks for keeper rows in the combined parquet and individual ``*_keeper.csv`` files.
    Extracts ``psxg`` (post-shot xG against) when available, or falls back to
    goals-against (``ga.1``) as a rough defensive proxy.

    Args:
        fbref_dir: Directory containing FBref files. Defaults to ``data/raw/fbref/``.

    Returns:
        Combined DataFrame with columns ``team``, ``date``, and available defensive cols.
    """
    if fbref_dir is None:
        fbref_dir = _FBREF_DIR

    dfs: list[pd.DataFrame] = []

    for combined_name in ("fbref_all_combined.parquet", "fbref_shooting_combined.parquet"):
        combined_path = fbref_dir / combined_name
        if combined_path.exists():
            try:
                raw = pd.read_parquet(combined_path)
                if "table" in raw.columns:
                    keeper = raw[raw["table"] == "keeper"].copy()
                    if not keeper.empty:
                        dfs.append(keeper)
            except Exception as exc:
                logger.warning(f"Failed to load keeper rows from {combined_name}: {exc}")

    csv_files = sorted(glob.glob(str(fbref_dir / "*_keeper.csv")))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if "table" not in df.columns:
                df["table"] = "keeper"
            dfs.append(df)
        except Exception as exc:
            logger.warning(f"Failed to load {csv_path}: {exc}")

    if not dfs:
        logger.warning("No FBref keeper data found")
        return pd.DataFrame(columns=["team", "date"])

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])

    dedup_cols = ["team", "date"]
    if "comp" in combined.columns:
        dedup_cols.append("comp")
    combined = combined.drop_duplicates(subset=dedup_cols)

    logger.info(f"FBref keeper data: {len(combined):,} rows, {combined['team'].nunique()} teams")
    return combined.reset_index(drop=True)


def get_team_fbref_window(
    fbref_df: pd.DataFrame,
    team: str,
    cut_date: pd.Timestamp,
    n: int = 10,
) -> pd.DataFrame:
    """Return the last *n* FBref rows for *team* strictly before *cut_date*.

    Args:
        fbref_df:  FBref DataFrame with ``team`` and ``date`` columns.
        team:      Canonical team name.
        cut_date:  Exclusive upper bound.
        n:         Number of rows to return.

    Returns:
        DataFrame sorted by date ascending (most recent last),
        or an empty DataFrame when no data exists.
    """
    team_rows = fbref_df[(fbref_df["team"] == team) & (fbref_df["date"] < cut_date)]
    if team_rows.empty:
        return pd.DataFrame()
    return team_rows.sort_values("date").tail(n)


def compute_tactical_features(
    fbref_shooting_df: pd.DataFrame,
    fbref_keeper_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    cut_date: pd.Timestamp,
    n: int = 10,
) -> dict:
    """Compute FBref-based tactical features for a matchup.

    - ``xg_diff_per_game``: team_a avg xG - team_b avg psxG (defensive quality).
      Returns NaN when xG data is unavailable (most nations lack xG columns).
    - ``sot_pct_diff``: team_a avg SoT% - team_b avg SoT%.
      Returns NaN when fewer than ``_MIN_ROWS`` exist for either team.

    Args:
        fbref_shooting_df: FBref shooting DataFrame with ``team``, ``date``, ``sotpct``.
        fbref_keeper_df:   FBref keeper DataFrame with ``team``, ``date``.
        team_a:            Canonical name of the primary team.
        team_b:            Canonical name of the opposing team.
        cut_date:          Exclusive upper bound.
        n:                 Window size (last n FBref matches).

    Returns:
        dict with keys: xg_diff_per_game, sot_pct_diff.
        Both may be NaN — this is expected and correct.
    """
    # --- SoT% diff (from shooting data) ---
    shoot_a = get_team_fbref_window(fbref_shooting_df, team_a, cut_date, n)
    shoot_b = get_team_fbref_window(fbref_shooting_df, team_b, cut_date, n)

    sot_a_avg: float | None = None
    sot_b_avg: float | None = None

    if len(shoot_a) >= _MIN_ROWS and "sotpct" in shoot_a.columns:
        vals = pd.to_numeric(shoot_a["sotpct"], errors="coerce").dropna()
        if len(vals) >= _MIN_ROWS:
            sot_a_avg = float(vals.mean())

    if len(shoot_b) >= _MIN_ROWS and "sotpct" in shoot_b.columns:
        vals = pd.to_numeric(shoot_b["sotpct"], errors="coerce").dropna()
        if len(vals) >= _MIN_ROWS:
            sot_b_avg = float(vals.mean())

    sot_pct_diff = (sot_a_avg - sot_b_avg) if (sot_a_avg is not None and sot_b_avg is not None) else np.nan

    # --- xG diff (requires xg column — not available in most scraped data) ---
    # Try shooting xG for team_a vs keeper psxg for team_b
    xg_a_avg: float | None = None
    psxg_b_avg: float | None = None

    if len(shoot_a) >= _MIN_ROWS:
        for col in ("xg", "npxg", "xG", "npxG"):
            if col in shoot_a.columns:
                vals = pd.to_numeric(shoot_a[col], errors="coerce").dropna()
                if len(vals) >= _MIN_ROWS:
                    xg_a_avg = float(vals.mean())
                    break

    keep_b = get_team_fbref_window(fbref_keeper_df, team_b, cut_date, n)
    if len(keep_b) >= _MIN_ROWS:
        for col in ("psxg", "psxG", "xga", "xGA"):
            if col in keep_b.columns:
                vals = pd.to_numeric(keep_b[col], errors="coerce").dropna()
                if len(vals) >= _MIN_ROWS:
                    psxg_b_avg = float(vals.mean())
                    break
        # Fallback: use goals-against as a rough proxy if psxg not available
        # We intentionally do NOT use ga as a proxy — it conflates defense with luck.
        # Keep xg_diff as NaN when true xG data is missing.

    xg_diff_per_game = (xg_a_avg - psxg_b_avg) if (xg_a_avg is not None and psxg_b_avg is not None) else np.nan

    return {
        "xg_diff_per_game": xg_diff_per_game,
        "sot_pct_diff":     sot_pct_diff,
    }


if __name__ == "__main__":
    from pathlib import Path

    fbref_dir = Path("data/raw/fbref")
    shooting = load_fbref_data(fbref_dir)
    keeper = load_fbref_keeper_data(fbref_dir)
    print(f"Shooting rows: {len(shooting):,}, teams: {shooting['team'].nunique()}")
    print(f"Keeper rows: {len(keeper):,}")

    cut = pd.Timestamp("2026-06-01")
    result = compute_tactical_features(shooting, keeper, "Brazil", "France", cut)
    print(f"\nBrazil vs France tactical features (cut {cut.date()}):")
    print(result)
