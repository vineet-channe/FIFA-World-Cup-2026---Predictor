"""Master data collection pipeline — Phase 1.

Runs every collection step in sequence, skipping steps whose output
files already exist. After collection, normalises team names and writes
cleaned parquets to data/processed/.

Usage:
    python scripts/01_collect_all_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from config.team_names import WC2026_CANONICAL_TEAMS, normalise_dataframe
from src.collection.elo_scraper import download_all_elo
from src.collection.fbref_scraper import discover_team_ids, download_all_fbref
from src.collection.fifa_rankings import download_rankings
from src.collection.kaggle_results import download_results
from src.collection.schedule import build_schedule
from src.collection.transfermarkt import download_all_squads

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip(step: str, path: Path) -> bool:
    if path.exists():
        print(f"[SKIP] {step} — output already exists ({path})")
        return True
    return False


def _done(step: str, df_or_list: pd.DataFrame | list) -> None:
    n = len(df_or_list)
    print(f"[DONE] {step} — {n:,} records")


def _file_summary(label: str, path: Path) -> None:
    if path.exists():
        size_kb = path.stat().st_size / 1024
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
                rows = len(df)
            else:
                import json
                with open(path) as f:
                    data = json.load(f)
                rows = len(data)
            print(f"  {label:<40} {rows:>8,} rows   {size_kb:>8.1f} KB   {path}")
        except Exception as exc:
            print(f"  {label:<40} (could not read: {exc})")
    else:
        print(f"  {label:<40} MISSING")


# ---------------------------------------------------------------------------
# Step 1 — Kaggle international results
# ---------------------------------------------------------------------------
def step_results() -> pd.DataFrame | None:
    out = settings.DATA_DIR / "raw" / "results" / "results.parquet"
    if _skip("download_results()", out):
        return None
    try:
        df = download_results()
        _done("download_results()", df)
        return df
    except Exception as exc:
        print(f"[FAIL] download_results(): {exc}")
        logger.error(f"download_results() failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 2 — FIFA rankings
# ---------------------------------------------------------------------------
def step_rankings() -> pd.DataFrame | None:
    out = settings.DATA_DIR / "raw" / "rankings" / "rankings.parquet"
    if _skip("download_rankings()", out):
        return None
    try:
        df = download_rankings()
        _done("download_rankings()", df)
        return df
    except Exception as exc:
        print(f"[FAIL] download_rankings(): {exc}")
        logger.error(f"download_rankings() failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 3 — Elo ratings (resume=True skips already-scraped teams)
# ---------------------------------------------------------------------------
def step_elo() -> pd.DataFrame | None:
    try:
        df = download_all_elo(resume=True)
        if df is not None and not df.empty:
            _done("download_all_elo()", df)
        return df
    except Exception as exc:
        print(f"[FAIL] download_all_elo(): {exc}")
        logger.error(f"download_all_elo() failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 4 — Transfermarkt squad values
# ---------------------------------------------------------------------------
def step_transfermarkt() -> pd.DataFrame | None:
    out = settings.DATA_DIR / "raw" / "transfermarkt" / "squad_values.parquet"
    if _skip("download_all_squads()", out):
        return None
    try:
        df = download_all_squads()
        _done("download_all_squads()", df)
        return df
    except Exception as exc:
        print(f"[FAIL] download_all_squads(): {exc}")
        logger.error(f"download_all_squads() failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 5 — FBref: discover team IDs
# ---------------------------------------------------------------------------
def step_fbref_ids() -> dict | None:
    out_dir = settings.DATA_DIR / "raw" / "fbref"
    cache = out_dir / "team_ids.json"
    if _skip("discover_team_ids()", cache):
        try:
            import json
            with open(cache, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        ids = discover_team_ids(out_dir)
        print(f"[DONE] discover_team_ids() — {len(ids):,} team IDs")
        return ids
    except Exception as exc:
        print(f"[FAIL] discover_team_ids(): {exc}")
        logger.error(f"discover_team_ids() failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Step 6 — FBref: scrape shooting/keeper/misc stats
# ---------------------------------------------------------------------------
def step_fbref_stats(team_ids: dict) -> pd.DataFrame | None:
    out_dir = settings.DATA_DIR / "raw" / "fbref"
    if not team_ids:
        print("[SKIP] download_all_fbref() — no team IDs available")
        return None
    try:
        df = download_all_fbref(WC2026_CANONICAL_TEAMS, team_ids, out_dir)
        if df is not None and not df.empty:
            _done("download_all_fbref()", df)
        return df
    except Exception as exc:
        print(f"[FAIL] download_all_fbref(): {exc}")
        logger.error(f"download_all_fbref() failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 7 — WC 2026 schedule
# ---------------------------------------------------------------------------
def step_schedule() -> list | None:
    out = settings.DATA_DIR / "raw" / "schedule" / "wc2026_schedule.json"
    if _skip("build_schedule()", out):
        return None
    try:
        schedule = build_schedule()
        _done("build_schedule()", schedule)
        return schedule
    except Exception as exc:
        print(f"[FAIL] build_schedule(): {exc}")
        logger.error(f"build_schedule() failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 8 — Normalise team names → data/processed/
# ---------------------------------------------------------------------------
def step_normalise() -> None:
    processed = settings.DATA_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # Results
    results_raw = settings.DATA_DIR / "raw" / "results" / "results.parquet"
    if results_raw.exists():
        results = pd.read_parquet(results_raw)
        results = normalise_dataframe(results, ["home_team", "away_team"])
        results.to_parquet(processed / "matches_clean.parquet", index=False, engine="pyarrow")
        logger.info(f"Normalised matches → {processed / 'matches_clean.parquet'}")
    else:
        logger.warning("results.parquet not found — skipping matches normalisation")

    # Elo
    elo_raw = settings.DATA_DIR / "raw" / "elo" / "elo_history.parquet"
    if elo_raw.exists():
        elo = pd.read_parquet(elo_raw)
        elo = normalise_dataframe(elo, ["team", "opponent"])
        elo.to_parquet(processed / "elo_clean.parquet", index=False, engine="pyarrow")
        logger.info(f"Normalised elo → {processed / 'elo_clean.parquet'}")
    else:
        logger.warning("elo_history.parquet not found — skipping elo normalisation")

    # Rankings
    rankings_raw = settings.DATA_DIR / "raw" / "rankings" / "rankings.parquet"
    if rankings_raw.exists():
        rankings = pd.read_parquet(rankings_raw)
        rankings = normalise_dataframe(rankings, ["country_full"])
        rankings.to_parquet(processed / "rankings_clean.parquet", index=False, engine="pyarrow")
        logger.info(f"Normalised rankings → {processed / 'rankings_clean.parquet'}")
    else:
        logger.warning("rankings.parquet not found — skipping rankings normalisation")

    print("\nAll datasets normalised → data/processed/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("WC 2026 — Phase 1 Data Collection")
    print("=" * 60)

    step_results()
    step_rankings()
    step_elo()
    team_ids = step_fbref_ids()
    step_fbref_stats(team_ids or {})
    step_transfermarkt()
    step_schedule()

    print("\n" + "=" * 60)
    print("Normalising team names...")
    print("=" * 60)
    step_normalise()

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("File summary")
    print("=" * 60)
    files = [
        ("results/results.parquet",           settings.DATA_DIR / "raw" / "results"      / "results.parquet"),
        ("results/results.csv",               settings.DATA_DIR / "raw" / "results"      / "results.csv"),
        ("rankings/rankings.parquet",         settings.DATA_DIR / "raw" / "rankings"     / "rankings.parquet"),
        ("elo/elo_history.parquet",           settings.DATA_DIR / "raw" / "elo"          / "elo_history.parquet"),
        ("transfermarkt/squad_values.parquet",settings.DATA_DIR / "raw" / "transfermarkt"/ "squad_values.parquet"),
        ("fbref/team_ids.json",               settings.DATA_DIR / "raw" / "fbref"        / "team_ids.json"),
        ("fbref/fbref_all_combined.parquet",    settings.DATA_DIR / "raw" / "fbref"    / "fbref_all_combined.parquet"),
        ("schedule/wc2026_schedule.json",     settings.DATA_DIR / "raw" / "schedule"     / "wc2026_schedule.json"),
        ("processed/matches_clean.parquet",   settings.DATA_DIR / "processed"            / "matches_clean.parquet"),
        ("processed/elo_clean.parquet",       settings.DATA_DIR / "processed"            / "elo_clean.parquet"),
        ("processed/rankings_clean.parquet",  settings.DATA_DIR / "processed"            / "rankings_clean.parquet"),
    ]
    for label, path in files:
        _file_summary(label, path)

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
