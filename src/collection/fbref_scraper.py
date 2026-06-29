"""Scrape per-match advanced stats for WC 2026 nations from FBref (DS-07).

FBref is protected by Cloudflare's JS challenge.  Plain HTTP clients (requests,
curl_cffi, headless Playwright) are all detected and blocked with HTTP 403.
The only reliable approach is to drive a **real, non-headless Chrome** browser
via ``nodriver``, which patches the Chromium binary to hide automation signals.

Two-phase design:
  1. discover_team_ids()  — return canonical-name → 8-char FBref hex ID map
  2. download_all_fbref() — scrape shooting/keeper/misc tables per team
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from io import StringIO
from pathlib import Path

import nodriver as uc
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings

__all__ = ["discover_team_ids", "download_all_fbref"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE = "https://fbref.com"
FBREF_HOME = f"{BASE}/en/"

# All-competition matchlogs URL (no comp filter — shows all international matches)
# season examples: "2024-2025", "2023-2024"
# all_comps variant includes per-match stat columns (Sh, SoT, xG for shooting etc.)
# The plain /matchlogs/{table}/ URL only returns the schedule/score table.
_MATCHLOG_URL = f"{BASE}/en/squads/{{team_id}}/{{season}}/matchlogs/all_comps/{{table}}/"

# Stat table slugs to download
STAT_TABLES: list[str] = ["shooting", "keeper", "misc"]

# Default seasons to try (most recent first).
# European teams use hyphenated seasons (2024-2025); CONMEBOL/AFC/CAF use
# calendar years (2024, 2023).  We try all formats — FBref returns a tiny
# page for seasons with no data, which the scraper gracefully skips.
DEFAULT_SEASONS = ["2024-2025", "2024", "2023-2024", "2023"]

# Regex for a valid YYYY-MM-DD date — used to keep only real match rows
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Known FBref national team IDs — stable permanent hex IDs.
#
# FBref blocks automated confederation-page scraping via Cloudflare (403),
# so we maintain a hardcoded seed map of the 48 WC 2026 nations.
# IDs can be verified at: https://fbref.com/en/squads/{id}/
# ---------------------------------------------------------------------------
_KNOWN_TEAM_IDS: dict[str, str] = {
    # ── UEFA ──────────────────────────────────────────────────────────────────
    # All IDs verified by navigating to competition/search pages with nodriver
    "France":               "b1b36dcd",
    "Spain":                "b561dd30",
    "England":              "1862c019",
    "Germany":              "c1e40422",
    "Portugal":             "4a1b4ea8",
    "Netherlands":          "5bb5024a",
    "Belgium":              "361422b9",
    "Croatia":              "7b08e376",
    "Switzerland":          "81021a70",
    "Austria":              "d5121f10",
    "Norway":               "599eba19",
    "Sweden":               "296f69e7",
    "Scotland":             "602d3994",
    "Turkey":               "ac6bcf92",
    "Czechia":              "2740937c",
    "Bosnia & Herzegovina": "6c5ef1c3",
    # ── CONMEBOL ──────────────────────────────────────────────────────────────
    "Argentina":            "f9fddd6e",
    "Brazil":               "304635c3",
    "Colombia":             "ab73cfe5",
    "Uruguay":              "870e020f",
    "Ecuador":              "123acaf8",
    "Paraguay":             "d2043442",
    # ── CAF ───────────────────────────────────────────────────────────────────
    "Morocco":              "af41ccda",
    "Senegal":              "9ab5c684",
    "Algeria":              "1e2dba57",
    "Egypt":                "b8889750",
    "Ivory Coast":          "24772b12",
    "Ghana":                "9349828d",
    "DR Congo":             "9be9f315",
    "Tunisia":              "a7c7562a",
    "South Africa":         "506f1741",
    "Cape Verde":           "31fa6fa6",   # FBref lists as "Cabo Verde"
    # ── AFC ───────────────────────────────────────────────────────────────────
    "Japan":                "ffcf1690",
    "South Korea":          "473f0fbf",
    "Iran":                 "6a08f71e",
    "Saudi Arabia":         "6e84edac",
    "Iraq":                 "ec843efd",
    "Jordan":               "3e22f0fa",
    "Uzbekistan":           "cd389e75",
    "Qatar":                "9b696ed1",
    # ── CONCACAF ──────────────────────────────────────────────────────────────
    "USA":                  "0f66725b",
    "Mexico":               "b009a548",
    "Canada":               "9c6d90a0",
    "Panama":               "6061a82d",
    "Haiti":                "61828292",
    "Curaçao":              "e0f5893a",
    # ── OFC / AFC-via-OFC ─────────────────────────────────────────────────────
    "New Zealand":          "259855f0",
    "Australia":            "b90bf4f9",
}


# ---------------------------------------------------------------------------
# Phase 1: discover team IDs (no network call needed)
# ---------------------------------------------------------------------------

def discover_team_ids(
    out_dir: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Return a canonical-name → FBref-team-ID mapping for all 48 WC 2026 nations.

    Uses the built-in ``_KNOWN_TEAM_IDS`` seed map (FBref blocks automated
    confederation-page scraping via Cloudflare).  Results are saved to
    ``{out_dir}/team_ids.json`` for inspection and manual correction.

    Pass ``force=True`` to overwrite an existing cache with the seed map.
    """
    out_dir = out_dir or (settings.DATA_DIR / "raw" / "fbref")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "team_ids.json"

    if cache.exists() and not force:
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} FBref team IDs from cache ({cache})")
        return data

    team_ids: dict[str, str] = dict(_KNOWN_TEAM_IDS)

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(team_ids, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(team_ids)} FBref team IDs → {cache}")
    return team_ids


# ---------------------------------------------------------------------------
# Phase 2: browser-based scraping helpers
# ---------------------------------------------------------------------------

def _team_to_snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_tables(html: str) -> list[pd.DataFrame]:
    """Extract all HTML tables from a page, returning a list of DataFrames."""
    try:
        return pd.read_html(StringIO(html))
    except Exception:
        return []


def _clean_table(df: pd.DataFrame, team_name: str, table: str) -> pd.DataFrame:
    """Flatten MultiIndex headers, normalise column names, keep match rows only."""
    # Flatten MultiIndex columns.
    # FBref uses two-level headers like ('Standard', 'Sh') or ('For France', 'Date').
    # We keep only the innermost level — it's the actual column name; the outer
    # level is just a visual grouping that's already implied by the table type.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(col[-1]) if (str(col[-1]) != "Unnamed" and "level" not in str(col[-1]).lower() and "unnamed" not in str(col[-1]).lower())
            else str(col[0])
            for col in df.columns
        ]

    # Normalise: cast to str first, then lowercase, spaces→underscores, %→pct
    df.columns = [
        re.sub(r"%", "pct", str(col).lower().strip().replace(" ", "_").replace("-", "_"))
        for col in df.columns
    ]

    # FBref MultiIndex flattening can yield duplicate names (e.g. two "match_report")
    if df.columns.duplicated().any():
        counts: dict[str, int] = {}
        unique_cols: list[str] = []
        for col in df.columns:
            if col not in counts:
                counts[col] = 0
                unique_cols.append(col)
            else:
                counts[col] += 1
                unique_cols.append(f"{col}_{counts[col]}")
        df.columns = unique_cols

    # Keep only rows with a valid date (filters out summary/header rows)
    date_cols = [c for c in df.columns if "date" in c]
    if not date_cols:
        return pd.DataFrame()
    date_col = date_cols[0]
    df = df[df[date_col].astype(str).str.match(_DATE_RE)].copy()
    if date_col != "date":
        df.rename(columns={date_col: "date"}, inplace=True)

    if df.empty:
        return pd.DataFrame()

    df["team"] = team_name
    df["table"] = table
    return df


async def _fetch_html(
    browser: uc.Browser,
    url: str,
    min_delay: float = 6.0,
) -> str:
    """Navigate to ``url`` with the real Chrome browser and return the page HTML.

    Applies a random delay before every request to respect FBref's rate limits.
    Returns an empty string on any failure.
    """
    await asyncio.sleep(min_delay + random.uniform(0, 3.0))
    try:
        tab = await browser.get(url)
        await asyncio.sleep(3.5)  # let dynamic content render
        html: str = await tab.get_content()
        return html
    except Exception as exc:
        logger.warning(f"Browser fetch failed for {url}: {exc}")
        return ""


async def _scrape_team_async(
    browser: uc.Browser,
    team_name: str,
    team_id: str,
    out_dir: Path,
    seasons: list[str],
    tables: list[str],
) -> list[pd.DataFrame]:
    """Scrape all stat tables for one team.  Returns list of non-empty DataFrames."""
    results: list[pd.DataFrame] = []

    for table in tables:
        for season in seasons:
            stem = f"{_team_to_snake(team_name)}_{season.replace('-', '_')}_{table}"
            cache = out_dir / f"{stem}.csv"

            # Use cached file if it already exists and has data
            if cache.exists():
                try:
                    cached = pd.read_csv(cache)
                    if not cached.empty:
                        logger.debug(f"  {team_name}/{table}/{season}: using cache")
                        results.append(cached)
                        break  # found data for this table — skip remaining seasons
                except Exception:
                    pass

            url = _MATCHLOG_URL.format(team_id=team_id, season=season, table=table)
            logger.debug(f"  {team_name}/{table}/{season}: {url}")

            html = await _fetch_html(browser, url)
            if not html or "Just a moment" in html:
                logger.warning(f"  {team_name}/{table}/{season}: Cloudflare block or empty")
                continue

            raw_tables = _parse_tables(html)
            if not raw_tables:
                logger.debug(f"  {team_name}/{table}/{season}: no tables in HTML")
                continue

            # Find the first table that has actual match rows
            df = pd.DataFrame()
            for raw in raw_tables:
                cleaned = _clean_table(raw.copy(), team_name, table)
                if not cleaned.empty:
                    df = cleaned
                    break

            if df.empty:
                logger.debug(f"  {team_name}/{table}/{season}: no match rows")
                continue

            df["season"] = season
            try:
                df.to_csv(cache, index=False)
            except Exception as exc:
                logger.warning(f"  Cache write failed: {exc}")

            logger.info(f"  {team_name}/{table}/{season}: {len(df)} rows")
            results.append(df)
            break  # got data — don't try older seasons

    return results


async def _run_scrape(
    wc2026_teams: list[str],
    team_ids: dict[str, str],
    out_dir: Path,
    seasons: list[str],
    tables: list[str],
) -> pd.DataFrame:
    """Main async scraping loop.  Starts ONE browser and reuses it for all teams."""

    logger.info("Starting real Chrome browser (headless=False required for Cloudflare)…")
    browser = await uc.start(headless=False)

    # Warm up: visit FBref homepage to establish session / solve initial challenge
    logger.info("Warming up FBref session on homepage…")
    try:
        tab = await browser.get(FBREF_HOME)
        await asyncio.sleep(7)
        title: str = await tab.get_content()
        if "Just a moment" in title:
            logger.warning("Cloudflare challenge not yet solved — waiting 10 more seconds…")
            await asyncio.sleep(10)
    except Exception as exc:
        logger.warning(f"Homepage warm-up failed: {exc}")

    all_dfs: list[pd.DataFrame] = []
    n = len(wc2026_teams)

    for i, team in enumerate(wc2026_teams, 1):
        print(f"[{i}/{n}] {team}")
        logger.info(f"[{i}/{n}] {team}")

        team_id = team_ids.get(team)
        if not team_id:
            logger.warning(f"  {team}: no FBref team ID — skipping")
            continue

        dfs = await _scrape_team_async(browser, team, team_id, out_dir, seasons, tables)
        all_dfs.extend(dfs)

    try:
        browser.stop()
    except Exception:
        pass

    if not all_dfs:
        return pd.DataFrame()

    # Drop duplicate column labels per chunk before concat (pandas raises otherwise)
    clean_dfs = [df.loc[:, ~df.columns.duplicated()] for df in all_dfs]
    return pd.concat(clean_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 2: public entry point
# ---------------------------------------------------------------------------

def download_all_fbref(
    wc2026_teams: list[str],
    team_ids: dict[str, str],
    out_dir: Path | None = None,
    seasons: list[str] | None = None,
    tables: list[str] | None = None,
) -> pd.DataFrame:
    """Scrape FBref match-log stats for all 48 WC 2026 nations.

    Launches a real (non-headless) Chrome window via ``nodriver`` to bypass
    Cloudflare.  The browser is shared across all requests; per-file CSV caches
    make the run resumable after interruption.

    Saves:
        data/raw/fbref/fbref_all_combined.parquet

    Returns the combined DataFrame with columns: team, table, season, date, …
    """
    out_dir = out_dir or (settings.DATA_DIR / "raw" / "fbref")
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = seasons or DEFAULT_SEASONS
    tables = tables or STAT_TABLES

    combined = asyncio.run(
        _run_scrape(wc2026_teams, team_ids, out_dir, seasons, tables)
    )

    if combined.empty:
        logger.warning("No FBref data collected — check network and team IDs")
        return combined

    # Normalise column types before writing parquet: object columns → string,
    # so pyarrow doesn't choke on mixed int/str values in columns like gf/ga.
    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = combined[col].astype(str)

    parquet_path = out_dir / "fbref_all_combined.parquet"
    combined.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(
        f"Saved {len(combined):,} rows for {combined['team'].nunique()} teams → {parquet_path}"
    )
    return combined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config.team_names import WC2026_CANONICAL_TEAMS

    out = settings.DATA_DIR / "raw" / "fbref"
    team_ids = discover_team_ids(out)
    print(f"Team IDs loaded: {len(team_ids)}")

    df = download_all_fbref(WC2026_CANONICAL_TEAMS, team_ids, out)
    print(f"Scraped {len(df):,} rows for {df['team'].nunique() if not df.empty else 0} teams")
