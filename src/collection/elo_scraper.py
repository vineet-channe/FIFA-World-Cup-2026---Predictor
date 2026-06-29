"""Scrape Elo ratings for all 48 WC 2026 nations from eloratings.net (DS-03)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings
from config.team_names import WC2026_CANONICAL_TEAMS, normalise

__all__ = ["WC2026_TEAMS", "scrape_elo_for_team", "download_all_elo"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WC2026_TEAMS: list[str] = WC2026_CANONICAL_TEAMS

_BASE_URL = "https://www.eloratings.net"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _USER_AGENT}


# ---------------------------------------------------------------------------
# Per-team URL slug helpers
# ---------------------------------------------------------------------------
_URL_OVERRIDES: dict[str, str] = {
    # Cases where the canonical name does not map cleanly to a URL slug
    "USA":                  "United_States",
    "South Korea":          "South_Korea",
    "Ivory Coast":          "Ivory_Coast",
    "DR Congo":             "DR_Congo",
    "Saudi Arabia":         "Saudi_Arabia",
    "New Zealand":          "New_Zealand",
    "South Africa":         "South_Africa",
    "Turkey":               "Turkey",          # not Türkiye on eloratings
    "Trinidad & Tobago":    "Trinidad_and_Tobago",
    "Bosnia & Herzegovina": "Bosnia_and_Herzegovina",
    "Czechia":              "Czechia",
    "North Macedonia":      "North_Macedonia",
    "Cape Verde":           "Cape_Verde",
    "Curaçao":              "Curacao",
}


def _team_to_slug(team_name: str) -> str:
    """Convert a canonical team name to the eloratings.net URL slug."""
    if team_name in _URL_OVERRIDES:
        return _URL_OVERRIDES[team_name]
    return team_name.replace(" ", "_")


def _team_to_filename(team_name: str) -> str:
    """Convert canonical name to a safe filesystem filename (snake_case)."""
    return team_name.lower().replace(" ", "_").replace("&", "and").replace("'", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_elo_for_team(team_name: str, delay: float = 2.0) -> pd.DataFrame:
    """Scrape full Elo history for one team from eloratings.net TSV endpoint.

    eloratings.net is a JavaScript SPA — HTML tables are never in the initial
    response.  The site publishes raw per-team data as tab-separated files at
    ``https://www.eloratings.net/{Slug}.tsv``.

    TSV column layout (no header row):
        0  year | 1 month | 2 day
        3  home_code | 4 away_code
        5  home_goals | 6 away_goals | 7 match_type
        8  qualifier (often empty)
        9  elo_change_home | 10 home_elo_after | 11 away_elo_after
        12+ optional extra columns

    Returns an empty DataFrame on HTTP error or parse failure (does NOT raise).
    """
    from collections import Counter

    slug = _team_to_slug(team_name)
    url = f"{_BASE_URL}/{slug}.tsv"

    time.sleep(delay)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
    except requests.RequestException as exc:
        logger.warning(f"Network error scraping {team_name}: {exc}")
        return pd.DataFrame()

    if resp.status_code != 200:
        logger.warning(f"  {team_name}: HTTP {resp.status_code} — skipping (url={url})")
        return pd.DataFrame()

    rows: list[list[str]] = []
    for line in resp.text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 12:
            rows.append(parts)

    if not rows:
        logger.warning(f"  {team_name}: TSV empty or too few columns at {url}")
        return pd.DataFrame()

    # Identify the team's 2-letter code: it appears in col-3 or col-4 of
    # every row in the file (since the file covers only this team's matches).
    code_counter: Counter[str] = Counter()
    for row in rows:
        code_counter[row[3]] += 1
        code_counter[row[4]] += 1
    team_code = code_counter.most_common(1)[0][0]

    records: list[dict] = []
    for row in rows:
        try:
            year, month, day = int(row[0]), int(row[1]), int(row[2])
            if row[3] == team_code:
                elo = int(row[10])         # home
            elif row[4] == team_code:
                elo = int(row[11])         # away
            else:
                continue
            records.append({
                "team":       normalise(team_name),
                "date":       pd.Timestamp(year=year, month=month, day=day),
                "elo_rating": elo,
            })
        except (ValueError, IndexError):
            continue

    if not records:
        logger.warning(f"  {team_name}: no usable rows in TSV (team_code={team_code!r})")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  {team_name}: {len(df)} records scraped (code={team_code!r})")
    return df


def download_all_elo(
    teams: list[str] = WC2026_TEAMS,
    resume: bool = True,
    delay: float = 2.0,
) -> pd.DataFrame:
    """Scrape Elo history for all WC 2026 teams and save to parquet.

    Args:
        teams:  List of canonical team names to scrape.
        resume: Skip teams whose CSV already exists in data/raw/elo/.
        delay:  Seconds to sleep between requests (never set to 0).

    Saves:
        data/raw/elo/{team_snake_case}.csv   — one per team
        data/raw/elo/elo_history.parquet     — combined

    Returns the combined DataFrame.
    """
    out = settings.DATA_DIR / "raw" / "elo"
    out.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    n = len(teams)

    for i, team in enumerate(teams, 1):
        csv_path = out / f"{_team_to_filename(team)}.csv"

        if resume and csv_path.exists():
            logger.info(f"[{i}/{n}] {team} — already scraped, loading from cache")
            try:
                cached = pd.read_csv(csv_path)
                if not cached.empty:
                    cached["team"] = normalise(team)
                    all_dfs.append(cached)
                    continue
            except Exception as exc:
                logger.warning(f"  Could not read cached CSV for {team}: {exc}")

        print(f"[{i}/{n}] Scraping {team}...")
        df = scrape_elo_for_team(team, delay=delay)

        if df.empty:
            # Save an empty CSV sentinel so resume knows we attempted this team
            pd.DataFrame(columns=["team", "elo_rating", "date"]).to_csv(
                csv_path, index=False
            )
            logger.warning(f"  {team}: empty result — sentinel CSV saved")
            continue

        df.to_csv(csv_path, index=False)
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No Elo data collected — check connectivity and team slugs.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    parquet_path = out / "elo_history.parquet"
    combined.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(
        f"Saved {len(combined):,} Elo records across "
        f"{combined['team'].nunique()} teams → {parquet_path}"
    )
    return combined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = download_all_elo(resume=True)
    print(f"\nTotal records:  {len(df):,}")
    print(f"Teams covered:  {df['team'].nunique()}")
    if "elo_rating" in df.columns:
        top = (
            df.sort_values("date", ascending=False)
            .drop_duplicates("team")[["team", "elo_rating", "date"]]
            .sort_values("elo_rating", ascending=False)
            .head(10)
        )
        print("\nTop-10 Elo ratings (most recent snapshot):")
        print(top.to_string(index=False))
