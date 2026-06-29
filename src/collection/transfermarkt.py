"""Scrape squad market values for all 48 WC 2026 nations from Transfermarkt (DS-06)."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings

__all__ = ["TM_TEAM_URLS", "parse_market_value", "scrape_squad_value", "download_all_squads"]

# ---------------------------------------------------------------------------
# Mimic a real Chrome browser to reduce 403 rate
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
}

# ---------------------------------------------------------------------------
# Transfermarkt URL slugs for all 48 WC 2026 nations
# Format: {canonical_name}: "{slug}/startseite/verein/{id}"
# ---------------------------------------------------------------------------
TM_TEAM_URLS: dict[str, str] = {
    # ── Group A ────────────────────────────────────────────────────────────
    "Mexico":               "mexiko/startseite/verein/6303",
    "South Africa":         "sudafrika/startseite/verein/3806",
    "South Korea":          "sudkorea/startseite/verein/3589",
    "Czechia":              "tschechien/startseite/verein/3445",

    # ── Group B ────────────────────────────────────────────────────────────
    "USA":                  "vereinigte-staaten/startseite/verein/3505",
    "Paraguay":             "paraguay/startseite/verein/3581",
    "Australia":            "australien/startseite/verein/3433",
    "Turkey":               "turkei/startseite/verein/3381",

    # ── Group C ────────────────────────────────────────────────────────────
    "Canada":               "kanada/startseite/verein/3510",
    "Bosnia & Herzegovina": "bosnien-herzegowina/startseite/verein/3446",
    "Qatar":                "katar/startseite/verein/14162",
    "Switzerland":          "schweiz/startseite/verein/3384",

    # ── Group D ────────────────────────────────────────────────────────────
    "Germany":              "deutschland/startseite/verein/3262",
    "Curaçao":              "curacao/startseite/verein/32364",
    "Ivory Coast":          "elfenbeinkuste/startseite/verein/3591",
    "Ecuador":              "ecuador/startseite/verein/5750",

    # ── Group E ────────────────────────────────────────────────────────────
    "Netherlands":          "niederlande/startseite/verein/3379",
    "Japan":                "japan/startseite/verein/3435",
    "Sweden":               "schweden/startseite/verein/3557",
    "Tunisia":              "tunesien/startseite/verein/3670",

    # ── Group F ────────────────────────────────────────────────────────────
    "Brazil":               "brasilien/startseite/verein/3439",
    "Morocco":              "marokko/startseite/verein/3575",
    "Scotland":             "schottland/startseite/verein/3380",
    "Haiti":                "haiti/startseite/verein/14161",

    # ── Group G ────────────────────────────────────────────────────────────
    "France":               "frankreich/startseite/verein/3377",
    "Senegal":              "senegal/startseite/verein/3499",
    "Iraq":                 "irak/startseite/verein/3560",
    "Norway":               "norwegen/startseite/verein/3440",

    # ── Group H ────────────────────────────────────────────────────────────
    "Spain":                "spanien/startseite/verein/3375",
    "Cape Verde":           "kap-verde/startseite/verein/4311",
    "Saudi Arabia":         "saudi-arabien/startseite/verein/3807",
    "Uruguay":              "uruguay/startseite/verein/3449",

    # ── Group I ────────────────────────────────────────────────────────────
    "Belgium":              "belgien/startseite/verein/3382",
    "Egypt":                "agypten/startseite/verein/3672",
    "Iran":                 "iran/startseite/verein/3582",
    "New Zealand":          "neuseeland/startseite/verein/9171",

    # ── Group J ────────────────────────────────────────────────────────────
    "England":              "england/startseite/verein/3299",
    "Croatia":              "kroatien/startseite/verein/3556",
    "Ghana":                "ghana/startseite/verein/3441",
    "Panama":               "panama/startseite/verein/3577",

    # ── Group K ────────────────────────────────────────────────────────────
    "Portugal":             "portugal/startseite/verein/3300",
    "DR Congo":             "demokratische-republik-kongo/startseite/verein/3854",
    "Uzbekistan":           "usbekistan/startseite/verein/3563",
    "Colombia":             "kolumbien/startseite/verein/3816",

    # ── Group L ────────────────────────────────────────────────────────────
    "Argentina":            "argentinien/startseite/verein/3437",
    "Algeria":              "algerien/startseite/verein/3614",
    "Austria":              "osterreich/startseite/verein/3383",
    "Jordan":               "jordanien/startseite/verein/15737",
}


# ---------------------------------------------------------------------------
# Value parser
# ---------------------------------------------------------------------------

def parse_market_value(value_str: str) -> float:
    """Convert Transfermarkt value strings to float EUR.

    Examples:
        "€1.23bn" → 1_230_000_000.0
        "€456m"   → 456_000_000.0
        "€12.5m"  → 12_500_000.0
        "€250k"   → 250_000.0
        "-"       → 0.0
    """
    if not value_str or value_str.strip() in ("-", "", "N/A", "n/a"):
        return 0.0

    cleaned = value_str.replace("€", "").replace(",", ".").strip()

    # Handle abbreviations case-insensitively
    lower = cleaned.lower()
    try:
        if "bn" in lower:
            return float(re.sub(r"[^\d.]", "", lower.replace("bn", ""))) * 1_000_000_000
        if "m" in lower:
            return float(re.sub(r"[^\d.]", "", lower.replace("m", ""))) * 1_000_000
        if "k" in lower:
            return float(re.sub(r"[^\d.]", "", lower.replace("k", ""))) * 1_000
        # Bare number
        return float(re.sub(r"[^\d.]", "", cleaned))
    except (ValueError, TypeError):
        logger.warning(f"Could not parse market value: {value_str!r}")
        return 0.0


# ---------------------------------------------------------------------------
# Single-team scraper
# ---------------------------------------------------------------------------

def scrape_squad_value(
    team_name: str,
    slug: str,
    delay: float = 3.0,
) -> dict[str, Any]:
    """Scrape the total squad market value for one national team.

    Returns a dict with keys: team, total_market_value_eur, url.
    On 403 or parse failure, total_market_value_eur is set to 0.0.
    """
    url = f"https://www.transfermarkt.com/{slug}"
    time.sleep(delay)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
    except requests.RequestException as exc:
        logger.warning(f"Network error scraping {team_name}: {exc}")
        return {"team": team_name, "total_market_value_eur": 0.0, "url": url}

    if resp.status_code == 403:
        logger.warning(f"  {team_name}: 403 Forbidden — set total_market_value_eur=0")
        return {"team": team_name, "total_market_value_eur": 0.0, "url": url}

    if resp.status_code != 200:
        logger.warning(f"  {team_name}: HTTP {resp.status_code} — skipping")
        return {"team": team_name, "total_market_value_eur": 0.0, "url": url}

    soup = BeautifulSoup(resp.text, "html.parser")

    total_value = 0.0

    # ── Strategy 1: data-header market value wrapper ───────────────────────
    value_box = soup.find("div", class_="data-header__market-value-wrapper")
    if value_box:
        raw = value_box.get_text(strip=True)
        total_value = parse_market_value(raw)

    # ── Strategy 2: "Total market value" text in info table ───────────────
    if total_value == 0.0:
        for tag in soup.find_all(["span", "td", "div"]):
            txt = tag.get_text(strip=True)
            if "total market value" in txt.lower():
                sibling = tag.find_next_sibling()
                if sibling:
                    total_value = parse_market_value(sibling.get_text(strip=True))
                    if total_value > 0:
                        break

    # ── Strategy 3: sum all player values in the squad table ──────────────
    if total_value == 0.0:
        player_values = []
        for a_tag in soup.select("table.items td.rechts.hauptlink a"):
            val_text = a_tag.get_text(strip=True)
            v = parse_market_value(val_text)
            if v > 0:
                player_values.append(v)
        if player_values:
            total_value = sum(player_values)
            logger.info(
                f"  {team_name}: summed {len(player_values)} player values → "
                f"€{total_value/1e6:.1f}M"
            )

    if total_value == 0.0:
        logger.warning(
            f"  {team_name}: could not parse market value from {url} — returning 0"
        )

    return {"team": team_name, "total_market_value_eur": total_value, "url": url}


# ---------------------------------------------------------------------------
# All-teams downloader
# ---------------------------------------------------------------------------

def download_all_squads(
    teams_dict: dict[str, str] = TM_TEAM_URLS,
    delay: float = 3.0,
) -> pd.DataFrame:
    """Scrape squad market values for all WC 2026 teams.

    Saves:
        data/raw/transfermarkt/squad_values.parquet

    Returns the DataFrame.
    """
    out = settings.DATA_DIR / "raw" / "transfermarkt"
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for team, slug in teams_dict.items():
        logger.info(f"Scraping {team}...")
        record = scrape_squad_value(team, slug, delay=delay)
        records.append(record)
        val_m = record["total_market_value_eur"] / 1e6
        print(f"  {team}: €{val_m:.1f}M")

    df = pd.DataFrame(records)
    df["total_market_value_eur"] = df["total_market_value_eur"].astype(float)

    parquet_path = out / "squad_values.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df)} team squad values → {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = download_all_squads()
    print("\nTop-10 squads by market value:")
    top = df.sort_values("total_market_value_eur", ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"  {row['team']:<20} €{row['total_market_value_eur']/1e6:,.1f}M")
