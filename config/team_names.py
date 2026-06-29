"""
Canonical team name mapping for all 48 WC 2026 qualified nations.

Every data collection module imports `normalise()` from here to ensure
consistent team names across Kaggle results, eloratings.net, FIFA rankings,
Transfermarkt, API-Football, and the official schedule PDF.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "CANONICAL_NAMES",
    "WC2026_CANONICAL_TEAMS",
    "WC2026_GROUPS",
    "normalise",
    "normalise_dataframe",
    "audit_mismatches",
]

# ---------------------------------------------------------------------------
# Canonical team names — the single source of truth
# ---------------------------------------------------------------------------
# Format: raw variant → canonical name
# Sources covered: Kaggle martj42, eloratings.net, cashncarry FIFA rankings,
#                  Transfermarkt, API-Football, FBref, StatsBomb,
#                  Official WC 2026 schedule PDF (FOX/Telemundo)

CANONICAL_NAMES: dict[str, str] = {
    # ── United States ──────────────────────────────────────────────────────
    "United States":                    "USA",
    "United States of America":         "USA",
    "US":                               "USA",
    "U.S.A.":                           "USA",
    "U.S.":                             "USA",
    "United States Men's National Soccer Team": "USA",

    # ── Turkey / Türkiye ───────────────────────────────────────────────────
    "Türkiye":                          "Turkey",
    "Turkiye":                          "Turkey",

    # ── South Korea ────────────────────────────────────────────────────────
    "Korea Republic":                   "South Korea",
    "Korea, Republic of":               "South Korea",
    "Republic of Korea":                "South Korea",

    # ── Iran ───────────────────────────────────────────────────────────────
    "IR Iran":                          "Iran",
    "Islamic Republic of Iran":         "Iran",

    # ── Ivory Coast ────────────────────────────────────────────────────────
    "Côte d'Ivoire":                    "Ivory Coast",
    "Cote d'Ivoire":                    "Ivory Coast",
    "Côte D'Ivoire":                    "Ivory Coast",

    # ── DR Congo ───────────────────────────────────────────────────────────
    "Democratic Republic of Congo":     "DR Congo",
    "Congo DR":                         "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Congo, Democratic Republic":       "DR Congo",

    # ── Czechia ────────────────────────────────────────────────────────────
    "Czech Republic":                   "Czechia",
    "Czech Rep.":                       "Czechia",

    # ── Bosnia & Herzegovina ───────────────────────────────────────────────
    "Bosnia and Herzegovina":           "Bosnia & Herzegovina",
    "Bosnia-Herzegovina":               "Bosnia & Herzegovina",
    "Bosnia & Herzegovina":             "Bosnia & Herzegovina",

    # ── Saudi Arabia ───────────────────────────────────────────────────────
    "Saudia Arabia":                    "Saudi Arabia",  # typo in schedule PDF
    "KSA":                              "Saudi Arabia",

    # ── Cape Verde ─────────────────────────────────────────────────────────
    "Cape Verde Islands":               "Cape Verde",
    "Cabo Verde":                       "Cape Verde",

    # ── South Africa ───────────────────────────────────────────────────────
    "RSA":                              "South Africa",

    # ── New Zealand ────────────────────────────────────────────────────────
    "NZ":                               "New Zealand",

    # ── Curaçao ────────────────────────────────────────────────────────────
    "Curacao":                          "Curaçao",
    "Curaçao":                          "Curaçao",

    # ── North Macedonia ────────────────────────────────────────────────────
    "FYR Macedonia":                    "North Macedonia",
    "Macedonia":                        "North Macedonia",

    # ── Trinidad & Tobago ──────────────────────────────────────────────────
    "Trinidad and Tobago":              "Trinidad & Tobago",

    # ── Haiti ──────────────────────────────────────────────────────────────
    "Haiti":                            "Haiti",

    # ── Scotland ───────────────────────────────────────────────────────────
    "Scotland":                         "Scotland",

    # ── Slovakia ───────────────────────────────────────────────────────────
    "Slovak Republic":                  "Slovakia",

    # ── Sweden ─────────────────────────────────────────────────────────────
    "Sverige":                          "Sweden",

    # ── Tunisia ────────────────────────────────────────────────────────────
    "Tunisia":                          "Tunisia",

    # ── Norway ─────────────────────────────────────────────────────────────
    "Norway":                           "Norway",

    # ── Iraq ───────────────────────────────────────────────────────────────
    "Iraq":                             "Iraq",

    # ── Jordan ─────────────────────────────────────────────────────────────
    "Jordan":                           "Jordan",

    # ── Algeria ────────────────────────────────────────────────────────────
    "Algeria":                          "Algeria",

    # ── Paraguay ───────────────────────────────────────────────────────────
    "Paraguay":                         "Paraguay",

    # ── Panama ─────────────────────────────────────────────────────────────
    "Panama":                           "Panama",

    # ── Ecuador ────────────────────────────────────────────────────────────
    "Ecuador":                          "Ecuador",

    # ── Colombia ───────────────────────────────────────────────────────────
    "Colombia":                         "Colombia",

    # ── Uruguay ────────────────────────────────────────────────────────────
    "Uruguay":                          "Uruguay",

    # ── Qatar ──────────────────────────────────────────────────────────────
    "Qatar":                            "Qatar",
}


# ---------------------------------------------------------------------------
# All 48 WC 2026 qualified nations — groups per official draw (Dec 2025)
# ---------------------------------------------------------------------------
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Africa", "South Korea",        "Czechia"],
    "B": ["USA",         "Paraguay",     "Australia",          "Turkey"],
    "C": ["Canada",      "Bosnia & Herzegovina", "Qatar",      "Switzerland"],
    "D": ["Germany",     "Curaçao",      "Ivory Coast",        "Ecuador"],
    "E": ["Netherlands", "Japan",        "Sweden",             "Tunisia"],
    "F": ["Brazil",      "Morocco",      "Scotland",           "Haiti"],
    "G": ["France",      "Senegal",      "Iraq",               "Norway"],
    "H": ["Spain",       "Cape Verde",   "Saudi Arabia",       "Uruguay"],
    "I": ["Belgium",     "Egypt",        "Iran",               "New Zealand"],
    "J": ["England",     "Croatia",      "Ghana",              "Panama"],
    "K": ["Portugal",    "DR Congo",     "Uzbekistan",         "Colombia"],
    "L": ["Argentina",   "Algeria",      "Austria",            "Jordan"],
}

WC2026_CANONICAL_TEAMS: list[str] = [
    team for teams in WC2026_GROUPS.values() for team in teams
]

assert len(WC2026_CANONICAL_TEAMS) == 48, (
    f"Expected 48 WC 2026 teams, got {len(WC2026_CANONICAL_TEAMS)}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(name: str) -> str:
    """Return canonical team name, or the input unchanged if not in the map."""
    return CANONICAL_NAMES.get(name, name)


def normalise_dataframe(df: pd.DataFrame, team_cols: list[str]) -> pd.DataFrame:
    """Apply normalise() to each specified column in-place and return df."""
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].map(normalise)
    return df


def audit_mismatches(
    df_results: pd.DataFrame,
    df_rankings: pd.DataFrame,
    df_elo: pd.DataFrame,
) -> dict[str, list[str]]:
    """Print team names appearing in results but not in rankings or elo data.

    Returns a dict with keys 'not_in_rankings' and 'not_in_elo', each
    containing a sorted list of unmatched names.
    """
    results_teams: set[str] = (
        set(df_results["home_team"].dropna().unique())
        | set(df_results["away_team"].dropna().unique())
    )
    ranking_teams: set[str] = set(df_rankings["country_full"].dropna().unique())
    elo_teams: set[str] = set(df_elo["team"].dropna().unique())

    not_in_rankings = sorted(results_teams - ranking_teams)
    not_in_elo = sorted(results_teams - elo_teams)

    if not_in_rankings:
        print("=== In results but NOT in rankings ===")
        for t in not_in_rankings[:30]:
            print(f"  {t!r}")
        if len(not_in_rankings) > 30:
            print(f"  ... and {len(not_in_rankings) - 30} more")
    else:
        print("Rankings: no mismatches")

    if not_in_elo:
        print("\n=== In results but NOT in elo ===")
        for t in not_in_elo[:30]:
            print(f"  {t!r}")
        if len(not_in_elo) > 30:
            print(f"  ... and {len(not_in_elo) - 30} more")
    else:
        print("Elo: no mismatches")

    return {"not_in_rankings": not_in_rankings, "not_in_elo": not_in_elo}
