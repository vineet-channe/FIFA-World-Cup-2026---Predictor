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
    "TEAM_ISO_CODES",
    "get_flag_emoji",
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
# ISO 3166-1 alpha-2 codes for all 48 WC 2026 nations + common others
# ---------------------------------------------------------------------------
# England, Scotland, Wales use GB-ENG/GB-SCT/GB-WLS subdivision codes — these
# do NOT map to standard flag emoji (which require exactly 2 letters), so they
# fall back to "⚽" in get_flag_emoji().

TEAM_ISO_CODES: dict[str, str] = {
    # Group A
    "Mexico":               "MX",
    "South Africa":         "ZA",
    "South Korea":          "KR",
    "Czechia":              "CZ",
    # Group B
    "USA":                  "US",
    "Paraguay":             "PY",
    "Australia":            "AU",
    "Turkey":               "TR",
    # Group C
    "Canada":               "CA",
    "Bosnia & Herzegovina": "BA",
    "Qatar":                "QA",
    "Switzerland":          "CH",
    # Group D
    "Germany":              "DE",
    "Curaçao":              "CW",
    "Ivory Coast":          "CI",
    "Ecuador":              "EC",
    # Group E
    "Netherlands":          "NL",
    "Japan":                "JP",
    "Sweden":               "SE",
    "Tunisia":              "TN",
    # Group F
    "Brazil":               "BR",
    "Morocco":              "MA",
    "Scotland":             "GB",   # GB-SCT not a 2-letter code → falls back to ⚽
    "Haiti":                "HT",
    # Group G
    "France":               "FR",
    "Senegal":              "SN",
    "Iraq":                 "IQ",
    "Norway":               "NO",
    # Group H
    "Spain":                "ES",
    "Cape Verde":           "CV",
    "Saudi Arabia":         "SA",
    "Uruguay":              "UY",
    # Group I
    "Belgium":              "BE",
    "Egypt":                "EG",
    "Iran":                 "IR",
    "New Zealand":          "NZ",
    # Group J
    "England":              "GB",   # GB-ENG not a 2-letter code → falls back to ⚽
    "Croatia":              "HR",
    "Ghana":                "GH",
    "Panama":               "PA",
    # Group K
    "Portugal":             "PT",
    "DR Congo":             "CD",
    "Uzbekistan":           "UZ",
    "Colombia":             "CO",
    # Group L
    "Argentina":            "AR",
    "Algeria":              "DZ",
    "Austria":              "AT",
    "Jordan":               "JO",
    # Extra common nations (not in WC 2026 field but appear in historical data)
    "Italy":                "IT",
    "Belgium":              "BE",
    "Poland":               "PL",
    "Denmark":              "DK",
    "Serbia":               "RS",
    "Ukraine":              "UA",
    "Romania":              "RO",
    "Hungary":              "HU",
    "Russia":               "RU",
    "Chile":                "CL",
    "Peru":                 "PE",
    "Bolivia":              "BO",
    "Venezuela":            "VE",
    "Costa Rica":           "CR",
    "Honduras":             "HN",
    "El Salvador":          "SV",
    "Jamaica":              "JM",
    "Trinidad & Tobago":    "TT",
    "Cameroon":             "CM",
    "Nigeria":              "NG",
    "Mali":                 "ML",
    "Zambia":               "ZM",
    "Zimbabwe":             "ZW",
    "Kenya":                "KE",
    "Tanzania":             "TZ",
    "Burkina Faso":         "BF",
    "Guinea":               "GN",
    "Mozambique":           "MZ",
    "Libya":                "LY",
    "Sudan":                "SD",
    "Ethiopia":             "ET",
    "China":                "CN",
    "India":                "IN",
    "Indonesia":            "ID",
    "Thailand":             "TH",
    "Vietnam":              "VN",
    "Philippines":          "PH",
    "Malaysia":             "MY",
    "Singapore":            "SG",
    "Pakistan":             "PK",
    "Afghanistan":          "AF",
    "Syria":                "SY",
    "Lebanon":              "LB",
    "Palestine":            "PS",
    "Kuwait":               "KW",
    "Bahrain":              "BH",
    "UAE":                  "AE",
    "Oman":                 "OM",
    "Yemen":                "YE",
    "Greece":               "GR",
    "Finland":              "FI",
    "Czech Republic":       "CZ",
    "Slovakia":             "SK",
    "Slovenia":             "SI",
    "Bulgaria":             "BG",
    "Albania":              "AL",
    "Kosovo":               "XK",
    "North Macedonia":      "MK",
    "Iceland":              "IS",
    "Ireland":              "IE",
    "Wales":                "GB",   # GB-WLS not a 2-letter code → falls back to ⚽
    "Northern Ireland":     "GB",
    "Luxembourg":           "LU",
    "Malta":                "MT",
    "Cyprus":               "CY",
    "Armenia":              "AM",
    "Azerbaijan":           "AZ",
    "Georgia":              "GE",
    "Kazakhstan":           "KZ",
    "Belarus":              "BY",
    "Moldova":              "MD",
    "Estonia":              "EE",
    "Latvia":               "LV",
    "Lithuania":            "LT",
    "North Korea":          "KP",
    "Taiwan":               "TW",
    "Hong Kong":            "HK",
    "Macau":                "MO",
    "Mongolia":             "MN",
    "Nepal":                "NP",
    "Sri Lanka":            "LK",
    "Bangladesh":           "BD",
    "Myanmar":              "MM",
    "Cambodia":             "KH",
    "Laos":                 "LA",
    "Brunei":               "BN",
    "Maldives":             "MV",
    "Bhutan":               "BT",
    "Timor-Leste":          "TL",
    "Cuba":                 "CU",
    "Dominican Republic":   "DO",
    "Guatemala":            "GT",
    "Nicaragua":            "NI",
    "Belize":               "BZ",
    "Guyana":               "GY",
    "Suriname":             "SR",
    "Barbados":             "BB",
    "Bahamas":              "BS",
    "Bermuda":              "BM",
    "Libya":                "LY",
    "Tunisia":              "TN",
    "Morocco":              "MA",
    "Egypt":                "EG",
    "Algeria":              "DZ",
    "South Sudan":          "SS",
    "Rwanda":               "RW",
    "Uganda":               "UG",
    "Angola":               "AO",
    "Namibia":              "NA",
    "Botswana":             "BW",
    "Lesotho":              "LS",
    "Eswatini":             "SZ",
    "Madagascar":           "MG",
    "Mauritius":            "MU",
    "Seychelles":           "SC",
    "Benin":                "BJ",
    "Togo":                 "TG",
    "Ghana":                "GH",
    "Niger":                "NE",
    "Chad":                 "TD",
    "Gabon":                "GA",
    "Congo":                "CG",
    "Equatorial Guinea":    "GQ",
    "São Tomé and Príncipe": "ST",
    "Cape Verde":           "CV",
    "Gambia":               "GM",
    "Guinea-Bissau":        "GW",
    "Sierra Leone":         "SL",
    "Liberia":              "LR",
    "Côte d'Ivoire":        "CI",
    "New Caledonia":        "NC",
    "Fiji":                 "FJ",
    "Papua New Guinea":     "PG",
    "Vanuatu":              "VU",
    "Solomon Islands":      "SB",
    "Tahiti":               "PF",
    "American Samoa":       "AS",
}

# Teams where the standard 2-letter ISO code does NOT produce a valid flag emoji
# (subdivisions like GB-ENG, GB-SCT, GB-WLS use 5-char codes)
_NO_FLAG_TEAMS: frozenset[str] = frozenset({"England", "Scotland", "Wales", "Northern Ireland"})


def get_flag_emoji(team_name: str) -> str:
    """Convert a canonical team name to a flag emoji via ISO 3166-1 alpha-2 code.

    Uses the Regional Indicator Symbol Letter approach: pair of Unicode code
    points 0x1F1E6–0x1F1FF corresponding to letters A–Z. Falls back to ⚽ if
    the team is unknown or uses a subdivision code that does not have a standard
    emoji (England, Scotland, Wales).

    Args:
        team_name: Canonical team name (as used in WC2026_CANONICAL_TEAMS).

    Returns:
        Two-character flag emoji string, or "⚽" as fallback.
    """
    if team_name in _NO_FLAG_TEAMS:
        return "⚽"
    code = TEAM_ISO_CODES.get(team_name)
    if not code or len(code) != 2:
        return "⚽"
    try:
        return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code.upper())
    except Exception:
        return "⚽"


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
