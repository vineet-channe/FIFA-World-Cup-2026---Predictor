"""Build the WC 2026 match schedule from the official FIFA schedule (DS-07).

Source: Official FOX/Telemundo WC 2026 broadcast schedule PDF.
All 104 matches: 72 group stage + 16 R32 + 8 R16 + 4 QF + 2 SF + 1 3rd + 1 Final.

Kickoff times are stored in UTC (converted from Pacific Daylight Time, UTC-7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import settings

__all__ = ["VENUES", "build_schedule", "load_schedule"]

# ---------------------------------------------------------------------------
# Official WC 2026 venues (16 total: 11 USA, 3 Mexico, 2 Canada)
# ---------------------------------------------------------------------------
VENUES: dict[str, dict[str, str]] = {
    # Mexico (3)
    "Estadio Azteca":           {"city": "Mexico City",     "country": "Mexico"},
    "Estadio Akron":            {"city": "Guadalajara",     "country": "Mexico"},
    "Estadio BBVA":             {"city": "Monterrey",       "country": "Mexico"},
    # Canada (2)
    "BMO Field":                {"city": "Toronto",         "country": "Canada"},
    "BC Place":                 {"city": "Vancouver",       "country": "Canada"},
    # USA (11)
    "SoFi Stadium":             {"city": "Inglewood",       "country": "USA"},
    "Levi's Stadium":           {"city": "Santa Clara",     "country": "USA"},
    "MetLife Stadium":          {"city": "East Rutherford", "country": "USA"},
    "Gillette Stadium":         {"city": "Foxborough",      "country": "USA"},
    "NRG Stadium":              {"city": "Houston",         "country": "USA"},
    "AT&T Stadium":             {"city": "Arlington",       "country": "USA"},
    "Lincoln Financial Field":  {"city": "Philadelphia",    "country": "USA"},
    "Estadio BBVA Bancomer":    {"city": "Monterrey",       "country": "Mexico"},  # alternate name
    "Mercedes-Benz Stadium":    {"city": "Atlanta",         "country": "USA"},
    "Lumen Field":              {"city": "Seattle",         "country": "USA"},
    "Hard Rock Stadium":        {"city": "Miami",           "country": "USA"},
    "Arrowhead Stadium":        {"city": "Kansas City",     "country": "USA"},
}

# ---------------------------------------------------------------------------
# Official WC 2026 groups (draw held December 2025)
# ---------------------------------------------------------------------------
_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Africa",       "South Korea",  "Czechia"],
    "B": ["USA",         "Paraguay",           "Australia",    "Turkey"],
    "C": ["Canada",      "Bosnia & Herzegovina","Qatar",       "Switzerland"],
    "D": ["Germany",     "Curaçao",            "Ivory Coast",  "Ecuador"],
    "E": ["Netherlands", "Japan",              "Sweden",       "Tunisia"],
    "F": ["Brazil",      "Morocco",            "Scotland",     "Haiti"],
    "G": ["France",      "Senegal",            "Iraq",         "Norway"],
    "H": ["Spain",       "Cape Verde",         "Saudi Arabia", "Uruguay"],
    "I": ["Belgium",     "Egypt",              "Iran",         "New Zealand"],
    "J": ["England",     "Croatia",            "Ghana",        "Panama"],
    "K": ["Portugal",    "DR Congo",           "Uzbekistan",   "Colombia"],
    "L": ["Argentina",   "Algeria",            "Austria",      "Jordan"],
}

# ---------------------------------------------------------------------------
# Group stage matches — all 72, with actual teams, exact dates and UTC times.
#
# Time conversion: All times in schedule PDF are Pacific Daylight Time (PDT = UTC-7)
# e.g. 12:00PM PDT = 19:00 UTC
#
# Format: (match_id, date_pt, kickoff_utc, group, venue, team_a, team_b)
# ---------------------------------------------------------------------------
_GROUP_MATCHES: list[tuple[str, str, str, str, str, str, str]] = [
    # ── Matchday 1 (June 11–17) ────────────────────────────────────────────
    ("WC2026_001", "2026-06-11", "2026-06-11T19:00:00Z", "A", "Estadio Azteca",         "Mexico",       "South Africa"),
    ("WC2026_002", "2026-06-11", "2026-06-12T02:00:00Z", "A", "Estadio Akron",          "South Korea",  "Czechia"),
    ("WC2026_003", "2026-06-12", "2026-06-12T19:00:00Z", "C", "BMO Field",              "Canada",       "Bosnia & Herzegovina"),
    ("WC2026_004", "2026-06-12", "2026-06-13T01:00:00Z", "B", "SoFi Stadium",           "USA",          "Paraguay"),
    ("WC2026_005", "2026-06-13", "2026-06-13T19:00:00Z", "C", "Levi's Stadium",         "Qatar",        "Switzerland"),
    ("WC2026_006", "2026-06-13", "2026-06-13T22:00:00Z", "F", "MetLife Stadium",        "Brazil",       "Morocco"),
    ("WC2026_007", "2026-06-13", "2026-06-14T01:00:00Z", "F", "Gillette Stadium",       "Haiti",        "Scotland"),
    ("WC2026_008", "2026-06-13", "2026-06-14T04:00:00Z", "B", "BC Place",               "Australia",    "Turkey"),
    ("WC2026_009", "2026-06-14", "2026-06-14T17:00:00Z", "D", "NRG Stadium",            "Germany",      "Curaçao"),
    ("WC2026_010", "2026-06-14", "2026-06-14T20:00:00Z", "E", "AT&T Stadium",           "Netherlands",  "Japan"),
    ("WC2026_011", "2026-06-14", "2026-06-14T23:00:00Z", "D", "Lincoln Financial Field","Ivory Coast",  "Ecuador"),
    ("WC2026_012", "2026-06-14", "2026-06-15T02:00:00Z", "E", "Estadio BBVA",           "Sweden",       "Tunisia"),
    ("WC2026_013", "2026-06-15", "2026-06-15T16:00:00Z", "H", "Mercedes-Benz Stadium",  "Spain",        "Cape Verde"),
    ("WC2026_014", "2026-06-15", "2026-06-15T19:00:00Z", "I", "Lumen Field",            "Belgium",      "Egypt"),
    ("WC2026_015", "2026-06-15", "2026-06-15T22:00:00Z", "H", "Hard Rock Stadium",      "Saudi Arabia", "Uruguay"),
    ("WC2026_016", "2026-06-15", "2026-06-16T01:00:00Z", "I", "SoFi Stadium",           "Iran",         "New Zealand"),
    ("WC2026_017", "2026-06-16", "2026-06-16T19:00:00Z", "G", "MetLife Stadium",        "France",       "Senegal"),
    ("WC2026_018", "2026-06-16", "2026-06-16T22:00:00Z", "G", "Gillette Stadium",       "Iraq",         "Norway"),
    ("WC2026_019", "2026-06-16", "2026-06-17T01:00:00Z", "L", "Arrowhead Stadium",      "Argentina",    "Algeria"),
    ("WC2026_020", "2026-06-16", "2026-06-17T04:00:00Z", "L", "Levi's Stadium",         "Austria",      "Jordan"),
    ("WC2026_021", "2026-06-17", "2026-06-17T17:00:00Z", "K", "NRG Stadium",            "Portugal",     "DR Congo"),
    ("WC2026_022", "2026-06-17", "2026-06-17T20:00:00Z", "J", "AT&T Stadium",           "England",      "Croatia"),
    ("WC2026_023", "2026-06-17", "2026-06-17T23:00:00Z", "J", "BMO Field",              "Ghana",        "Panama"),
    ("WC2026_024", "2026-06-17", "2026-06-18T02:00:00Z", "K", "Estadio Azteca",         "Uzbekistan",   "Colombia"),

    # ── Matchday 2 (June 18–23) ────────────────────────────────────────────
    ("WC2026_025", "2026-06-18", "2026-06-18T16:00:00Z", "A", "Mercedes-Benz Stadium",  "Czechia",      "South Africa"),
    ("WC2026_026", "2026-06-18", "2026-06-18T19:00:00Z", "C", "SoFi Stadium",           "Switzerland",  "Bosnia & Herzegovina"),
    ("WC2026_027", "2026-06-18", "2026-06-18T22:00:00Z", "C", "BC Place",               "Canada",       "Qatar"),
    ("WC2026_028", "2026-06-18", "2026-06-19T01:00:00Z", "A", "Estadio Akron",          "Mexico",       "South Korea"),
    ("WC2026_029", "2026-06-19", "2026-06-19T19:00:00Z", "B", "Lumen Field",            "USA",          "Australia"),
    ("WC2026_030", "2026-06-19", "2026-06-19T22:00:00Z", "F", "Gillette Stadium",       "Scotland",     "Morocco"),
    ("WC2026_031", "2026-06-19", "2026-06-20T01:30:00Z", "F", "Lincoln Financial Field","Brazil",       "Haiti"),
    ("WC2026_032", "2026-06-19", "2026-06-20T03:00:00Z", "B", "Levi's Stadium",         "Turkey",       "Paraguay"),
    ("WC2026_033", "2026-06-20", "2026-06-20T17:00:00Z", "E", "NRG Stadium",            "Netherlands",  "Sweden"),
    ("WC2026_034", "2026-06-20", "2026-06-20T20:00:00Z", "D", "BMO Field",              "Germany",      "Ivory Coast"),
    ("WC2026_035", "2026-06-20", "2026-06-21T00:00:00Z", "D", "Arrowhead Stadium",      "Ecuador",      "Curaçao"),
    ("WC2026_036", "2026-06-20", "2026-06-21T04:00:00Z", "E", "Estadio BBVA",           "Tunisia",      "Japan"),
    ("WC2026_037", "2026-06-21", "2026-06-21T16:00:00Z", "H", "Mercedes-Benz Stadium",  "Spain",        "Saudi Arabia"),
    ("WC2026_038", "2026-06-21", "2026-06-21T19:00:00Z", "I", "SoFi Stadium",           "Belgium",      "Iran"),
    ("WC2026_039", "2026-06-21", "2026-06-21T22:00:00Z", "H", "Hard Rock Stadium",      "Uruguay",      "Cape Verde"),
    ("WC2026_040", "2026-06-21", "2026-06-22T01:00:00Z", "I", "BC Place",               "New Zealand",  "Egypt"),
    ("WC2026_041", "2026-06-22", "2026-06-22T17:00:00Z", "L", "AT&T Stadium",           "Argentina",    "Austria"),
    ("WC2026_042", "2026-06-22", "2026-06-22T21:00:00Z", "G", "Lincoln Financial Field","France",       "Iraq"),
    ("WC2026_043", "2026-06-22", "2026-06-23T00:00:00Z", "G", "MetLife Stadium",        "Norway",       "Senegal"),
    ("WC2026_044", "2026-06-22", "2026-06-23T03:00:00Z", "L", "Levi's Stadium",         "Jordan",       "Algeria"),
    ("WC2026_045", "2026-06-23", "2026-06-23T17:00:00Z", "K", "NRG Stadium",            "Portugal",     "Uzbekistan"),
    ("WC2026_046", "2026-06-23", "2026-06-23T20:00:00Z", "J", "Gillette Stadium",       "England",      "Ghana"),
    ("WC2026_047", "2026-06-23", "2026-06-23T23:00:00Z", "J", "BMO Field",              "Panama",       "Croatia"),
    ("WC2026_048", "2026-06-23", "2026-06-24T02:00:00Z", "K", "Estadio Akron",          "Colombia",     "DR Congo"),

    # ── Matchday 3 (June 24–27, simultaneous pairs) ────────────────────────
    # Group C (simultaneous)
    ("WC2026_049", "2026-06-24", "2026-06-24T19:00:00Z", "C", "BC Place",               "Switzerland",  "Canada"),
    ("WC2026_050", "2026-06-24", "2026-06-24T19:00:00Z", "C", "Lumen Field",            "Bosnia & Herzegovina", "Qatar"),
    # Group F (simultaneous)
    ("WC2026_051", "2026-06-24", "2026-06-24T22:00:00Z", "F", "Hard Rock Stadium",      "Scotland",     "Brazil"),
    ("WC2026_052", "2026-06-24", "2026-06-24T22:00:00Z", "F", "Mercedes-Benz Stadium",  "Morocco",      "Haiti"),
    # Group A (simultaneous)
    ("WC2026_053", "2026-06-24", "2026-06-25T01:00:00Z", "A", "Estadio Azteca",         "Czechia",      "Mexico"),
    ("WC2026_054", "2026-06-24", "2026-06-25T01:00:00Z", "A", "Estadio BBVA",           "South Africa", "South Korea"),
    # Group D (simultaneous)
    ("WC2026_055", "2026-06-25", "2026-06-25T20:00:00Z", "D", "MetLife Stadium",        "Ecuador",      "Germany"),
    ("WC2026_056", "2026-06-25", "2026-06-25T20:00:00Z", "D", "Lincoln Financial Field","Curaçao",      "Ivory Coast"),
    # Group E (simultaneous)
    ("WC2026_057", "2026-06-25", "2026-06-25T23:00:00Z", "E", "Arrowhead Stadium",      "Tunisia",      "Netherlands"),
    ("WC2026_058", "2026-06-25", "2026-06-25T23:00:00Z", "E", "AT&T Stadium",           "Japan",        "Sweden"),
    # Group B (simultaneous)
    ("WC2026_059", "2026-06-25", "2026-06-26T02:00:00Z", "B", "SoFi Stadium",           "Turkey",       "USA"),
    ("WC2026_060", "2026-06-25", "2026-06-26T02:00:00Z", "B", "Levi's Stadium",         "Paraguay",     "Australia"),
    # Group G (simultaneous)
    ("WC2026_061", "2026-06-26", "2026-06-26T19:00:00Z", "G", "Gillette Stadium",       "Norway",       "France"),
    ("WC2026_062", "2026-06-26", "2026-06-26T19:00:00Z", "G", "BMO Field",              "Senegal",      "Iraq"),
    # Group H (simultaneous)
    ("WC2026_063", "2026-06-26", "2026-06-27T00:00:00Z", "H", "Estadio Akron",          "Uruguay",      "Spain"),
    ("WC2026_064", "2026-06-26", "2026-06-27T00:00:00Z", "H", "NRG Stadium",            "Cape Verde",   "Saudi Arabia"),
    # Group I (simultaneous)
    ("WC2026_065", "2026-06-26", "2026-06-27T03:00:00Z", "I", "BC Place",               "New Zealand",  "Belgium"),
    ("WC2026_066", "2026-06-26", "2026-06-27T03:00:00Z", "I", "Lumen Field",            "Egypt",        "Iran"),
    # Group J (simultaneous)
    ("WC2026_067", "2026-06-27", "2026-06-27T21:00:00Z", "J", "MetLife Stadium",        "Panama",       "England"),
    ("WC2026_068", "2026-06-27", "2026-06-27T21:00:00Z", "J", "Lincoln Financial Field","Croatia",      "Ghana"),
    # Group K (simultaneous)
    ("WC2026_069", "2026-06-27", "2026-06-27T23:30:00Z", "K", "Hard Rock Stadium",      "Colombia",     "Portugal"),
    ("WC2026_070", "2026-06-27", "2026-06-27T23:30:00Z", "K", "Mercedes-Benz Stadium",  "DR Congo",     "Uzbekistan"),
    # Group L (simultaneous)
    ("WC2026_071", "2026-06-27", "2026-06-28T03:00:00Z", "L", "AT&T Stadium",           "Jordan",       "Argentina"),
    ("WC2026_072", "2026-06-27", "2026-06-28T03:00:00Z", "L", "Arrowhead Stadium",      "Algeria",      "Austria"),
]

# ---------------------------------------------------------------------------
# Round of 32 — bracket positions as team_a / team_b (official bracket)
# ---------------------------------------------------------------------------
# Format: (match_id, date_pt, kickoff_utc, team_a_bracket, team_b_bracket, venue)
_R32_MATCHES: list[tuple[str, str, str, str, str, str]] = [
    ("WC2026_073", "2026-06-28", "2026-06-28T19:00:00Z", "2A",       "2B",       "SoFi Stadium"),
    ("WC2026_074", "2026-06-29", "2026-06-29T17:00:00Z", "1C",       "2F",       "NRG Stadium"),
    ("WC2026_075", "2026-06-29", "2026-06-29T20:30:00Z", "1E",       "3ABCDF",   "Gillette Stadium"),
    ("WC2026_076", "2026-06-29", "2026-06-30T01:00:00Z", "1F",       "2C",       "Estadio BBVA"),
    ("WC2026_077", "2026-06-30", "2026-06-30T17:00:00Z", "2E",       "2I",       "AT&T Stadium"),
    ("WC2026_078", "2026-06-30", "2026-06-30T21:00:00Z", "1I",       "3CDFGH",   "MetLife Stadium"),
    ("WC2026_079", "2026-06-30", "2026-07-01T01:00:00Z", "1A",       "3CEFHI",   "Estadio Azteca"),
    ("WC2026_080", "2026-07-01", "2026-07-01T16:00:00Z", "1L",       "3EHIJK",   "Mercedes-Benz Stadium"),
    ("WC2026_081", "2026-07-01", "2026-07-01T20:00:00Z", "1G",       "3AEHIJ",   "Lumen Field"),
    ("WC2026_082", "2026-07-01", "2026-07-02T00:00:00Z", "1D",       "3BEFIJ",   "Levi's Stadium"),
    ("WC2026_083", "2026-07-02", "2026-07-02T19:00:00Z", "1H",       "2J",       "SoFi Stadium"),
    ("WC2026_084", "2026-07-02", "2026-07-02T23:00:00Z", "2K",       "2L",       "BMO Field"),
    ("WC2026_085", "2026-07-02", "2026-07-03T03:00:00Z", "1B",       "3EFGIJ",   "BC Place"),
    ("WC2026_086", "2026-07-03", "2026-07-03T18:00:00Z", "2D",       "2G",       "AT&T Stadium"),
    ("WC2026_087", "2026-07-03", "2026-07-03T22:00:00Z", "1J",       "2H",       "Hard Rock Stadium"),
    ("WC2026_088", "2026-07-03", "2026-07-04T01:30:00Z", "1K",       "3DEIJL",   "Arrowhead Stadium"),
]

# ---------------------------------------------------------------------------
# Round of 16 — 8 matches (teams TBD from R32 results)
# ---------------------------------------------------------------------------
_R16_MATCHES: list[tuple[str, str, str]] = [
    ("WC2026_089", "2026-07-04", "2026-07-04T17:00:00Z", "NRG Stadium"),
    ("WC2026_090", "2026-07-04", "2026-07-04T21:00:00Z", "Lincoln Financial Field"),
    ("WC2026_091", "2026-07-05", "2026-07-05T20:00:00Z", "MetLife Stadium"),
    ("WC2026_092", "2026-07-05", "2026-07-06T00:00:00Z", "Estadio Azteca"),
    ("WC2026_093", "2026-07-06", "2026-07-06T19:00:00Z", "AT&T Stadium"),
    ("WC2026_094", "2026-07-06", "2026-07-07T00:00:00Z", "Lumen Field"),
    ("WC2026_095", "2026-07-07", "2026-07-07T16:00:00Z", "Mercedes-Benz Stadium"),
    ("WC2026_096", "2026-07-07", "2026-07-07T20:00:00Z", "BC Place"),
]

# ---------------------------------------------------------------------------
# Quarter-finals — 4 matches
# ---------------------------------------------------------------------------
_QF_MATCHES: list[tuple[str, str, str]] = [
    ("WC2026_097", "2026-07-09", "2026-07-09T20:00:00Z", "Gillette Stadium"),
    ("WC2026_098", "2026-07-10", "2026-07-10T19:00:00Z", "SoFi Stadium"),
    ("WC2026_099", "2026-07-11", "2026-07-11T21:00:00Z", "Hard Rock Stadium"),
    ("WC2026_100", "2026-07-11", "2026-07-12T01:00:00Z", "Arrowhead Stadium"),
]

# ---------------------------------------------------------------------------
# Semi-finals — 2 matches
# ---------------------------------------------------------------------------
_SF_MATCHES: list[tuple[str, str, str]] = [
    ("WC2026_101", "2026-07-14", "2026-07-14T19:00:00Z", "AT&T Stadium"),
    ("WC2026_102", "2026-07-15", "2026-07-15T19:00:00Z", "Mercedes-Benz Stadium"),
]

# Third-place play-off — 1 match
_3RD_PLACE: list[tuple[str, str, str]] = [
    ("WC2026_103", "2026-07-18", "2026-07-18T21:00:00Z", "Hard Rock Stadium"),
]

# Final — 1 match
_FINAL: list[tuple[str, str, str]] = [
    ("WC2026_104", "2026-07-19", "2026-07-19T19:00:00Z", "MetLife Stadium"),
]


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def _venue_info(name: str) -> dict[str, str]:
    return VENUES.get(name, {"city": "", "country": ""})


def build_schedule() -> list[dict[str, Any]]:
    """Build all 104 WC 2026 matches and save to JSON.

    Saves:
        data/raw/schedule/wc2026_schedule.json

    Returns the list of match dicts.
    """
    schedule: list[dict[str, Any]] = []

    # ── Group stage (72 matches) ───────────────────────────────────────────
    for match_id, date, kickoff, group, venue, team_a, team_b in _GROUP_MATCHES:
        vi = _venue_info(venue)
        schedule.append({
            "match_id":    match_id,
            "round":       "Group Stage",
            "group":       group,
            "match_date":  date,
            "kickoff_utc": kickoff,
            "team_a":      team_a,
            "team_b":      team_b,
            "venue":       venue,
            "city":        vi["city"],
            "country":     vi["country"],
            "played":      False,
            "result_a":    None,
            "result_b":    None,
        })

    # ── Round of 32 (16 matches) ───────────────────────────────────────────
    for match_id, date, kickoff, team_a, team_b, venue in _R32_MATCHES:
        vi = _venue_info(venue)
        schedule.append({
            "match_id":    match_id,
            "round":       "Round of 32",
            "group":       None,
            "match_date":  date,
            "kickoff_utc": kickoff,
            "team_a":      team_a,   # bracket position, e.g. "1A", "2B"
            "team_b":      team_b,
            "venue":       venue,
            "city":        vi["city"],
            "country":     vi["country"],
            "played":      False,
            "result_a":    None,
            "result_b":    None,
        })

    # ── R16, QF, SF, 3rd, Final (all TBD) ─────────────────────────────────
    knockout_rounds = [
        (_R16_MATCHES,  "Round of 16"),
        (_QF_MATCHES,   "Quarter-final"),
        (_SF_MATCHES,   "Semi-final"),
        (_3RD_PLACE,    "Third-place play-off"),
        (_FINAL,        "Final"),
    ]
    for slots, round_name in knockout_rounds:
        for match_id, date, kickoff, venue in slots:
            vi = _venue_info(venue)
            schedule.append({
                "match_id":    match_id,
                "round":       round_name,
                "group":       None,
                "match_date":  date,
                "kickoff_utc": kickoff,
                "team_a":      "TBD",
                "team_b":      "TBD",
                "venue":       venue,
                "city":        vi["city"],
                "country":     vi["country"],
                "played":      False,
                "result_a":    None,
                "result_b":    None,
            })

    # Sanity checks
    ids = [m["match_id"] for m in schedule]
    assert len(set(ids)) == len(ids), "Duplicate match IDs detected!"
    assert len(schedule) == 104, f"Expected 104 matches, got {len(schedule)}"

    out = settings.DATA_DIR / "raw" / "schedule"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "wc2026_schedule.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, default=str)

    logger.info(f"Schedule saved: {len(schedule)} matches → {dest}")
    return schedule


def load_schedule(path: str | None = None) -> list[dict[str, Any]]:
    """Load the WC 2026 schedule from JSON.

    Args:
        path: Optional path override.
              Defaults to data/raw/schedule/wc2026_schedule.json.
    """
    dest = (
        Path(path)
        if path
        else settings.DATA_DIR / "raw" / "schedule" / "wc2026_schedule.json"
    )
    if not dest.exists():
        raise FileNotFoundError(
            f"Schedule not found at {dest} — run build_schedule() first."
        )
    with open(dest, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    schedule = build_schedule()
    print(f"\nTotal matches: {len(schedule)}")

    from collections import Counter
    round_counts = Counter(m["round"] for m in schedule)
    order = [
        "Group Stage", "Round of 32", "Round of 16",
        "Quarter-final", "Semi-final", "Third-place play-off", "Final",
    ]
    for rnd in order:
        print(f"  {rnd:<28} {round_counts.get(rnd, 0):>3} matches")

    print(f"\nFirst match:  {schedule[0]['team_a']} vs {schedule[0]['team_b']}"
          f" ({schedule[0]['match_date']}) @ {schedule[0]['venue']}")
    print(f"Final match:  {schedule[-1]['team_a']} vs {schedule[-1]['team_b']}"
          f" ({schedule[-1]['match_date']}) @ {schedule[-1]['venue']}")
