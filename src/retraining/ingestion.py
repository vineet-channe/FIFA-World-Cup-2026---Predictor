"""
Pull actual WC 2026 results from football-data.org.
All team names are normalised via config.team_names before returning.
"""
import json
import time

import pandas as pd
import requests
from loguru import logger

from config.settings import settings
from config.team_names import normalise

BASE_URL = "https://api.football-data.org/v4"
WC_COMPETITION_CODE = "WC"
WC_SEASON = 2026
RATE_LIMIT_DELAY = 7.0

RESULTS_PATH = settings.DATA_DIR / "raw" / "api_football" / "wc2026_results.json"

STATUS_MAP: dict[str, str] = {
    "FINISHED": "FT",
    "IN_PLAY": "1H",
    "PAUSED": "HT",
    "SCHEDULED": "NS",
    "TIMED": "NS",
    "POSTPONED": "NS",
    "CANCELLED": "NS",
    "SUSPENDED": "NS",
    "AWARDED": "FT",
}

STAGE_MAP: dict[str, str] = {
    "GROUP_STAGE": "Group Stage",
    "LAST_32": "Round of 32",
    "ROUND_OF_32": "Round of 32",
    "LAST_16": "Round of 16",
    "ROUND_OF_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-finals",
    "SEMI_FINALS": "Semi-finals",
    "THIRD_PLACE": "3rd Place",
    "FINAL": "Final",
}

if not settings.FOOTBALL_DATA_API_KEY:
    logger.warning(
        "FOOTBALL_DATA_API_KEY is not set in .env — "
        "ingestion will fail and the fixture fallback will be used instead."
    )


def _get(path: str, params: dict | None = None) -> dict:
    """Single API call to football-data.org. Respects rate limit delay."""
    headers = {"X-Auth-Token": settings.FOOTBALL_DATA_API_KEY}
    time.sleep(RATE_LIMIT_DELAY)
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)

    if resp.status_code == 429:
        logger.warning("Rate limited by football-data.org — waiting 60s and retrying")
        time.sleep(60)
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)

    if resp.status_code != 200:
        raise ValueError(
            f"football-data.org {path} returned {resp.status_code}: {resp.text[:200]}"
        )
    return resp.json()


def _parse_match(match: dict) -> dict:
    """Normalise one football-data.org match into our internal fixture format."""
    stage = match.get("stage", "GROUP_STAGE")
    group_raw = match.get("group") or ""
    group = group_raw.replace("GROUP_", "").strip() if group_raw else None

    round_label = STAGE_MAP.get(stage, "Group Stage")
    if stage == "GROUP_STAGE" and group:
        round_str = f"Group Stage - {group}"
    else:
        round_str = round_label

    score = match.get("score", {})
    ft = score.get("fullTime", {}) or {}
    home_score = ft.get("home")
    away_score = ft.get("away")
    winner = score.get("winner")

    status_raw = match.get("status", "SCHEDULED")
    internal_status = STATUS_MAP.get(status_raw, "NS")

    return {
        "fixture_id": match["id"],
        "date": match.get("utcDate", ""),
        "round": round_str,
        "status": internal_status,
        "home_team": normalise(match.get("homeTeam", {}).get("name", "")),
        "away_team": normalise(match.get("awayTeam", {}).get("name", "")),
        "home_score": home_score,
        "away_score": away_score,
        "home_winner": (winner == "HOME_TEAM") if winner is not None else None,
        "away_winner": (winner == "AWAY_TEAM") if winner is not None else None,
    }


def _fetch_all_matches(params: dict | None = None) -> list[dict]:
    data = _get(
        f"/competitions/{WC_COMPETITION_CODE}/matches",
        params={"season": WC_SEASON, **(params or {})},
    )
    return [_parse_match(m) for m in data.get("matches", [])]


def get_fixtures(status: str = "FT") -> list[dict]:
    """
    Fetch WC 2026 fixtures filtered by internal status code.
    status: "FT" = finished | "NS" = scheduled | "1H" = in play
    """
    all_matches = _fetch_all_matches()
    filtered = [f for f in all_matches if f["status"] == status]
    logger.info(
        f"Fetched {len(filtered)} fixtures with status={status} "
        f"(from {len(all_matches)} total)"
    )
    return filtered


def get_all_fixtures() -> list[dict]:
    """Return all WC 2026 fixtures (finished and scheduled)."""
    all_matches = _fetch_all_matches()
    logger.info(f"Fetched {len(all_matches)} total fixtures")
    return all_matches


def get_all_completed_fixtures() -> list[dict]:
    """Fetch all finished WC 2026 matches — group stage + any completed knockout matches."""
    return get_fixtures(status="FT")


def _standings_from_group_fixtures(fixtures: list[dict]) -> dict[str, list[dict]]:
    """Build per-group standings when the API only returns a TOTAL table."""
    tables: dict[str, dict[str, dict]] = {}

    for f in fixtures:
        if f["status"] != "FT" or not f.get("round", "").startswith("Group Stage - "):
            continue
        group = f["round"].replace("Group Stage - ", "").strip()
        if not group:
            continue
        if group not in tables:
            tables[group] = {}
        for team, gf, ga in [
            (f["home_team"], f["home_score"], f["away_score"]),
            (f["away_team"], f["away_score"], f["home_score"]),
        ]:
            if team not in tables[group]:
                tables[group][team] = {
                    "team": team,
                    "points": 0,
                    "gd": 0,
                    "gf": 0,
                    "ga": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                }
            row = tables[group][team]
            row["gf"] += gf or 0
            row["ga"] += ga or 0
            row["gd"] = row["gf"] - row["ga"]

        ta, tb = f["home_team"], f["away_team"]
        sa, sb = f["home_score"] or 0, f["away_score"] or 0
        pts_a = 3 if sa > sb else (1 if sa == sb else 0)
        pts_b = 3 if sb > sa else (1 if sa == sb else 0)
        for team, pts in [(ta, pts_a), (tb, pts_b)]:
            tables[group][team]["points"] += pts
            if pts == 3:
                tables[group][team]["wins"] += 1
            elif pts == 1:
                tables[group][team]["draws"] += 1
            else:
                tables[group][team]["losses"] += 1

    result: dict[str, list[dict]] = {}
    for group, teams in tables.items():
        ranked = sorted(
            teams.values(),
            key=lambda x: (x["points"], x["gd"], x["gf"]),
            reverse=True,
        )
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i
        result[group] = ranked
    return result


def get_standings() -> dict[str, list[dict]]:
    """
    Fetch group stage standings. Returns dict mapping group letter → ordered team list.
    Only populated once the group stage has begun.
    """
    data = _get(
        f"/competitions/{WC_COMPETITION_CODE}/standings",
        params={"season": WC_SEASON},
    )

    result: dict[str, list[dict]] = {}
    for standing_block in data.get("standings", []):
        group_raw = standing_block.get("group") or ""
        group = group_raw.replace("GROUP_", "").strip()
        if not group:
            continue
        table = standing_block.get("table", [])
        result[group] = [
            {
                "team": normalise(entry["team"]["name"]),
                "rank": entry["position"],
                "points": entry["points"],
                "gd": entry["goalDifference"],
                "gf": entry["goalsFor"],
                "ga": entry["goalsAgainst"],
                "wins": entry["won"],
                "draws": entry["draw"],
                "losses": entry["lost"],
            }
            for entry in sorted(table, key=lambda x: x["position"])
        ]

    if not result:
        logger.info(
            "No per-group standings in API response — deriving from group fixtures"
        )
        result = _standings_from_group_fixtures(_fetch_all_matches())

    logger.info(f"Fetched standings for {len(result)} groups")
    return result


def get_bracket() -> list[dict]:
    """
    Fetch Round of 32 fixtures. Returns all 16 R32 fixtures (real team names
    if the group stage is complete, placeholder names otherwise).
    """
    data = _get(
        f"/competitions/{WC_COMPETITION_CODE}/matches",
        params={"season": WC_SEASON, "stage": "LAST_32"},
    )
    bracket = [_parse_match(m) for m in data.get("matches", [])]
    logger.info(f"Fetched {len(bracket)} R32 bracket entries")
    return bracket


def enrich_knockout_fixtures(
    knockout: list[dict],
    all_fixtures: list[dict],
) -> list[dict]:
    """
    Prefer the main /matches feed when the stage-specific endpoint is stale.

    football-data.org sometimes returns R32 as SCHEDULED on the LAST_32
    endpoint even after full-time results exist on the competition matches list.
    """
    by_id = {f["fixture_id"]: f for f in all_fixtures}
    enriched: list[dict] = []
    for match in knockout:
        latest = by_id.get(match["fixture_id"])
        if (
            latest
            and latest["status"] == "FT"
            and latest.get("home_score") is not None
        ):
            enriched.append(latest)
        else:
            enriched.append(match)
    return enriched


def fixture_to_result_entry(fixture: dict) -> dict:
    """Normalise one completed fixture into an actual_results row."""
    return {
        "team_a": fixture["home_team"],
        "team_b": fixture["away_team"],
        "score_a": fixture["home_score"],
        "score_b": fixture["away_score"],
        "round": fixture["round"],
        "played": True,
        "winner": (
            fixture["home_team"]
            if fixture.get("home_winner")
            else fixture["away_team"]
            if fixture.get("away_winner")
            else None
        ),
    }


def completed_fixtures(all_fixtures: list[dict]) -> list[dict]:
    """All finished matches with a recorded score."""
    return [
        f
        for f in all_fixtures
        if f["status"] == "FT" and f.get("home_score") is not None
    ]


def save_raw_results(fixtures: list[dict]) -> None:
    """Cache raw normalised fixtures for reproducibility / audit."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "pulled_at": pd.Timestamp.now().isoformat(),
                "source": "football-data.org",
                "fixtures": fixtures,
            },
            f,
            indent=2,
        )
