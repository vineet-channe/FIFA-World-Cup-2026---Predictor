"""
Local WC 2026 fixture data when API-Football is unavailable.

Builds deterministic results from pre-tournament simulation predictions
and the official schedule. Cached to data/raw/api_football/wc2026_results.json.
"""
from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from config.settings import settings
from config.team_names import WC2026_GROUPS
from src.simulation.group_stage import get_group_standings, select_best_third_place
from src.simulation.knockout import build_r32_bracket

SIM_PATH = settings.DATA_DIR / "predictions" / "tournament_simulation.json"
PRE_TOURNAMENT_PATH = settings.DATA_DIR / "predictions/snapshots/pre_tournament.json"
SCHEDULE_PATH = settings.DATA_DIR / "raw" / "schedule" / "wc2026_schedule.json"
CACHE_PATH = settings.DATA_DIR / "raw" / "api_football" / "wc2026_results.json"

# Number of R32 matches treated as completed on catch-up (July 4 2026 scenario).
R32_COMPLETED_COUNT = 8


def _fixture_id(match_id: str) -> int:
    return abs(hash(match_id)) % 900_000_000 + 100_000


def _elo_ratings_on_date(cut: pd.Timestamp) -> dict[str, float]:
    elo_df = pd.read_parquet(settings.DATA_DIR / "processed" / "elo_clean.parquet")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    ratings: dict[str, float] = {}
    for teams in WC2026_GROUPS.values():
        for team in teams:
            team_elo = elo_df[(elo_df["team"] == team) & (elo_df["date"] < cut)]
            ratings[team] = (
                float(team_elo.sort_values("date")["elo_rating"].iloc[-1])
                if len(team_elo)
                else 1500.0
            )
    return ratings


def _scores_from_prediction(mp: dict) -> tuple[int, int, bool, bool]:
    pa = mp["p_team_a_win"]
    pd_ = mp["p_draw"]
    pb = mp["p_team_b_win"]
    sa = max(0, round(mp.get("expected_score_a", 1.0)))
    sb = max(0, round(mp.get("expected_score_b", 1.0)))

    if pa >= pd_ and pa >= pb:
        if sa <= sb:
            sa = sb + 1
        return sa, sb, True, False
    if pb >= pd_ and pb >= pa:
        if sb <= sa:
            sb = sa + 1
        return sa, sb, False, True
    if sa == sb:
        sa = sb = max(sa, 1)
    return sa, sb, False, False


def _knockout_scores(team_a: str, team_b: str, seed: int) -> tuple[int, int, str]:
    rng = abs(hash((team_a, team_b, seed))) % 1000
    if rng % 5 == 0:
        sa = sb = 1 + (rng % 2)
        winner = team_a if abs(hash(team_a)) >= abs(hash(team_b)) else team_b
    elif abs(hash(team_a)) >= abs(hash(team_b)):
        sa, sb = 2 + rng % 2, rng % 2
        winner = team_a
    else:
        sa, sb = rng % 2, 2 + rng % 2
        winner = team_b
    return sa, sb, winner


def _build_group_tables(
    match_preds: dict,
    elo_ratings: dict[str, float],
) -> tuple[dict[str, list[str]], dict]:
    group_standings: dict[str, list[str]] = {}
    group_tables: dict[str, dict] = {}

    for letter, teams in WC2026_GROUPS.items():
        table = {
            t: {
                "points": 0,
                "gd": 0,
                "gf": 0,
                "ga": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "h2h": {
                    other: {"points": 0, "gd": 0, "gf": 0}
                    for other in teams
                    if other != t
                },
            }
            for t in teams
        }
        matches = []
        for mp in match_preds.values():
            if mp.get("group") != letter:
                continue
            ta, tb = mp["team_a"], mp["team_b"]
            sa, sb, _, _ = _scores_from_prediction(mp)
            pts_a = 3 if sa > sb else (1 if sa == sb else 0)
            pts_b = 3 if sb > sa else (1 if sa == sb else 0)
            for team, pts, gf, ga in [(ta, pts_a, sa, sb), (tb, pts_b, sb, sa)]:
                table[team]["points"] += pts
                table[team]["gf"] += gf
                table[team]["ga"] += ga
                table[team]["gd"] += gf - ga
                if pts == 3:
                    table[team]["wins"] += 1
                elif pts == 1:
                    table[team]["draws"] += 1
                else:
                    table[team]["losses"] += 1
            matches.append({"team_a": ta, "team_b": tb, "score_a": sa, "score_b": sb})

        group_result = {"matches": matches, "table": table}
        ranked = get_group_standings(group_result, elo_ratings)
        group_standings[letter] = ranked
        group_tables[letter] = table

    return group_standings, group_tables


def build_local_fixtures() -> list[dict]:
    source_path = PRE_TOURNAMENT_PATH if PRE_TOURNAMENT_PATH.exists() else SIM_PATH
    with open(source_path) as f:
        sim = json.load(f)
    if "snapshot_metadata" in sim:
        sim = {k: v for k, v in sim.items() if k != "snapshot_metadata"}
    match_preds = sim.get("match_predictions", {})
    elo_ratings = _elo_ratings_on_date(pd.Timestamp("2026-07-01"))

    fixtures: list[dict] = []
    for mid, mp in sorted(match_preds.items()):
        if not mp.get("round", "").startswith("Group Stage"):
            continue
        group = mp.get("group") or mp.get("round", "").replace("Group Stage - ", "").strip()
        sa, sb, hw, aw = _scores_from_prediction(mp)
        fixtures.append(
            {
                "fixture_id": _fixture_id(mid),
                "match_id": mid,
                "date": f"{mp['match_date']}T19:00:00+00:00",
                "round": f"Group Stage - {group}",
                "status": "FT",
                "home_team": mp["team_a"],
                "away_team": mp["team_b"],
                "home_score": sa,
                "away_score": sb,
                "home_winner": hw,
                "away_winner": aw,
            }
        )

    group_standings, group_tables = _build_group_tables(match_preds, elo_ratings)

    third_place = []
    for letter, ranked in group_standings.items():
        third = ranked[2]
        t = group_tables[letter][third]
        third_place.append(
            {
                "team": third,
                "points": t["points"],
                "gd": t["gd"],
                "gf": t["gf"],
                "elo": elo_ratings.get(third, 1500.0),
            }
        )
    best_thirds = select_best_third_place(third_place, elo_ratings)
    r32_pairs = build_r32_bracket(group_standings, best_thirds)

    with open(SCHEDULE_PATH) as f:
        schedule = json.load(f)
    r32_schedule = [m for m in schedule if m.get("round") == "Round of 32"]

    for i, (team_a, team_b) in enumerate(r32_pairs):
        sched = r32_schedule[i] if i < len(r32_schedule) else {}
        kickoff = sched.get("kickoff_utc", "2026-07-04T20:00:00Z")
        mid = sched.get("match_id", f"WC2026_R32_{i+1:02d}")
        played = i < R32_COMPLETED_COUNT
        if played:
            sa, sb, winner = _knockout_scores(team_a, team_b, i)
            hw = winner == team_a
            aw = winner == team_b
            status = "FT"
        else:
            sa = sb = None
            hw = aw = None
            status = "NS"

        fixtures.append(
            {
                "fixture_id": _fixture_id(mid),
                "match_id": mid,
                "date": kickoff,
                "round": "Round of 32",
                "status": status,
                "home_team": team_a,
                "away_team": team_b,
                "home_score": sa,
                "away_score": sb,
                "home_winner": hw if played else None,
                "away_winner": aw if played else None,
            }
        )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(
            {
                "pulled_at": pd.Timestamp.now().isoformat(),
                "source": "local_fallback",
                "fixtures": fixtures,
            },
            f,
            indent=2,
        )
    logger.info(f"Built {len(fixtures)} local fixtures → {CACHE_PATH}")
    return fixtures


def load_cached_fixtures() -> list[dict]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        fixtures = data.get("fixtures", [])
        if fixtures:
            logger.info(f"Loaded {len(fixtures)} fixtures from cache")
            return fixtures
    return build_local_fixtures()


def standings_from_fixtures(fixtures: list[dict]) -> dict[str, list[dict]]:
    tables: dict[str, dict[str, dict]] = {}

    for f in fixtures:
        if f["status"] != "FT" or "Group Stage" not in f.get("round", ""):
            continue
        group = f["round"].replace("Group Stage - ", "").strip()
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
            t = tables[group][team]
            t["gf"] += gf
            t["ga"] += ga
            t["gd"] = t["gf"] - t["ga"]

        ta, tb = f["home_team"], f["away_team"]
        sa, sb = f["home_score"], f["away_score"]
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

    standings: dict[str, list[dict]] = {}
    for group, teams in tables.items():
        ranked = sorted(
            teams.values(),
            key=lambda x: (x["points"], x["gd"], x["gf"]),
            reverse=True,
        )
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i
        standings[group] = ranked
    return standings
