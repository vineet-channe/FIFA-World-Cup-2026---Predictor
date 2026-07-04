"""Teams router — per-team summary and detailed profile endpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings
from config.team_names import WC2026_CANONICAL_TEAMS, TEAM_ISO_CODES
from src.api.schemas import (
    FormMatch,
    RadarStats,
    TeamDetail,
    TeamProbabilities,
    TeamSummary,
    TournamentPath,
)
from src.api.sim_loader import load_simulation

router = APIRouter()

_WC_CUT = pd.Timestamp("2026-06-11")

# Kit colours for all 48 WC 2026 teams — primary kit, switched to away if too
# dark to read against --ink (#10140F). Values must be visible as accent strips.
_KIT_COLORS: dict[str, str] = {
    "Mexico":               "#006847",
    "South Africa":         "#007A4D",
    "South Korea":          "#C60C30",
    "Czechia":              "#D7141A",
    "USA":                  "#B22234",
    "Paraguay":             "#D52B1E",
    "Australia":            "#FFD700",
    "Turkey":               "#E30A17",
    "Canada":               "#FF0000",
    "Bosnia & Herzegovina": "#002395",
    "Qatar":                "#8D1B3D",
    "Switzerland":          "#FF0000",
    "Germany":              "#FFFFFF",
    "Curaçao":              "#003DA5",
    "Ivory Coast":          "#F77F00",
    "Ecuador":              "#FFD100",
    "Netherlands":          "#FF6600",
    "Japan":                "#BC002D",
    "Sweden":               "#006AA7",
    "Tunisia":              "#E70013",
    "Brazil":               "#FFD700",
    "Morocco":              "#C1272D",
    "Scotland":             "#003F87",
    "Haiti":                "#00209F",
    "France":               "#002395",
    "Senegal":              "#00853F",
    "Iraq":                 "#007A3D",
    "Norway":               "#EF2B2D",
    "Spain":                "#AA151B",
    "Cape Verde":           "#003893",
    "Saudi Arabia":         "#006C35",
    "Uruguay":              "#5EB6E4",
    "Belgium":              "#EF3340",
    "Egypt":                "#CE1126",
    "Iran":                 "#239F40",
    "New Zealand":          "#FFFFFF",
    "England":              "#FFFFFF",
    "Croatia":              "#FF0000",
    "Ghana":                "#FCD116",
    "Panama":               "#DA121A",
    "Portugal":             "#006600",
    "DR Congo":             "#007FFF",
    "Uzbekistan":           "#1EB53A",
    "Colombia":             "#FCD116",
    "Argentina":            "#74ACDF",
    "Algeria":              "#006233",
    "Austria":              "#ED2939",
    "Jordan":               "#007A3D",
}


# ---------------------------------------------------------------------------
# Data helpers — loaded lazily and cached in module scope
# ---------------------------------------------------------------------------

_elo_df:      pd.DataFrame | None = None
_rankings_df: pd.DataFrame | None = None
_squad_df:    pd.DataFrame | None = None
_matches_df:  pd.DataFrame | None = None


def _load_sim() -> dict:
    return load_simulation()


def _load_elo() -> pd.DataFrame:
    global _elo_df
    if _elo_df is None:
        _elo_df = pd.read_parquet(settings.DATA_DIR / "processed" / "elo_clean.parquet")
        _elo_df["date"] = pd.to_datetime(_elo_df["date"])
    return _elo_df


def _load_rankings() -> pd.DataFrame:
    global _rankings_df
    if _rankings_df is None:
        _rankings_df = pd.read_parquet(settings.DATA_DIR / "processed" / "rankings_clean.parquet")
        _rankings_df["rank_date"] = pd.to_datetime(_rankings_df["rank_date"])
    return _rankings_df


def _load_squad() -> pd.DataFrame:
    global _squad_df
    if _squad_df is None:
        _squad_df = pd.read_parquet(settings.DATA_DIR / "raw" / "transfermarkt" / "squad_values.parquet")
    return _squad_df


def _load_matches() -> pd.DataFrame:
    global _matches_df
    if _matches_df is None:
        _matches_df = pd.read_parquet(settings.DATA_DIR / "processed" / "matches_clean.parquet")
        _matches_df["date"] = pd.to_datetime(_matches_df["date"])
    return _matches_df


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def normalise_to_100(value: float, all_values: list[float]) -> float:
    """Min-max normalise a single value against a list, scaled to 0–100."""
    finite = [v for v in all_values if not math.isnan(v)]
    if not finite or max(finite) == min(finite):
        return 50.0
    lo, hi = min(finite), max(finite)
    clamped = max(lo, min(hi, value if not math.isnan(value) else lo))
    return round((clamped - lo) / (hi - lo) * 100.0, 1)


def _get_elo_for(team: str) -> float:
    elo_df = _load_elo()
    rows = elo_df[(elo_df["team"] == team) & (elo_df["date"] < _WC_CUT)]
    if rows.empty:
        return 1500.0
    return float(rows.sort_values("date")["elo_rating"].iloc[-1])


def _get_fifa_pts_for(team: str) -> float:
    rk = _load_rankings()
    rows = rk[(rk["country_full"] == team) & (rk["rank_date"] < _WC_CUT)]
    if rows.empty:
        return 1000.0
    return float(rows.sort_values("rank_date")["total_points"].iloc[-1])


def _get_squad_value_for(team: str) -> float:
    sq = _load_squad()
    rows = sq[sq["team"] == team]
    if rows.empty:
        return 1e8
    val = float(rows["total_market_value_eur"].iloc[0])
    return val if val > 0 else 1e8


def _get_wc_apps(team: str) -> int:
    m = _load_matches()
    wc = m[
        (m["tournament"] == "FIFA World Cup")
        & ((m["home_team"] == team) | (m["away_team"] == team))
        & (m["date"] < _WC_CUT)
    ]
    return int(wc["date"].dt.year.nunique()) if not wc.empty else 0


def _get_ppg(team: str) -> float:
    m = _load_matches()
    recent = m[
        ((m["home_team"] == team) | (m["away_team"] == team))
        & (m["date"] < _WC_CUT)
    ].sort_values("date").tail(10)
    if recent.empty:
        return 1.0
    pts, n = 0, 0
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        hs = _safe_int(row.get("home_score"))
        as_ = _safe_int(row.get("away_score"))
        if is_home:
            pts += 3 if hs > as_ else (1 if hs == as_ else 0)
        else:
            pts += 3 if as_ > hs else (1 if as_ == hs else 0)
        n += 1
    return pts / n if n else 1.0


def _get_form_timeline(team: str) -> list[FormMatch]:
    """Return last 10 competitive matches as FormMatch objects."""
    competitive = {
        "FIFA World Cup", "World Cup qualifier", "Copa America",
        "UEFA Euro", "African Cup of Nations", "AFCON", "Gold Cup",
        "CONCACAF Gold Cup", "CONCACAF Nations League", "UEFA Nations League",
        "AFC Asian Cup", "OFC Nations Cup", "Confederations Cup",
    }
    m = _load_matches()
    recent = m[
        ((m["home_team"] == team) | (m["away_team"] == team))
        & (m["tournament"].isin(competitive))
        & (m["date"] < _WC_CUT)
    ].sort_values("date").tail(10)

    results = []
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        opp = str(row["away_team"] if is_home else row["home_team"])
        hs  = _safe_int(row.get("home_score"))
        as_ = _safe_int(row.get("away_score"))
        if is_home:
            result = "W" if hs > as_ else ("D" if hs == as_ else "L")
            score  = f"{hs}-{as_}"
        else:
            result = "W" if as_ > hs else ("D" if as_ == hs else "L")
            score  = f"{as_}-{hs}"
        results.append(
            FormMatch(
                opponent=opp,
                result=result,
                score=score,
                date=str(row["date"])[:10],
                competition=str(row.get("tournament", "")),
            )
        )
    return results


def _safe_int(v: object) -> int:
    """Convert float/NaN/None to int safely."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return 0 if (f != f) else int(f)
    except (TypeError, ValueError):
        return 0


def _build_team_summary(team: str) -> TeamSummary:
    return TeamSummary(
        name=team,
        iso_code=TEAM_ISO_CODES.get(team),
        kit_color=_KIT_COLORS.get(team, "#1F6F4A"),
        elo=_get_elo_for(team),
        fifa_points=_get_fifa_pts_for(team),
        squad_value_eur=_get_squad_value_for(team),
        wc_appearances=_get_wc_apps(team),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/teams", response_model=list[TeamSummary])
def get_teams() -> list[TeamSummary]:
    """Return summary data for all 48 WC 2026 teams."""
    return [_build_team_summary(t) for t in WC2026_CANONICAL_TEAMS]


@router.get("/team/{team_name}", response_model=TeamDetail)
def get_team(team_name: str) -> TeamDetail:
    """Return full team profile including radar stats, form, and tournament path."""
    if team_name not in WC2026_CANONICAL_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    sim = _load_sim()
    tp  = sim["team_probabilities"].get(team_name, {})

    summary = _build_team_summary(team_name)

    probs = TeamProbabilities(
        p_champion=tp.get("p_champion", 0.0),
        p_final=tp.get("p_final", 0.0),
        p_semi=tp.get("p_semi", 0.0),
        p_quarter=tp.get("p_quarter", 0.0),
        p_r16=tp.get("p_r16", 0.0),
        p_r32=tp.get("p_r32", 0.0),
        p_advance_groups=tp.get("p_advance_groups", 0.0),
        avg_goals_for_pg=tp.get("avg_goals_for_pg", 1.0),
        avg_goals_against_pg=tp.get("avg_goals_against_pg", 1.0),
    )

    # Radar — compute all 48 values to normalise against
    all_elos   = [_get_elo_for(t) for t in WC2026_CANONICAL_TEAMS]
    all_ppg    = [_get_ppg(t) for t in WC2026_CANONICAL_TEAMS]
    all_apps   = [float(_get_wc_apps(t)) for t in WC2026_CANONICAL_TEAMS]
    all_sq     = [math.log1p(_get_squad_value_for(t)) for t in WC2026_CANONICAL_TEAMS]
    all_adv    = [sim["team_probabilities"].get(t, {}).get("p_quarter", 0.0) for t in WC2026_CANONICAL_TEAMS]
    all_champ  = [sim["team_probabilities"].get(t, {}).get("p_advance_groups", 0.0) for t in WC2026_CANONICAL_TEAMS]

    idx = WC2026_CANONICAL_TEAMS.index(team_name)
    radar = RadarStats(
        elo_strength=normalise_to_100(all_elos[idx], all_elos),
        recent_form=normalise_to_100(all_ppg[idx], all_ppg),
        h2h_dominance=normalise_to_100(all_ppg[idx], all_ppg),  # PPG proxy for H2H dominance
        squad_value=normalise_to_100(all_sq[idx], all_sq),
        tournament_experience=normalise_to_100(all_apps[idx], all_apps),
        advance_probability=normalise_to_100(all_adv[idx], all_adv),
    )

    form = _get_form_timeline(team_name)

    path = TournamentPath(
        advance_groups=tp.get("p_advance_groups", 0.0),
        r16=tp.get("p_r16", 0.0),
        quarter=tp.get("p_quarter", 0.0),
        semi=tp.get("p_semi", 0.0),
        final=tp.get("p_final", 0.0),
        champion=tp.get("p_champion", 0.0),
    )

    return TeamDetail(
        summary=summary,
        probabilities=probs,
        radar_stats=radar,
        form_timeline=form,
        tournament_path=path,
    )
