"""Simulation router — tournament results, groups, metadata."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings
from config.team_names import WC2026_GROUPS
from src.api.schemas import (
    AccuracyStats,
    GroupData,
    MatchPrediction,
    MatchPredictionAtStage,
    MatchPredictionHistory,
    MatchResultCard,
    MetaInfo,
    RoundData,
    TournamentBracket,
)
from src.api.prediction_lookup import enrich_completed_match, invalidate_prediction_index
from src.api.sim_loader import load_simulation
from src.retraining.snapshots import (
    list_snapshots,
    load_snapshot,
    get_probability_history,
)

router = APIRouter()

_comp_cache:  dict | None = None


def _load_live_sim() -> dict:
    """Load live simulation; return empty dict if not yet generated."""
    return load_simulation(required=False)


def _strip_projection(name: str) -> str:
    return name[1:] if name.startswith("~") else name


def _match_pair_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted([_strip_projection(team_a), _strip_projection(team_b)]))


def _round_key(round_str: str) -> str:
    """Map round string to a short key for the frontend tabs."""
    r = round_str.lower()
    if "group" in r:
        return "groups"
    if "32" in r:
        return "r32"
    if "16" in r:
        return "r16"
    if "quarter" in r:
        return "qf"
    if "semi" in r:
        return "sf"
    if "3rd" in r or "third" in r:
        return "3rd"
    if "final" in r:
        return "final"
    return "other"


ROUND_ORDER = ["groups", "r32", "r16", "qf", "sf", "3rd", "final"]
ROUND_LABELS = {
    "groups": "Group Stage",
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarter-finals",
    "sf": "Semi-finals",
    "3rd": "3rd Place",
    "final": "Final",
}


def _outcome_correct(
    pred: dict,
    actual_winner: str | None,
    score_a: int,
    score_b: int,
) -> bool | None:
    if actual_winner is None:
        return None
    p_a = pred.get("p_team_a_win", 0)
    p_d = pred.get("p_draw", 0)
    p_b = pred.get("p_team_b_win", 0)
    if p_a >= p_d and p_a >= p_b:
        predicted = pred.get("team_a")
    elif p_b >= p_d and p_b >= p_a:
        predicted = pred.get("team_b")
    else:
        predicted = "draw"
    if score_a == score_b:
        return predicted == "draw"
    return predicted == actual_winner


def _meta_from_sim(sim: dict) -> MetaInfo:
    comp = _load_comparison()
    meta = sim.get("metadata", {})
    actual = sim.get("actual_results", {})
    preds = sim.get("match_predictions", {})
    matches_total = len(actual) + sum(
        1 for v in preds.values() if v.get("status") == "predicted"
    )
    if not matches_total:
        matches_total = len(preds) or None
    return MetaInfo(
        model_version=meta.get("model", "ensemble_v1"),
        last_simulation=meta.get("run_at", "unknown"),
        n_simulations=int(meta.get("n_simulations", 0)),
        ensemble_brier_wc2022=comp.get("ensemble_wc2022_brier"),
        current_round=meta.get("tournament_phase"),
        matches_played=len(actual) if actual else None,
        matches_total=matches_total,
        last_updated=meta.get("data_as_of") or meta.get("run_at"),
    )


def _load_sim() -> dict:
    return load_simulation()


def _load_comparison() -> dict:
    global _comp_cache
    if _comp_cache is None:
        p = settings.DATA_DIR / "processed" / "model_comparison.json"
        if p.exists():
            with open(p) as f:
                _comp_cache = json.load(f)
        else:
            _comp_cache = {}
    return _comp_cache


@router.get("/simulation")
def get_simulation() -> dict:
    """Full passthrough of tournament_simulation.json."""
    return _load_sim()


@router.get("/groups", response_model=list[GroupData])
def get_groups() -> list[GroupData]:
    """Return each group with teams, advance probabilities, and match rows."""
    sim = _load_sim()
    group_preds = sim.get("group_predictions", {})

    if group_preds:
        return _groups_from_pre_tournament(sim)

    return _groups_from_live_state(sim)


def _groups_from_pre_tournament(sim: dict) -> list[GroupData]:
    """Pre-tournament simulation — group_predictions populated (Phase 4)."""
    group_preds = sim.get("group_predictions", {})
    match_preds = sim.get("match_predictions", {})
    results: list[GroupData] = []

    for letter in sorted(group_preds.keys()):
        gp = group_preds[letter]
        teams = gp.get("teams", [])
        advance_probs = gp.get("advance_probs", {})
        group_matches: list[MatchPrediction] = []

        for mid, mp in match_preds.items():
            if mp.get("group") != letter:
                continue
            try:
                group_matches.append(_match_from_prediction(mid, mp, letter))
            except Exception as exc:
                logger.warning(f"Skipping match {mid}: {exc}")

        results.append(
            GroupData(
                letter=letter,
                teams=teams,
                advance_probs=advance_probs,
                matches=group_matches,
            )
        )
    return results


def _live_group_letters(sim: dict) -> list[str]:
    """Group letters present in live data (FIFA draw may differ from WC2026_GROUPS)."""
    letters: set[str] = set(sim.get("group_stage_standings", {}).keys())
    for mp in sim.get("match_predictions", {}).values():
        r = mp.get("round", "")
        if r.startswith("Group Stage - "):
            letter = r.replace("Group Stage - ", "").strip()
            if len(letter) == 1:
                letters.add(letter)
    if letters:
        return sorted(letters)
    return sorted(WC2026_GROUPS.keys())


def _teams_for_live_group(
    letter: str,
    group_match_rows: list[dict],
    standings: dict,
) -> list[str]:
    """Teams in this group from actual results, ordered by final standings."""
    from_matches = {
        t for mp in group_match_rows for t in (mp["team_a"], mp["team_b"])
    }
    gs = standings.get(letter, {})
    ordered = [gs.get(k) for k in ("1st", "2nd", "3rd", "4th") if gs.get(k)]
    ordered = [t for t in ordered if t in from_matches]
    for t in sorted(from_matches):
        if t not in ordered:
            ordered.append(t)
    return ordered


def _groups_from_live_state(sim: dict) -> list[GroupData]:
    """Post-group-stage — build from standings, team_probs, and actual results."""
    standings = sim.get("group_stage_standings", {})
    team_probs = sim.get("team_probabilities", {})
    match_preds = sim.get("match_predictions", {})
    results: list[GroupData] = []

    for letter in _live_group_letters(sim):
        round_label = f"Group Stage - {letter}"
        raw_rows = [
            mp
            for mp in match_preds.values()
            if mp.get("round") == round_label
        ]
        teams = _teams_for_live_group(letter, raw_rows, standings)

        advance_probs = {
            team: float(team_probs.get(team, {}).get("p_r32", 0.0))
            for team in teams
        }

        group_matches: list[MatchPrediction] = []
        for mid, mp in match_preds.items():
            if mp.get("round") != round_label:
                continue
            try:
                group_matches.append(_match_from_result(mid, mp, letter))
            except Exception as exc:
                logger.warning(f"Skipping match {mid}: {exc}")

        group_matches.sort(key=lambda m: (m.match_date, m.team_a))

        results.append(
            GroupData(
                letter=letter,
                teams=teams,
                advance_probs=advance_probs,
                matches=group_matches,
            )
        )
    return results


def _match_from_prediction(mid: str, mp: dict, letter: str) -> MatchPrediction:
    return MatchPrediction(
        match_id=str(mid),
        team_a=mp["team_a"],
        team_b=mp["team_b"],
        round=mp.get("round", "Group Stage"),
        group=mp.get("group", letter),
        match_date=mp.get("match_date", ""),
        p_team_a_win=mp.get("p_team_a_win", 0.333),
        p_draw=mp.get("p_draw", 0.333),
        p_team_b_win=mp.get("p_team_b_win", 0.333),
        expected_score_a=mp.get("expected_score_a", 1.0),
        expected_score_b=mp.get("expected_score_b", 1.0),
    )


def _match_from_result(mid: str, mp: dict, letter: str) -> MatchPrediction:
    """Build a match row from a completed result or pending prediction."""
    if mp.get("status") == "predicted":
        return _match_from_prediction(mid, mp, letter)

    sa = mp.get("score_a", mp.get("expected_score_a", 0)) or 0
    sb = mp.get("score_b", mp.get("expected_score_b", 0)) or 0
    winner = mp.get("winner")

    if winner == mp["team_a"]:
        p_a, p_d, p_b = 1.0, 0.0, 0.0
    elif winner == mp["team_b"]:
        p_a, p_d, p_b = 0.0, 0.0, 1.0
    elif sa == sb:
        p_a, p_d, p_b = 0.0, 1.0, 0.0
    else:
        p_a, p_d, p_b = 0.333, 0.333, 0.333

    return MatchPrediction(
        match_id=str(mid),
        team_a=mp["team_a"],
        team_b=mp["team_b"],
        round=mp.get("round", f"Group Stage - {letter}"),
        group=letter,
        match_date=mp.get("match_date", mp.get("date", ""))[:10],
        p_team_a_win=p_a,
        p_draw=p_d,
        p_team_b_win=p_b,
        expected_score_a=float(sa),
        expected_score_b=float(sb),
    )


@router.get("/snapshots")
def get_snapshots():
    """List all saved prediction snapshots in chronological order."""
    return list_snapshots()


@router.get("/snapshots/{stage}")
def get_snapshot(stage: str):
    """Load a specific named snapshot by stage."""
    try:
        return load_snapshot(stage)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {stage}")


@router.get("/probability-history/{team}")
def get_probability_history_endpoint(team: str):
    """Championship probability over time for one team — used for the history chart."""
    history = get_probability_history(team)
    if not history:
        raise HTTPException(status_code=404, detail=f"No snapshot data for: {team}")
    return {"team": team, "history": history}


@router.get("/meta", response_model=MetaInfo)
def get_meta() -> MetaInfo:
    """Return sidebar/header status metadata."""
    try:
        sim = _load_sim()
    except HTTPException:
        comp = _load_comparison()
        return MetaInfo(
            model_version="ensemble_v1",
            last_simulation="unknown",
            n_simulations=0,
            ensemble_brier_wc2022=comp.get("ensemble_wc2022_brier"),
        )
    return _meta_from_sim(sim)


@router.get("/tournament-bracket", response_model=TournamentBracket)
def get_tournament_bracket() -> TournamentBracket:
    """
    Returns all matches grouped by round, with actual results for completed
    matches and live predictions for upcoming ones.
    """
    sim = _load_live_sim()
    if not sim:
        return TournamentBracket(
            current_round="Pre-tournament",
            last_updated="",
            rounds=[],
        )

    meta = sim.get("metadata", {})
    actual = sim.get("actual_results", {})
    preds = sim.get("match_predictions", {})

    all_matches: dict[str, dict] = {}

    for mid, res in actual.items():
        row = {
            "match_id": mid,
            "team_a": res.get("team_a", ""),
            "team_b": res.get("team_b", ""),
            "round": res.get("round", ""),
            "group": None,
            "match_date": res.get("match_date"),
            "status": "completed",
            "actual_score_a": res.get("score_a"),
            "actual_score_b": res.get("score_b"),
            "actual_winner": res.get("winner"),
            "p_team_a_win": None,
            "p_draw": None,
            "p_team_b_win": None,
            "expected_score_a": None,
            "expected_score_b": None,
            "model_correct": None,
            "has_pre_match_prediction": False,
        }
        all_matches[mid] = enrich_completed_match(
            mid, row["team_a"], row["team_b"], row
        )

    for mid, pred in preds.items():
        status = pred.get("status", "predicted")
        round_key = _round_key(pred.get("round", ""))
        pair_key = (round_key, _match_pair_key(
            pred.get("team_a", ""), pred.get("team_b", "")
        ))
        if mid not in all_matches and status == "predicted":
            duplicate = any(
                _round_key(m["round"]) == round_key
                and _match_pair_key(m["team_a"], m["team_b"]) == pair_key[1]
                for m in all_matches.values()
            )
            if duplicate:
                continue
        if mid in all_matches:
            prev = all_matches[mid]
            if pred.get("p_team_a_win") is not None:
                prev["p_team_a_win"] = pred.get("p_team_a_win")
                prev["p_draw"] = pred.get("p_draw")
                prev["p_team_b_win"] = pred.get("p_team_b_win")
                prev["expected_score_a"] = pred.get("expected_score_a", 1.0)
                prev["expected_score_b"] = pred.get("expected_score_b", 1.0)
                prev["has_pre_match_prediction"] = True
            if prev.get("has_pre_match_prediction"):
                prev["model_correct"] = _outcome_correct(
                    {
                        **pred,
                        "team_a": prev["team_a"],
                        "team_b": prev["team_b"],
                        "p_team_a_win": prev["p_team_a_win"],
                        "p_draw": prev["p_draw"],
                        "p_team_b_win": prev["p_team_b_win"],
                    },
                    prev["actual_winner"],
                    prev.get("actual_score_a") or 0,
                    prev.get("actual_score_b") or 0,
                )
            elif (
                prev.get("p_team_a_win") is not None
                and prev.get("p_draw") is not None
                and prev.get("p_team_b_win") is not None
            ):
                prev["has_pre_match_prediction"] = True
                prev["model_correct"] = _outcome_correct(
                    {
                        "team_a": prev["team_a"],
                        "team_b": prev["team_b"],
                        "p_team_a_win": prev["p_team_a_win"],
                        "p_draw": prev["p_draw"],
                        "p_team_b_win": prev["p_team_b_win"],
                    },
                    prev["actual_winner"],
                    prev.get("actual_score_a") or 0,
                    prev.get("actual_score_b") or 0,
                )
            if pred.get("group"):
                prev["group"] = pred.get("group")
        else:
            all_matches[mid] = {
                "match_id": mid,
                "team_a": pred.get("team_a", ""),
                "team_b": pred.get("team_b", ""),
                "round": pred.get("round", ""),
                "group": pred.get("group"),
                "match_date": pred.get("match_date"),
                "status": status,
                "actual_score_a": None,
                "actual_score_b": None,
                "actual_winner": None,
                "p_team_a_win": pred.get("p_team_a_win", 0.33),
                "p_draw": pred.get("p_draw", 0.33),
                "p_team_b_win": pred.get("p_team_b_win", 0.34),
                "expected_score_a": pred.get("expected_score_a", 1.0),
                "expected_score_b": pred.get("expected_score_b", 1.0),
                "model_correct": None,
                "has_pre_match_prediction": pred.get("p_team_a_win") is not None,
            }

    by_round: dict[str, list] = {}
    for m in all_matches.values():
        key = _round_key(m["round"])
        by_round.setdefault(key, []).append(m)

    rounds: list[RoundData] = []
    for key in ROUND_ORDER:
        matches = sorted(
            by_round.get(key, []),
            key=lambda x: (x.get("match_date") or "", x["match_id"]),
        )
        if not matches:
            continue
        completed = sum(1 for m in matches if m["status"] == "completed")
        rounds.append(
            RoundData(
                round_name=ROUND_LABELS.get(key, key),
                round_key=key,
                matches=[MatchResultCard(**m) for m in matches],
                completed=completed,
                total=len(matches),
            )
        )

    return TournamentBracket(
        current_round=meta.get("tournament_phase", "Round of 32"),
        last_updated=meta.get("data_as_of", meta.get("run_at", "")),
        rounds=rounds,
    )


@router.get("/match-prediction-history", response_model=MatchPredictionHistory)
def get_match_prediction_history(team_a: str, team_b: str) -> MatchPredictionHistory:
    """
    Returns predictions for a specific matchup across all saved named snapshots,
    plus the actual result if the match has been played.
    """
    sim = _load_live_sim()
    actual = sim.get("actual_results", {})

    actual_result = None
    for res in actual.values():
        ta, tb = res.get("team_a", ""), res.get("team_b", "")
        if {ta, tb} == {team_a, team_b}:
            actual_result = res
            break

    predictions: list[MatchPredictionAtStage] = []
    for snap_meta in list_snapshots():
        if snap_meta.get("type") != "milestone":
            continue
        stage = snap_meta["stage"]
        try:
            snap_data = load_snapshot(stage)
        except FileNotFoundError:
            continue

        for pred in snap_data.get("match_predictions", {}).values():
            ta, tb = pred.get("team_a", ""), pred.get("team_b", "")
            if {ta, tb} == {team_a, team_b}:
                if ta == team_a:
                    p_a = pred.get("p_team_a_win", 0.33)
                    p_b = pred.get("p_team_b_win", 0.34)
                    sa = pred.get("expected_score_a", 1.0)
                    sb = pred.get("expected_score_b", 1.0)
                else:
                    p_a = pred.get("p_team_b_win", 0.34)
                    p_b = pred.get("p_team_a_win", 0.33)
                    sa = pred.get("expected_score_b", 1.0)
                    sb = pred.get("expected_score_a", 1.0)
                predictions.append(
                    MatchPredictionAtStage(
                        stage=stage,
                        saved_at=snap_meta.get("saved_at", ""),
                        p_team_a_win=p_a,
                        p_draw=pred.get("p_draw", 0.33),
                        p_team_b_win=p_b,
                        expected_score_a=sa,
                        expected_score_b=sb,
                    )
                )
                break

    if actual_result is None:
        for pred in sim.get("match_predictions", {}).values():
            ta, tb = pred.get("team_a", ""), pred.get("team_b", "")
            if {ta, tb} == {team_a, team_b} and pred.get("status") == "predicted":
                if ta == team_a:
                    p_a = pred.get("p_team_a_win", 0.33)
                    p_b = pred.get("p_team_b_win", 0.34)
                    sa = pred.get("expected_score_a", 1.0)
                    sb = pred.get("expected_score_b", 1.0)
                else:
                    p_a = pred.get("p_team_b_win", 0.34)
                    p_b = pred.get("p_team_a_win", 0.33)
                    sa = pred.get("expected_score_b", 1.0)
                    sb = pred.get("expected_score_a", 1.0)
                predictions.append(
                    MatchPredictionAtStage(
                        stage="current",
                        saved_at=sim.get("metadata", {}).get("run_at", ""),
                        p_team_a_win=p_a,
                        p_draw=pred.get("p_draw", 0.33),
                        p_team_b_win=p_b,
                        expected_score_a=sa,
                        expected_score_b=sb,
                    )
                )
                break

    played = actual_result is not None
    if played:
        ta = actual_result.get("team_a", "")
        if ta == team_a:
            score_a = actual_result.get("score_a")
            score_b = actual_result.get("score_b")
        else:
            score_a = actual_result.get("score_b")
            score_b = actual_result.get("score_a")
        winner = actual_result.get("winner")
    else:
        score_a = score_b = winner = None

    return MatchPredictionHistory(
        team_a=team_a,
        team_b=team_b,
        actual_score_a=score_a,
        actual_score_b=score_b,
        actual_winner=winner,
        played=played,
        predictions=predictions,
    )


@router.get("/accuracy", response_model=AccuracyStats)
def get_accuracy() -> AccuracyStats:
    """
    Live accuracy of the model on completed WC 2026 matches.
    Shows correct outcome % and rolling Brier score by round.
    """
    import numpy as np
    from sklearn.metrics import brier_score_loss

    sim = _load_live_sim()
    actual = sim.get("actual_results", {})
    preds = sim.get("match_predictions", {})

    total = correct = 0
    brier_scores: list[float] = []
    correct_by_round: dict[str, list] = {}
    brier_by_round: dict[str, list] = {}

    for mid, res in actual.items():
        if res.get("score_a") is None:
            continue
        pred = preds.get(mid)
        if not pred:
            continue

        ta, tb = res.get("team_a", ""), res.get("team_b", "")
        sa, sb = int(res.get("score_a", 0)), int(res.get("score_b", 0))
        winner = res.get("winner")

        p = np.array([
            pred.get("p_team_b_win", 0.33),
            pred.get("p_draw", 0.33),
            pred.get("p_team_a_win", 0.34),
        ])
        if sa > sb:
            y = 2
        elif sa == sb:
            y = 1
        else:
            y = 0

        bs = float(np.mean([
            brier_score_loss([int(y == c)], [p[c]]) for c in range(3)
        ]))
        brier_scores.append(bs)

        predicted_winner = (
            ta if p[2] >= p[0] and p[2] >= p[1]
            else tb if p[0] >= p[1] and p[0] >= p[2]
            else "draw"
        )
        is_correct = (
            (predicted_winner == "draw" and sa == sb)
            or (predicted_winner == winner and winner is not None)
        )
        total += 1
        if is_correct:
            correct += 1

        rnd = _round_key(res.get("round", ""))
        correct_by_round.setdefault(rnd, []).append(int(is_correct))
        brier_by_round.setdefault(rnd, []).append(bs)

    return AccuracyStats(
        total_played=total,
        correct_outcome=correct,
        correct_pct=correct / total if total > 0 else 0.0,
        current_brier=float(np.mean(brier_scores)) if brier_scores else None,
        brier_by_round={k: float(np.mean(v)) for k, v in brier_by_round.items()},
        correct_by_round={k: float(np.mean(v)) for k, v in correct_by_round.items()},
    )
