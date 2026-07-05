"""Story page API — prediction evolution, movers reasoning, accuracy, match explanations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from config.settings import settings
from src.retraining.snapshots import load_snapshot

router = APIRouter()

STAGE_ORDER = [
    "pre_tournament",
    "post_group_stage",
    "post_r32",
    "post_r16",
    "post_qf",
    "post_sf",
]

STAGE_DISPLAY_SHORT = {
    "pre_tournament": "Pre-Tournament",
    "post_group_stage": "Entering R32",
    "post_r32": "Entering R16",
    "post_r16": "Entering QF",
    "post_qf": "Entering SF",
    "post_sf": "Entering Final",
}

STAGE_DISPLAY_LONG = {
    "pre_tournament": "Pre-Tournament",
    "post_group_stage": "After Group Stage → Round of 32",
    "post_r32": "After Round of 32 → Round of 16",
    "post_r16": "After Round of 16 → Quarter-finals",
    "post_qf": "After Quarter-finals → Semi-finals",
    "post_sf": "After Semi-finals → Final",
}


def _stage_labels(stage: str) -> dict[str, str]:
    long_label = STAGE_DISPLAY_LONG.get(stage, stage)
    short_label = STAGE_DISPLAY_SHORT.get(stage, stage)
    return {
        "display": long_label,
        "display_short": short_label,
        "display_long": long_label,
    }

STAGE_PREDICTS_ROUND = {
    "pre_tournament": "Group Stage",
    "post_group_stage": "Round of 32",
    "post_r32": "Round of 16",
    "post_r16": "Quarter-finals",
    "post_qf": "Semi-finals",
    "post_sf": "Final",
}

STAGE_CUT_DATES = {
    "pre_tournament": "2026-06-11",
    "post_group_stage": "2026-07-02",
    "post_r32": "2026-07-05",
    "post_r16": "2026-07-09",
    "post_qf": "2026-07-12",
    "post_sf": "2026-07-15",
}

CONTEXT_ONLY_FEATURES = {"elo_a", "elo_b", "h2h_n_matches", "round_importance"}
EXCLUDED_REDUNDANT_FEATURES = {"market_value_ratio"}
INVERSE_DIFF_FEATURES = {"goals_conceded_diff"}
CENTERED_AT_HALF_FEATURES = {"h2h_win_rate", "h2h_neutral_win_rate"}

FEATURE_SCALE = {
    "elo_diff": 400.0,
    "elo_trajectory_diff": 150.0,
    "fifa_pts_diff": 800.0,
    "squad_log_value_diff": 3.0,
    "ppg_diff": 2.0,
    "goals_scored_diff": 1.5,
    "goals_conceded_diff": 1.5,
    "clean_sheet_diff": 0.5,
    "win_pct_diff": 0.5,
    "neutral_win_pct_diff": 0.4,
    "h2h_win_rate": 0.3,
    "h2h_goal_diff_avg": 1.5,
    "h2h_neutral_win_rate": 0.3,
    "wc_appearances_diff": 10.0,
    "avg_wc_finish_diff": 2.0,
    "rest_days_diff": 3.0,
}

FEATURE_DISPLAY = {
    "elo_diff": ("Elo rating gap", "Long-run team strength difference"),
    "elo_a": ("Team A Elo", "Absolute strength rating"),
    "elo_b": ("Team B Elo", "Absolute strength rating"),
    "elo_trajectory_diff": ("Elo momentum (90 days)", "Who is rising or falling in strength"),
    "fifa_pts_diff": ("FIFA ranking points gap", "Official ranking points difference"),
    "squad_log_value_diff": ("Squad market value gap", "Talent depth difference (log scale)"),
    "market_value_ratio": ("Squad value ratio", "How many times richer in talent"),
    "ppg_diff": ("Points per game gap", "Recent competitive results (last 10)"),
    "goals_scored_diff": ("Goals scored gap", "Attacking output per game (last 10)"),
    "goals_conceded_diff": ("Goals conceded gap", "Defensive solidity per game (last 10)"),
    "clean_sheet_diff": ("Clean sheet % gap", "Shutout frequency (last 10)"),
    "win_pct_diff": ("Win % gap", "Raw winning frequency (last 10)"),
    "neutral_win_pct_diff": ("Neutral-venue win % gap", "Form at neutral venues — WC conditions"),
    "h2h_win_rate": ("Head-to-head win rate", "Historical record vs this opponent"),
    "h2h_goal_diff_avg": ("H2H avg goal margin", "How decisively they beat each other"),
    "h2h_neutral_win_rate": ("H2H neutral win rate", "Direct meetings at neutral venues"),
    "h2h_n_matches": ("H2H meetings count", "How reliable the H2H data is"),
    "wc_appearances_diff": ("WC experience gap", "World Cups played difference"),
    "avg_wc_finish_diff": ("WC depth-of-run gap", "How far each historically advances"),
    "rest_days_diff": ("Rest days gap", "Recovery time difference"),
    "round_importance": ("Match importance", "Group stage vs knockout weight"),
}


def _load_all_milestone_snapshots() -> dict[str, dict]:
    """Load every existing milestone snapshot keyed by stage, in order."""
    out: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        try:
            out[stage] = load_snapshot(stage)
        except FileNotFoundError:
            continue
    return out


def _match_pred_for_pair(snapshot: dict, team_a: str, team_b: str) -> dict | None:
    """Find a match prediction for a team pair in a snapshot (order-agnostic).

    Returns the prediction dict normalised so team_a matches the requested team_a.
    Bracket projections (~ prefix) are matched but flagged via ``is_projection``.
    """
    for pred in snapshot.get("match_predictions", {}).values():
        ta, tb = pred.get("team_a", ""), pred.get("team_b", "")
        ta_clean, tb_clean = ta.lstrip("~"), tb.lstrip("~")
        if {ta_clean, tb_clean} != {team_a, team_b}:
            continue

        is_projection = ta.startswith("~") or tb.startswith("~")
        if ta_clean == team_a:
            return {**pred, "is_projection": is_projection}

        return {
            **pred,
            "team_a": tb,
            "team_b": ta,
            "p_team_a_win": pred.get("p_team_b_win", 0.33),
            "p_team_b_win": pred.get("p_team_a_win", 0.33),
            "expected_score_a": pred.get("expected_score_b", 1.0),
            "expected_score_b": pred.get("expected_score_a", 1.0),
            "is_projection": is_projection,
        }
    return None


def _brier_one(p_bwin: float, p_draw: float, p_awin: float, outcome: int) -> float:
    """Multi-class Brier for one match. outcome: 2=a wins, 1=draw, 0=b wins."""
    probs = [p_bwin, p_draw, p_awin]
    return float(np.mean([(probs[c] - (1 if outcome == c else 0)) ** 2 for c in range(3)]))


def _outcome_of(score_a: int, score_b: int) -> int:
    if score_a > score_b:
        return 2
    if score_a < score_b:
        return 0
    return 1


def _favourite_of(pred: dict) -> str:
    """'a', 'b', or 'draw' — which outcome the prediction favoured."""
    pa = pred.get("p_team_a_win", 0)
    pd_ = pred.get("p_draw", 0)
    pb = pred.get("p_team_b_win", 0)
    if pa >= pd_ and pa >= pb:
        return "a"
    if pb >= pd_ and pb >= pa:
        return "b"
    return "draw"


@router.get("/story/evolution")
def get_evolution():
    """
    Championship/final/semi odds for every team at every available milestone,
    plus per-stage metadata for the timeline header.
    """
    snaps = _load_all_milestone_snapshots()
    if not snaps:
        raise HTTPException(404, "No snapshots available")

    stages_meta = []
    teams: dict[str, list] = {}

    for stage, snap in snaps.items():
        meta = snap.get("snapshot_metadata", {})
        inner = snap.get("metadata", {})
        stages_meta.append({
            "stage": stage,
            **_stage_labels(stage),
            "saved_at": meta.get("saved_at", ""),
            "teams_alive": sum(
                1
                for v in snap.get("team_probabilities", {}).values()
                if v.get("p_champion", 0) > 0
            ),
            "matches_completed": meta.get("matches_completed", 0),
            "reconstructed": bool(inner.get("reconstructed", False)),
            "model_version": meta.get("model_version", ""),
        })

        for team, v in snap.get("team_probabilities", {}).items():
            teams.setdefault(team, []).append({
                "stage": stage,
                "p_champion": v.get("p_champion", 0),
                "p_final": v.get("p_final", 0),
                "p_semi": v.get("p_semi", 0),
                "eliminated": v.get("p_champion", 0) == 0 and stage != "pre_tournament",
                "eliminated_in": v.get("eliminated_in"),
            })

    return {"stages": stages_meta, "teams": teams}


@router.get("/story/movers")
def get_movers(
    from_stage: str = Query(...),
    to_stage: str = Query(...),
    top_n: int = 6,
):
    snaps = _load_all_milestone_snapshots()
    if from_stage not in snaps or to_stage not in snaps:
        raise HTTPException(404, f"Snapshot missing: need both {from_stage} and {to_stage}")

    prev, curr = snaps[from_stage], snaps[to_stage]
    prev_tp = prev.get("team_probabilities", {})
    curr_tp = curr.get("team_probabilities", {})

    prev_result_keys = set(prev.get("actual_results", {}).keys())
    new_results = {
        k: v
        for k, v in curr.get("actual_results", {}).items()
        if k not in prev_result_keys
    }

    def team_record_in(results: dict, team: str) -> dict:
        w = d = l = gf = ga = 0
        matches = []
        for res in results.values():
            ta, tb = res.get("team_a"), res.get("team_b")
            if team not in (ta, tb):
                continue
            sa, sb = res.get("score_a", 0) or 0, res.get("score_b", 0) or 0
            mine, theirs = (sa, sb) if team == ta else (sb, sa)
            opp = tb if team == ta else ta
            gf += mine
            ga += theirs
            if mine > theirs:
                w += 1
                verdict = "W"
            elif mine < theirs:
                l += 1
                verdict = "L"
            else:
                d += 1
                verdict = "D"
            matches.append({
                "opponent": opp,
                "score": f"{mine}-{theirs}",
                "result": verdict,
                "round": res.get("round", ""),
            })
        return {"wins": w, "draws": d, "losses": l, "gf": gf, "ga": ga, "matches": matches}

    elo_df = pd.read_parquet(settings.DATA_DIR / "processed" / "elo_clean.parquet")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    prev_date = pd.Timestamp(prev.get("snapshot_metadata", {}).get("saved_at", "2026-06-11")[:10])
    curr_date = pd.Timestamp(curr.get("snapshot_metadata", {}).get("saved_at", "2026-07-05")[:10])

    def elo_change(team: str) -> float | None:
        rows = elo_df[elo_df["team"] == team].sort_values("date")
        before = rows[rows["date"] <= prev_date]["elo_rating"]
        after = rows[rows["date"] <= curr_date]["elo_rating"]
        if len(before) and len(after):
            return round(float(after.iloc[-1]) - float(before.iloc[-1]), 1)
        return None

    changes = []
    for team in prev_tp:
        p0 = prev_tp[team].get("p_champion", 0)
        p1 = curr_tp.get(team, {}).get("p_champion", 0)
        changes.append({
            "team": team,
            "before": p0,
            "after": p1,
            "delta": p1 - p0,
            "eliminated_in": curr_tp.get(team, {}).get("eliminated_in"),
            "record": team_record_in(new_results, team),
            "elo_change": elo_change(team),
        })

    risers = sorted([c for c in changes if c["delta"] > 0], key=lambda x: -x["delta"])[:top_n]
    fallers = sorted([c for c in changes if c["delta"] < 0], key=lambda x: x["delta"])[:top_n]

    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "from_display": STAGE_DISPLAY_LONG.get(from_stage),
        "from_display_short": STAGE_DISPLAY_SHORT.get(from_stage),
        "from_display_long": STAGE_DISPLAY_LONG.get(from_stage),
        "to_display": STAGE_DISPLAY_LONG.get(to_stage),
        "to_display_short": STAGE_DISPLAY_SHORT.get(to_stage),
        "to_display_long": STAGE_DISPLAY_LONG.get(to_stage),
        "new_results_count": len(new_results),
        "risers": risers,
        "fallers": fallers,
    }


@router.get("/story/accuracy")
def get_accuracy_report():
    snaps = _load_all_milestone_snapshots()

    live_path = settings.DATA_DIR / "predictions" / "tournament_simulation.json"
    with open(live_path) as f:
        live = json.load(f)
    all_results = live.get("actual_results", {})

    per_stage = []
    all_scored = []

    for stage in STAGE_ORDER:
        if stage not in snaps:
            continue
        snap = snaps[stage]
        target_round = STAGE_PREDICTS_ROUND.get(stage)
        if not target_round:
            continue

        round_results = {
            k: v
            for k, v in all_results.items()
            if v.get("round", "").startswith(target_round)
            and v.get("score_a") is not None
        }
        if not round_results:
            per_stage.append({
                "stage": stage,
                **_stage_labels(stage),
                "predicts_round": target_round,
                "n_matches": 0,
                "status": "round not yet played",
            })
            continue

        n = correct = 0
        briers = []
        upsets = []

        for res in round_results.values():
            ta, tb = res["team_a"], res["team_b"]
            pred = _match_pred_for_pair(snap, ta, tb)
            if pred is None or pred.get("is_projection"):
                continue

            sa, sb = int(res["score_a"]), int(res["score_b"])
            outcome = _outcome_of(sa, sb)
            fav = _favourite_of(pred)
            fav_correct = (
                (fav == "a" and outcome == 2)
                or (fav == "b" and outcome == 0)
                or (fav == "draw" and outcome == 1)
            )
            fav_prob = max(pred["p_team_a_win"], pred["p_draw"], pred["p_team_b_win"])
            bs = _brier_one(
                pred["p_team_b_win"], pred["p_draw"], pred["p_team_a_win"], outcome
            )

            n += 1
            briers.append(bs)
            if fav_correct:
                correct += 1
            else:
                fav_team = ta if fav == "a" else tb if fav == "b" else "Draw"
                actual = ta if outcome == 2 else tb if outcome == 0 else "Draw"
                upsets.append({
                    "team_a": ta,
                    "team_b": tb,
                    "score": f"{sa}-{sb}",
                    "round": res.get("round", ""),
                    "model_favoured": fav_team,
                    "confidence": round(fav_prob, 3),
                    "actual": actual,
                })
            all_scored.append({"conf": fav_prob, "correct": fav_correct, "brier": bs})

        per_stage.append({
            "stage": stage,
            **_stage_labels(stage),
            "predicts_round": target_round,
            "reconstructed": bool(snap.get("metadata", {}).get("reconstructed", False)),
            "n_matches": n,
            "correct": correct,
            "correct_pct": round(correct / n, 3) if n else None,
            "brier": round(float(np.mean(briers)), 4) if briers else None,
            "upsets": upsets,
        })

    buckets = [(0.34, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]
    calibration = []
    for lo, hi in buckets:
        in_bucket = [s for s in all_scored if lo <= s["conf"] < hi]
        if in_bucket:
            calibration.append({
                "bucket": f"{int(lo * 100)}–{int(hi * 100) if hi <= 1 else 100}%",
                "midpoint": round((lo + min(hi, 1.0)) / 2, 3),
                "n": len(in_bucket),
                "predicted_rate": round(float(np.mean([s["conf"] for s in in_bucket])), 3),
                "actual_rate": round(float(np.mean([s["correct"] for s in in_bucket])), 3),
            })

    overall_brier = (
        round(float(np.mean([s["brier"] for s in all_scored])), 4) if all_scored else None
    )
    overall_correct = (
        round(float(np.mean([s["correct"] for s in all_scored])), 3) if all_scored else None
    )

    return {
        "per_stage": per_stage,
        "calibration": calibration,
        "overall": {
            "n_matches": len(all_scored),
            "correct_pct": overall_correct,
            "brier": overall_brier,
            "wc2022_benchmark_brier": 0.1897,
        },
    }


@router.get("/story/match-explanation")
def get_match_explanation(
    team_a: str = Query(...),
    team_b: str = Query(...),
    stage: str = Query("post_r32"),
):
    """
    Feature-level explanation of a matchup prediction as of a given stage.
    Rebuilds the exact feature vector the model saw at that point in time.
    """
    from src.api.predict_service import build_feature_vectors
    from src.features.pipeline import FEATURE_COLS_TREES

    cut_date = STAGE_CUT_DATES.get(stage)
    if cut_date is None:
        raise HTTPException(400, f"Unknown stage: {stage}")

    try:
        X_trees, _X_linear = build_feature_vectors(team_a, team_b, cut_date=cut_date)
    except Exception as e:
        raise HTTPException(422, f"Could not build features: {e}") from e

    row = X_trees[0]
    raw_values = dict(zip(FEATURE_COLS_TREES, row))

    context = {
        "team_a_elo": round(float(raw_values.get("elo_a", 0)), 0)
        if raw_values.get("elo_a") is not None
        and not (isinstance(raw_values.get("elo_a"), float) and np.isnan(raw_values.get("elo_a")))
        else None,
        "team_b_elo": round(float(raw_values.get("elo_b", 0)), 0)
        if raw_values.get("elo_b") is not None
        and not (isinstance(raw_values.get("elo_b"), float) and np.isnan(raw_values.get("elo_b")))
        else None,
        "h2h_meetings": int(raw_values.get("h2h_n_matches", 0))
        if raw_values.get("h2h_n_matches") is not None
        and not (
            isinstance(raw_values.get("h2h_n_matches"), float)
            and np.isnan(raw_values.get("h2h_n_matches"))
        )
        else None,
        "round_importance": float(raw_values.get("round_importance", 0))
        if raw_values.get("round_importance") is not None
        and not (
            isinstance(raw_values.get("round_importance"), float)
            and np.isnan(raw_values.get("round_importance"))
        )
        else None,
    }

    features = []
    for col, val in raw_values.items():
        if col in CONTEXT_ONLY_FEATURES or col in EXCLUDED_REDUNDANT_FEATURES:
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        val = float(val)
        display, description = FEATURE_DISPLAY.get(col, (col, ""))

        if col in CENTERED_AT_HALF_FEATURES:
            centered = val - 0.5
        elif col in INVERSE_DIFF_FEATURES:
            centered = -val
        else:
            centered = val

        if abs(centered) < 1e-9:
            favours = "neutral"
        else:
            favours = "a" if centered > 0 else "b"

        scale = FEATURE_SCALE.get(col, 1.0)
        normalised_magnitude = abs(centered) / scale if scale else abs(centered)

        features.append({
            "feature": col,
            "display": display,
            "description": description,
            "value": round(val, 3),
            "favours": favours,
            "normalised_magnitude": round(min(normalised_magnitude, 1.0), 3),
        })

    features.sort(key=lambda f: -f["normalised_magnitude"])

    snaps = _load_all_milestone_snapshots()
    recorded = None
    if stage in snaps:
        raw = _match_pred_for_pair(snaps[stage], team_a, team_b)
        if raw is not None:
            recorded = {k: v for k, v in raw.items() if k != "is_projection"}

    live_path = settings.DATA_DIR / "predictions" / "tournament_simulation.json"
    with open(live_path) as f:
        live = json.load(f)
    actual = None
    for res in live.get("actual_results", {}).values():
        if {res.get("team_a"), res.get("team_b")} == {team_a, team_b} and res.get("score_a") is not None:
            sa, sb = res["score_a"], res["score_b"]
            if res["team_a"] != team_a:
                sa, sb = sb, sa
            actual = {
                "score_a": sa,
                "score_b": sb,
                "winner": res.get("winner"),
                "round": res.get("round", ""),
            }
            break

    n_a = sum(1 for f in features if f["favours"] == "a")
    n_b = sum(1 for f in features if f["favours"] == "b")
    n_total = len(features)

    if recorded is not None:
        pa, pd_, pb = recorded["p_team_a_win"], recorded["p_draw"], recorded["p_team_b_win"]
        fav_conf = max(pa, pd_, pb)
        fav_team = team_a if pa == fav_conf else team_b if pb == fav_conf else "Draw"
        call_sentence = f"The model's call: {fav_team} at {fav_conf:.0%}."
    else:
        call_sentence = (
            "No probability estimate exists for this exact matchup at this "
            "stage — it was not a scheduled fixture in this snapshot."
        )

    summary = (
        f"As of {cut_date}, {n_a} of {n_total} comparative features favoured "
        f"{team_a}, {n_b} favoured {team_b}. {call_sentence}"
    )

    labels = _stage_labels(stage)
    return {
        "team_a": team_a,
        "team_b": team_b,
        "stage": stage,
        "stage_display": labels["display_long"],
        **labels,
        "cut_date": cut_date,
        "context": context,
        "features": features,
        "summary": summary,
        "recorded_prediction": recorded,
        "actual_result": actual,
        "reconstructed": bool(
            stage in snaps and snaps[stage].get("metadata", {}).get("reconstructed", False)
        ),
        "notes": (
            "Feature vector reconstructed as of the cut date using the same "
            "point-in-time pipeline used in training. Raw team ratings "
            "(elo_a, elo_b) and non-comparative metadata (H2H meeting count, "
            "match importance weight) are shown as context only and excluded "
            "from the favours comparison. market_value_ratio is omitted as "
            "redundant with squad_log_value_diff. Features are sorted by "
            "magnitude relative to each feature's typical range, not raw units."
        ),
    }
