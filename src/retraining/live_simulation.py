"""
Simulates only the REMAINING knockout matches from the current real tournament state.
Completed matches are treated as fixed facts — not re-simulated.
"""
import json
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from config.team_names import WC2026_CANONICAL_TEAMS
from config.settings import settings
from src.api.prediction_lookup import lookup_pre_match_prediction
from src.models.ensemble import predict_ensemble
from src.models.dixon_coles import DixonColesModel
from src.features.elo_features import get_elo_on_date
from src.simulation.scoreline import sample_scoreline
from src.simulation.penalties import simulate_penalties

SIM_PATH = settings.DATA_DIR / "predictions" / "tournament_simulation.json"
WC2026_TEAMS = WC2026_CANONICAL_TEAMS


def build_tournament_state(
    fixtures: list[dict],
    standings: dict[str, list[dict]],
    bracket: list[dict],
) -> dict:
    """Build a structured representation of the current tournament state."""
    group_stage_complete = all(
        f["status"] == "FT"
        for f in fixtures
        if "Group Stage" in f.get("round", "")
    )

    qualified: dict[str, dict] = {}
    for group, teams in standings.items():
        sorted_teams = sorted(teams, key=lambda t: t["rank"])
        qualified[group] = {
            "1st": sorted_teams[0]["team"] if len(sorted_teams) > 0 else "TBD",
            "2nd": sorted_teams[1]["team"] if len(sorted_teams) > 1 else "TBD",
            "3rd": sorted_teams[2]["team"] if len(sorted_teams) > 2 else "TBD",
        }

    r32_matches = []
    for f in bracket:
        r32_matches.append(
            {
                "fixture_id": f["fixture_id"],
                "team_a": f["home_team"],
                "team_b": f["away_team"],
                "status": f["status"],
                "score_a": f["home_score"],
                "score_b": f["away_score"],
                "winner": (
                    f["home_team"]
                    if f.get("home_winner")
                    else f["away_team"]
                    if f.get("away_winner")
                    else None
                ),
            }
        )

    eliminated = set()
    for m in r32_matches:
        if m["status"] == "FT" and m["winner"]:
            loser = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
            eliminated.add(loser)

    return {
        "group_stage_complete": group_stage_complete,
        "group_standings": qualified,
        "r32_matches": r32_matches,
        "eliminated_teams": list(eliminated),
        "current_round": "Round of 32",
    }


def _precompute_caches(
    remaining_teams: list[str],
    ensemble,
    dc_model: DixonColesModel,
    elo_df: pd.DataFrame,
    cut_date: str = "2026-07-04",
) -> tuple[dict, dict, dict]:
    """Pre-compute ensemble probabilities + DC lambdas for all remaining team pairs."""
    from src.api.predict_service import build_feature_vectors, calibrate_lambdas

    prob_cache: dict[tuple, np.ndarray] = {}
    dc_cache: dict[tuple, tuple] = {}
    elo_cache: dict[str, float] = {}

    cut = pd.Timestamp(cut_date)
    for team in remaining_teams:
        elo_cache[team] = get_elo_on_date(elo_df, team, cut)

    logger.info(f"Pre-computing caches for {len(remaining_teams)} teams...")
    for i, ta in enumerate(remaining_teams):
        for tb in remaining_teams[i + 1 :]:
            X_trees, X_linear = build_feature_vectors(ta, tb, cut_date=cut_date)
            proba = predict_ensemble(ensemble, X_trees, X_linear)[0]

            dc = dc_model.predict(ta, tb, neutral=True)
            lam_a = dc["lambda_a"]
            lam_b = dc["lambda_b"]
            lam_a, lam_b = calibrate_lambdas(
                lam_a, lam_b, float(proba[2]), float(proba[0])
            )

            prob_cache[(ta, tb)] = proba
            prob_cache[(tb, ta)] = np.array([proba[2], proba[1], proba[0]])
            dc_cache[(ta, tb)] = (lam_a, lam_b)
            dc_cache[(tb, ta)] = (lam_b, lam_a)

    return prob_cache, dc_cache, elo_cache


def _compute_forecast(
    team_a: str,
    team_b: str,
    ensemble,
    dc_model: DixonColesModel,
    cut_date: str = "2026-06-11",
) -> dict:
    """Run the ensemble for one fixture — used when no archived forecast exists."""
    from src.api.predict_service import (
        build_feature_vectors,
        calibrate_lambdas,
        simulate_match_scoreline,
    )

    X_trees, X_linear = build_feature_vectors(team_a, team_b, cut_date=cut_date)
    proba = predict_ensemble(ensemble, X_trees, X_linear)[0]
    dc = dc_model.predict(team_a, team_b, neutral=True)
    lam_a, lam_b = calibrate_lambdas(
        dc["lambda_a"], dc["lambda_b"], float(proba[2]), float(proba[0])
    )
    exp_a, exp_b = simulate_match_scoreline(proba, lam_a, lam_b)
    return {
        "p_team_a_win": float(proba[2]),
        "p_draw": float(proba[1]),
        "p_team_b_win": float(proba[0]),
        "expected_score_a": round(exp_a, 2),
        "expected_score_b": round(exp_b, 2),
    }


def _attach_forecast(
    entry: dict,
    team_a: str,
    team_b: str,
    match_id: str,
    prev_preds: dict,
    ensemble,
    dc_model: DixonColesModel,
) -> dict:
    """Ensure every completed match row carries model probabilities in the JSON."""
    prev = prev_preds.get(str(match_id), {})
    if prev.get("status") == "predicted":
        for key in (
            "p_team_a_win",
            "p_draw",
            "p_team_b_win",
            "expected_score_a",
            "expected_score_b",
        ):
            if key in prev:
                entry[key] = prev[key]
        if entry.get("p_team_a_win") is not None:
            entry["forecast_source"] = "archived"
            return entry

    archived = lookup_pre_match_prediction(str(match_id), team_a, team_b)
    if archived:
        entry.update(archived)
        entry["forecast_source"] = "archived"
        return entry

    entry.update(_compute_forecast(team_a, team_b, ensemble, dc_model))
    entry["forecast_source"] = "model"
    return entry


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    prob_cache: dict,
    dc_cache: dict,
    elo_cache: dict,
) -> tuple[str, int, int, bool]:
    """Simulate one knockout match. Returns (winner, score_a, score_b, went_to_penalties)."""
    proba = prob_cache.get((team_a, team_b), np.array([1 / 3, 1 / 3, 1 / 3]))
    outcome = int(np.random.choice([0, 1, 2], p=proba))

    lam_a, lam_b = dc_cache.get((team_a, team_b), (1.3, 1.1))
    score_a, score_b = sample_scoreline(outcome, lam_a, lam_b)

    penalties = False
    if outcome == 1:
        winner = simulate_penalties(team_a, team_b, elo_cache)
        penalties = True
    else:
        winner = team_a if outcome == 2 else team_b

    return winner, score_a, score_b, penalties


def run_live_simulation(
    state: dict,
    ensemble,
    dc_model: DixonColesModel,
    elo_df: pd.DataFrame,
    n_sim: int = 10_000,
    actual_results: dict | None = None,
) -> dict:
    """
    Simulate the remaining tournament from the current real state.
    Completed matches are fixed facts; only future matches are simulated.
    """
    r32 = state["r32_matches"]
    completed_r32 = {m["fixture_id"]: m for m in r32 if m["status"] == "FT"}
    pending_r32 = [m for m in r32 if m["status"] != "FT"]

    remaining_teams = list(
        {
            t
            for m in r32
            for t in (m["team_a"], m["team_b"])
            if t and t not in ("TBD", "")
        }
    )

    prob_cache, dc_cache, elo_cache = _precompute_caches(
        remaining_teams, ensemble, dc_model, elo_df
    )

    counts = {
        team: {
            "champion": 0,
            "final": 0,
            "semi": 0,
            "quarter": 0,
            "r16": 0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "matches_played": 0,
        }
        for team in WC2026_TEAMS
    }

    pending_acc: dict[int, dict] = {
        m["fixture_id"]: {
            "sum_a": 0,
            "sum_b": 0,
            "n": 0,
            "team_a": m["team_a"],
            "team_b": m["team_b"],
        }
        for m in pending_r32
    }

    logger.info(
        f"Running {n_sim:,} simulations | "
        f"{len(completed_r32)} completed R32 matches fixed | "
        f"{len(pending_r32)} R32 matches to simulate"
    )

    for _ in tqdm(range(n_sim), desc="Simulating"):
        r16_teams = []

        for m in completed_r32.values():
            r16_teams.append(m["winner"])
            for t, gf, ga in [
                (m["team_a"], m["score_a"], m["score_b"]),
                (m["team_b"], m["score_b"], m["score_a"]),
            ]:
                if t in counts:
                    counts[t]["goals_for"] += gf or 0
                    counts[t]["goals_against"] += ga or 0
                    counts[t]["matches_played"] += 1

        for m in pending_r32:
            ta, tb = m["team_a"], m["team_b"]
            if ta == "TBD" or tb == "TBD":
                continue
            winner, sa, sb, _ = _simulate_knockout_match(
                ta, tb, prob_cache, dc_cache, elo_cache
            )
            r16_teams.append(winner)
            pending_acc[m["fixture_id"]]["sum_a"] += sa
            pending_acc[m["fixture_id"]]["sum_b"] += sb
            pending_acc[m["fixture_id"]]["n"] += 1
            for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                if t in counts:
                    counts[t]["goals_for"] += gf
                    counts[t]["goals_against"] += ga
                    counts[t]["matches_played"] += 1

        def sim_round(teams: list[str]) -> list[str]:
            winners = []
            n = len(teams) - (len(teams) % 2)
            for i in range(0, n, 2):
                ta, tb = teams[i], teams[i + 1]
                if ta not in elo_cache or tb not in elo_cache:
                    winners.append(ta)
                    continue
                w, sa, sb, _ = _simulate_knockout_match(
                    ta, tb, prob_cache, dc_cache, elo_cache
                )
                winners.append(w)
                for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
            return winners

        qf_teams = sim_round(r16_teams)
        sf_teams = sim_round(qf_teams)
        final_teams = sim_round(sf_teams)

        for t in r16_teams:
            if t in counts:
                counts[t]["r16"] += 1
        for t in qf_teams:
            if t in counts:
                counts[t]["quarter"] += 1
        for t in sf_teams:
            if t in counts:
                counts[t]["semi"] += 1
        for t in final_teams:
            if t in counts:
                counts[t]["final"] += 1

        if len(final_teams) >= 2:
            winner, sa, sb, _ = _simulate_knockout_match(
                final_teams[0], final_teams[1], prob_cache, dc_cache, elo_cache
            )
        elif len(final_teams) == 1:
            winner = final_teams[0]
        else:
            winner = None
        if winner and winner in counts:
            counts[winner]["champion"] += 1

    team_probs: dict[str, dict] = {}
    for team in WC2026_TEAMS:
        c = counts[team]
        mp = max(c["matches_played"], 1)
        in_r32 = any(team in (m["team_a"], m["team_b"]) for m in r32)
        eliminated_in = None
        if team in state.get("eliminated_teams", []):
            eliminated_in = "Round of 32"

        team_probs[team] = {
            "p_champion": c["champion"] / n_sim,
            "p_final": c["final"] / n_sim,
            "p_semi": c["semi"] / n_sim,
            "p_quarter": c["quarter"] / n_sim,
            "p_r16": c["r16"] / n_sim,
            "p_r32": 1.0 if in_r32 else 0.0,
            "p_advance_groups": 1.0 if in_r32 else 0.0,
            "avg_goals_for_pg": c["goals_for"] / mp,
            "avg_goals_against_pg": c["goals_against"] / mp,
            **({"eliminated_in": eliminated_in} if eliminated_in else {}),
        }

    match_preds: dict[str, dict] = {}

    prev_preds: dict[str, dict] = {}
    if SIM_PATH.exists():
        try:
            with open(SIM_PATH) as f:
                prev_preds = json.load(f).get("match_predictions", {})
        except (json.JSONDecodeError, OSError):
            prev_preds = {}

    if actual_results:
        for mid, res in actual_results.items():
            entry = {**res, "status": "completed"}
            entry = _attach_forecast(
                entry,
                res["team_a"],
                res["team_b"],
                str(mid),
                prev_preds,
                ensemble,
                dc_model,
            )
            match_preds[str(mid)] = entry

    for m in pending_r32:
        fid = m["fixture_id"]
        acc = pending_acc.get(fid, {})
        n = acc.get("n", 0)
        ta, tb = m["team_a"], m["team_b"]
        proba = prob_cache.get((ta, tb), np.array([1 / 3, 1 / 3, 1 / 3]))
        match_preds[str(fid)] = {
            "team_a": ta,
            "team_b": tb,
            "round": "Round of 32",
            "status": "predicted",
            "p_team_a_win": float(proba[2]),
            "p_draw": float(proba[1]),
            "p_team_b_win": float(proba[0]),
            "expected_score_a": round(acc["sum_a"] / n, 2) if n else 1.3,
            "expected_score_b": round(acc["sum_b"] / n, 2) if n else 1.1,
        }

    output = {
        "metadata": {
            "n_simulations": n_sim,
            "random_seed": 42,
            "run_at": pd.Timestamp.now().isoformat(),
            "data_as_of": pd.Timestamp.now().isoformat(),
            "model": "ensemble_v1 + lgbm_live + dixon_coles_v1",
            "tournament_phase": state.get("current_round", "Round of 32"),
        },
        "actual_results": actual_results or {},
        "group_stage_standings": state.get("group_standings", {}),
        "team_probabilities": team_probs,
        "match_predictions": match_preds,
        "group_predictions": {},
    }

    with open(SIM_PATH, "w") as f:
        json.dump(output, f, indent=2)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    bak = SIM_PATH.parent / "history" / f"{ts}_simulation.json"
    bak.parent.mkdir(parents=True, exist_ok=True)
    with open(bak, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Live simulation complete. Saved to {SIM_PATH} + backup {bak.name}")
    return output
