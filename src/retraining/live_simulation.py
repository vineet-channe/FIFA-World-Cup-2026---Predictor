"""
Simulates only the REMAINING knockout matches from the current real tournament state.
Completed matches are treated as fixed facts — not re-simulated.
"""
import json
from collections import Counter

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
from src.simulation.knockout import (
    QF_FROM_R16,
    R16_FROM_R32,
    SF_FROM_QF,
    order_r32_matches,
)

SIM_PATH = settings.DATA_DIR / "predictions" / "tournament_simulation.json"
WC2026_TEAMS = WC2026_CANONICAL_TEAMS


def _strip_projection(name: str) -> str:
    """Remove ~ prefix from projected team names in bracket accumulators."""
    return name[1:] if name.startswith("~") else name


def _match_pair_key(team_a: str, team_b: str) -> tuple[str, str]:
    """Order-independent key for deduplicating the same fixture."""
    return tuple(sorted([_strip_projection(team_a), _strip_projection(team_b)]))


def _build_completed_knockout(
    actual_results: dict | None,
    round_label: str,
) -> dict[tuple[str, str], dict]:
    """Index completed knockout matches by team pair."""
    out: dict[tuple[str, str], dict] = {}
    if not actual_results:
        return out
    for res in actual_results.values():
        if res.get("round") != round_label or not res.get("played"):
            continue
        key = _match_pair_key(res["team_a"], res["team_b"])
        out[key] = res
    return out


def _orient_completed_match(
    ta: str,
    tb: str,
    res: dict,
) -> tuple[str, int, int]:
    """Map stored scores to the bracket orientation (ta, tb)."""
    winner = res.get("winner")
    sa = int(res.get("score_a") or 0)
    sb = int(res.get("score_b") or 0)
    if res["team_a"] == ta and res["team_b"] == tb:
        return winner, sa, sb
    if res["team_a"] == tb and res["team_b"] == ta:
        return winner, sb, sa
    return winner, sa, sb


def _knockout_eliminations(actual_results: dict | None) -> dict[str, str]:
    """Map eliminated team -> knockout round they were knocked out in."""
    out: dict[str, str] = {}
    if not actual_results:
        return out
    for res in actual_results.values():
        if not res.get("played") or not res.get("winner"):
            continue
        rnd = res.get("round", "")
        if rnd not in ("Round of 32", "Round of 16", "Quarter-finals", "Semi-finals"):
            continue
        ta, tb, winner = res["team_a"], res["team_b"], res["winner"]
        loser = tb if winner == ta else ta if winner == tb else None
        if loser:
            out[loser] = rnd
    return out


_KNOCKOUT_ROUND_ORDER = [
    "Round of 32",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "3rd Place",
    "Final",
]
_KNOCKOUT_ROUND_COUNTS = {
    "Round of 32": 16,
    "Round of 16": 8,
    "Quarter-finals": 4,
    "Semi-finals": 2,
    "3rd Place": 1,
    "Final": 1,
}


def _tournament_phase(actual_results: dict | None, fallback: str) -> str:
    """Infer current tournament phase from completed knockout results."""
    if not actual_results:
        return fallback
    completed: dict[str, int] = {}
    for res in actual_results.values():
        if res.get("played"):
            rnd = res.get("round", "")
            completed[rnd] = completed.get(rnd, 0) + 1

    for rnd in reversed(_KNOCKOUT_ROUND_ORDER):
        n = completed.get(rnd, 0)
        if n <= 0:
            continue
        expected = _KNOCKOUT_ROUND_COUNTS.get(rnd)
        if expected and n >= expected:
            idx = _KNOCKOUT_ROUND_ORDER.index(rnd)
            if idx + 1 < len(_KNOCKOUT_ROUND_ORDER):
                return _KNOCKOUT_ROUND_ORDER[idx + 1]
        return rnd
    return fallback


def _empty_slot_acc() -> dict:
    """One accumulator per bracket slot across all simulation runs."""
    return {
        "teams_a": Counter(),
        "teams_b": Counter(),
        "team_wins": Counter(),
        "wins_a": 0,
        "wins_b": 0,
        "draws": 0,
        "sum_a": 0.0,
        "sum_b": 0.0,
        "n": 0,
    }


def _record_slot(
    acc: dict,
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
    winner: str,
) -> None:
    """Record one simulated match result into a bracket slot accumulator."""
    acc["teams_a"][team_a] += 1
    acc["teams_b"][team_b] += 1
    acc["team_wins"][winner] += 1
    if winner == team_a:
        acc["wins_a"] += 1
    elif winner == team_b:
        acc["wins_b"] += 1
    else:
        acc["draws"] += 1
    acc["sum_a"] += score_a
    acc["sum_b"] += score_b
    acc["n"] += 1


def _acc_to_prediction(
    acc: dict,
    round_name: str,
    min_certainty: float = 0.40,
) -> dict | None:
    """Convert a bracket slot accumulator into a match_predictions entry."""
    if acc["n"] == 0:
        return None

    team_a, count_a = acc["teams_a"].most_common(1)[0] if acc["teams_a"] else ("TBD", 0)
    team_b, count_b = acc["teams_b"].most_common(1)[0] if acc["teams_b"] else ("TBD", 0)

    cert_a = count_a / acc["n"]
    cert_b = count_b / acc["n"]

    team_wins: Counter = acc.get("team_wins", Counter())
    w_a = team_wins.get(team_a, 0)
    w_b = team_wins.get(team_b, 0)
    if w_a + w_b > 0:
        p_a = w_a / (w_a + w_b)
        p_b = w_b / (w_a + w_b)
        p_d = 0.0
    else:
        total = acc["wins_a"] + acc["wins_b"] + acc["draws"]
        p_a = acc["wins_a"] / total if total else 0.33
        p_b = acc["wins_b"] / total if total else 0.34
        p_d = acc["draws"] / total if total else 0.33

    return {
        "team_a": team_a if cert_a >= min_certainty else f"~{team_a}",
        "team_b": team_b if cert_b >= min_certainty else f"~{team_b}",
        "round": round_name,
        "status": "predicted",
        "p_team_a_win": round(p_a, 3),
        "p_draw": round(p_d, 3),
        "p_team_b_win": round(p_b, 3),
        "expected_score_a": round(acc["sum_a"] / acc["n"], 2),
        "expected_score_b": round(acc["sum_b"] / acc["n"], 2),
        "team_a_certainty": round(cert_a, 3),
        "team_b_certainty": round(cert_b, 3),
    }


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

    r32_complete = len(r32_matches) == 16 and all(
        m["status"] == "FT" for m in r32_matches
    )
    if r32_complete:
        current_round = "Round of 16"
    elif group_stage_complete:
        current_round = "Round of 32"
    else:
        current_round = "Group Stage"

    return {
        "group_stage_complete": group_stage_complete,
        "group_standings": qualified,
        "r32_matches": r32_matches,
        "eliminated_teams": list(eliminated),
        "current_round": current_round,
    }


def _precompute_caches(
    remaining_teams: list[str],
    ensemble,
    dc_model: DixonColesModel,
    elo_df: pd.DataFrame,
    cut_date: str = "2026-06-11",
    lgbm_model=None,
    lgbm_blend_weight: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Pre-compute blended probabilities + DC lambdas for all remaining team pairs.

    When lgbm_model is provided and lgbm_blend_weight > 0, blends the retrained
    LightGBM (which knows about actual WC 2026 results) with the frozen stacking
    ensemble (which has broader historical calibration).
    """
    from src.api.predict_service import build_feature_vectors, calibrate_lambdas
    from src.features.pipeline import FEATURE_COLS_TREES

    prob_cache: dict[tuple, np.ndarray] = {}
    dc_cache:   dict[tuple, tuple]      = {}
    elo_cache:  dict[str, float]        = {}

    cut = pd.Timestamp(cut_date)
    for team in remaining_teams:
        elo_cache[team] = get_elo_on_date(elo_df, team, cut)

    use_blend = lgbm_model is not None and lgbm_blend_weight > 0.0
    if use_blend:
        logger.info(
            f"Pre-computing blended caches for {len(remaining_teams)} teams — "
            f"LightGBM {lgbm_blend_weight:.0%} / Ensemble {1 - lgbm_blend_weight:.0%}"
        )
    else:
        logger.info(
            f"Pre-computing caches for {len(remaining_teams)} teams "
            f"(ensemble only — no WC 2026 data yet)..."
        )

    for i, ta in enumerate(remaining_teams):
        for tb in remaining_teams[i + 1 :]:
            X_trees, X_linear = build_feature_vectors(ta, tb, cut_date=cut_date)

            # Ensemble prediction
            ens_proba = predict_ensemble(ensemble, X_trees, X_linear)[0]

            # Blend with LightGBM if available
            if use_blend:
                try:
                    X_lgbm = pd.DataFrame(
                        [X_trees[0]], columns=FEATURE_COLS_TREES
                    )
                    X_lgbm = X_lgbm.fillna(X_lgbm.median())
                    lgbm_proba = lgbm_model.predict_proba(X_lgbm)[0]
                    blended = (
                        lgbm_blend_weight * lgbm_proba
                        + (1 - lgbm_blend_weight) * ens_proba
                    )
                    blended = blended / blended.sum()
                    proba = blended
                except Exception as _exc:
                    logger.warning(
                        f"LightGBM blend failed for {ta} vs {tb}: {_exc} — "
                        f"falling back to ensemble"
                    )
                    proba = ens_proba
            else:
                proba = ens_proba

            dc = dc_model.predict(ta, tb, neutral=True)
            lam_a = dc["lambda_a"]
            lam_b = dc["lambda_b"]
            lam_a, lam_b = calibrate_lambdas(
                lam_a, lam_b, float(proba[2]), float(proba[0])
            )

            prob_cache[(ta, tb)] = proba
            prob_cache[(tb, ta)] = np.array([proba[2], proba[1], proba[0]])
            dc_cache[(ta, tb)]   = (lam_a, lam_b)
            dc_cache[(tb, ta)]   = (lam_b, lam_a)

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
    lgbm_model=None,
    lgbm_blend_weight: float = 0.0,
    save_live: bool = True,
) -> dict:
    """
    Simulate the remaining tournament from the current real state.
    Completed matches are fixed facts; only future matches are simulated.

    Set save_live=False when rebuilding a historical snapshot so this does not
    overwrite data/predictions/tournament_simulation.json.
    """
    r32 = state["r32_matches"]
    ordered_r32 = order_r32_matches(r32)
    completed_r32 = {m["fixture_id"]: m for m in ordered_r32 if m["status"] == "FT"}
    pending_r32 = [m for m in ordered_r32 if m["status"] != "FT"]
    completed_r16 = _build_completed_knockout(actual_results, "Round of 16")
    completed_qf = _build_completed_knockout(actual_results, "Quarter-finals")
    completed_sf = _build_completed_knockout(actual_results, "Semi-finals")
    completed_tpp = _build_completed_knockout(actual_results, "3rd Place")
    completed_fin = _build_completed_knockout(actual_results, "Final")
    knockout_elim = _knockout_eliminations(actual_results)

    remaining_teams = list(
        {
            t
            for m in r32
            for t in (m["team_a"], m["team_b"])
            if t and t not in ("TBD", "")
        }
    )

    prob_cache, dc_cache, elo_cache = _precompute_caches(
        remaining_teams,
        ensemble,
        dc_model,
        elo_df,
        lgbm_model=lgbm_model,
        lgbm_blend_weight=lgbm_blend_weight,
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

    r16_acc = {i: _empty_slot_acc() for i in range(8)}
    qf_acc = {i: _empty_slot_acc() for i in range(4)}
    sf_acc = {i: _empty_slot_acc() for i in range(2)}
    tpp_acc = _empty_slot_acc()
    fin_acc = _empty_slot_acc()

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
        pending_sim_results: dict[int, tuple[str, int, int]] = {}

        def _sim(ta: str, tb: str) -> tuple[str, int, int]:
            winner, sa, sb, _ = _simulate_knockout_match(
                ta, tb, prob_cache, dc_cache, elo_cache
            )
            return winner, sa, sb

        r32_winners: list[str | None] = [None] * 16

        for m in pending_r32:
            ta, tb = m["team_a"], m["team_b"]
            if ta == "TBD" or tb == "TBD":
                continue
            winner, sa, sb = _sim(ta, tb)
            pending_sim_results[m["fixture_id"]] = (winner, sa, sb)
            acc = pending_acc[m["fixture_id"]]
            acc["sum_a"] += sa
            acc["sum_b"] += sb
            acc["n"] += 1

        for slot, m in enumerate(ordered_r32):
            fid = m["fixture_id"]
            if m["status"] == "FT":
                r32_winners[slot] = completed_r32[fid]["winner"]
            elif fid in pending_sim_results:
                r32_winners[slot] = pending_sim_results[fid][0]

        for m in completed_r32.values():
            for t, gf, ga in [
                (m["team_a"], m["score_a"], m["score_b"]),
                (m["team_b"], m["score_b"], m["score_a"]),
            ]:
                if t in counts:
                    counts[t]["goals_for"] += gf or 0
                    counts[t]["goals_against"] += ga or 0
                    counts[t]["matches_played"] += 1

        for m in pending_r32:
            fid = m["fixture_id"]
            if fid not in pending_sim_results:
                continue
            sa, sb = pending_sim_results[fid][1], pending_sim_results[fid][2]
            ta, tb = m["team_a"], m["team_b"]
            for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                if t in counts:
                    counts[t]["goals_for"] += gf
                    counts[t]["goals_against"] += ga
                    counts[t]["matches_played"] += 1

        r16_participants: list[str] = []

        # ── R16 (fixed FIFA bracket) ──────────────────────────────────────────
        r16_winners: list[str | None] = []
        sf_losers: list[str] = []

        for slot, (idx_a, idx_b) in enumerate(R16_FROM_R32):
            ta, tb = r32_winners[idx_a], r32_winners[idx_b]
            if not ta or not tb:
                r16_winners.append(None)
                continue
            r16_participants.extend([ta, tb])
            pair_key = _match_pair_key(ta, tb)
            if pair_key in completed_r16:
                winner, sa, sb = _orient_completed_match(
                    ta, tb, completed_r16[pair_key]
                )
                _record_slot(r16_acc[slot], ta, tb, sa, sb, winner)
                r16_winners.append(winner)
                for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
                continue
            if ta not in elo_cache or tb not in elo_cache:
                r16_winners.append(ta)
                continue
            winner, sa, sb, _ = _simulate_knockout_match(
                ta, tb, prob_cache, dc_cache, elo_cache
            )
            _record_slot(r16_acc[slot], ta, tb, sa, sb, winner)
            r16_winners.append(winner)
            for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                if t in counts:
                    counts[t]["goals_for"] += gf
                    counts[t]["goals_against"] += ga
                    counts[t]["matches_played"] += 1

        # ── QF ────────────────────────────────────────────────────────────────
        qf_winners: list[str | None] = []

        for slot, (idx_a, idx_b) in enumerate(QF_FROM_R16):
            ta, tb = r16_winners[idx_a], r16_winners[idx_b]
            if not ta or not tb:
                qf_winners.append(None)
                continue
            pair_key = _match_pair_key(ta, tb)
            if pair_key in completed_qf:
                winner, sa, sb = _orient_completed_match(
                    ta, tb, completed_qf[pair_key]
                )
                _record_slot(qf_acc[slot], ta, tb, sa, sb, winner)
                qf_winners.append(winner)
                for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
                continue
            if ta not in elo_cache or tb not in elo_cache:
                qf_winners.append(ta)
                continue
            winner, sa, sb, _ = _simulate_knockout_match(
                ta, tb, prob_cache, dc_cache, elo_cache
            )
            _record_slot(qf_acc[slot], ta, tb, sa, sb, winner)
            qf_winners.append(winner)
            for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                if t in counts:
                    counts[t]["goals_for"] += gf
                    counts[t]["goals_against"] += ga
                    counts[t]["matches_played"] += 1

        # ── SF ────────────────────────────────────────────────────────────────
        sf_winners: list[str | None] = []

        for slot, (idx_a, idx_b) in enumerate(SF_FROM_QF):
            ta, tb = qf_winners[idx_a], qf_winners[idx_b]
            if not ta or not tb:
                sf_winners.append(None)
                continue
            pair_key = _match_pair_key(ta, tb)
            if pair_key in completed_sf:
                winner, sa, sb = _orient_completed_match(
                    ta, tb, completed_sf[pair_key]
                )
                loser = tb if winner == ta else ta
                _record_slot(sf_acc[slot], ta, tb, sa, sb, winner)
                sf_winners.append(winner)
                sf_losers.append(loser)
                for team in (ta, tb):
                    if team in counts:
                        counts[team]["semi"] += 1
                for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
                continue
            if ta not in elo_cache or tb not in elo_cache:
                sf_winners.append(ta)
                continue
            winner, sa, sb, _ = _simulate_knockout_match(
                ta, tb, prob_cache, dc_cache, elo_cache
            )
            loser = tb if winner == ta else ta
            _record_slot(sf_acc[slot], ta, tb, sa, sb, winner)
            sf_winners.append(winner)
            sf_losers.append(loser)
            for team in (ta, tb):
                if team in counts:
                    counts[team]["semi"] += 1
            for t, gf, ga in [(ta, sa, sb), (tb, sb, sa)]:
                if t in counts:
                    counts[t]["goals_for"] += gf
                    counts[t]["goals_against"] += ga
                    counts[t]["matches_played"] += 1

        # ── 3rd place ─────────────────────────────────────────────────────────
        if len(sf_losers) >= 2:
            ta_3, tb_3 = sf_losers[0], sf_losers[1]
            pair_key = _match_pair_key(ta_3, tb_3)
            if pair_key in completed_tpp:
                winner_3, sa_3, sb_3 = _orient_completed_match(
                    ta_3, tb_3, completed_tpp[pair_key]
                )
                _record_slot(tpp_acc, ta_3, tb_3, sa_3, sb_3, winner_3)
                for t, gf, ga in [(ta_3, sa_3, sb_3), (tb_3, sb_3, sa_3)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
            elif ta_3 in elo_cache and tb_3 in elo_cache:
                winner_3, sa_3, sb_3, _ = _simulate_knockout_match(
                    ta_3, tb_3, prob_cache, dc_cache, elo_cache
                )
                _record_slot(tpp_acc, ta_3, tb_3, sa_3, sb_3, winner_3)

        # ── Final ─────────────────────────────────────────────────────────────
        if len(sf_winners) >= 2:
            ta_f, tb_f = sf_winners[0], sf_winners[1]
            pair_key = _match_pair_key(ta_f, tb_f)
            if pair_key in completed_fin:
                winner_f, sa_f, sb_f = _orient_completed_match(
                    ta_f, tb_f, completed_fin[pair_key]
                )
                _record_slot(fin_acc, ta_f, tb_f, sa_f, sb_f, winner_f)
                if winner_f in counts:
                    counts[winner_f]["champion"] += 1
                for team in (ta_f, tb_f):
                    if team in counts:
                        counts[team]["final"] += 1
                for t, gf, ga in [(ta_f, sa_f, sb_f), (tb_f, sb_f, sa_f)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
            elif ta_f and tb_f and ta_f in elo_cache and tb_f in elo_cache:
                winner_f, sa_f, sb_f, _ = _simulate_knockout_match(
                    ta_f, tb_f, prob_cache, dc_cache, elo_cache
                )
                _record_slot(fin_acc, ta_f, tb_f, sa_f, sb_f, winner_f)
                if winner_f in counts:
                    counts[winner_f]["champion"] += 1
                for team in (ta_f, tb_f):
                    if team in counts:
                        counts[team]["final"] += 1
                for t, gf, ga in [(ta_f, sa_f, sb_f), (tb_f, sb_f, sa_f)]:
                    if t in counts:
                        counts[t]["goals_for"] += gf
                        counts[t]["goals_against"] += ga
                        counts[t]["matches_played"] += 1
            elif sf_winners[0] and sf_winners[0] in counts:
                counts[sf_winners[0]]["champion"] += 1

        for t in r16_participants:
            if t in counts:
                counts[t]["r16"] += 1
        for t in qf_winners:
            if t and t in counts:
                counts[t]["quarter"] += 1

    team_probs: dict[str, dict] = {}
    for team in WC2026_TEAMS:
        c = counts[team]
        mp = max(c["matches_played"], 1)
        in_r32 = any(team in (m["team_a"], m["team_b"]) for m in r32)
        eliminated_in = knockout_elim.get(team)
        if not eliminated_in and team in state.get("eliminated_teams", []):
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
            entry = {
                **res,
                "status": "completed",
                "actual_score_a": res.get("score_a"),
                "actual_score_b": res.get("score_b"),
                "actual_winner": res.get("winner"),
            }
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

    downstream: dict[str, dict] = {}

    for i, acc in r16_acc.items():
        pred = _acc_to_prediction(acc, "Round of 16")
        if pred:
            downstream[f"r16_{i + 1}"] = pred

    for i, acc in qf_acc.items():
        pred = _acc_to_prediction(acc, "Quarter-finals")
        if pred:
            downstream[f"qf_{i + 1}"] = pred

    for i, acc in sf_acc.items():
        pred = _acc_to_prediction(acc, "Semi-finals")
        if pred:
            downstream[f"sf_{i + 1}"] = pred

    pred_fin = _acc_to_prediction(fin_acc, "Final")
    if pred_fin:
        downstream["final_match"] = pred_fin

    pred_3rd = _acc_to_prediction(tpp_acc, "3rd Place")
    if pred_3rd:
        downstream["3rd_place"] = pred_3rd

    completed_pairs = {
        _match_pair_key(m["team_a"], m["team_b"])
        for m in match_preds.values()
        if m.get("status") == "completed"
    }
    downstream = {
        mid: pred
        for mid, pred in downstream.items()
        if _match_pair_key(pred["team_a"], pred["team_b"]) not in completed_pairs
    }

    match_preds.update(downstream)

    logger.info(
        f"match_predictions: "
        f"{sum(1 for v in match_preds.values() if 'Group Stage' in v.get('round', ''))} group | "
        f"{sum(1 for v in match_preds.values() if v.get('round') == 'Round of 32')} r32 | "
        f"{sum(1 for v in match_preds.values() if v.get('round') == 'Round of 16')} r16 | "
        f"{sum(1 for v in match_preds.values() if v.get('round') == 'Quarter-finals')} qf | "
        f"{sum(1 for v in match_preds.values() if v.get('round') == 'Semi-finals')} sf | "
        f"{sum(1 for v in match_preds.values() if v.get('round') in ('3rd Place', 'Final'))} 3rd/final"
    )

    output = {
        "metadata": {
            "n_simulations": n_sim,
            "random_seed": 42,
            "run_at": pd.Timestamp.now().isoformat(),
            "data_as_of": pd.Timestamp.now().isoformat(),
            "model": (
                f"lgbm_{lgbm_blend_weight:.0%}+ensemble_{1-lgbm_blend_weight:.0%}"
                f"+dixon_coles_v1"
                if lgbm_blend_weight > 0
                else "ensemble_v1+dixon_coles_v1"
            ),
            "tournament_phase": _tournament_phase(
                actual_results, state.get("current_round", "Round of 32")
            ),
        },
        "actual_results": actual_results or {},
        "group_stage_standings": state.get("group_standings", {}),
        "team_probabilities": team_probs,
        "match_predictions": match_preds,
        "group_predictions": {},
    }

    if save_live:
        with open(SIM_PATH, "w") as f:
            json.dump(output, f, indent=2)

        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        bak = SIM_PATH.parent / "history" / f"{ts}_simulation.json"
        bak.parent.mkdir(parents=True, exist_ok=True)
        with open(bak, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Live simulation complete. Saved to {SIM_PATH} + backup {bak.name}")
    else:
        logger.info("Simulation complete (save_live=False — live file not overwritten)")

    return output
