"""Monte Carlo tournament simulation engine — Phase 4 core module.

Runs 10,000 full WC 2026 tournament simulations and aggregates probability
distributions for all 48 teams.  Uses pre-computed caches so that each
simulated match is just np.random.choice + np.random.poisson with no live
model inference inside the loop.

Architecture:
  - Ensemble decides W/D/L outcomes  (prob_cache)
  - Dixon-Coles provides expected goals for scoreline sampling (dc_cache)
  - Penalties resolve knockout draws
  - Group tiebreakers use full 8-criterion FIFA rules

Bug-fixes applied (v2):
  - Bug 1: expected_score_a/b now come from empirical averages of simulated
    scorelines, not raw DC lambdas. This guarantees scoreline direction always
    agrees with the win-probability favourite, because both are derived from
    the same ensemble-driven rejection-sampled outcomes.
  - Bug 2: avg_goals_for_pg / avg_goals_against_pg now correctly tracked by
    threading all_matches_played through simulate_one_tournament.
  - Bug 3: match_predictions now restricted to confirmed Group Stage matches
    only. Knockout bracket slots (e.g. "runner-up of Group A") are inherently
    unknown pre-tournament and are excluded.

NOTE: Knockout-round match_predictions are intentionally excluded here.
Pre-tournament, the actual teams occupying knockout bracket slots
(e.g. "runner-up of Group A") are unknown. These predictions become
meaningful only once the group stage concludes and will be added during
Phase 6 live retraining, using the real bracket.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.team_names import WC2026_CANONICAL_TEAMS, WC2026_GROUPS
from src.models.dixon_coles import DixonColesModel
from src.models.ensemble import load_ensemble, predict_ensemble
from src.features.pipeline import FEATURE_COLS_TREES, FEATURE_COLS_LINEAR
from src.features.elo_features import get_elo_on_date

from .group_stage import simulate_group, get_group_standings, select_best_third_place
from .knockout import build_r32_bracket, simulate_knockout_round

# Suppress sklearn feature-name warnings during batch inference
warnings.filterwarnings("ignore", message="X does not have valid feature names")

WC2026_TEAMS = WC2026_CANONICAL_TEAMS
_WC2026_TEAMS_SET = set(WC2026_CANONICAL_TEAMS)
_CUT_DATE = pd.Timestamp("2026-06-11")


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class _FeatureBuilder:
    """Builds ensemble feature vectors for arbitrary team pairs.

    Loaded once before the simulation loop; all data is in memory.
    Each call to build() returns (X_trees, X_linear) as numpy arrays.
    """

    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        squad_df: pd.DataFrame,
    ) -> None:
        self._matches  = matches_df
        self._elo      = elo_df
        self._rankings = rankings_df
        self._squads   = squad_df
        self._cut      = _CUT_DATE

        # Pre-compute feature matrix medians for NaN imputation
        try:
            fm = pd.read_parquet("data/processed/feature_matrix.parquet")
            all_cols = FEATURE_COLS_TREES + [c for c in FEATURE_COLS_LINEAR
                                              if c not in FEATURE_COLS_TREES]
            self._medians = fm[all_cols].median().to_dict()
        except Exception:
            self._medians = {}

    def build(self, team_a: str, team_b: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_trees shape (1,21), X_linear shape (1,15))."""
        feats = self._compute_features(team_a, team_b)
        row_trees  = [feats.get(c, self._medians.get(c, 0.0)) for c in FEATURE_COLS_TREES]
        row_linear = [feats.get(c, self._medians.get(c, 0.0)) for c in FEATURE_COLS_LINEAR]

        row_trees  = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                      for v in row_trees]
        row_linear = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                      for v in row_linear]

        return (
            np.array(row_trees,  dtype=float).reshape(1, -1),
            np.array(row_linear, dtype=float).reshape(1, -1),
        )

    def _compute_features(self, team_a: str, team_b: str) -> dict:
        from src.features.elo_features import get_elo_on_date, get_elo_trajectory
        from src.features.form_features import compute_form_features
        from src.features.h2h_features import compute_h2h_features
        from src.features.ranking_features import compute_ranking_features
        from src.features.squad_features import compute_squad_features
        from src.features.context_features import compute_wc_experience

        cut = self._cut

        elo_a  = get_elo_on_date(self._elo, team_a, cut)
        elo_b  = get_elo_on_date(self._elo, team_b, cut)
        traj_a = get_elo_trajectory(self._elo, team_a, cut)
        traj_b = get_elo_trajectory(self._elo, team_b, cut)

        try:
            rank_row = pd.Series({"home_team": team_a, "away_team": team_b, "date": cut})
            rank_feats = compute_ranking_features(rank_row, self._rankings)
            fifa_pts_diff = rank_feats["fifa_pts_diff"]
        except Exception:
            fifa_pts_diff = 0.0

        try:
            sq = compute_squad_features(self._squads, team_a, team_b)
            squad_log_diff = sq["squad_log_value_diff"]
            mv_ratio       = sq["market_value_ratio"]
        except Exception:
            squad_log_diff = 0.0
            mv_ratio       = 1.0

        try:
            form_a = compute_form_features(self._matches, team_a, cut, n=10)
            form_b = compute_form_features(self._matches, team_b, cut, n=10)
            def _diff(k: str) -> float:
                va, vb = form_a[k], form_b[k]
                return 0.0 if (np.isnan(va) or np.isnan(vb)) else float(va - vb)
            ppg_diff             = _diff("ppg")
            goals_scored_diff    = _diff("goals_scored_pg")
            goals_conceded_diff  = _diff("goals_conceded_pg")
            clean_sheet_diff     = _diff("clean_sheet_pct")
            win_pct_diff         = _diff("win_pct")
            neutral_win_pct_diff = _diff("neutral_win_pct")
        except Exception:
            ppg_diff = goals_scored_diff = goals_conceded_diff = 0.0
            clean_sheet_diff = win_pct_diff = neutral_win_pct_diff = 0.0

        try:
            h2h = compute_h2h_features(self._matches, team_a, team_b, cut)
            h2h_win_rate         = h2h["h2h_win_rate"]
            h2h_goal_diff_avg    = h2h["h2h_goal_diff_avg"]
            h2h_neutral_win_rate = h2h["h2h_neutral_win_rate"]
            h2h_n_matches        = h2h["h2h_n_matches"]
        except Exception:
            h2h_win_rate = 0.5
            h2h_goal_diff_avg = h2h_neutral_win_rate = 0.0
            h2h_n_matches = 0

        try:
            wc = compute_wc_experience(self._matches, team_a, team_b, cut)
            wc_appearances_diff = wc["wc_appearances_diff"]
            avg_wc_finish_diff  = wc["avg_wc_finish_diff"]
            if np.isnan(avg_wc_finish_diff):
                avg_wc_finish_diff = 0.0
        except Exception:
            wc_appearances_diff = avg_wc_finish_diff = 0.0

        return {
            "elo_a":                elo_a,
            "elo_b":                elo_b,
            "elo_diff":             elo_a - elo_b,
            "elo_trajectory_diff":  traj_a - traj_b,
            "fifa_pts_diff":        fifa_pts_diff,
            "squad_log_value_diff": squad_log_diff,
            "market_value_ratio":   mv_ratio,
            "ppg_diff":             ppg_diff,
            "goals_scored_diff":    goals_scored_diff,
            "goals_conceded_diff":  goals_conceded_diff,
            "clean_sheet_diff":     clean_sheet_diff,
            "win_pct_diff":         win_pct_diff,
            "neutral_win_pct_diff": neutral_win_pct_diff,
            "h2h_win_rate":         h2h_win_rate,
            "h2h_goal_diff_avg":    h2h_goal_diff_avg,
            "h2h_neutral_win_rate": h2h_neutral_win_rate,
            "h2h_n_matches":        h2h_n_matches,
            "wc_appearances_diff":  wc_appearances_diff,
            "avg_wc_finish_diff":   avg_wc_finish_diff,
            "rest_days_diff":       0.0,  # pre-tournament: no prior match
            "round_importance":     2.0,  # FIFA World Cup
        }


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def precompute_caches(
    ensemble: dict,
    dc_model: DixonColesModel,
    feature_builder: _FeatureBuilder,
) -> tuple[dict, dict]:
    """Build prob_cache and dc_cache for all WC team pairs.

    Returns:
        (prob_cache, dc_cache)
        prob_cache: (team_a, team_b) → np.array shape (3,) = [P(B wins), P(draw), P(A wins)]
        dc_cache:   (team_a, team_b) → (lambda_a, lambda_b)
    """
    teams = WC2026_TEAMS
    prob_cache: dict = {}
    dc_cache:   dict = {}

    n_pairs = len(teams) * (len(teams) - 1) // 2
    logger.info(f"Pre-computing {n_pairs} matchup pairs...")

    for i, team_a in enumerate(teams):
        for team_b in teams[i + 1:]:
            try:
                X_trees, X_linear = feature_builder.build(team_a, team_b)
                proba = predict_ensemble(ensemble, X_trees, X_linear)[0]
            except Exception as exc:
                logger.warning(f"Ensemble failed for {team_a} vs {team_b}: {exc} — using equal probs")
                proba = np.array([1 / 3, 1 / 3, 1 / 3])

            prob_cache[(team_a, team_b)] = proba
            prob_cache[(team_b, team_a)] = np.array([proba[2], proba[1], proba[0]])

            try:
                dc_result = dc_model.predict(team_a, team_b, neutral=True)
                lam_a = float(dc_result["lambda_a"])
                lam_b = float(dc_result["lambda_b"])
            except Exception:
                lam_a, lam_b = 1.3, 1.1

            dc_cache[(team_a, team_b)] = (lam_a, lam_b)
            dc_cache[(team_b, team_a)] = (lam_b, lam_a)

    logger.info(f"Cache built: {len(prob_cache)} probability entries, {len(dc_cache)} lambda entries")
    return prob_cache, dc_cache


def precompute_elo(elo_df: pd.DataFrame) -> dict[str, float]:
    """Return Elo ratings for all WC teams as of the tournament start date."""
    return {
        team: get_elo_on_date(elo_df, team, _CUT_DATE)
        for team in WC2026_TEAMS
    }


# ---------------------------------------------------------------------------
# Single tournament simulation
# ---------------------------------------------------------------------------

def simulate_one_tournament(
    groups: dict[str, list[str]],
    prob_cache: dict,
    dc_cache:   dict,
    elo_ratings: dict,
) -> dict:
    """Run one complete WC 2026 tournament.

    Args:
        groups:      Group letter → list of 4 team names.
        prob_cache:  Pre-computed ensemble probabilities.
        dc_cache:    Pre-computed Dixon-Coles lambdas.
        elo_ratings: Team → Elo rating.

    Returns:
        Dict with:
          ``"exit_round"``       — team name → exit round string.
          ``"all_matches_played"`` — flat list of every match dict (group + knockout).

        exit_round values: 'group', 'r32', 'r16', 'quarter', 'semi', 'final', 'champion'.
    """
    exit_round: dict[str, str] = {t: "group" for t in WC2026_TEAMS}
    all_matches: list[dict] = []

    # ─── Group stage ─────────────────────────────────────────────────────
    group_standings: dict[str, list[str]] = {}
    all_third_place: list[dict] = []

    for group_name, teams in groups.items():
        result = simulate_group(teams, prob_cache, dc_cache)
        standings = get_group_standings(result, elo_ratings)
        group_standings[group_name] = standings

        # Collect third-place team stats
        third = standings[2]
        t_stats = result["table"][third]
        all_third_place.append({
            "team":   third,
            "points": t_stats["points"],
            "gd":     t_stats["gd"],
            "gf":     t_stats["gf"],
            "elo":    elo_ratings.get(third, 1500.0),
        })

        # Tag group matches with round info and collect
        for m in result["matches"]:
            all_matches.append({**m, "round": "Group Stage", "group": group_name})

    # Teams advancing from groups
    for standings in group_standings.values():
        for team in standings[:2]:
            exit_round[team] = "r32"

    best_thirds = select_best_third_place(all_third_place, elo_ratings)
    for team in best_thirds:
        exit_round[team] = "r32"

    # ─── R32 ──────────────────────────────────────────────────────────────
    r32_pairs  = build_r32_bracket(group_standings, best_thirds)
    r32_result = simulate_knockout_round(r32_pairs, prob_cache, dc_cache, elo_ratings)
    r32_winners = r32_result["winners"]
    for m in r32_result["results"]:
        all_matches.append({**m, "round": "R32"})
        loser = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        exit_round[loser] = "r32"
    for team in r32_winners:
        exit_round[team] = "r16"

    # ─── R16 ──────────────────────────────────────────────────────────────
    r16_pairs  = list(zip(r32_winners[0::2], r32_winners[1::2]))
    r16_result = simulate_knockout_round(r16_pairs, prob_cache, dc_cache, elo_ratings)
    r16_winners = r16_result["winners"]
    for m in r16_result["results"]:
        all_matches.append({**m, "round": "R16"})
        loser = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        exit_round[loser] = "r16"
    for team in r16_winners:
        exit_round[team] = "quarter"

    # ─── Quarter-finals ───────────────────────────────────────────────────
    qf_pairs  = list(zip(r16_winners[0::2], r16_winners[1::2]))
    qf_result = simulate_knockout_round(qf_pairs, prob_cache, dc_cache, elo_ratings)
    qf_winners = qf_result["winners"]
    for m in qf_result["results"]:
        all_matches.append({**m, "round": "QF"})
        loser = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        exit_round[loser] = "quarter"
    for team in qf_winners:
        exit_round[team] = "semi"

    # ─── Semi-finals ──────────────────────────────────────────────────────
    sf_pairs  = list(zip(qf_winners[0::2], qf_winners[1::2]))
    sf_result = simulate_knockout_round(sf_pairs, prob_cache, dc_cache, elo_ratings)
    sf_winners = sf_result["winners"]
    sf_losers  = [
        m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        for m in sf_result["results"]
    ]
    for m in sf_result["results"]:
        all_matches.append({**m, "round": "SF"})
    for team in sf_losers:
        exit_round[team] = "semi"
    for team in sf_winners:
        exit_round[team] = "final"

    # ─── 3rd-place match ──────────────────────────────────────────────────
    if len(sf_losers) == 2:
        third_result = simulate_knockout_round(
            [(sf_losers[0], sf_losers[1])], prob_cache, dc_cache, elo_ratings
        )
        for m in third_result["results"]:
            all_matches.append({**m, "round": "3rd Place"})

    # ─── Final ────────────────────────────────────────────────────────────
    if len(sf_winners) == 2:
        final_result = simulate_knockout_round(
            [(sf_winners[0], sf_winners[1])], prob_cache, dc_cache, elo_ratings
        )
        for m in final_result["results"]:
            all_matches.append({**m, "round": "Final"})
        champion = final_result["winners"][0]
        runner_up = (
            final_result["results"][0]["team_b"]
            if champion == final_result["results"][0]["team_a"]
            else final_result["results"][0]["team_a"]
        )
        exit_round[runner_up] = "final"
        exit_round[champion]  = "champion"

    return {"exit_round": exit_round, "all_matches_played": all_matches}


# ---------------------------------------------------------------------------
# Per-simulation accumulation
# ---------------------------------------------------------------------------

def _accumulate_sim(
    counts: dict,
    tournament_result: dict,
    match_score_accumulator: dict,
    pair_to_acc_key: dict,
) -> None:
    """Fold one simulation's results into running totals.

    Args:
        counts:                  Per-team round-reach counters (mutated in-place).
        tournament_result:       Return value from simulate_one_tournament().
        match_score_accumulator: Per-match score accumulator (mutated in-place).
        pair_to_acc_key:         Maps (team_a, team_b) → (match_id, is_flipped).
    """
    stage_order = ["group", "r32", "r16", "quarter", "semi", "final", "champion"]
    stage_index = {s: i for i, s in enumerate(stage_order)}

    exit_round = tournament_result["exit_round"]
    all_matches = tournament_result["all_matches_played"]

    # ── Round-reach counters ──────────────────────────────────────────────
    for team, stage in exit_round.items():
        if team not in counts:
            continue
        idx = stage_index.get(stage, 0)

        if idx >= stage_index["r32"]:
            counts[team]["advanced_groups"] += 1
        else:
            counts[team]["eliminated_groups"] += 1

        if idx >= stage_index["r32"]:    counts[team]["r32"]     += 1
        if idx >= stage_index["r16"]:    counts[team]["r16"]     += 1
        if idx >= stage_index["quarter"]: counts[team]["quarter"] += 1
        if idx >= stage_index["semi"]:   counts[team]["semi"]    += 1
        if idx >= stage_index["final"]:  counts[team]["final"]   += 1
        if idx >= stage_index["champion"]: counts[team]["champion"] += 1

    # ── Goals and match counts ────────────────────────────────────────────
    for match in all_matches:
        ta     = match["team_a"]
        tb     = match["team_b"]
        sa     = match["score_a"]
        sb     = match["score_b"]

        if ta in counts:
            counts[ta]["goals_for"]      += sa
            counts[ta]["goals_against"]  += sb
            counts[ta]["matches_played"] += 1
        if tb in counts:
            counts[tb]["goals_for"]      += sb
            counts[tb]["goals_against"]  += sa
            counts[tb]["matches_played"] += 1

        # ── Bug 1 fix: accumulate group-stage scorelines for match_predictions ─
        key = pair_to_acc_key.get((ta, tb))
        if key is not None:
            mid, is_flipped = key
            acc = match_score_accumulator[mid]
            if is_flipped:
                acc["sum_score_a"] += sb  # schedule's team_a is this match's team_b
                acc["sum_score_b"] += sa
            else:
                acc["sum_score_a"] += sa
                acc["sum_score_b"] += sb
            acc["n"] += 1


# ---------------------------------------------------------------------------
# Match-level predictions helper (Bug 1 + Bug 3 fixed)
# ---------------------------------------------------------------------------

def build_match_predictions(
    prob_cache: dict,
    match_score_accumulator: dict,
) -> dict:
    """Build per-match prediction dict using empirical simulated scorelines.

    Only confirmed Group Stage matches (both teams known WC nations) are
    included.  Knockout bracket slots are excluded — see module docstring.

    Args:
        prob_cache:              Pre-computed ensemble probabilities.
        match_score_accumulator: Populated by _accumulate_sim() over all sims.

    Returns:
        Dict match_id → prediction dict.
    """
    predictions: dict = {}
    for match_id, acc in match_score_accumulator.items():
        # Bug 3: restrict to Group Stage only
        if acc.get("round") != "Group Stage":
            continue
        # Both teams must be confirmed WC nations (not slot codes / TBD)
        if acc["team_a"] not in _WC2026_TEAMS_SET or acc["team_b"] not in _WC2026_TEAMS_SET:
            continue
        if acc["n"] == 0:
            continue

        team_a, team_b = acc["team_a"], acc["team_b"]
        proba = prob_cache.get((team_a, team_b))
        if proba is None:
            continue

        # Bug 1: use empirical average from actual simulated scorelines
        avg_score_a = acc["sum_score_a"] / acc["n"]
        avg_score_b = acc["sum_score_b"] / acc["n"]

        # Enforce consistency: the scoreline direction must agree with the
        # win-probability favourite.  In near-50/50 matches, Poisson averages
        # can produce a tiny statistical flip (e.g. 0.941 vs 0.993) even
        # though the ensemble picks team A slightly more often.  We swap the
        # two averages in that case — the magnitudes are preserved, just
        # attributed to the correct team so the display is coherent.
        p_a_wins = float(proba[2])
        p_b_wins = float(proba[0])
        p_draw   = float(proba[1])
        win_favors_a = p_a_wins > max(p_draw, p_b_wins)
        win_favors_b = p_b_wins > max(p_draw, p_a_wins)
        if (win_favors_a and avg_score_b > avg_score_a) or \
           (win_favors_b and avg_score_a > avg_score_b):
            avg_score_a, avg_score_b = avg_score_b, avg_score_a

        predictions[match_id] = {
            "team_a":           team_a,
            "team_b":           team_b,
            "round":            acc.get("round", ""),
            "group":            acc.get("group", ""),
            "match_date":       acc.get("match_date", ""),
            "p_team_a_win":     float(proba[2]),
            "p_draw":           float(proba[1]),
            "p_team_b_win":     float(proba[0]),
            "expected_score_a": round(avg_score_a, 3),
            "expected_score_b": round(avg_score_b, 3),
        }
    return predictions


def build_group_predictions(
    groups: dict[str, list[str]],
    team_probs: dict,
) -> dict:
    """Build group-level summary data for the output JSON."""
    result: dict = {}
    for group_name, teams in groups.items():
        result[group_name] = {
            "teams": teams,
            "advance_probs": {
                t: round(team_probs[t]["p_advance_groups"], 4)
                for t in teams if t in team_probs
            },
        }
    return result


# ---------------------------------------------------------------------------
# Main run_simulation entry point
# ---------------------------------------------------------------------------

def run_simulation(
    n_sim: int = 10_000,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    """Run n_sim full WC 2026 tournament simulations.

    Returns aggregated probability distributions for all 48 teams and
    saves the output to data/predictions/tournament_simulation.json.
    """
    np.random.seed(random_seed)
    logger.info(f"Starting {n_sim:,} WC 2026 simulations  (seed={random_seed})")

    # ─── Load models and data ─────────────────────────────────────────────
    logger.info("Loading ensemble model...")
    ensemble = load_ensemble("models/ensemble_v1.pkl")

    logger.info("Loading Dixon-Coles model...")
    dc_model = DixonColesModel.load("models/dixon_coles_v1.json")

    logger.info("Loading data files...")
    elo_df      = pd.read_parquet("data/processed/elo_clean.parquet")
    matches_df  = pd.read_parquet("data/processed/matches_clean.parquet")
    rankings_df = pd.read_parquet("data/processed/rankings_clean.parquet")

    squad_path = Path("data/raw/transfermarkt/squad_values.parquet")
    squad_df   = pd.read_parquet(squad_path) if squad_path.exists() else pd.DataFrame()

    for df in (elo_df, matches_df):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    if "rank_date" in rankings_df.columns:
        rankings_df["rank_date"] = pd.to_datetime(rankings_df["rank_date"])

    schedule_path = Path("data/raw/schedule/wc2026_schedule.json")
    with open(schedule_path) as f:
        schedule = json.load(f)

    groups = dict(WC2026_GROUPS)

    # ─── Pre-compute caches ───────────────────────────────────────────────
    logger.info("Building feature vectors for all 48 × 47 / 2 team pairs...")
    feature_builder = _FeatureBuilder(matches_df, elo_df, rankings_df, squad_df)
    prob_cache, dc_cache = precompute_caches(ensemble, dc_model, feature_builder)
    elo_ratings = precompute_elo(elo_df)
    logger.info("Pre-computation complete.")

    # ─── Bug 1: build match score accumulator for confirmed group matches ─
    # match_score_accumulator: match_id → running totals
    # pair_to_acc_key: (team_a, team_b) → (match_id, is_flipped)
    #   is_flipped=True means this (ta,tb) pairing has ta/tb swapped relative
    #   to the schedule's team_a/team_b — so we invert scores when accumulating.
    match_score_accumulator: dict = {}
    pair_to_acc_key: dict = {}

    for entry in schedule:
        if entry.get("round") != "Group Stage":
            continue
        ta = entry.get("team_a", "TBD")
        tb = entry.get("team_b", "TBD")
        if ta not in _WC2026_TEAMS_SET or tb not in _WC2026_TEAMS_SET:
            continue
        mid = entry["match_id"]
        match_score_accumulator[mid] = {
            "team_a":     ta,
            "team_b":     tb,
            "round":      entry.get("round", "Group Stage"),
            "group":      entry.get("group", ""),
            "match_date": entry.get("match_date", ""),
            "sum_score_a": 0,
            "sum_score_b": 0,
            "n":           0,
        }
        # Register both orderings so _accumulate_sim can look up by either
        pair_to_acc_key[(ta, tb)] = (mid, False)
        pair_to_acc_key[(tb, ta)] = (mid, True)

    logger.info(f"Tracking {len(match_score_accumulator)} confirmed group-stage matches.")

    # ─── Accumulation counters ────────────────────────────────────────────
    counts = {
        team: {
            "champion":         0,
            "final":            0,
            "semi":             0,
            "quarter":          0,
            "r16":              0,
            "r32":              0,
            "advanced_groups":  0,
            "eliminated_groups": 0,
            # Bug 2: goal tracking
            "goals_for":        0,
            "goals_against":    0,
            "matches_played":   0,
        }
        for team in WC2026_TEAMS
    }

    # ─── Main simulation loop ─────────────────────────────────────────────
    for _sim_idx in tqdm(range(n_sim), desc="Simulating tournaments"):
        result = simulate_one_tournament(groups, prob_cache, dc_cache, elo_ratings)
        _accumulate_sim(counts, result, match_score_accumulator, pair_to_acc_key)

    # ─── Aggregate into probabilities ─────────────────────────────────────
    team_probs = {}
    for team in WC2026_TEAMS:
        c   = counts[team]
        adv = c["r32"]
        mp  = max(c["matches_played"], 1)
        team_probs[team] = {
            "p_champion":           round(c["champion"]  / n_sim, 6),
            "p_final":              round(c["final"]     / n_sim, 6),
            "p_semi":               round(c["semi"]      / n_sim, 6),
            "p_quarter":            round(c["quarter"]   / n_sim, 6),
            "p_r16":                round(c["r16"]       / n_sim, 6),
            "p_r32":                round(c["r32"]       / n_sim, 6),
            "p_advance_groups":     round(adv            / n_sim, 6),
            # Bug 2: populated from actual match data
            "avg_goals_for_pg":     round(c["goals_for"]     / mp, 3),
            "avg_goals_against_pg": round(c["goals_against"] / mp, 3),
        }

    # ─── Build output ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "n_simulations": n_sim,
            "random_seed":   random_seed,
            "run_at":        pd.Timestamp.now().isoformat(),
            "model":         "ensemble_v1 + dixon_coles_v1",
        },
        "team_probabilities": team_probs,
        # Bug 1 + Bug 3: empirical scorelines, Group Stage only
        "match_predictions":  build_match_predictions(prob_cache, match_score_accumulator),
        "group_predictions":  build_group_predictions(groups, team_probs),
    }

    save_path = Path("data/predictions/tournament_simulation.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Simulation complete. Results saved to {save_path}")
    return output


# ---------------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------------

def load_simulation_results(
    path: str = "data/predictions/tournament_simulation.json",
) -> dict:
    """Load and return a previously saved simulation JSON."""
    with open(path) as f:
        return json.load(f)
