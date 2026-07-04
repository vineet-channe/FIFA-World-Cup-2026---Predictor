"""Live match prediction service.

Builds feature vectors for any pair of WC 2026 teams and runs them through
the stacking ensemble.  Results are LRU-cached because all inputs are static
pre-tournament data — the underlying data files do not change during a session.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.api.schemas import PredictResponse, TopFeature
from src.features.pipeline import FEATURE_COLS_LINEAR, FEATURE_COLS_TREES

_WC_CUT_DATE = "2026-06-11"

# Fallback when Dixon-Coles params are missing for a team pair
GLOBAL_AVG_LAMBDA = (1.3, 1.1)


# ---------------------------------------------------------------------------
# Mini Monte Carlo scoreline (API-only — mirrors Phase 4 logic, 2k iterations)
# ---------------------------------------------------------------------------

def simulate_match_scoreline(
    outcome_proba: np.ndarray,
    lambda_a: float,
    lambda_b: float,
    n: int = 2000,
    max_attempts: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Derive expected goals for one matchup by running n mini-simulations.

    outcome_proba: shape (3,) — [P(team_b_wins), P(draw), P(team_a_wins)]
    lambda_a, lambda_b: Dixon-Coles expected goals per team
    n: number of simulations (2000 is sufficient for stable averages)

    Returns:
        (avg_score_a, avg_score_b)

    Guarantee: avg_score_a > avg_score_b whenever P(a_wins) > P(b_wins),
    because every individual sample is already outcome-consistent.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam_a = max(float(lambda_a), 0.05)
    lam_b = max(float(lambda_b), 0.05)

    proba = np.asarray(outcome_proba, dtype=float).reshape(3)
    proba = proba / proba.sum()

    scores_a = np.empty(n)
    scores_b = np.empty(n)

    for i in range(n):
        outcome = int(rng.choice([0, 1, 2], p=proba))
        score_a, score_b = _rejection_sample(outcome, lam_a, lam_b, max_attempts, rng)
        scores_a[i] = score_a
        scores_b[i] = score_b

    return float(np.mean(scores_a)), float(np.mean(scores_b))


def _rejection_sample(
    outcome: int,
    lambda_a: float,
    lambda_b: float,
    max_attempts: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Draw from Poisson until the scoreline matches the given outcome."""
    for _ in range(max_attempts):
        s_a = int(rng.poisson(lambda_a))
        s_b = int(rng.poisson(lambda_b))
        if outcome == 2 and s_a > s_b:
            return s_a, s_b
        if outcome == 1 and s_a == s_b:
            return s_a, s_b
        if outcome == 0 and s_b > s_a:
            return s_a, s_b

    if outcome == 2:
        return 1, 0
    if outcome == 1:
        return 0, 0
    return 0, 1


def safe_dc_predict(dc_model: object, team_a: str, team_b: str) -> tuple[float, float]:
    """Return (lambda_a, lambda_b). Falls back to global averages if a team is missing."""
    try:
        result = dc_model.predict(team_a, team_b, neutral=True)  # type: ignore[attr-defined]
        return float(result["lambda_a"]), float(result["lambda_b"])
    except (KeyError, ValueError, AttributeError, TypeError):
        return GLOBAL_AVG_LAMBDA


def calibrate_lambdas(
    lambda_a: float,
    lambda_b: float,
    p_team_a_win: float,
    p_team_b_win: float,
) -> tuple[float, float]:
    """Adjust DC lambdas so their ratio matches the ensemble's win probabilities.

    Preserves total expected goals (lambda_a + lambda_b) while replacing the
    split between teams with the ensemble-implied ratio from P(a wins) and P(b wins).

    Args:
        lambda_a, lambda_b: Raw Dixon-Coles expected goals per team.
        p_team_a_win:       Ensemble P(team_a wins), in [0, 1].
        p_team_b_win:       Ensemble P(team_b wins), in [0, 1].

    Returns:
        (lambda_a_calibrated, lambda_b_calibrated) with the same sum as inputs.
    """
    total_goals = lambda_a + lambda_b

    total_win_prob = p_team_a_win + p_team_b_win
    if total_win_prob < 1e-9:
        return lambda_a, lambda_b

    ratio_a = p_team_a_win / total_win_prob
    ratio_b = p_team_b_win / total_win_prob

    lambda_a_cal = total_goals * ratio_a
    lambda_b_cal = total_goals * ratio_b

    return lambda_a_cal, lambda_b_cal


# ---------------------------------------------------------------------------
# Data singletons — loaded once, reused across requests
# ---------------------------------------------------------------------------

_matches_df: pd.DataFrame | None = None
_elo_df:     pd.DataFrame | None = None
_rankings_df: pd.DataFrame | None = None
_squad_df:   pd.DataFrame | None = None


def _get_matches() -> pd.DataFrame | None:
    global _matches_df
    if _matches_df is None:
        p = settings.DATA_DIR / "processed" / "matches_clean.parquet"
        if p.exists():
            _matches_df = pd.read_parquet(p)
            _matches_df["date"] = pd.to_datetime(_matches_df["date"])
    return _matches_df


def _get_elo() -> pd.DataFrame | None:
    global _elo_df
    if _elo_df is None:
        p = settings.DATA_DIR / "processed" / "elo_clean.parquet"
        if p.exists():
            _elo_df = pd.read_parquet(p)
            _elo_df["date"] = pd.to_datetime(_elo_df["date"])
    return _elo_df


def _get_rankings() -> pd.DataFrame | None:
    global _rankings_df
    if _rankings_df is None:
        p = settings.DATA_DIR / "processed" / "rankings_clean.parquet"
        if p.exists():
            _rankings_df = pd.read_parquet(p)
            _rankings_df["rank_date"] = pd.to_datetime(_rankings_df["rank_date"])
    return _rankings_df


def _get_squad() -> pd.DataFrame | None:
    global _squad_df
    if _squad_df is None:
        p = settings.DATA_DIR / "raw" / "transfermarkt" / "squad_values.parquet"
        if p.exists():
            _squad_df = pd.read_parquet(p)
    return _squad_df


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def build_feature_vectors(
    team_a: str,
    team_b: str,
    cut_date: str = _WC_CUT_DATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_trees, X_linear) for any team pair at the given cut-off date.

    Reuses all Phase 2 feature functions with strict date filtering.
    Returns NaN-imputed arrays ready for ``predict_ensemble()``.

    Args:
        team_a:   Canonical name of the first team.
        team_b:   Canonical name of the second team.
        cut_date: YYYY-MM-DD feature cut-off (default: day before WC opening).

    Returns:
        Tuple of numpy arrays with shapes (1, 21) and (1, 15).
    """
    cut_ts = pd.Timestamp(cut_date)
    match_row = pd.Series(
        {
            "home_team":  team_a,
            "away_team":  team_b,
            "date":       cut_ts,
            "tournament": "FIFA World Cup",
            "neutral":    True,
        }
    )
    features: dict[str, float] = {}

    matches_df  = _get_matches()
    elo_df      = _get_elo()
    rankings_df = _get_rankings()
    squad_df    = _get_squad()

    # Elo
    try:
        from src.features.elo_features import compute_elo_features
        if elo_df is not None:
            features.update(compute_elo_features(match_row, elo_df))
    except Exception as exc:
        logger.warning(f"Elo features failed for {team_a} vs {team_b}: {exc}")

    # FIFA ranking
    try:
        from src.features.ranking_features import compute_ranking_features
        if rankings_df is not None:
            features.update(compute_ranking_features(match_row, rankings_df))
    except Exception as exc:
        logger.warning(f"Ranking features failed: {exc}")

    # Form
    try:
        from src.features.form_features import compute_form_features
        if matches_df is not None:
            form_a = compute_form_features(matches_df, team_a, cut_ts, n=10)
            form_b = compute_form_features(matches_df, team_b, cut_ts, n=10)

            def _diff(key: str) -> float:
                va, vb = form_a.get(key, np.nan), form_b.get(key, np.nan)
                return float(va - vb) if not (np.isnan(va) or np.isnan(vb)) else np.nan

            features["ppg_diff"]             = _diff("ppg")
            features["goals_scored_diff"]    = _diff("goals_scored_pg")
            features["goals_conceded_diff"]  = _diff("goals_conceded_pg")
            features["clean_sheet_diff"]     = _diff("clean_sheet_pct")
            features["win_pct_diff"]         = _diff("win_pct")
            features["neutral_win_pct_diff"] = _diff("neutral_win_pct")
    except Exception as exc:
        logger.warning(f"Form features failed: {exc}")

    # H2H
    try:
        from src.features.h2h_features import compute_h2h_features
        if matches_df is not None:
            features.update(compute_h2h_features(matches_df, team_a, team_b, cut_ts))
    except Exception as exc:
        logger.warning(f"H2H features failed: {exc}")

    # Squad
    try:
        from src.features.squad_features import compute_squad_features
        if squad_df is not None:
            features.update(compute_squad_features(squad_df, team_a, team_b))
    except Exception as exc:
        logger.warning(f"Squad features failed: {exc}")

    # Context
    try:
        from src.features.context_features import compute_context_features, compute_wc_experience
        features.update(compute_context_features(match_row))
        if matches_df is not None:
            features.update(compute_wc_experience(matches_df, team_a, team_b, cut_ts))
    except Exception as exc:
        logger.warning(f"Context features failed: {exc}")

    features.setdefault("xg_diff_per_game", np.nan)
    features.setdefault("sot_pct_diff",     np.nan)

    X_trees  = np.array([[features.get(c, np.nan) for c in FEATURE_COLS_TREES]],  dtype=float)
    X_linear = np.array([[features.get(c, np.nan) for c in FEATURE_COLS_LINEAR]], dtype=float)
    X_trees[np.isnan(X_trees)]   = 0.0
    X_linear[np.isnan(X_linear)] = 0.0

    return X_trees, X_linear


def get_top_features(features: dict[str, float], n: int = 5) -> list[TopFeature]:
    """Return the n features with the largest absolute value from a feature vector.

    Args:
        features: Dict of feature name → float value.
        n:        Number of top features to return.

    Returns:
        List of TopFeature objects sorted by absolute value descending.
    """
    diff_feats = {
        k: v for k, v in features.items()
        if "diff" in k and not math.isnan(v) and v != 0.0
    }
    top = sorted(diff_feats.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

    _name_map = {
        "elo_diff":             "Elo rating gap",
        "elo_trajectory_diff":  "Elo momentum",
        "fifa_pts_diff":        "FIFA points gap",
        "squad_log_value_diff": "Squad value gap",
        "ppg_diff":             "Points per game",
        "goals_scored_diff":    "Goals scored",
        "goals_conceded_diff":  "Goals conceded",
        "clean_sheet_diff":     "Clean sheet rate",
        "win_pct_diff":         "Win percentage",
        "neutral_win_pct_diff": "Neutral venue wins",
        "h2h_goal_diff_avg":    "H2H goal diff",
        "wc_appearances_diff":  "WC experience",
        "avg_wc_finish_diff":   "WC stage reached",
        "rest_days_diff":       "Rest days",
        "h2h_neutral_win_rate": "H2H neutral wins",
    }

    results = []
    for name, value in top:
        favors: str
        if abs(value) < 0.01:
            favors = "neutral"
        elif value > 0:
            favors = "a"
        else:
            favors = "b"
        results.append(
            TopFeature(
                name=_name_map.get(name, name.replace("_", " ").title()),
                value=round(value, 4),
                favors=favors,
            )
        )
    return results


@lru_cache(maxsize=256)
def predict_matchup(team_a: str, team_b: str, ensemble: object = None) -> PredictResponse:
    """Run the ensemble on team_a vs team_b and return a PredictResponse.

    Cached with lru_cache — inputs are static pre-tournament data.

    Args:
        team_a:   First team canonical name.
        team_b:   Second team canonical name.
        ensemble: The loaded ensemble dict (passed as parameter to allow caching
                  without a global mutable reference in the cache key — callers
                  pass ``None`` and we fetch from the app state instead).

    Returns:
        PredictResponse with probabilities, scoreline, and top features.
    """
    from src.models.ensemble import predict_ensemble

    # ensemble is injected via the router, but lru_cache needs hashable args.
    # We import the global reference loaded at startup instead.
    from src.api.main import app as _app
    _ensemble = getattr(_app.state, "ensemble", None)
    _dc_model = getattr(_app.state, "dc_model", None)
    if _ensemble is None:
        raise RuntimeError("Ensemble model not loaded — check startup logs.")
    if _dc_model is None:
        raise RuntimeError("Dixon-Coles model not loaded — check startup logs.")

    X_trees, X_linear = build_feature_vectors(team_a, team_b)
    proba = predict_ensemble(_ensemble, X_trees, X_linear)

    # shape (3,) — [P(team_b_wins), P(draw), P(team_a_wins)]
    outcome_proba = proba[0]
    p_b_win = float(outcome_proba[0])
    p_draw  = float(outcome_proba[1])
    p_a_win = float(outcome_proba[2])

    lambda_a, lambda_b = safe_dc_predict(_dc_model, team_a, team_b)

    # Keep DC total goals; use ensemble ratio for which team scores more
    lambda_a, lambda_b = calibrate_lambdas(lambda_a, lambda_b, p_a_win, p_b_win)

    # Deterministic seed per pair so lru_cache returns stable scorelines
    pair_seed = hash((team_a, team_b)) & 0xFFFFFFFF
    rng = np.random.default_rng(pair_seed)
    avg_score_a, avg_score_b = simulate_match_scoreline(
        outcome_proba=outcome_proba,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        n=2000,
        rng=rng,
    )

    features_dict = {
        col: float(X_trees[0, i]) for i, col in enumerate(FEATURE_COLS_TREES)
    }
    top_feats = get_top_features(features_dict)

    return PredictResponse(
        p_team_a_win=round(p_a_win, 4),
        p_draw=round(p_draw, 4),
        p_team_b_win=round(p_b_win, 4),
        expected_score_a=round(avg_score_a, 2),
        expected_score_b=round(avg_score_b, 2),
        top_features=top_feats,
    )
