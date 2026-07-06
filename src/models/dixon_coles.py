"""Dixon-Coles Poisson scoreline model.

Trains directly on matches_clean.parquet — NOT on the feature matrix.
Uses time-decay weighting to emphasise recent matches.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market".

Expected WC 2018 Brier: 0.200 – 0.215.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import poisson

from src.models.metrics import brier_score_multi

MODEL_PATH = Path("models/dixon_coles_v1.json")


def _dc_tau(x: int, y: int, lx: float, ly: float, rho: float) -> float:
    """Dixon-Coles draw-bias correction for low-scoring results."""
    if x == 0 and y == 0:
        return 1 - lx * ly * rho
    if x == 0 and y == 1:
        return 1 + lx * rho
    if x == 1 and y == 0:
        return 1 + ly * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _neg_log_likelihood(
    params: np.ndarray,
    teams: list[str],
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Vectorised negative log-likelihood for faster optimisation.

    Uses numpy arrays of team indices instead of string lookups.
    """
    n = len(teams)
    attack   = params[:n]
    defence  = params[n:2 * n]
    home_adv = params[2 * n]
    rho      = params[2 * n + 1]

    lambda_h = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    lambda_a = np.exp(attack[away_idx] + defence[home_idx])

    log_p_h = poisson.logpmf(home_goals, lambda_h)
    log_p_a = poisson.logpmf(away_goals, lambda_a)

    # Dixon-Coles tau correction (vectorised for low scores)
    tau = np.ones(len(home_goals))
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1 - lambda_h[m00] * lambda_a[m00] * rho
    tau[m01] = 1 + lambda_h[m01] * rho
    tau[m10] = 1 + lambda_a[m10] * rho
    tau[m11] = 1 - rho
    tau = np.clip(tau, 1e-10, None)

    ll = np.sum(weights * (np.log(tau) + log_p_h + log_p_a))
    return -float(ll)


class DixonColesModel:
    """Fitted Dixon-Coles Poisson model for match outcome prediction.

    Usage::

        dc = DixonColesModel().fit(matches_df)
        dc.predict("Brazil", "France")
        dc.save()
    """

    WC2018_START: ClassVar[str] = "2018-06-14"
    WC2018_END:   ClassVar[str] = "2018-07-15"

    def __init__(self) -> None:
        self.attack_:   dict[str, float] = {}
        self.defence_:  dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.rho_:      float = 0.0
        self.teams_:    list[str] = []

    def fit(
        self,
        matches_df: pd.DataFrame,
        min_date: str | None = None,
        recency_weight: float = 0.002,
    ) -> "DixonColesModel":
        """Fit the Dixon-Coles model using L-BFGS-B optimisation.

        Args:
            matches_df:     matches_clean.parquet (not the feature matrix).
            min_date:       Ignore matches before this date (ISO string).
            recency_weight: Decay coefficient for time-weighting.
                            weight = exp(-recency_weight * days_before_last_match).
        """
        df = matches_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Competitive matches with valid scores only
        df = df.dropna(subset=["home_score", "away_score"])
        if "is_competitive" in df.columns:
            df = df[df["is_competitive"]]

        if min_date:
            df = df[df["date"] >= pd.Timestamp(min_date)]

        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("No matches remain after filtering — check min_date and is_competitive.")

        # Time-decay weights (more recent → higher weight)
        last_date = df["date"].max()
        days_before = (last_date - df["date"]).dt.days.to_numpy(dtype=float)
        weights = np.exp(-recency_weight * days_before)
        weights = weights / weights.sum()

        self.teams_ = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(self.teams_)
        team_to_idx = {t: i for i, t in enumerate(self.teams_)}

        logger.info(
            f"Dixon-Coles fitting on {len(df):,} matches "
            f"({len(self.teams_)} teams)..."
        )

        home_idx   = np.array([team_to_idx[t] for t in df["home_team"]], dtype=int)
        away_idx   = np.array([team_to_idx[t] for t in df["away_team"]], dtype=int)
        home_goals = df["home_score"].astype(int).to_numpy()
        away_goals = df["away_score"].astype(int).to_numpy()

        x0 = np.zeros(2 * n + 2)
        x0[2 * n]     =  0.3   # home advantage prior
        x0[2 * n + 1] = -0.1   # rho prior

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(
                self.teams_,
                home_idx,
                away_idx,
                home_goals,
                away_goals,
                weights,
            ),
            method="L-BFGS-B",
            options={"maxiter": 2000, "disp": False},
        )

        if not result.success:
            logger.warning(f"Optimiser did not converge: {result.message}")

        self.attack_  = dict(zip(self.teams_, result.x[:n]))
        self.defence_ = dict(zip(self.teams_, result.x[n:2 * n]))
        self.home_adv_ = float(result.x[2 * n])
        self.rho_      = float(result.x[2 * n + 1])

        logger.info(
            f"Fitted.  home_adv={self.home_adv_:.3f}  rho={self.rho_:.3f}"
        )
        return self

    def predict(
        self,
        team_a: str,
        team_b: str,
        n_sim: int = 10_000,
        neutral: bool = True,
    ) -> dict:
        """Predict W/D/L probabilities + expected scoreline.

        Args:
            team_a:  Home/team-A name.
            team_b:  Away/team-B name.
            n_sim:   Monte Carlo samples.
            neutral: If True, suppress home advantage (WC matches are neutral).

        Returns:
            Dict with prob_home_win, prob_draw, prob_away_win,
            predicted_score_a, predicted_score_b, lambda_a, lambda_b.
        """
        atk_a = self.attack_.get(team_a, 0.0)
        def_b = self.defence_.get(team_b, 0.0)
        atk_b = self.attack_.get(team_b, 0.0)
        def_a = self.defence_.get(team_a, 0.0)

        ha = 0.0 if neutral else self.home_adv_
        lambda_a = float(np.exp(atk_a + def_b + ha))
        lambda_b = float(np.exp(atk_b + def_a))

        rng = np.random.default_rng(42)
        goals_a = rng.poisson(lambda_a, n_sim)
        goals_b = rng.poisson(lambda_b, n_sim)

        # Apply DC tau correction to sampled 0-0, 0-1, 1-0, 1-1 scores
        tau_mask = (goals_a <= 1) & (goals_b <= 1)
        tau_vals = np.array([
            _dc_tau(int(ga), int(gb), lambda_a, lambda_b, self.rho_)
            for ga, gb in zip(goals_a[tau_mask], goals_b[tau_mask])
        ])
        # Re-weight using tau — accept/reject style
        keep = np.ones(n_sim, dtype=bool)
        tau_indices = np.where(tau_mask)[0]
        u = rng.uniform(size=len(tau_indices))
        reject_prob = np.clip(1 - tau_vals, 0, 1)
        keep[tau_indices[u < reject_prob]] = False

        ga_k = goals_a[keep]
        gb_k = goals_b[keep]

        if len(ga_k) == 0:
            # Fallback: use raw Poisson if all samples rejected
            ga_k, gb_k = goals_a, goals_b

        prob_home = float((ga_k > gb_k).mean())
        prob_draw = float((ga_k == gb_k).mean())
        prob_away = float((ga_k < gb_k).mean())

        # Most likely scoreline from full simulation
        from collections import Counter
        sc = Counter(zip(goals_a.tolist(), goals_b.tolist()))
        ml_score = sc.most_common(1)[0][0]

        return {
            "prob_home_win":     prob_home,
            "prob_draw":         prob_draw,
            "prob_away_win":     prob_away,
            "predicted_score_a": int(ml_score[0]),
            "predicted_score_b": int(ml_score[1]),
            "lambda_a":          lambda_a,
            "lambda_b":          lambda_b,
        }

    def evaluate_on_wc2018(self, matches_df: pd.DataFrame) -> float:
        """Compute multi-class Brier score on WC 2018 matches.

        Args:
            matches_df: matches_clean.parquet.

        Returns:
            Brier score (float, lower is better).
        """
        df = matches_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        wc18 = df[
            (df["date"] >= self.WC2018_START)
            & (df["date"] <= self.WC2018_END)
            & df["home_score"].notna()
            & df["away_score"].notna()
        ]

        if wc18.empty:
            raise ValueError("No WC 2018 matches found in matches_df.")

        y_true = []
        y_proba = []

        for _, row in wc18.iterrows():
            pred = self.predict(str(row["home_team"]), str(row["away_team"]), neutral=True)
            # Outcome encoding: 2=home win, 1=draw, 0=away win
            hg, ag = int(row["home_score"]), int(row["away_score"])
            if hg > ag:
                outcome = 2
            elif hg == ag:
                outcome = 1
            else:
                outcome = 0
            y_true.append(outcome)
            y_proba.append([
                pred["prob_away_win"],
                pred["prob_draw"],
                pred["prob_home_win"],
            ])

        return brier_score_multi(np.array(y_true), np.array(y_proba))

    def save(self, path: Path | str = MODEL_PATH) -> None:
        """Serialise fitted params to JSON (portable, not pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "attack":   self.attack_,
            "defence":  self.defence_,
            "home_adv": self.home_adv_,
            "rho":      self.rho_,
            "teams":    self.teams_,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved Dixon-Coles params → {path}")

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "DixonColesModel":
        """Load fitted params from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        m = cls()
        m.attack_   = data["attack"]
        m.defence_  = data["defence"]
        m.home_adv_ = data["home_adv"]
        m.rho_      = data["rho"]
        m.teams_    = data["teams"]
        return m


if __name__ == "__main__":
    import mlflow

    matches = pd.read_parquet("data/processed/matches_clean.parquet")

    dc = DixonColesModel().fit(matches)

    print("Brazil vs France:", dc.predict("Brazil", "France"))
    print("England vs Germany:", dc.predict("England", "Germany"))

    brier = dc.evaluate_on_wc2018(matches)
    print(f"Dixon-Coles WC 2018 Brier: {brier:.4f}")

    mlflow.set_experiment("wc2026_phase3")
    with mlflow.start_run(run_name="dixon_coles_v1"):
        mlflow.log_metric("brier_val", brier)
        mlflow.log_params({
            "home_adv": round(dc.home_adv_, 4),
            "rho":      round(dc.rho_, 4),
            "n_teams":  len(dc.teams_),
        })

    dc.save()
