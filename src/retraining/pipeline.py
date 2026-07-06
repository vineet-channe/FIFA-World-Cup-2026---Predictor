"""
LiveRetrainPipeline: orchestrates the full post-match update cycle.
Ingests results → updates Elo → extends features → retrains LightGBM →
validates → simulates remaining matches → saves snapshots.
Target: complete in under 4 minutes per run.
"""
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from loguru import logger
from config.settings import settings
from src.models.ensemble import load_ensemble
from src.models.dixon_coles import DixonColesModel
from src.retraining.ingestion import (
    get_all_fixtures,
    get_standings,
    get_bracket,
    save_raw_results,
    enrich_knockout_fixtures,
    completed_fixtures,
    fixture_to_result_entry,
)
from src.retraining.elo_updater import update_elo_with_results
from src.retraining.retrain import (
    build_extended_feature_matrix,
    retrain_lightgbm,
    validate_on_recent_wc,
)
from src.retraining.live_simulation import build_tournament_state, run_live_simulation
from src.retraining.monitor import should_deploy
from src.retraining.snapshots import (
    save_snapshot,
    save_history_entry,
    should_save_named_snapshot,
    list_snapshots,
)


def _safe_mlflow(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.warning(f"MLflow log failed ({exc})")


class LiveRetrainPipeline:
    def __init__(self):
        logger.info("Loading models...")
        self.ensemble = load_ensemble(str(settings.MODEL_DIR / "ensemble_v1.pkl"))
        self.dc_model = DixonColesModel.load(
            str(settings.MODEL_DIR / "dixon_coles_v1.json")
        )
        self.current_lgbm_path = settings.MODEL_DIR / "lightgbm_v1.pkl"
        self.current_lgbm = joblib.load(self.current_lgbm_path)
        logger.info("Models loaded.")

    def run(self, n_sim: int = 10_000) -> dict:
        t0 = time.time()
        logger.info("=== LiveRetrainPipeline starting ===")

        run_ctx = nullcontext()
        try:
            mlflow.set_experiment("wc2026_phase6")
            run_ctx = mlflow.start_run(
                run_name=f"live_retrain_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
            )
        except Exception as exc:
            logger.warning(
                f"MLflow unavailable ({exc}) — continuing without experiment tracking"
            )

        with run_ctx:
            all_fixtures = get_all_fixtures()
            fixtures = completed_fixtures(all_fixtures)
            standings = get_standings()
            bracket = enrich_knockout_fixtures(get_bracket(), all_fixtures)
            save_raw_results(all_fixtures)
            _safe_mlflow(mlflow.log_metric, "n_completed_fixtures", len(fixtures))
            _safe_mlflow(mlflow.log_metric, "n_total_fixtures", len(all_fixtures))
            logger.info(
                f"Ingested: {len(fixtures)} completed fixtures "
                f"({sum(1 for f in fixtures if f['round'] == 'Round of 32')} R32)"
            )

            elo_df = pd.read_parquet(settings.DATA_DIR / "processed" / "elo_clean.parquet")
            elo_df["date"] = pd.to_datetime(elo_df["date"])
            elo_df = update_elo_with_results(fixtures, elo_df)

            rankings_df = pd.read_parquet(
                settings.DATA_DIR / "processed" / "rankings_clean.parquet"
            )
            squad_df = pd.read_parquet(
                settings.DATA_DIR / "raw" / "transfermarkt" / "squad_values.parquet"
            )
            fbref_path = settings.DATA_DIR / "raw" / "fbref"
            from src.features.tactical_features import (
                load_fbref_data,
                load_fbref_keeper_data,
            )

            fbref_shooting = load_fbref_data(fbref_path)
            fbref_keeper = load_fbref_keeper_data(fbref_path)

            fm_extended = build_extended_feature_matrix(
                fixtures,
                elo_df,
                rankings_df,
                squad_df,
                fbref_shooting,
                fbref_keeper,
            )

            new_lgbm, lgbm_path = retrain_lightgbm(fm_extended)

            brier = validate_on_recent_wc(new_lgbm, fm_extended, n=10)
            deploy, reason = should_deploy(brier)
            logger.info(f"Guardrail: {reason}")
            _safe_mlflow(mlflow.log_metric, "brier_recent_wc", brier if brier else -1)
            _safe_mlflow(mlflow.log_param, "deployed", deploy)

            if deploy:
                self.current_lgbm = new_lgbm
                self.current_lgbm_path = lgbm_path
            else:
                logger.warning(f"Using previous model: {self.current_lgbm_path}")

            # Count WC 2026 matches in training data to compute blend weight
            wc2026_mask = (
                fm_extended["tournament"].str.contains("FIFA World Cup", na=False)
                & (fm_extended["match_date"].dt.year == 2026)
            )
            n_wc2026_matches = int(wc2026_mask.sum())
            lgbm_blend_weight = min(n_wc2026_matches / 100, 0.75)

            logger.info(
                f"WC 2026 matches in training: {n_wc2026_matches} | "
                f"Blend: LightGBM {lgbm_blend_weight:.0%} / "
                f"Ensemble {1 - lgbm_blend_weight:.0%}"
            )
            _safe_mlflow(mlflow.log_metric, "n_wc2026_matches", n_wc2026_matches)
            _safe_mlflow(mlflow.log_metric, "lgbm_blend_weight", lgbm_blend_weight)

            state = build_tournament_state(fixtures, standings, bracket)

            actual_results = {
                str(f["fixture_id"]): fixture_to_result_entry(f) for f in fixtures
            }

            output = run_live_simulation(
                state,
                self.ensemble,
                self.dc_model,
                elo_df,
                n_sim=n_sim,
                actual_results=actual_results,
                lgbm_model=self.current_lgbm,
                lgbm_blend_weight=lgbm_blend_weight,
            )
            _safe_mlflow(
                mlflow.log_metric,
                "teams_remaining",
                sum(
                    1
                    for v in output["team_probabilities"].values()
                    if v.get("p_champion", 0) > 0
                ),
            )

            model_ver = (
                lgbm_path.name if deploy else Path(self.current_lgbm_path).name
            )
            save_history_entry(output, model_version=model_ver)

            existing = {s["stage"] for s in list_snapshots() if s["type"] == "milestone"}
            milestone = should_save_named_snapshot(output, existing)
            if milestone:
                save_snapshot(stage=milestone, simulation=output, model_version=model_ver)
                _safe_mlflow(mlflow.log_param, "milestone_snapshot", milestone)

            # Reload API models so /api/predict uses the updated LightGBM blend
            try:
                import requests as _requests
                api_port = getattr(settings, "API_PORT", 8000)
                _requests.post(
                    f"http://localhost:{api_port}/api/reload-models",
                    timeout=5,
                )
                logger.info("API models reloaded — predictor will use updated blend")
            except Exception as _e:
                logger.warning(
                    f"Could not auto-reload API models: {_e} — restart 'make api' to apply blend"
                )

            elapsed = time.time() - t0
            logger.info(f"=== Pipeline complete in {elapsed:.1f}s ===")
            _safe_mlflow(mlflow.log_metric, "pipeline_duration_s", elapsed)

        return output
