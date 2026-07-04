from pathlib import Path

try:
    from pydantic_settings import BaseSettings
    _USE_PYDANTIC = True
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore[no-redef]
        _USE_PYDANTIC = True
    except (ImportError, Exception):
        _USE_PYDANTIC = False

if _USE_PYDANTIC:
    _base = BaseSettings  # type: ignore[name-defined]
else:
    class _base:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        class Config:
            env_file = ".env"
            extra = "ignore"

class Settings(_base):
    # Project
    PROJECT_NAME: str = "wc2026-predictor"
    DATA_DIR: Path = Path("./data")
    MODEL_DIR: Path = Path("./models")
    LOG_DIR: Path = Path("./logs")

    # APIs
    API_FOOTBALL_KEY: str = ""
    FOOTBALL_DATA_API_KEY: str = ""
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""

    # Database
    DATABASE_URL: str = "sqlite:///./wc2026.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # Simulation
    MONTE_CARLO_N: int = 10_000
    MONTE_CARLO_WORKERS: int = 8
    RANDOM_SEED: int = 42

    # Retraining
    WC_SAMPLE_WEIGHT_GROUP: float = 2.0
    WC_SAMPLE_WEIGHT_KNOCKOUT: float = 3.0
    BRIER_ALERT_THRESHOLD: float = 0.22
    BRIER_ROLLBACK_THRESHOLD: float = 0.24

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

settings = Settings()