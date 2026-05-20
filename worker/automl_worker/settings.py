import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "/app/output"))
    max_candidates_per_family: int = int(os.getenv("AUTOML_MAX_CANDIDATES_PER_FAMILY", "3"))
    cv_folds: int = int(os.getenv("AUTOML_CV_FOLDS", "3"))
    random_state: int = int(os.getenv("AUTOML_RANDOM_STATE", "42"))


settings = Settings()
