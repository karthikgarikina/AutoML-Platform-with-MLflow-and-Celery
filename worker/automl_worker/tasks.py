import traceback
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from automl_worker.celery_app import celery_app
from automl_worker.data_profile import generate_data_profile
from automl_worker.deployment import create_deployment_package
from automl_worker.explainability import generate_feature_importance, generate_shap_summary
from automl_worker.reporting import generate_summary_report
from automl_worker.settings import settings
from automl_worker.status import atomic_write_json, update_status, utc_now
from automl_worker.training import train_automl


def _load_dataset(input_path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    df = pd.read_csv(input_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("Dataset has no rows after dropping missing target values.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    if X.empty:
        raise ValueError("Dataset must contain at least one feature column.")
    return df, y


@celery_app.task(name="automl_worker.tasks.run_automl_job", bind=True)
def run_automl_job(self, **payload: Any) -> dict[str, Any]:
    job_id = payload["job_id"]
    output_dir = settings.output_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = utc_now()
    update_status(settings.output_dir, job_id, "RUNNING", start_time=start_time, error=None)

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        target_column = payload["target_column"]
        task_type = payload["task_type"]
        input_path = Path(payload["input_path"])
        df, y = _load_dataset(input_path, target_column)
        X = df.drop(columns=[target_column])

        data_summary = generate_data_profile(df, target_column, output_dir / "data_profile.html")
        training_result = train_automl(
            X,
            y,
            task_type=task_type,
            output_dir=output_dir,
            job_id=job_id,
            time_budget_seconds=int(payload["time_budget_seconds"]),
            max_candidates_per_family=settings.max_candidates_per_family,
            cv_folds=settings.cv_folds,
            random_state=settings.random_state,
        )

        generate_feature_importance(
            training_result.best_pipeline,
            X,
            y,
            scoring=training_result.metric,
            output_path=output_dir / "feature_importance.png",
        )
        generate_shap_summary(
            training_result.best_pipeline,
            X,
            task_type=task_type,
            output_path=output_dir / "shap_summary.png",
        )
        generate_summary_report(
            output_dir=output_dir,
            job_id=job_id,
            data_summary=data_summary,
            best_model_name=training_result.best_model_name,
            best_model_score=training_result.best_model_score,
            metric=training_result.metric,
            mlflow_run_id=training_result.mlflow_run_id,
            leaderboard=training_result.leaderboard,
        )
        create_deployment_package(output_dir)

        results = {
            "job_id": job_id,
            "best_model_name": training_result.best_model_name,
            "best_model_score": training_result.best_model_score,
            "evaluation_metric": training_result.metric,
            "mlflow_run_id": training_result.mlflow_run_id,
        }
        atomic_write_json(output_dir / "results.json", results)
        update_status(settings.output_dir, job_id, "SUCCESS", end_time=utc_now(), error=None)
        return results
    except Exception as exc:
        atomic_write_json(
            output_dir / "failure.json",
            {
                "job_id": job_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        update_status(settings.output_dir, job_id, "FAILED", end_time=utc_now(), error=str(exc))
        raise
