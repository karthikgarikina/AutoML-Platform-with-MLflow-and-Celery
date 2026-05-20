import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


@dataclass
class CandidateResult:
    model_id: str
    model_type: str
    mean_cv_score: float
    metric: str
    std_cv_score: float
    mlflow_run_id: str
    params: dict[str, Any]
    duration_seconds: float
    status: str = "SUCCESS"
    error: str | None = None


@dataclass
class TrainingResult:
    best_pipeline: Pipeline
    best_model_id: str
    best_model_name: str
    best_model_score: float
    metric: str
    mlflow_run_id: str
    leaderboard: pd.DataFrame


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _sample_params(space: dict[str, list[Any]], max_candidates: int, random_state: int) -> list[dict[str, Any]]:
    if not space:
        return [{}]
    sampled = list(ParameterSampler(space, n_iter=min(max_candidates, _space_size(space)), random_state=random_state))
    return sampled or [{}]


def _space_size(space: dict[str, list[Any]]) -> int:
    size = 1
    for values in space.values():
        size *= max(len(values), 1)
    return size


def _classification_specs(random_state: int) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1200, solver="lbfgs", random_state=random_state),
            {"C": [0.1, 1.0, 3.0, 10.0]},
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {"n_estimators": [80, 140], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
        ),
        (
            "SVC",
            SVC(probability=True, random_state=random_state),
            {"C": [0.5, 1.0, 2.0], "kernel": ["rbf", "linear"], "gamma": ["scale"]},
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=random_state),
            {"n_estimators": [70, 120], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
        (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(random_state=random_state, n_jobs=1),
            {"n_estimators": [80, 140], "max_depth": [None, 8, 14], "min_samples_leaf": [1, 2]},
        ),
        (
            "VotingEnsembleClassifier",
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1200, C=1.0, random_state=random_state)),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=120,
                            max_depth=None,
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                    ("svc", SVC(probability=True, C=1.0, gamma="scale", random_state=random_state)),
                ],
                voting="soft",
            ),
            {},
        ),
        (
            "StackingEnsembleClassifier",
            StackingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=80,
                            max_depth=None,
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                    ("svc", SVC(probability=True, C=1.0, gamma="scale", random_state=random_state)),
                ],
                final_estimator=LogisticRegression(max_iter=1200, random_state=random_state),
                cv=3,
            ),
            {},
        ),
    ]


def _regression_specs(random_state: int) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "Ridge",
            Ridge(random_state=random_state),
            {"alpha": [0.1, 1.0, 5.0, 10.0]},
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            {"n_estimators": [80, 140], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
        ),
        (
            "SVR",
            SVR(),
            {"C": [0.5, 1.0, 2.0], "kernel": ["rbf", "linear"], "gamma": ["scale"]},
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(random_state=random_state),
            {"n_estimators": [70, 120], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
        (
            "ExtraTreesRegressor",
            ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            {"n_estimators": [80, 140], "max_depth": [None, 8, 14], "min_samples_leaf": [1, 2]},
        ),
        (
            "VotingEnsembleRegressor",
            VotingRegressor(
                estimators=[
                    ("ridge", Ridge(alpha=1.0, random_state=random_state)),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=120,
                            max_depth=None,
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                    ("gbr", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
                ]
            ),
            {},
        ),
        (
            "StackingEnsembleRegressor",
            StackingRegressor(
                estimators=[
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=80,
                            max_depth=None,
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                    ("gbr", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
                ],
                final_estimator=Ridge(alpha=1.0, random_state=random_state),
                cv=3,
            ),
            {},
        ),
    ]


def _cv_strategy(task_type: str, y: pd.Series, requested_folds: int, random_state: int):
    if task_type == "classification":
        class_counts = y.value_counts(dropna=False)
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        folds = min(requested_folds, min_class_count)
        if folds >= 2:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    folds = min(requested_folds, len(y))
    folds = max(2, folds)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _apply_params(estimator: Any, params: dict[str, Any]) -> Any:
    estimator = clone(estimator)
    if params:
        estimator.set_params(**params)
    return estimator


def train_automl(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: str,
    output_dir: Path,
    job_id: str,
    time_budget_seconds: int,
    max_candidates_per_family: int,
    cv_folds: int,
    random_state: int,
) -> TrainingResult:
    metric = "f1_macro" if task_type == "classification" else "r2"
    cv = _cv_strategy(task_type, y, cv_folds, random_state)
    preprocessor = build_preprocessor(X)
    specs = _classification_specs(random_state) if task_type == "classification" else _regression_specs(random_state)
    deadline = time.monotonic() + max(10, time_budget_seconds)

    mlflow.set_experiment(f"automl-job-{job_id}")
    candidate_results: list[CandidateResult] = []
    best_score = -np.inf
    best_pipeline: Pipeline | None = None
    best_model_id = ""
    best_model_name = ""
    best_run_id = ""

    for family_index, (model_type, estimator, param_space) in enumerate(specs):
        params_list = _sample_params(param_space, max_candidates_per_family, random_state + family_index)
        if "Ensemble" in model_type:
            params_list = [{}]

        for param_index, params in enumerate(params_list, start=1):
            if param_index > 1 and time.monotonic() > deadline:
                break
            model_id = f"{model_type.lower()}-{param_index:02d}"
            started = time.monotonic()
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", _apply_params(estimator, params)),
                ]
            )
            with mlflow.start_run(run_name=model_id) as run:
                run_id = run.info.run_id
                mlflow.set_tag("job_id", job_id)
                mlflow.set_tag("task_type", task_type)
                mlflow.set_tag("model_type", model_type)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param(
                    "feature_engineering_steps",
                    "SimpleImputer, StandardScaler, OneHotEncoder",
                )
                for key, value in params.items():
                    mlflow.log_param(key, value)
                try:
                    scores = cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring=metric,
                        n_jobs=1,
                        error_score="raise",
                    )
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    duration = float(time.monotonic() - started)
                    mlflow.log_metric(metric, mean_score)
                    mlflow.log_metric("mean_cv_score", mean_score)
                    mlflow.log_metric("std_cv_score", std_score)
                    mlflow.log_metric("duration_seconds", duration)
                    mlflow.set_tag("candidate_status", "SUCCESS")
                    candidate_results.append(
                        CandidateResult(
                            model_id=model_id,
                            model_type=model_type,
                            mean_cv_score=mean_score,
                            metric=metric,
                            std_cv_score=std_score,
                            mlflow_run_id=run_id,
                            params=params,
                            duration_seconds=duration,
                        )
                    )
                    if mean_score > best_score:
                        best_score = mean_score
                        best_pipeline = clone(pipeline)
                        best_model_id = model_id
                        best_model_name = model_type
                        best_run_id = run_id
                except Exception as exc:
                    duration = float(time.monotonic() - started)
                    mlflow.set_tag("candidate_status", "FAILED")
                    mlflow.log_param("error", str(exc)[:240])
                    candidate_results.append(
                        CandidateResult(
                            model_id=model_id,
                            model_type=model_type,
                            mean_cv_score=float("nan"),
                            metric=metric,
                            std_cv_score=float("nan"),
                            mlflow_run_id=run_id,
                            params=params,
                            duration_seconds=duration,
                            status="FAILED",
                            error=str(exc),
                        )
                    )

    if best_pipeline is None:
        raise RuntimeError("No candidate model completed successfully.")

    best_pipeline.fit(X, y)
    best_model_path = output_dir / "best_model.pkl"
    with best_model_path.open("wb") as model_file:
        pickle.dump(best_pipeline, model_file)

    leaderboard = pd.DataFrame([result.__dict__ for result in candidate_results])
    leaderboard = leaderboard.sort_values("mean_cv_score", ascending=False, na_position="last")
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)

    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_run", "true")
        mlflow.log_artifact(str(best_model_path), artifact_path="model")
        mlflow.log_artifact(str(output_dir / "leaderboard.csv"), artifact_path="reports")

    return TrainingResult(
        best_pipeline=best_pipeline,
        best_model_id=best_model_id,
        best_model_name=best_model_name,
        best_model_score=float(best_score),
        metric=metric,
        mlflow_run_id=best_run_id,
        leaderboard=leaderboard,
    )
