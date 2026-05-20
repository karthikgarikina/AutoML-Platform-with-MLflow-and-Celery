from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance


def _feature_names(pipeline: Any, X: pd.DataFrame) -> list[str]:
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        return list(X.columns)


def generate_feature_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps.get("model")
    names = _feature_names(pipeline, X)
    values = None
    title = "Feature Importance"

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or len(values) != len(names):
        sample_size = min(len(X), 200)
        sample_X = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        sample_y = y.loc[sample_X.index]
        result = permutation_importance(
            pipeline,
            sample_X,
            sample_y,
            scoring=scoring,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        names = list(X.columns)
        values = result.importances_mean
        title = "Permutation Feature Importance"

    importance = (
        pd.DataFrame({"feature": names, "importance": values})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("importance", ascending=False)
        .head(20)
    )
    if importance.empty:
        importance = pd.DataFrame({"feature": ["No importances available"], "importance": [0.0]})

    plt.figure(figsize=(9, max(4, min(10, len(importance) * 0.42))))
    sns.barplot(data=importance, x="importance", y="feature", color="#0f766e")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_shap_summary(
    pipeline: Any,
    X: pd.DataFrame,
    *,
    task_type: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_size = min(len(X), 40)
    sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X.copy()
    columns = list(X.columns)

    try:
        import shap

        def predict_fn(raw: Any) -> np.ndarray:
            frame = pd.DataFrame(raw, columns=columns)
            model = pipeline.named_steps.get("model")
            if task_type == "classification" and hasattr(model, "predict_proba"):
                probabilities = pipeline.predict_proba(frame)
                if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                    return probabilities[:, 1]
                return probabilities.ravel()
            return np.asarray(pipeline.predict(frame))

        masker = shap.maskers.Independent(sample, max_samples=min(40, len(sample)))
        explainer = shap.Explainer(predict_fn, masker, feature_names=columns)
        shap_values = explainer(sample, max_evals=max(2 * len(columns) + 1, 50), silent=True)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        # A valid PNG is still generated so downstream reporting never breaks.
        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.text(
            0.02,
            0.72,
            "SHAP summary plot could not be generated for this pipeline.",
            fontsize=14,
            weight="bold",
        )
        plt.text(0.02, 0.52, f"Reason: {str(exc)[:220]}", fontsize=10, wrap=True)
        plt.text(
            0.02,
            0.34,
            "The feature importance plot in this report provides a model-level explanation fallback.",
            fontsize=11,
            wrap=True,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
