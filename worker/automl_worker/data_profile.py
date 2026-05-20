import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _figure_to_base64() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _distribution_plot(series: pd.Series, column: str) -> str:
    clean = series.dropna()
    plt.figure(figsize=(7, 4))
    if pd.api.types.is_numeric_dtype(clean):
        sns.histplot(clean, kde=True, color="#0f766e")
        plt.xlabel(column)
        plt.ylabel("Count")
    else:
        counts = clean.astype(str).value_counts().head(12)
        sns.barplot(x=counts.values, y=counts.index, color="#0f766e")
        plt.xlabel("Count")
        plt.ylabel(column)
    plt.title(f"Distribution: {column}")
    return _figure_to_base64()


def _correlation_plot(df: pd.DataFrame) -> str | None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(max(7, numeric.shape[1] * 0.8), max(5, numeric.shape[1] * 0.65)))
    sns.heatmap(corr, cmap="vlag", center=0, annot=numeric.shape[1] <= 8, fmt=".2f")
    plt.title("Numeric Feature Correlations")
    return _figure_to_base64()


def generate_data_profile(df: pd.DataFrame, target_column: str, output_path: Path) -> dict[str, int | str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = (missing["missing_count"] / max(len(df), 1) * 100).round(2)

    variables = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null": [int(df[col].notna().sum()) for col in df.columns],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
            "role": ["target" if col == target_column else "feature" for col in df.columns],
        }
    )

    describe_html = df.describe(include="all").fillna("").to_html(classes="data-table")
    variables_html = variables.to_html(index=False, classes="data-table")
    missing_html = missing.reset_index(names="column").to_html(index=False, classes="data-table")

    distribution_sections = []
    for column in df.columns[:16]:
        try:
            encoded = _distribution_plot(df[column], column)
            distribution_sections.append(
                f"""
                <figure>
                  <img src="data:image/png;base64,{encoded}" alt="Distribution plot for {html.escape(column)}" />
                  <figcaption>{html.escape(column)}</figcaption>
                </figure>
                """
            )
        except Exception as exc:
            distribution_sections.append(
                f"<p>Distribution plot unavailable for {html.escape(column)}: {html.escape(str(exc))}</p>"
            )

    corr_image = _correlation_plot(df)
    corr_html = (
        f'<img src="data:image/png;base64,{corr_image}" alt="Correlation heatmap" />'
        if corr_image
        else "<p>Not enough numeric columns for a correlation heatmap.</p>"
    )

    html_body = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Data Profile</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; background: #f8fafc; }}
        h1, h2 {{ color: #0f766e; }}
        section {{ margin: 24px 0; padding: 20px; background: #fff; border: 1px solid #d9e0ea; border-radius: 8px; }}
        .data-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        .data-table th, .data-table td {{ border: 1px solid #d9e0ea; padding: 8px; text-align: left; }}
        .data-table th {{ background: #eefdfa; }}
        img {{ max-width: 100%; height: auto; }}
        figure {{ margin: 18px 0; }}
      </style>
    </head>
    <body>
      <h1>Data Profile</h1>
      <section>
        <h2>Data Summary</h2>
        <p>Rows: {len(df)} | Columns: {len(df.columns)} | Target column: {html.escape(target_column)}</p>
        {describe_html}
      </section>
      <section>
        <h2>Variables</h2>
        {variables_html}
      </section>
      <section>
        <h2>Missing Values</h2>
        {missing_html}
      </section>
      <section>
        <h2>Distributions</h2>
        {''.join(distribution_sections)}
      </section>
      <section>
        <h2>Correlations</h2>
        {corr_html}
      </section>
    </body>
    </html>
    """
    output_path.write_text(html_body, encoding="utf-8")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "target_column": target_column,
    }
