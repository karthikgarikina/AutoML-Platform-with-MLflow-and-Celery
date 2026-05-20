import base64
import html
from pathlib import Path
from typing import Any

import pandas as pd


def _image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate_summary_report(
    *,
    output_dir: Path,
    job_id: str,
    data_summary: dict[str, Any],
    best_model_name: str,
    best_model_score: float,
    metric: str,
    mlflow_run_id: str,
    leaderboard: pd.DataFrame,
) -> Path:
    feature_importance = _image_to_base64(output_dir / "feature_importance.png")
    shap_summary = _image_to_base64(output_dir / "shap_summary.png")
    leaderboard_html = leaderboard.to_html(index=False, classes="data-table")
    best_row = leaderboard[leaderboard["mlflow_run_id"] == mlflow_run_id]
    best_params = "{}"
    if not best_row.empty and "params" in best_row:
        best_params = str(best_row.iloc[0]["params"])

    report_path = output_dir / "summary_report.html"
    report_path.write_text(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>AutoML Summary Report</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; background: #f8fafc; }}
            h1, h2 {{ color: #0f766e; }}
            section {{ margin: 24px 0; padding: 20px; background: #fff; border: 1px solid #d9e0ea; border-radius: 8px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
            .metric {{ padding: 14px; border: 1px solid #d9e0ea; border-radius: 8px; background: #fbfcfe; }}
            .metric span {{ display: block; color: #667085; font-size: 12px; text-transform: uppercase; font-weight: bold; }}
            .metric strong {{ display: block; margin-top: 6px; word-break: break-word; }}
            .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
            .data-table th, .data-table td {{ border: 1px solid #d9e0ea; padding: 8px; text-align: left; }}
            .data-table th {{ background: #eefdfa; }}
            .download-grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .download-link {{ display: inline-flex; align-items: center; min-height: 40px; padding: 0 14px; border: 1px solid #bdd7d3; border-radius: 8px; color: #115e59; background: #f1fbfa; font-weight: bold; text-decoration: none; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #d9e0ea; border-radius: 8px; }}
          </style>
        </head>
        <body>
          <h1>AutoML Summary Report</h1>
          <section>
            <h2>Data Summary</h2>
            <div class="metric-grid">
              <div class="metric"><span>Job ID</span><strong>{html.escape(job_id)}</strong></div>
              <div class="metric"><span>Rows</span><strong>{data_summary.get("rows")}</strong></div>
              <div class="metric"><span>Columns</span><strong>{data_summary.get("columns")}</strong></div>
              <div class="metric"><span>Missing cells</span><strong>{data_summary.get("missing_cells")}</strong></div>
            </div>
            <p>Full profiling output is available in <code>data_profile.html</code>.</p>
          </section>
          <section>
            <h2>Best Model</h2>
            <div class="metric-grid">
              <div class="metric"><span>Algorithm</span><strong>{html.escape(best_model_name)}</strong></div>
              <div class="metric"><span>{html.escape(metric)}</span><strong>{best_model_score:.6f}</strong></div>
              <div class="metric"><span>MLflow run</span><strong>{html.escape(mlflow_run_id)}</strong></div>
              <div class="metric"><span>Key parameters</span><strong>{html.escape(best_params)}</strong></div>
            </div>
          </section>
          <section>
            <h2>Leaderboard</h2>
            {leaderboard_html}
          </section>
          <section>
            <h2>Feature Importance</h2>
            <img src="data:image/png;base64,{feature_importance}" alt="Feature importance plot" />
          </section>
          <section>
            <h2>Explainability</h2>
            <img src="data:image/png;base64,{shap_summary}" alt="SHAP summary plot" />
          </section>
          <section>
            <h2>Downloads</h2>
            <div class="download-grid">
              <a class="download-link" href="leaderboard.csv?download=1" download>Leaderboard CSV</a>
              <a class="download-link" href="best_model.pkl?download=1" download>Best Model Pickle</a>
              <a class="download-link" href="data_profile.html?download=1" download>Data Profile HTML</a>
              <a class="download-link" href="deployment/model.pkl?download=1" download>Deployment Model</a>
              <a class="download-link" href="deployment/main.py?download=1" download>Prediction API</a>
              <a class="download-link" href="deployment/requirements.txt?download=1" download>Deployment Requirements</a>
              <a class="download-link" href="deployment/Dockerfile?download=1" download>Deployment Dockerfile</a>
            </div>
          </section>
        </body>
        </html>
        """,
        encoding="utf-8",
    )
    return report_path
