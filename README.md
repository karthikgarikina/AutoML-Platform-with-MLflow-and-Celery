# AutoML Platform with MLflow and Celery

This project is a small MLOps platform for tabular data. Upload a CSV, choose the target column, and the system runs an asynchronous AutoML job that profiles the data, trains multiple model families, tracks every run in MLflow, explains the best model, and packages it for deployment.

## What It Builds

- `api`: FastAPI REST API plus an operator UI at `http://localhost:8000`.
- `worker`: Celery worker that performs profiling, feature engineering, model search, MLflow logging, reports, SHAP, and deployment packaging.
- `redis`: Celery broker and result backend.
- `mlflow`: MLflow Tracking Server at `http://localhost:5000`.
- `uploads/`: runtime CSV inputs, ignored by git.
- `output/`: runtime job artifacts, ignored by git except `.gitkeep`.

The API stays responsive while the worker handles long-running ML tasks in the background.

## Quick Start

Clone the repository, then create local environment files from the examples:

```bash
git clone https://github.com/karthikgarikina/AutoML-Platform-with-MLflow-and-Celery
cp .env.example .env
cp api/.env.example api/.env
cp worker/.env.example worker/.env
```

Start the full platform:

```bash
docker compose up --build
```

Open:

- UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

If a port is already busy, stop the conflicting local container/process first so the project can use the expected ports: `8000`, `5000`, and `6379`.

## Run Your First Job

Use the UI, or submit the bundled sample dataset:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@examples/iris_sample.csv" \
  -F "target_column=species" \
  -F "task_type=classification" \
  -F "time_budget_seconds=60"
```

Response:

```json
{
  "job_id": "unique-job-id",
  "status": "PENDING"
}
```

Check status:

```bash
curl http://localhost:8000/jobs/{job_id}
```

Fetch final results:

```bash
curl http://localhost:8000/jobs/{job_id}/results
```

## API Summary

### `POST /jobs`

Multipart form fields:

- `file`: CSV dataset.
- `target_column`: target variable name.
- `task_type`: `classification` or `regression`.
- `time_budget_seconds`: maximum search time.

Returns `202 Accepted` with `job_id` and `PENDING` status.

### `GET /jobs/{job_id}`

Returns job status:

```json
{
  "job_id": "string",
  "status": "PENDING | RUNNING | SUCCESS | FAILED",
  "start_time": "ISO-8601 or null",
  "end_time": "ISO-8601 or null"
}
```

### `GET /jobs/{job_id}/results`

Returns `404` until the job succeeds. On success:

```json
{
  "job_id": "string",
  "best_model_name": "string",
  "best_model_score": 0.0,
  "evaluation_metric": "f1_macro or r2",
  "mlflow_run_id": "string"
}
```

## What Happens Inside

1. FastAPI receives the CSV and saves it to `uploads/{job_id}/input.csv`.
2. FastAPI queues a Celery task through Redis.
3. The worker reads the CSV and generates a data profile.
4. The worker builds a preprocessing pipeline with imputation, scaling, and encoding.
5. The worker evaluates linear, tree-based, SVM, boosting, voting ensemble, and stacking ensemble pipelines.
6. Each candidate run logs params, metrics, tags, and artifacts to MLflow under `automl-job-{job_id}`.
7. The best pipeline is serialized and explained with feature importance and SHAP.
8. The worker writes the final report and deployment package to `output/{job_id}/`.

## Generated Artifacts

Each successful job creates:

- `data_profile.html`: data summary, missing values, distributions, correlations.
- `leaderboard.csv`: all evaluated pipelines and scores.
- `best_model.pkl`: pickled sklearn `Pipeline` with preprocessing and model.
- `feature_importance.png`: feature importance chart.
- `shap_summary.png`: SHAP explanation plot.
- `summary_report.html`: human-readable report with embedded leaderboard and plots.
- `deployment/`: self-contained prediction service package.

The UI opens the summary report and data profile. Download links for the leaderboard, model, and deployment files are inside `summary_report.html`.

## Deployment Package

After a job succeeds:

```bash
cd output/{job_id}/deployment
docker build -t automl-predict:{job_id} .
docker run -p 8080:8080 automl-predict:{job_id}
```

Predict with:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data":{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}}'
```

`deployment/main.py` loads `model.pkl` and exposes `/predict`; it is the serving code for the trained best model.

## Environment

Required variables are documented in `.env.example`:

- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `MLFLOW_TRACKING_URI`
- `API_PORT`
- `REDIS_PORT`
- `MLFLOW_PORT`
- `UPLOAD_DIR`
- `OUTPUT_DIR`
- `MAX_UPLOAD_BYTES`
- `AUTOML_MAX_CANDIDATES_PER_FAMILY`
- `AUTOML_CV_FOLDS`
- `AUTOML_RANDOM_STATE`

Service-level examples live in `api/.env.example` and `worker/.env.example`.

## Git Hygiene

Runtime data is intentionally ignored:

- `uploads/*`
- `output/*`

Only `.gitkeep` and `.gitignore` files are committed in those folders.

## Checks

```bash
python -m unittest discover -s tests
python -m compileall api worker tests mlflow
docker compose config
```

## Conclusion

This platform demonstrates the complete backend MLOps loop: submit data, run AutoML asynchronously, track experiments, explain the chosen model, and export a deployable prediction service. It is intentionally containerized and reproducible so it can be evaluated or extended without manual setup.
