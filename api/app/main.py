import csv
import io
import shutil
import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.celery_client import celery_app
from app.config import get_settings
from app.job_store import (
    atomic_write_json,
    job_dir,
    read_json,
    request_path,
    results_path,
    status_path,
    utc_now,
)

app = FastAPI(
    title="AutoML Platform",
    description="A Celery-backed AutoML platform with MLflow tracking.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


def _validate_csv_upload(file: UploadFile, data: bytes, target_column: str) -> list[str]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    try:
        text = data[:8192].decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV header: {exc}") from exc

    header = [column.strip() for column in header]
    if not header or any(not column for column in header):
        raise HTTPException(status_code=400, detail="CSV header must contain named columns.")
    if target_column not in header:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' was not found in the CSV header.",
        )
    return header


@app.post("/jobs", status_code=202)
async def start_job(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...),
    time_budget_seconds: int = Form(...),
) -> JSONResponse:
    settings = get_settings()
    normalized_task_type = task_type.strip().lower()
    if normalized_task_type not in {"classification", "regression"}:
        raise HTTPException(
            status_code=400,
            detail="task_type must be either 'classification' or 'regression'.",
        )
    if time_budget_seconds < 10:
        raise HTTPException(status_code=400, detail="time_budget_seconds must be at least 10.")
    if time_budget_seconds > 24 * 60 * 60:
        raise HTTPException(status_code=400, detail="time_budget_seconds is too large.")

    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded CSV exceeds MAX_UPLOAD_BYTES.")
    header = _validate_csv_upload(file, data, target_column)

    job_id = uuid.uuid4().hex
    upload_job_dir = settings.upload_dir / job_id
    output_job_dir = job_dir(settings.output_dir, job_id)
    upload_job_dir.mkdir(parents=True, exist_ok=True)
    output_job_dir.mkdir(parents=True, exist_ok=True)

    csv_path = upload_job_dir / "input.csv"
    csv_path.write_bytes(data)

    request_payload = {
        "job_id": job_id,
        "original_filename": file.filename,
        "input_path": str(csv_path),
        "target_column": target_column,
        "task_type": normalized_task_type,
        "time_budget_seconds": time_budget_seconds,
        "columns": header,
        "created_at": utc_now(),
    }
    atomic_write_json(request_path(settings.output_dir, job_id), request_payload)
    atomic_write_json(
        status_path(settings.output_dir, job_id),
        {
            "job_id": job_id,
            "status": "PENDING",
            "start_time": None,
            "end_time": None,
            "error": None,
        },
    )

    celery_app.send_task(
        "automl_worker.tasks.run_automl_job",
        kwargs=request_payload,
        task_id=job_id,
        queue="automl",
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "PENDING"})


def _status_from_celery(job_id: str) -> str:
    state = AsyncResult(job_id, app=celery_app).state
    return {
        "PENDING": "PENDING",
        "RECEIVED": "PENDING",
        "STARTED": "RUNNING",
        "RETRY": "RUNNING",
        "SUCCESS": "SUCCESS",
        "FAILURE": "FAILED",
        "REVOKED": "FAILED",
    }.get(state, state)


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, str | None]:
    settings = get_settings()
    payload = read_json(status_path(settings.output_dir, job_id))
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    celery_status = _status_from_celery(job_id)
    status = payload.get("status", celery_status)
    if status == "PENDING" and celery_status == "RUNNING":
        status = "RUNNING"
    return {
        "job_id": job_id,
        "status": status,
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
    }


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str) -> dict[str, str | float]:
    settings = get_settings()
    status = read_json(status_path(settings.output_dir, job_id))
    results = read_json(results_path(settings.output_dir, job_id))
    if status is None or status.get("status") != "SUCCESS" or results is None:
        raise HTTPException(status_code=404, detail="Completed job results were not found.")
    return {
        "job_id": job_id,
        "best_model_name": results["best_model_name"],
        "best_model_score": float(results["best_model_score"]),
        "evaluation_metric": results["evaluation_metric"],
        "mlflow_run_id": results["mlflow_run_id"],
    }


@app.get("/jobs/{job_id}/artifacts/{artifact_path:path}", include_in_schema=False)
def get_job_artifact(
    job_id: str,
    artifact_path: str,
    download: bool = Query(False),
) -> FileResponse:
    settings = get_settings()
    root = job_dir(settings.output_dir, job_id).resolve()
    candidate = (root / artifact_path).resolve()
    if not candidate.is_file() or root not in candidate.parents:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if download:
        return FileResponse(candidate, filename=candidate.name)
    return FileResponse(candidate)


@app.delete("/jobs/{job_id}", include_in_schema=False)
def delete_job(job_id: str) -> dict[str, str]:
    settings = get_settings()
    root = job_dir(settings.output_dir, job_id)
    upload_root = settings.upload_dir / job_id
    if not root.exists() and not upload_root.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(upload_root, ignore_errors=True)
    return {"job_id": job_id, "status": "deleted"}
