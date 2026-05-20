import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def job_dir(output_dir: Path, job_id: str) -> Path:
    return output_dir / job_id


def status_path(output_dir: Path, job_id: str) -> Path:
    return job_dir(output_dir, job_id) / "status.json"


def results_path(output_dir: Path, job_id: str) -> Path:
    return job_dir(output_dir, job_id) / "results.json"


def request_path(output_dir: Path, job_id: str) -> Path:
    return job_dir(output_dir, job_id) / "job_request.json"
