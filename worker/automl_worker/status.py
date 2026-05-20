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


def update_status(
    output_dir: Path,
    job_id: str,
    status: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    error: str | None = None,
) -> None:
    status_file = output_dir / job_id / "status.json"
    current = read_json(status_file) or {
        "job_id": job_id,
        "status": "PENDING",
        "start_time": None,
        "end_time": None,
        "error": None,
    }
    current.update(
        {
            "job_id": job_id,
            "status": status,
            "start_time": start_time if start_time is not None else current.get("start_time"),
            "end_time": end_time if end_time is not None else current.get("end_time"),
            "error": error,
        }
    )
    atomic_write_json(status_file, current)
