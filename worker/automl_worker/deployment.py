import shutil
from pathlib import Path


DEPLOYMENT_MAIN = '''import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

app = FastAPI(title="AutoML Prediction Service", version="1.0.0")


class PredictionRequest(BaseModel):
    data: dict[str, Any] | list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    rows = payload.data if isinstance(payload.data, list) else [payload.data]
    if not rows:
        raise HTTPException(status_code=400, detail="At least one row is required.")
    frame = pd.DataFrame(rows)
    predictions = model.predict(frame)
    response: dict[str, Any] = {"predictions": predictions.tolist()}
    if hasattr(model, "predict_proba"):
        try:
            response["probabilities"] = model.predict_proba(frame).tolist()
        except Exception:
            pass
    return response
'''


DEPLOYMENT_REQUIREMENTS = """fastapi==0.115.12
uvicorn[standard]==0.34.2
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
joblib==1.4.2
"""


DEPLOYMENT_DOCKERFILE = """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""


def create_deployment_package(output_dir: Path) -> Path:
    deployment_dir = output_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "best_model.pkl", deployment_dir / "model.pkl")
    (deployment_dir / "main.py").write_text(DEPLOYMENT_MAIN, encoding="utf-8")
    (deployment_dir / "requirements.txt").write_text(DEPLOYMENT_REQUIREMENTS, encoding="utf-8")
    (deployment_dir / "Dockerfile").write_text(DEPLOYMENT_DOCKERFILE, encoding="utf-8")
    return deployment_dir
