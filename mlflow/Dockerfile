FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow==2.22.0 \
    && mkdir -p /mlflow/mlruns /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "file:/mlflow/mlruns", "--default-artifact-root", "/mlflow/artifacts", "--serve-artifacts", "--host", "0.0.0.0", "--port", "5000"]
