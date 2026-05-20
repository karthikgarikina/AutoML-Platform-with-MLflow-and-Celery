FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential grep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY automl_worker ./automl_worker

CMD ["celery", "-A", "automl_worker.celery_app", "worker", "--loglevel=info", "--hostname=worker@worker", "--concurrency=1"]
