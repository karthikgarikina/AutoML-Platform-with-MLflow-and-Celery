from celery import Celery

from automl_worker.settings import settings


celery_app = Celery(
    "automl_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["automl_worker.tasks"],
)

celery_app.conf.update(
    task_default_queue="automl",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=7 * 24 * 60 * 60,
    broker_connection_retry_on_startup=True,
)
