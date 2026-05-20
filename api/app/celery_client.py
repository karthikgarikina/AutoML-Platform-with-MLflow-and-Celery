from celery import Celery

from app.config import get_settings


def make_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "automl_api",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    app.conf.update(
        task_default_queue="automl",
        task_track_started=True,
        result_expires=7 * 24 * 60 * 60,
        broker_connection_retry_on_startup=True,
    )
    return app


celery_app = make_celery_app()
