from celery import Celery
from src.config.settings import Settings
from src.pipelines.generation_pipeline import pipeline

settings = Settings()
celery_app = Celery(
    "visual_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_BROKER_URL,
)

@celery_app.task(name="src.services.async_worker.generate_task")
def generate_task(prompt: str) -> dict:
    url, typ = pipeline.generate(prompt)
    return {"url": url, "type": typ}