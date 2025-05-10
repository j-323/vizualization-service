from functools import lru_cache
from fastapi import Depends
from celery import Celery
from src.config.settings import Settings
from src.utils.cache import RedisCache
from src.utils.logging import get_logger
from src.utils.metrics import Metrics
from src.pipelines.generation_pipeline import pipeline

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_cache(settings: Settings = Depends(get_settings)):
    return RedisCache(settings.REDIS_URL)

@lru_cache()
def get_logger():
    return get_logger("visual-agent")

@lru_cache()
def get_metrics():
    return Metrics()

@lru_cache()
def get_pipeline():
    return pipeline

@lru_cache()
def get_celery(settings: Settings = Depends(get_settings)):
    return Celery(
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_BROKER_URL,
    )