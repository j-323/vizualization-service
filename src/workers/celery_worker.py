from src.services.async_worker import celery_app

if __name__ == "__main__":
    celery_app.start()