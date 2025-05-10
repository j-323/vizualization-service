from pydantic import BaseSettings, AnyUrl

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"

    IMAGE_MODEL_A_URL: AnyUrl
    IMAGE_MODEL_A_KEY: str
    IMAGE_MODEL_B_URL: AnyUrl
    IMAGE_MODEL_B_KEY: str
    VIDEO_MODEL_URL: AnyUrl
    VIDEO_MODEL_KEY: str

    REDIS_URL: AnyUrl
    CELERY_BROKER_URL: AnyUrl

    class Config:
        env_file = ".env"