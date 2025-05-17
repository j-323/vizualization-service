import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.config.settings import Settings
from src.models.base import ModelClient

settings = Settings()

class APIClientMixin:
    """
    Общие методы для HTTP-клиентов: заголовки, запросы, retry.
    """
    def _get_headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _post(self, url: str, payload: dict, headers: dict, timeout: float) -> str:
        """
        Отправляет POST и возвращает поле "url" из JSON, либо кидает.
        retry с backoff при HTTPError.
        """
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "url" not in data:
            raise ValueError(f"Unexpected response format: {data}")
        return data["url"]

class APIModelA(ModelClient, APIClientMixin):
    def __init__(self):
        super().__init__("ImageA_API", rate_limit=5)

    def _call_model(self, prompt: str) -> str:
        return self._post(
            settings.IMAGE_MODEL_A_URL,
            payload={"prompt": prompt},
            headers=self._get_headers(settings.IMAGE_MODEL_A_KEY),
            timeout=30.0,
        )

class APIModelB(ModelClient, APIClientMixin):
    def __init__(self):
        super().__init__("ImageB_API", rate_limit=3)

    def _call_model(self, prompt: str) -> str:
        return self._post(
            settings.IMAGE_MODEL_B_URL,
            payload={"prompt": prompt, "style": "photo"},
            headers=self._get_headers(settings.IMAGE_MODEL_B_KEY),
            timeout=60.0,
        )

class APIVideoC(ModelClient, APIClientMixin):
    def __init__(self):
        super().__init__("Video_API", rate_limit=1)

    def _call_model(self, prompt: str) -> str:
        return self._post(
            settings.VIDEO_MODEL_URL,
            payload={"prompt": prompt, "format": "mp4"},
            headers=self._get_headers(settings.VIDEO_MODEL_KEY),
            timeout=120.0,
        )