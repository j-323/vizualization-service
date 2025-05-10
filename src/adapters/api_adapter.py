import httpx
from src.config.settings import Settings
from src.models.base import ModelClient

settings = Settings()

class APIModelA(ModelClient):
    def __init__(self):
        super().__init__("ImageA_API", rate_limit=5)

    def _call_model(self, prompt: str) -> str:
        r = httpx.post(
            settings.IMAGE_MODEL_A_URL,
            json={"prompt": prompt},
            headers={"Authorization": f"Bearer {settings.IMAGE_MODEL_A_KEY}"},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()["url"]

class APIModelB(ModelClient):
    def __init__(self):
        super().__init__("ImageB_API", rate_limit=3)

    def _call_model(self, prompt: str) -> str:
        r = httpx.post(
            settings.IMAGE_MODEL_B_URL,
            json={"prompt": prompt},
            headers={"Authorization": f"Bearer {settings.IMAGE_MODEL_B_KEY}"},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["url"]

class APIVideoC(ModelClient):
    def __init__(self):
        super().__init__("Video_API", rate_limit=1)

    def _call_model(self, prompt: str) -> str:
        r = httpx.post(
            settings.VIDEO_MODEL_URL,
            json={"prompt": prompt},
            headers={"Authorization": f"Bearer {settings.VIDEO_MODEL_KEY}"},
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()["url"]