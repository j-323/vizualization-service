import time
from abc import ABC, abstractmethod
from src.utils.retry import retry
from src.utils.rate_limiter import TokenBucket
from prometheus_client import Histogram, Counter

REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency per model', ['model'])
REQUEST_COUNT   = Counter('request_count', 'Count per model', ['model'])

class ModelClient(ABC):
    def __init__(self, name: str, rate_limit: float = None, rate_period: float = 1.0):
        self.name = name
        if rate_limit:
            self._bucket = TokenBucket(rate_limit, rate_period)
        else:
            self._bucket = None
        self._last_health = {"ok": True, "ts": 0}

    @abstractmethod
    def _call_model(self, prompt: str) -> str:
        pass

    def generate(self, prompt: str) -> str:
        start = time.time()
        if self._bucket:
            self._bucket.consume()
        result = self._generate_with_retry(prompt)
        latency = time.time() - start
        REQUEST_LATENCY.labels(model=self.name).observe(latency)
        REQUEST_COUNT.labels(model=self.name).inc()
        return result

    @retry(exceptions=(Exception,), tries=3, delay=1, backoff=2)
    def _generate_with_retry(self, prompt: str) -> str:
        return self._call_model(prompt)

    def health_check(self, force: bool = False) -> bool:
        now = time.time()
        if force or now - self._last_health["ts"] > 60:
            try:
                self._perform_health()
                self._last_health = {"ok": True, "ts": now}
            except:
                self._last_health = {"ok": False, "ts": now}
        return self._last_health["ok"]

    def _perform_health(self):
        self._call_model("__health_check__")