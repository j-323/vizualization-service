import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Tuple, List, Dict, Optional

from src.registry.model_registry import select_candidates
from src.models.base import ModelClient
from src.models.evaluator import CLIPScorer
from src.utils.cache import RedisCache

class GenerationPipeline:
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        max_workers: int = 3,
        top_k: int = 2,
        timeout: int = 60
    ):
        self.logger = logging.getLogger(__name__)
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.top_k = top_k
        self.timeout = timeout
        self.scorer = CLIPScorer()

    def generate(self, prompt: str) -> Tuple[str, str]:
        start_ts = time.time()
        img_or_vid = self._determine_type(prompt)

        # 1) Попробовать из кэша
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                self.logger.debug(f"Cache hit: {cached}")
                return cached, img_or_vid

        # 2) Получаем кандидатов и фильтруем по health_check
        candidates = [
            m for m in select_candidates(prompt, img_or_vid)
            if m.health_check()
        ]

        # 3) Параллельно запускаем top_k моделей
        results = self._generate_concurrent(candidates[: self.top_k], prompt)

        # 4) Выбираем финальную ссылку
        if img_or_vid == "image":
            url = self._select_best_image(results, prompt)
        else:
            url = self._select_video(results, candidates, prompt)

        # 5) Кэшируем и логируем
        if self.cache:
            self.cache.set(prompt, url)
        elapsed = time.time() - start_ts
        self.logger.info(f"Generated {img_or_vid} in {elapsed:.2f}s → {url}")
        return url, img_or_vid

    def _determine_type(self, prompt: str) -> str:
        low = prompt.lower()
        return "video" if "mp4" in low or "анимац" in low else "image"

    def _generate_concurrent(
        self,
        models: List[ModelClient],
        prompt: str
    ) -> Dict[str, Optional[str]]:
        futures = {self.executor.submit(m.generate, prompt): m.name for m in models}
        urls: Dict[str, Optional[str]] = {}
        try:
            for fut in as_completed(futures, timeout=self.timeout):
                name = futures[fut]
                try:
                    urls[name] = fut.result()
                except Exception as e:
                    self.logger.warning(f"Model {name} failed: {e}")
                    urls[name] = None
        except TimeoutError:
            self.logger.warning("Timeout reached during concurrent generation")
        return urls

    def _select_best_image(
        self,
        results: Dict[str, Optional[str]],
        prompt: str
    ) -> str:
        scores: Dict[str, float] = {}
        for name, url in results.items():
            if url:
                try:
                    scores[name] = self.scorer.score(url, prompt)
                except Exception as e:
                    self.logger.warning(f"Scoring failed for {name}: {e}")
        if scores:
            best = max(scores, key=scores.get)
            return results[best]  # type: ignore
        return self._fallback_sequence(prompt)

    def _select_video(
        self,
        results: Dict[str, Optional[str]],
        candidates: List[ModelClient],
        prompt: str
    ) -> str:
        # возвращаем первый успешный
        for url in results.values():
            if url:
                return url
        # иначе fallback по очереди
        return self._fallback_sequence(prompt)

    def _fallback_sequence(self, prompt: str) -> str:
        self.logger.debug("Entering fallback sequence")
        for model in select_candidates(prompt, self._determine_type(prompt))[self.top_k :]:
            try:
                url = model.generate(prompt)
                self.logger.debug(f"Fallback succeeded with {model.name}")
                return url
            except Exception as e:
                self.logger.warning(f"Fallback model {model.name} failed: {e}")
        raise RuntimeError("Все модели недоступны или вернули ошибку")


# DI-экземпляр
from src.utils.cache import RedisCache
from src.config.settings import Settings

settings = Settings()
pipeline = GenerationPipeline(
    cache=RedisCache(settings.REDIS_URL),
    max_workers=4,
    top_k=3,
    timeout=90
)