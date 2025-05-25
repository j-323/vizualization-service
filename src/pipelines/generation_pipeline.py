import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
from typing import Tuple, List, Dict, Optional

from src.registry.model_registry import select_candidates
from src.models.base import ModelClient
from src.models.evaluator import CLIPScorer
from src.utils.cache import RedisCache

logger = logging.getLogger(__name__)

class GenerationPipeline:
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        max_workers: int = 3,
        top_k: int = 2,
        timeout: int = 60
    ):
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.top_k = top_k
        self.timeout = timeout
        self.scorer = CLIPScorer()

    def generate(self, prompt: str) -> Tuple[str, str]:
        start_ts = time.time()
        kind = "video" if "mp4" in prompt.lower() or "анимац" in prompt.lower() else "image"

        # кэширование
        if self.cache:
            hit = self.cache.get(prompt)
            if hit:
                logger.debug(f"cache hit for '{prompt}'")
                return hit, kind

        # выбираем кандидатов с пройденным health_check
        candidates = [
            m for m in select_candidates(prompt, kind) if m.health_check()
        ]

        # параллельная генерация первых top_k моделей
        urls: Dict[str, Optional[str]] = {}
        futures = {self.executor.submit(m.generate, prompt): m for m in candidates[:self.top_k]}

        # для видео — берём первый успешно завершившийся
        if kind == "video":
            done, _ = wait(futures, timeout=self.timeout, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    url = fut.result()
                    urls[futures[fut].name] = url
                    break
                except Exception as e:
                    logger.warning(f"{futures[fut].name} failed: {e}")
        else:
            # для изображений — ждём всех завершений или таймаут
            for fut in as_completed(futures, timeout=self.timeout):
                name = futures[fut].name
                try:
                    urls[name] = fut.result()
                except Exception as e:
                    logger.warning(f"{name} failed: {e}")

        # выбор финального URL
        if kind == "image":
            url = self._select_best_image(urls, prompt)
        else:
            url = next(iter(urls.values()), None) or self._fallback(prompt, kind, candidates)

        if not url:
            url = self._fallback(prompt, kind, candidates)

        # кэш и лог
        if self.cache:
            self.cache.set(prompt, url)
        elapsed = time.time() - start_ts
        logger.info(f"Generated {kind} in {elapsed:.2f}s: {url}")
        return url, kind

    def _select_best_image(self, results: Dict[str, Optional[str]], prompt: str) -> Optional[str]:
        scores = {}
        for name, path in results.items():
            if path:
                try:
                    scores[name] = self.scorer.score(path, prompt)
                except Exception as e:
                    logger.warning(f"scoring {name} failed: {e}")
        if scores:
            best = max(scores, key=scores.get)
            return results[best]
        return None

    def _fallback(
        self,
        prompt: str,
        kind: str,
        candidates: List[ModelClient]
    ) -> str:
        logger.debug("fallback sequence start")
        for model in candidates[self.top_k:]:
            try:
                return model.generate(prompt)
            except Exception as e:
                logger.warning(f"fallback model {model.name} failed: {e}")
        raise RuntimeError("All models failed")