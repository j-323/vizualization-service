import concurrent.futures
from src.registry.model_registry import select_candidates
from src.models.evaluator import CLIPScorer

class GenerationPipeline:
    def __init__(self):
        self.scorer = CLIPScorer()
        self.timeout = 60

    def generate(self, prompt: str) -> tuple[str,str]:
        typ = "video" if "mp4" in prompt.lower() or "анимац" in prompt.lower() else "image"
        candidates = select_candidates(prompt, typ)

        urls = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
            futures = {exe.submit(m.generate, prompt): m for m in candidates[:2]}
            for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                m = futures[fut]
                try:
                    urls[m.name] = fut.result()
                except Exception:
                    urls[m.name] = None

        if typ == "image":
            scores = {n: self.scorer.score(u, prompt) for n,u in urls.items() if u}
            best = max(scores, key=scores.get)
            return urls[best], "image"

        for url in urls.values():
            if url:
                return url, "video"

        for m in candidates[2:]:
            try:
                return m.generate(prompt), typ
            except Exception:
                continue

        raise RuntimeError("Все модели недоступны или вернули ошибку")

# DI-экземпляр
pipeline = GenerationPipeline()