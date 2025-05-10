import pytest
from src.pipelines.generation_pipeline import pipeline

class Dummy:
    def __init__(self): self.name="Dummy"; self.type="image"
    def generate(self, prompt): return "/tmp/x.png"

def test_pipeline_returns_tuple(monkeypatch):
    monkeypatch.setattr("src.registry.model_registry.select_candidates", lambda p,t: [Dummy()])
    url, t = pipeline.generate("x")
    assert url.endswith(".png")
    assert t == "image"