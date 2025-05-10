import pytest
from src.adapters import api_adapter, local_adapter

class DummyResponse:
    def __init__(self, url): self.url=url
    def raise_for_status(self): pass
    def json(self): return {"url": self.url}

@pytest.fixture(autouse=True)
def patch_requests(monkeypatch):
    class M:
        @staticmethod
        def post(u, json, headers, timeout):
            return DummyResponse("http://example.com/test.png")
    monkeypatch.setattr(api_adapter, "httpx", M)

def test_generate_image_a():
    url = api_adapter.APIModelA().generate("привет")
    assert url.endswith(".png")

def test_generate_image_b():
    url = api_adapter.APIModelB().generate("привет")
    assert "http" in url

def test_generate_video_api():
    url = api_adapter.APIVideoC().generate("привет")
    assert url.endswith(".mp4")

def test_local_adapters(tmp_path):
    p1 = local_adapter.LocalImageC().generate("x")
    assert p1.endswith(".png")
    p2 = local_adapter.LocalVideoD().generate("x")
    assert p2.endswith(".mp4")