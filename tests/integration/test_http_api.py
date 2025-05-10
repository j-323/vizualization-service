from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_generate_sync(monkeypatch):
    monkeypatch.setattr("src.pipelines.generation_pipeline.pipeline.generate", lambda p: ("http://x.png","image"))
    resp = client.post("/generate/", json={"prompt":"x"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "image"