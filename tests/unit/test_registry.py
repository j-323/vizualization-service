import pytest
from src.registry.model_registry import select_candidates

def test_select_photo():
    m = select_candidates("Фотореалистичный закат", "image")[0]
    assert m.name == "ImageB_API"

def test_select_animation():
    m = select_candidates("Сделай анимацию mp4", "video")[0]
    assert m.name in ("Video_API","VideoD_Local")

def test_default_fallback():
    m = select_candidates("любой запрос", "image")[0]
    assert m.name == "ImageA_API"