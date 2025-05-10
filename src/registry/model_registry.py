from typing import List
from src.adapters.api_adapter import APIModelA, APIModelB, APIVideoC
from src.adapters.local_adapter import LocalImageC, LocalVideoD
from src.models.base import ModelClient

ALL_MODELS: List[ModelClient] = [
    APIModelA(), APIModelB(), APIVideoC(),
    LocalImageC(), LocalVideoD(),
]

MODEL_TAGS = {
    "ImageA_API":    {"type": "image", "quality": "fast"},
    "ImageB_API":    {"type": "image", "quality": "photo"},
    "ImageC_Local":  {"type": "image", "quality": "artistic"},
    "Video_API":     {"type": "video", "quality": "fast"},
    "VideoD_Local":  {"type": "video", "quality": "anim"},
}

def select_candidates(prompt: str, wanted_type: str) -> List[ModelClient]:
    low = prompt.lower()
    scored = []
    for m in ALL_MODELS:
        tags = MODEL_TAGS[m.name]
        if tags["type"] != wanted_type: 
            continue
        score = 0
        if "фотореалист" in low and tags["quality"] == "photo": score += 2
        if "анимац" in low and tags["quality"] == "anim": score += 2
        if tags["quality"] == "fast": score += 1
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]