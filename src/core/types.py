from pydantic import BaseModel, HttpUrl

class PromptRequest(BaseModel):
    prompt: str

class GenerationResult(BaseModel):
    url: HttpUrl
    type: str  # "image" or "video"