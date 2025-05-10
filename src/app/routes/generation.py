from fastapi import APIRouter, Depends, HTTPException
from src.core.types import PromptRequest, GenerationResult
from src.app.dependencies import get_pipeline, get_celery
from celery.result import AsyncResult

router = APIRouter()

@router.post("/", response_model=GenerationResult)
def generate_sync(
    req: PromptRequest,
    pipeline=Depends(get_pipeline),
):
    try:
        url, typ = pipeline.generate(req.prompt)
        return GenerationResult(url=url, type=typ)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/async", response_model=dict)
def generate_async(req: PromptRequest, celery_app=Depends(get_celery)):
    task = celery_app.send_task("src.services.async_worker.generate_task", args=[req.prompt])
    return {"task_id": task.id}

@router.get("/status/{task_id}", response_model=GenerationResult)
def get_status(task_id: str, celery_app=Depends(get_celery)):
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        raise HTTPException(202, "Pending")
    elif res.state == "FAILURE":
        raise HTTPException(500, str(res.result))
    return GenerationResult(**res.result)