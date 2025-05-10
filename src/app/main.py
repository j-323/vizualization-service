from fastapi import FastAPI
from src.app.routes.generation import router as gen_router

app = FastAPI(title="Visual Agent API")
app.include_router(gen_router, prefix="/generate")