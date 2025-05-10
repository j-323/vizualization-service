from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from src.config.settings import Settings
from src.adapters.api_adapter import APIModelA, APIModelB, APIVideoC
from src.adapters.local_adapter import LocalImageC, LocalVideoD

settings = Settings()

llm = OpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.OPENAI_MODEL_NAME,
    temperature=0.0,
)

tools = [
    Tool("ImageA_API",    APIModelA().generate,    "fast статическое изображение через API A"),
    Tool("ImageB_API",    APIModelB().generate,    "фотореалистичное изображение через API B"),
    Tool("ImageC_Local",  LocalImageC().generate,  "художественное изображение локально"),
    Tool("Video_API_C",   APIVideoC().generate,    "короткое MP4 видео через API C"),
    Tool("VideoD_Local",  LocalVideoD().generate,  "анимационное видео локально"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def ask_agent(prompt: str) -> str:
    return agent.run(prompt)