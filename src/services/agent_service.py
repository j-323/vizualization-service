import logging
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from src.config.settings import Settings
from src.adapters.api_adapter import APIModelA, APIModelB, APIVideoC
from src.adapters.local_adapter import LocalImageC, LocalVideoD
from src.prompts.agent_prompt import build_agent_prompt
from src.pipelines.generation_pipeline import pipeline

settings = Settings()
logger = logging.getLogger(__name__)

# LLM-инструмент
llm = OpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0.0,
)

tools = [
    Tool("ImageA_API", APIModelA().generate, "быстрая статическая"),
    Tool("ImageB_API", APIModelB().generate, "фотореалистичная"),
    Tool("ImageC_Local", LocalImageC().generate, "художественная"),
    Tool("Video_API", APIVideoC().generate, "короткое MP4"),
    Tool("VideoD_Local", LocalVideoD().generate, "анимационное MP4"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

def ask_agent(user_request: str) -> str:
    prompt = build_agent_prompt(user_request)
    try:
        return agent.run(prompt)
    except Exception as e:
        logger.error(f"agent.run failed: {e}", exc_info=True)
        # fallback на прямой pipeline
        url, kind = pipeline.generate(user_request)
        return url