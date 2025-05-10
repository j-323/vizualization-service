from src.prompts.agent_prompt import build_agent_prompt
from src.services.agent_service import agent 

def ask_agent(user_request: str) -> str:
    prompt = build_agent_prompt(user_request)
    return agent.run(prompt)