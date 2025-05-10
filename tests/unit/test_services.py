import pytest
from src.services.agent_service import ask_agent, agent

def test_agent_stub(monkeypatch):
    monkeypatch.setattr(agent, "run", lambda p: "RESULT")
    assert ask_agent("любой") == "RESULT"