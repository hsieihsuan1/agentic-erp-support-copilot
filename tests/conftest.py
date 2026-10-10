import pytest

from erp_copilot.agent import Agent, approve_all
from erp_copilot.llm import MockLLM


@pytest.fixture
def agent() -> Agent:
    return Agent(MockLLM(), customer_id="C-1001", approver=approve_all)
