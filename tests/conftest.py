"""Pytest configuration and fixtures for autogen-voting tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.votingai.config import MODEL


@pytest.fixture
def mock_model_client() -> MagicMock:
    """Mock OpenAI model client for testing."""
    mock_client = MagicMock(spec=OpenAIChatCompletionClient)
    mock_client.model = MODEL
    return mock_client


@pytest.fixture
def sample_agents(mock_model_client: MagicMock) -> list[AssistantAgent]:
    """Create sample agents for testing."""
    return [
        AssistantAgent("Agent1", mock_model_client, system_message="First agent for testing"),
        AssistantAgent("Agent2", mock_model_client, system_message="Second agent for testing"),
        AssistantAgent("Agent3", mock_model_client, system_message="Third agent for testing"),
    ]


@pytest.fixture
def mock_async_response() -> AsyncMock:
    """Mock async response for agent communication."""
    mock_response = AsyncMock()
    mock_response.response = MagicMock()
    return mock_response
