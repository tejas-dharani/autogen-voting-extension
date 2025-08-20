"""Pytest configuration and fixtures for voting ai tests."""

import os
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.votingai.config import MODEL


def requires_openai_api_key(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to skip tests that require OpenAI API key if not available."""
    api_key = os.getenv("OPENAI_API_KEY")
    # Skip if no key, empty key, or if it looks like a dummy/invalid key
    should_skip = (
        not api_key or api_key.strip() == "" or "sk-proj-" in api_key and "5zMA" in api_key  # Skip known dummy keys
    )
    return pytest.mark.skipif(  # type: ignore[no-any-return]
        should_skip,
        reason="Valid OpenAI API key not provided. Set OPENAI_API_KEY environment variable to run this test.",
    )(func)


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
