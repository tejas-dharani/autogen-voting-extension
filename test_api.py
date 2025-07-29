#!/usr/bin/env python3
"""Simple test to verify OpenAI API key works with AutoGen."""

import asyncio
import os
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def test_api() -> bool:
    """Test OpenAI API integration."""

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        return False

    try:
        print("🔧 Testing OpenAI API connection...")

        # Create model client
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

        # Create simple agent
        agent = AssistantAgent(
            name="TestAgent",
            model_client=model_client,
            system_message="You are a helpful assistant. Keep responses very short."
        )

        print("🚀 Running simple agent test...")

        test_message = TextMessage(content="Say 'API test successful' and nothing else.", source="user")
        cancellation_token = CancellationToken()

        response: Any = await agent.on_messages([test_message], cancellation_token)

        print("✅ API test successful!")
        print(f"   Agent response: {response}")
        print(f"   Response type: {type(response)}")

        # Safely extract content using getattr with fallback
        content = getattr(response, 'content', None)
        if content is not None:
            print(f"   Content: {content}")
        else:
            chat_message = getattr(response, 'chat_message', None)
            if chat_message is not None:
                print(f"   Chat message: {chat_message}")
            else:
                print(f"   Raw response: {response}")

        return True

    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api())
    if success:
        print("\n🎉 Your OpenAI API key is working correctly!")
        print("   You can now run the full benchmark suite:")
        print("   python run_benchmarks.py --code-review")
    else:
        print("\n💡 Please check your OpenAI API key setup")
