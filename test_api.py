#!/usr/bin/env python3
"""Simple test to verify OpenAI API key works with AutoGen."""

import asyncio
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

async def test_api():
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
        
        # Simple test using the newer API
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        
        test_message = TextMessage(content="Say 'API test successful' and nothing else.", source="user")
        cancellation_token = CancellationToken()
        
        response = await agent.on_messages([test_message], cancellation_token)
        
        print(f"✅ API test successful!")
        print(f"   Agent response: {response}")
        print(f"   Response type: {type(response)}")
        
        # Try to extract content
        if hasattr(response, 'content'):
            print(f"   Content: {response.content}")
        elif hasattr(response, 'chat_message'):
            print(f"   Chat message: {response.chat_message}")
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