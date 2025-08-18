# AutoGen Voting Extension Examples

This directory contains comprehensive examples demonstrating various use cases for the AutoGen Voting Extension.

## Prerequisites

Before running these examples, make sure you have:

1. **Installed the extension**:
   ```bash
   pip install autogen-agentchat autogen-ext[openai]
   pip install votingai
   ```

2. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Examples Overview

### 1. [basic_example.py](basic_example.py)
**Basic voting setup** - Simple majority voting with three agents making a technology decision.

**Key Features:**
- Majority voting method
- Reasoning requirements
- Discussion rounds
- Basic termination conditions

**Run with:**
```bash
python examples/basic_example.py
```

### 2. [code_review_example.py](code_review_example.py)
**Code review workflow** - Qualified majority voting for PR approval with specialized reviewer agents.

**Key Features:**
- Qualified majority (2/3 approval)
- Specialized agent roles
- Code review scenarios
- Auto-proposer selection

### 3. [architecture_decision_example.py](architecture_decision_example.py)
**Architecture decisions** - Ranked choice voting for complex technical decisions.

**Key Features:**
- Ranked choice voting
- Multiple architecture options
- Extended discussion rounds
- Unanimous fallback

### 4. [content_moderation_example.py](content_moderation_example.py)
**Content moderation** - Simple majority with abstention support for content approval.

**Key Features:**
- Majority voting with abstentions
- Content safety scenarios
- Quick decision cycles
- Legal compliance considerations

### 5. [feature_prioritization_example.py](feature_prioritization_example.py)
**Feature prioritization** - Unanimous consensus for high-stakes feature decisions.

**Key Features:**
- Unanimous voting requirement
- Product management scenarios
- Stakeholder alignment
- Extended discussion support

## Running Examples

### Individual Examples
```bash
# Run specific example
python examples/basic_example.py

# Run with different voting methods
python examples/architecture_decision_example.py
```

### All Examples
```bash
# Run all examples (requires API key)
python -m examples.run_all_examples
```

## Configuration Options

Each example demonstrates different configuration options:

### Voting Methods
- `VotingMethod.MAJORITY` - >50% approval required
- `VotingMethod.PLURALITY` - Most votes wins
- `VotingMethod.UNANIMOUS` - All voters must agree
- `VotingMethod.QUALIFIED_MAJORITY` - Configurable threshold
- `VotingMethod.RANKED_CHOICE` - Ranked preferences

### Advanced Settings
- `qualified_majority_threshold` - Set custom thresholds (0.5-1.0)
- `allow_abstentions` - Enable/disable abstention votes
- `require_reasoning` - Mandate reasoning for votes
- `max_discussion_rounds` - Limit discussion cycles
- `auto_propose_speaker` - Auto-select proposer
- `emit_team_events` - Enable event streaming

## Customization

### Creating Your Own Examples

1. **Copy a base example** that matches your use case
2. **Modify agent roles** and system messages
3. **Adjust voting configuration** for your requirements
4. **Update the proposal/task** content
5. **Add custom termination conditions** if needed

### Example Template
```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination

from votingai import VotingGroupChat, VotingMethod

async def my_voting_example():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialized agents
    agents = [
        AssistantAgent("Agent1", model_client, system_message="Your role here"),
        AssistantAgent("Agent2", model_client, system_message="Your role here"),
        # Add more agents as needed
    ]
    
    # Configure voting team
    voting_team = VotingGroupChat(
        participants=agents,
        voting_method=VotingMethod.MAJORITY,  # Choose appropriate method
        # Add other configuration options
        termination_condition=MaxMessageTermination(20)
    )
    
    # Define your proposal
    task = """
    Your proposal description here...
    
    Please vote APPROVE or REJECT with detailed reasoning.
    """
    
    # Run the voting process
    result = await voting_team.run(task=task)
    print(f"Decision: {result}")

if __name__ == "__main__":
    asyncio.run(my_voting_example())
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Import Errors**
   ```bash
   pip install autogen-agentchat autogen-ext[openai] votingai
   ```

3. **Timeout Issues**
   - Increase `MaxMessageTermination` limit
   - Reduce `max_discussion_rounds`
   - Simplify the proposal/task

4. **Voting Not Converging**
   - Check voting method appropriateness
   - Review agent system messages
   - Consider allowing abstentions

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable detailed logging
logging.getLogger("autogen_agentchat").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)
```

## Contributing Examples

Have a great use case example? We'd love to include it!

1. Create your example following the template above
2. Add documentation explaining the use case
3. Test thoroughly with different scenarios
4. Submit a pull request with your example

---

For more information, see the main [README.md](../README.md) and [documentation](https://votingai.readthedocs.io/).