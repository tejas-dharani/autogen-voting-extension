"""
Basic example of using the AutoGen Voting Extension.

This demonstrates how to use the voting extension as a standalone package.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_voting import VotingGroupChat, VotingMethod


async def main():
    """Example of code review voting with qualified majority."""
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Create reviewers with different expertise
    senior_dev = AssistantAgent(
        "SeniorDev", model_client, system_message="Senior developer focused on architecture and best practices."
    )
    security_expert = AssistantAgent(
        "SecurityExpert", model_client, system_message="Security specialist reviewing for vulnerabilities."
    )
    performance_engineer = AssistantAgent(
        "PerformanceEngineer",
        model_client,
        system_message="Performance engineer optimizing for speed and efficiency.",
    )

    # Create voting team for code review
    voting_team = VotingGroupChat(
        participants=[senior_dev, security_expert, performance_engineer],
        voting_method=VotingMethod.QUALIFIED_MAJORITY,
        qualified_majority_threshold=0.67,
        require_reasoning=True,
        max_discussion_rounds=2,
        termination_condition=MaxMessageTermination(20),
    )

    # Review code changes
    task = "Proposal: Approve code change for merge with caching implementation"

    result = await voting_team.run(task=task)
    print(result)


async def architecture_decision_example():
    """Example of architecture decision with unanimous consensus."""
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Create architecture team
    tech_lead = AssistantAgent(
        "TechLead", model_client, system_message="Technical lead with expertise in distributed systems."
    )
    solution_architect = AssistantAgent(
        "SolutionArchitect", model_client, system_message="Solution architect focused on enterprise patterns."
    )
    devops_engineer = AssistantAgent(
        "DevOpsEngineer", model_client, system_message="DevOps engineer focused on deployment and operations."
    )

    # Create voting team requiring unanimous consensus
    voting_team = VotingGroupChat(
        participants=[tech_lead, solution_architect, devops_engineer],
        voting_method=VotingMethod.UNANIMOUS,
        max_discussion_rounds=3,
        auto_propose_speaker="TechLead",
        termination_condition=MaxMessageTermination(30),
    )

    task = "Proposal: Choose microservices communication pattern from available options"

    result = await voting_team.run(task=task)
    print(result)


async def content_moderation_example():
    """Example of content moderation with simple majority."""
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Create moderation team
    community_manager = AssistantAgent(
        "CommunityManager", model_client, system_message="Community manager maintaining positive environment."
    )
    safety_specialist = AssistantAgent(
        "SafetySpecialist", model_client, system_message="Safety specialist focused on harmful content detection."
    )
    legal_advisor = AssistantAgent(
        "LegalAdvisor", model_client, system_message="Legal advisor focused on compliance and risk."
    )

    # Create voting team for content moderation
    voting_team = VotingGroupChat(
        participants=[community_manager, safety_specialist, legal_advisor],
        voting_method=VotingMethod.MAJORITY,
        allow_abstentions=True,
        max_discussion_rounds=1,
        termination_condition=MaxMessageTermination(15),
    )

    task = "Proposal: Moderate user forum post about platform feedback"

    result = await voting_team.run(task=task)
    print(result)


"""A group chat team that enables democratic consensus through configurable voting mechanisms.

    Perfect for code reviews, architecture decisions, content moderation, and any scenario
    requiring group consensus with transparent decision-making processes.

    Args:
        participants (List[ChatAgent]): The agents participating in the voting process.
        voting_method (VotingMethod, optional): Method used for determining consensus. Defaults to VotingMethod.MAJORITY.
        qualified_majority_threshold (float, optional): Threshold for qualified majority voting (0.5-1.0). Defaults to 0.67.
        allow_abstentions (bool, optional): Whether agents can abstain from voting. Defaults to True.
        require_reasoning (bool, optional): Whether votes must include reasoning. Defaults to False.
        max_discussion_rounds (int, optional): Maximum rounds of discussion before final decision. Defaults to 3.
        auto_propose_speaker (str, optional): Agent name to automatically select as proposer. Defaults to None.
        termination_condition (TerminationCondition, optional): Condition for ending the chat. Defaults to None.
        max_turns (int, optional): Maximum number of turns before forcing termination. Defaults to None.
        runtime (AgentRuntime, optional): The agent runtime to use. Defaults to None.
        custom_message_types (List[type[BaseAgentEvent | BaseChatMessage]], optional): Additional message types for the chat. Defaults to None.
        emit_team_events (bool, optional): Whether to emit team events for UI integration. Defaults to False.

    Raises:
        ValueError: If fewer than 2 participants, invalid thresholds, or missing auto_propose_speaker.

    Examples:

    Code review voting with qualified majority:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import VotingGroupChat, VotingMethod
            from autogen_agentchat.conditions import MaxMessageTermination


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # Create reviewers with different expertise
                senior_dev = AssistantAgent(
                    "SeniorDev", model_client, system_message="Senior developer focused on architecture and best practices."
                )
                security_expert = AssistantAgent(
                    "SecurityExpert", model_client, system_message="Security specialist reviewing for vulnerabilities."
                )
                performance_engineer = AssistantAgent(
                    "PerformanceEngineer",
                    model_client,
                    system_message="Performance engineer optimizing for speed and efficiency.",
                )

                # Create voting team for code review
                voting_team = VotingGroupChat(
                    participants=[senior_dev, security_expert, performance_engineer],
                    voting_method=VotingMethod.QUALIFIED_MAJORITY,
                    qualified_majority_threshold=0.67,
                    require_reasoning=True,
                    max_discussion_rounds=2,
                    termination_condition=MaxMessageTermination(20),
                )

                # Review code changes
                task = "Proposal: Approve code change for merge with caching implementation"

                result = await voting_team.run(task=task)
                print(result)


            asyncio.run(main())

    Architecture decision with unanimous consensus:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import VotingGroupChat, VotingMethod
            from autogen_agentchat.conditions import MaxMessageTermination


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # Create architecture team
                tech_lead = AssistantAgent(
                    "TechLead", model_client, system_message="Technical lead with expertise in distributed systems."
                )
                solution_architect = AssistantAgent(
                    "SolutionArchitect", model_client, system_message="Solution architect focused on enterprise patterns."
                )
                devops_engineer = AssistantAgent(
                    "DevOpsEngineer", model_client, system_message="DevOps engineer focused on deployment and operations."
                )

                # Create voting team requiring unanimous consensus
                voting_team = VotingGroupChat(
                    participants=[tech_lead, solution_architect, devops_engineer],
                    voting_method=VotingMethod.UNANIMOUS,
                    max_discussion_rounds=3,
                    auto_propose_speaker="TechLead",
                    termination_condition=MaxMessageTermination(30),
                )

                task = "Proposal: Choose microservices communication pattern from available options"

                result = await voting_team.run(task=task)
                print(result)


            asyncio.run(main())

    Content moderation with simple majority:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import VotingGroupChat, VotingMethod
            from autogen_agentchat.conditions import MaxMessageTermination


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # Create moderation team
                community_manager = AssistantAgent(
                    "CommunityManager", model_client, system_message="Community manager maintaining positive environment."
                )
                safety_specialist = AssistantAgent(
                    "SafetySpecialist", model_client, system_message="Safety specialist focused on harmful content detection."
                )
                legal_advisor = AssistantAgent(
                    "LegalAdvisor", model_client, system_message="Legal advisor focused on compliance and risk."
                )

                # Create voting team for content moderation
                voting_team = VotingGroupChat(
                    participants=[community_manager, safety_specialist, legal_advisor],
                    voting_method=VotingMethod.MAJORITY,
                    allow_abstentions=True,
                    max_discussion_rounds=1,
                    termination_condition=MaxMessageTermination(15),
                )

                task = "Proposal: Moderate user forum post about platform feedback"

                result = await voting_team.run(task=task)
                print(result)


            asyncio.run(main())
    """

if __name__ == "__main__":
    import asyncio
    
    print("Running AutoGen Voting Extension examples...")
    print("Make sure you have set your OpenAI API key in the environment.")
    print("Install dependencies with: pip install autogen-agentchat autogen-ext")
    print("Then install this extension with: pip install -e .")
    print()

    # Run all examples
    print("=== Code Review Example ===")
    asyncio.run(main())
    
    print("\n=== Architecture Decision Example ===")
    asyncio.run(architecture_decision_example())
    
    print("\n=== Content Moderation Example ===")
    asyncio.run(content_moderation_example())
