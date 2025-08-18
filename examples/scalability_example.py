"""Example demonstrating scalability testing with varying agent counts."""

import asyncio
import os

# Add the src directory to the path
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import ChatAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from votingai import VotingGroupChat, VotingMethod
from src.votingai.config import MODEL


def create_scalable_personas(agent_count: int, domain: str = "code_review") -> list[dict[str, str]]:
    """Create persona configurations for different agent counts."""

    base_personas = {
        "code_review": [
            {"name": "SeniorDev", "role": "Senior developer focused on architecture and best practices"},
            {"name": "SecurityExpert", "role": "Security specialist reviewing for vulnerabilities"},
            {"name": "PerformanceEngineer", "role": "Performance engineer optimizing for speed and efficiency"},
            {"name": "CodeQualityReviewer", "role": "Code quality reviewer focusing on maintainability"},
            {"name": "UIUXSpecialist", "role": "UI/UX specialist reviewing user-facing changes"},
            {"name": "DatabaseExpert", "role": "Database specialist reviewing data access patterns"},
            {"name": "DevOpsEngineer", "role": "DevOps engineer reviewing deployment and operations"},
            {"name": "AccessibilityExpert", "role": "Accessibility expert ensuring inclusive design"},
            {"name": "TechnicalWriter", "role": "Technical writer reviewing documentation and clarity"},
            {"name": "TestingSpecialist", "role": "Testing specialist focusing on test coverage and quality"},
        ],
        "architecture": [
            {"name": "SolutionArchitect", "role": "Solution architect with enterprise experience"},
            {"name": "TechLead", "role": "Technical lead familiar with current system"},
            {"name": "DevOpsEngineer", "role": "DevOps engineer considering operational complexity"},
            {"name": "DataArchitect", "role": "Data architect specializing in data systems"},
            {"name": "SecurityArchitect", "role": "Security architect focused on secure design"},
            {"name": "CloudArchitect", "role": "Cloud architect specializing in scalable infrastructure"},
            {"name": "IntegrationSpecialist", "role": "Integration specialist focused on system connections"},
            {"name": "PerformanceArchitect", "role": "Performance architect optimizing for scale"},
            {"name": "CostOptimizer", "role": "Cost optimization specialist managing technical debt"},
            {"name": "ComplianceOfficer", "role": "Compliance officer ensuring regulatory requirements"},
        ],
    }

    personas = base_personas.get(domain, base_personas["code_review"])
    return personas[:agent_count]


async def test_voting_scalability(
    agent_counts: list[int], voting_method: VotingMethod = VotingMethod.MAJORITY, domain: str = "code_review"
) -> dict[int, dict[str, Any]]:
    """Test voting performance with different numbers of agents."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return {}

    model_client = OpenAIChatCompletionClient(model=MODEL)

    # Code review task
    task = """
    Review this code change for merge approval:

    ```python
    def process_payments(payments):
        results = []
        for payment in payments:
            if payment.amount > 1000:
                # High-value payment processing
                result = external_api.process_large_payment(payment)
                if result.success:
                    db.update_payment_status(payment.id, 'completed')
                    send_notification(payment.user_id, 'payment_success')
                else:
                    db.update_payment_status(payment.id, 'failed')
                    send_notification(payment.user_id, 'payment_failed')
            else:
                # Standard payment processing
                result = internal_processor.process(payment)
                db.update_payment_status(payment.id, result.status)
            results.append(result)
        return results
    ```

    Decision: Should this code be approved for merge? Consider security, performance, and maintainability.
    """

    results: dict[int, dict[str, Any]] = {}

    for count in agent_counts:
        print(f"\n=== Testing with {count} agents ===")

        # Create agents
        personas = create_scalable_personas(count, domain)
        agents: list[ChatAgent] = []

        for persona in personas:
            agent = AssistantAgent(name=persona["name"], model_client=model_client, system_message=persona["role"])
            agents.append(agent)

        # Create voting team
        voting_team = VotingGroupChat(
            participants=agents,
            voting_method=voting_method,
            require_reasoning=True,
            max_discussion_rounds=2,
            termination_condition=MaxMessageTermination(50),
        )

        # Measure performance
        start_time = asyncio.get_event_loop().time()

        try:
            await voting_team.run(task=task)
            end_time = asyncio.get_event_loop().time()

            duration = end_time - start_time
            results[count] = {"duration": duration, "success": True, "agents": count}

            print(f"  Completed in {duration:.2f} seconds")
            print("  Result available: True")

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            results[count] = {"duration": duration, "success": False, "error": str(e), "agents": count}

            print(f"  Failed after {duration:.2f} seconds: {e}")

    return results


async def compare_voting_methods_by_scale() -> None:
    """Compare how different voting methods perform at scale."""

    print("=== Voting Method Scalability Comparison ===")

    agent_counts = [3, 5, 7]
    voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]

    all_results: dict[str, dict[int, dict[str, Any]]] = {}

    for method in voting_methods:
        print(f"\n--- Testing {method.value} ---")
        results = await test_voting_scalability(agent_counts, method)
        all_results[method.value] = results

    # Print comparison
    print("\n=== Scalability Comparison Results ===")
    print(f"{'Method':<20} {'3 Agents':<12} {'5 Agents':<12} {'7 Agents':<12}")
    print("-" * 60)

    for method_name, results in all_results.items():
        durations: list[str] = []
        for count in agent_counts:
            if count in results and results[count]["success"]:
                durations.append(f"{results[count]['duration']:.1f}s")
            else:
                durations.append("FAILED")

        print(f"{method_name:<20} {durations[0]:<12} {durations[1]:<12} {durations[2]:<12}")


async def test_consensus_difficulty() -> None:
    """Test how different scenarios affect consensus difficulty."""

    print("=== Consensus Difficulty Testing ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    model_client = OpenAIChatCompletionClient(model=MODEL)

    # Create diverse agent perspectives
    personas = [
        {"name": "SecurityFirst", "role": "Security-first developer who prioritizes safety over speed"},
        {"name": "SpeedOptimizer", "role": "Performance engineer who prioritizes speed over security"},
        {"name": "BusinessFocused", "role": "Business-focused developer who prioritizes features over technical debt"},
        {"name": "QualityGuardian", "role": "Quality-focused developer who prioritizes maintainability"},
        {"name": "PragmaticDev", "role": "Pragmatic developer who balances all concerns equally"},
    ]

    agents: list[ChatAgent] = []
    for persona in personas:
        agent = AssistantAgent(name=persona["name"], model_client=model_client, system_message=persona["role"])
        agents.append(agent)

    # Test scenarios with different expected difficulty
    scenarios = [
        {
            "name": "Easy Consensus - Clear Bug Fix",
            "task": "Should we fix this obvious null pointer exception? Fix: add null check before accessing object.property",
            "expected_difficulty": "easy",
        },
        {
            "name": "Medium Consensus - Performance vs Security Trade-off",
            "task": "Should we cache user authentication tokens in memory (faster) or validate each request (more secure)?",
            "expected_difficulty": "medium",
        },
        {
            "name": "Hard Consensus - Architecture Decision",
            "task": "Should we rewrite our monolith as microservices? Consider: team size (5 people), complexity, deployment overhead, debugging difficulty",
            "expected_difficulty": "hard",
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Expected difficulty: {scenario['expected_difficulty']}")

        # Test with different voting methods
        for method in [VotingMethod.MAJORITY, VotingMethod.UNANIMOUS]:
            voting_team = VotingGroupChat(
                participants=agents,
                voting_method=method,
                require_reasoning=True,
                max_discussion_rounds=3,
                termination_condition=MaxMessageTermination(40),
            )

            start_time = asyncio.get_event_loop().time()

            try:
                await voting_team.run(task=scenario["task"])
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time

                print(f"  {method.value}: {duration:.1f}s (SUCCESS)")

            except Exception as e:
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                print(f"  {method.value}: {duration:.1f}s (FAILED - {str(e)[:50]}...)")


def main() -> None:
    """Run scalability examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Run scalability examples")
    parser.add_argument(
        "--test",
        choices=["basic", "methods", "consensus", "all"],
        default="basic",
        help="Which scalability test to run",
    )

    args = parser.parse_args()

    if args.test == "basic" or args.test == "all":
        asyncio.run(test_voting_scalability([3, 5, 7]))

    if args.test == "methods" or args.test == "all":
        asyncio.run(compare_voting_methods_by_scale())

    if args.test == "consensus" or args.test == "all":
        asyncio.run(test_consensus_difficulty())


if __name__ == "__main__":
    main()
