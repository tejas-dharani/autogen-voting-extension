"""Benchmark runner for comparing VotingGroupChat vs standard GroupChat."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_voting import VotingGroupChat, VotingMethod

from .metrics import BenchmarkMetrics, ComparisonResults, MetricsCollector
from .scenarios import BenchmarkScenario, ScenarioType, get_all_scenarios


class BenchmarkRunner:
    """Runs comparative benchmarks between voting and standard group chats."""

    def __init__(self, model_name: str = "gpt-4o-mini", results_dir: str = "benchmark_results"):
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.metrics_collector = MetricsCollector()

    def create_model_client(self) -> OpenAIChatCompletionClient:
        """Create a model client for agents."""
        return OpenAIChatCompletionClient(model=self.model_name)

    def create_agents(self, personas: list[dict[str, str]]) -> list[AssistantAgent]:
        """Create agents based on persona specifications."""
        model_client = self.create_model_client()
        agents: list[AssistantAgent] = []

        for persona in personas:
            agent = AssistantAgent(name=persona["name"], model_client=model_client, system_message=persona["role"])
            agents.append(agent)

        return agents

    async def run_voting_scenario(
        self,
        scenario: BenchmarkScenario,
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        qualified_majority_threshold: float = 0.67,
        max_discussion_rounds: int = 2,
        require_reasoning: bool = True,
    ) -> BenchmarkMetrics:
        """Run a scenario using VotingGroupChat."""
        print(f"debug: Starting voting scenario '{scenario.name}'")
        print(f"debug: Scenario type: {scenario.scenario_type.value}")

        agents = self.create_agents(scenario.agent_personas)
        print(f"debug: Created {len(agents)} agents: {[a.name for a in agents]}")
        print(f"debug: Agent personas: {[p['role'][:50] + '...' for p in scenario.agent_personas]}")

        metrics = self.metrics_collector.start_collection()
        print("debug: Started metrics collection")

        # Configure voting team
        voting_team = VotingGroupChat(
            participants=agents,  # type: ignore
            voting_method=voting_method,
            qualified_majority_threshold=qualified_majority_threshold,
            require_reasoning=require_reasoning,
            max_discussion_rounds=max_discussion_rounds,
            termination_condition=MaxMessageTermination(30),
        )
        print("debug: Configured voting team:")
        print(f"debug:   Method: {voting_method.value}")
        print(f"debug:   Threshold: {qualified_majority_threshold}")
        print(f"debug:   Max rounds: {max_discussion_rounds}")
        print(f"debug:   Require reasoning: {require_reasoning}")
        print("debug:   Max turns: 30")

        # Set metrics collector on the voting team
        voting_team.set_metrics_collector(self.metrics_collector)
        print("debug: Set metrics collector on voting team")

        try:
            # Run the scenario
            print(f"debug: Task prompt: {scenario.task_prompt[:200]}...")
            print("debug: Running voting team...")
            result = await voting_team.run(task=scenario.task_prompt)
            print(f"debug: Scenario completed with result type: {type(result)}")
            print(f"debug: Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

            # Extract voting results from the team's manager
            print("debug: Extracting voting results from team manager...")
            try:
                # Access the voting manager through the runtime's instantiated agents
                runtime = voting_team._runtime  # type: ignore[attr-defined]
                manager_name = voting_team._group_chat_manager_name  # type: ignore[attr-defined]
                print(f"debug: Looking for manager '{manager_name}' in runtime")

                # Check all instantiated agents
                instantiated_agents = getattr(runtime, "_instantiated_agents", {})
                agent_keys = list(instantiated_agents.keys()) if instantiated_agents else []
                print(f"debug: Available agents ({len(agent_keys)}): {agent_keys}")

                # Try to find the manager in instantiated agents
                manager: Any = None
                for agent_id, agent_instance in instantiated_agents.items():
                    agent_type_name = getattr(type(agent_instance), "__name__", "Unknown")  # type: ignore[arg-type]
                    print(f"debug: Checking agent: {agent_id} -> {agent_type_name}")
                    if manager_name in str(agent_id) or "VotingGroupChatManager" in agent_type_name:
                        manager = agent_instance
                        manager_type_name = getattr(type(manager), "__name__", "Unknown")  # type: ignore[arg-type]
                        print(f"debug: Found manager: {manager_type_name}")
                        break

                if manager and hasattr(manager, "votes_cast"):
                    votes_cast = getattr(manager, "votes_cast", {})
                    print(f"debug: Found votes cast: {len(votes_cast)} votes")
                    votes_cast_dict = cast(dict[str, Any], votes_cast)
                    for agent, vote_data in votes_cast_dict.items():
                        vote_data_dict = cast(dict[str, Any], vote_data)
                        vote_obj = vote_data_dict["vote"]
                        vote_val = vote_obj.value if hasattr(vote_obj, "value") else str(vote_obj)
                        print(f"debug:   {agent}: {vote_val}")

                    # Votes are already recorded by the manager during the process
                    # Don't double-count here - just verify the metrics match
                    print("debug: Verifying votes in metrics match manager votes...")
                    current_vote_counts: dict[str, int] = {}
                    for _agent_name, vote_data in votes_cast_dict.items():
                        vote_data_dict = cast(dict[str, Any], vote_data)
                        vote_obj = vote_data_dict["vote"]
                        vote_value = vote_obj.value if hasattr(vote_obj, "value") else str(vote_obj)
                        current_vote_counts[vote_value] = current_vote_counts.get(vote_value, 0) + 1
                    print(f"debug: Manager vote counts: {current_vote_counts}")
                    print(f"debug: Metrics vote counts: {metrics.final_vote_counts}")

                    if votes_cast_dict:
                        metrics.decision_reached = True
                        metrics.consensus_type = voting_method.value
                        print(f"debug: Successfully extracted {len(votes_cast_dict)} votes")
                        print(f"debug: Set decision_reached=True, consensus_type={voting_method.value}")
                    else:
                        print("debug: No votes were cast")

                    # Also check current phase and proposal
                    current_phase = getattr(manager, "current_phase", "unknown")
                    current_proposal = getattr(manager, "current_proposal", None)
                    print(f"debug: Final phase: {current_phase}")
                    proposal_id = "none"
                    if isinstance(current_proposal, dict):
                        proposal_dict = cast(dict[str, Any], current_proposal)
                        proposal_id = proposal_dict.get("id", "none")
                    print(f"debug: Proposal ID: {proposal_id}")
                else:
                    print("debug: Could not access voting manager or votes_cast")
                    if manager:
                        print(f"debug: Manager has votes_cast: {hasattr(manager, 'votes_cast')}")
                        manager_attrs = dir(manager)
                        vote_attrs = [attr for attr in manager_attrs if "vote" in attr.lower()]
                        print(f"debug: Manager vote attributes: {vote_attrs}")
                    else:
                        print("debug: Manager not found")

            except Exception as e:
                print(f"debug: Error extracting voting results: {e}")
                import traceback

                traceback.print_exc()

            # Mark completion
            metrics.complete_benchmark()

        except Exception as e:
            print(f"debug: Error in voting scenario: {e}")
            import traceback

            traceback.print_exc()
            metrics.complete_benchmark()

        finally:
            self.metrics_collector.stop_collection()

        return metrics

    async def run_standard_scenario(self, scenario: BenchmarkScenario, max_turns: int = 20) -> BenchmarkMetrics:
        """Run a scenario using standard GroupChat."""
        print(f"debug: Starting standard scenario '{scenario.name}'")

        agents = self.create_agents(scenario.agent_personas)
        print(f"debug: Created {len(agents)} agents")
        metrics = self.metrics_collector.start_collection()

        # Configure standard group chat (using RoundRobinGroupChat as standard comparison)
        standard_team = RoundRobinGroupChat(
            participants=agents,  # type: ignore
            termination_condition=MaxMessageTermination(max_turns),
        )
        print("debug: Configured RoundRobinGroupChat")

        try:
            # Run the scenario
            print("debug: Running standard team...")
            result = await standard_team.run(task=scenario.task_prompt)
            print("debug: Standard scenario completed")

            # Extract messages and estimate metrics from result
            if hasattr(result, "messages") and result.messages:
                print(f"debug: Found {len(result.messages)} messages in result")
                for message in result.messages:
                    source = getattr(message, "source", "unknown")
                    if source != "user":
                        # Estimate tokens
                        content = getattr(message, "content", "")
                        estimated_tokens = len(str(content)) // 4
                        metrics.add_message(source)
                        metrics.add_tokens(estimated_tokens)
                        metrics.add_api_call()
                        print(f"debug: Recorded message from {source}, {estimated_tokens} tokens")

            # Basic completion tracking
            metrics.decision_reached = True
            metrics.consensus_type = "sequential"

            # Mark completion
            metrics.complete_benchmark()
            print(
                f"debug: Final metrics: {metrics.total_messages} messages, {metrics.token_usage} tokens, {metrics.api_calls} API calls"
            )

        except Exception as e:
            print(f"debug: Error in standard scenario: {e}")
            import traceback

            traceback.print_exc()
            metrics.complete_benchmark()

        finally:
            self.metrics_collector.stop_collection()

        return metrics

    async def run_comparison(
        self,
        scenario: BenchmarkScenario,
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        save_results: bool = True,
    ) -> ComparisonResults:
        """Run a complete comparison between voting and standard approaches."""

        print(f"Running comparison for scenario: {scenario.name}")

        # Run voting approach
        print("  Running voting approach...")
        voting_metrics = await self.run_voting_scenario(scenario, voting_method)

        # Run standard approach
        print("  Running standard approach...")
        standard_metrics = await self.run_standard_scenario(scenario)

        # Create comparison results
        results = ComparisonResults(
            voting_metrics=voting_metrics,
            standard_metrics=standard_metrics,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
        )

        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.results_dir / f"{scenario.name}_{voting_method.value}_{timestamp}.json"
            results.save_to_file(str(filename))
            print(f"  Results saved to: {filename}")

        return results

    async def run_all_scenarios(
        self, scenario_type: ScenarioType | None = None, voting_methods: list[VotingMethod] | None = None
    ) -> list[ComparisonResults]:
        """Run all scenarios with specified parameters."""

        if voting_methods is None:
            voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]

        scenarios = get_all_scenarios()
        if scenario_type:
            scenarios = [s for s in scenarios if s.scenario_type == scenario_type]

        all_results: list[ComparisonResults] = []

        for scenario in scenarios:
            for voting_method in voting_methods:
                try:
                    result = await self.run_comparison(scenario, voting_method)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error running scenario {scenario.name} with {voting_method.value}: {e}")

        # Save summary results
        summary_filename = self.results_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data: dict[str, Any] = {
            "total_scenarios": len(all_results),
            "timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in all_results],
        }

        with open(summary_filename, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"Summary results saved to: {summary_filename}")
        return all_results

    def analyze_results(self, results: list[ComparisonResults]) -> dict[str, Any]:
        """Analyze benchmark results and provide insights."""

        analysis: dict[str, Any] = {
            "total_comparisons": len(results),
            "scenario_types": {},
            "voting_method_performance": {},
            "efficiency_metrics": {
                "avg_time_ratio": 0.0,
                "avg_message_ratio": 0.0,
                "avg_token_ratio": 0.0,
            },
            "decision_quality": {
                "voting_success_rate": 0.0,
                "standard_success_rate": 0.0,
            },
        }

        # Group by scenario type
        scenario_types: dict[str, list[ComparisonResults]] = {}
        for result in results:
            scenario_type = result.scenario_name.split("_")[0]  # Simple grouping
            if scenario_type not in scenario_types:
                scenario_types[scenario_type] = []
            scenario_types[scenario_type].append(result)
        analysis["scenario_types"] = scenario_types

        # Calculate efficiency averages
        if results:
            time_ratios = [r.efficiency_comparison["time_ratio"] for r in results]
            message_ratios = [r.efficiency_comparison["message_ratio"] for r in results]
            token_ratios = [r.efficiency_comparison["token_ratio"] for r in results]

            efficiency_metrics = analysis["efficiency_metrics"]
            if isinstance(efficiency_metrics, dict):
                efficiency_metrics["avg_time_ratio"] = float(sum(time_ratios) / len(time_ratios))
                efficiency_metrics["avg_message_ratio"] = float(sum(message_ratios) / len(message_ratios))
                efficiency_metrics["avg_token_ratio"] = float(sum(token_ratios) / len(token_ratios))

            # Decision quality
            voting_successes = sum(1 for r in results if r.voting_metrics.decision_reached)
            standard_successes = sum(1 for r in results if r.standard_metrics.decision_reached)

            decision_quality = analysis["decision_quality"]
            if isinstance(decision_quality, dict):
                decision_quality["voting_success_rate"] = float(voting_successes / len(results))
                decision_quality["standard_success_rate"] = float(standard_successes / len(results))

        return analysis


async def main():
    """Run a sample benchmark comparison."""
    runner = BenchmarkRunner()

    # Run a subset of scenarios for testing
    results = await runner.run_all_scenarios(scenario_type=ScenarioType.CODE_REVIEW)

    # Analyze results
    analysis = runner.analyze_results(results)
    print("\nBenchmark Analysis:")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
