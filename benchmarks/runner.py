"""Benchmark runner for comparing VotingGroupChat vs standard GroupChat."""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
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
    
    def create_agents(self, personas: List[Dict[str, str]]) -> List[AssistantAgent]:
        """Create agents based on persona specifications."""
        model_client = self.create_model_client()
        agents: List[AssistantAgent] = []
        
        for persona in personas:
            agent = AssistantAgent(
                name=persona["name"],
                model_client=model_client,
                system_message=persona["role"]
            )
            agents.append(agent)
        
        return agents
    
    async def run_voting_scenario(
        self,
        scenario: BenchmarkScenario,
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        qualified_majority_threshold: float = 0.67,
        max_discussion_rounds: int = 2,
        require_reasoning: bool = True
    ) -> BenchmarkMetrics:
        """Run a scenario using VotingGroupChat."""
        
        agents = self.create_agents(scenario.agent_personas)
        metrics = self.metrics_collector.start_collection()
        
        # Configure voting team  
        voting_team = VotingGroupChat(
            participants=agents,  # type: ignore
            voting_method=voting_method,
            qualified_majority_threshold=qualified_majority_threshold,
            require_reasoning=require_reasoning,
            max_discussion_rounds=max_discussion_rounds,
            termination_condition=MaxMessageTermination(30),
        )
        
        try:
            # Run the scenario
            result = await voting_team.run(task=scenario.task_prompt)
            
            # Extract voting results
            voting_results = getattr(result, 'voting_results', None)
            if voting_results is not None:
                for agent_name, vote in voting_results.items():
                    metrics.add_vote(agent_name, vote)
                metrics.decision_reached = True
                metrics.consensus_type = voting_method.value
            
            # Mark completion
            metrics.complete_benchmark()
            
        except Exception as e:
            print(f"Error in voting scenario: {e}")
            metrics.complete_benchmark()
        
        finally:
            self.metrics_collector.stop_collection()
        
        return metrics
    
    async def run_standard_scenario(
        self,
        scenario: BenchmarkScenario,
        max_turns: int = 20
    ) -> BenchmarkMetrics:
        """Run a scenario using standard GroupChat."""
        
        agents = self.create_agents(scenario.agent_personas)
        metrics = self.metrics_collector.start_collection()
        
        # Configure standard group chat (using RoundRobinGroupChat as standard comparison)
        standard_team = RoundRobinGroupChat(
            participants=agents,  # type: ignore
            termination_condition=MaxMessageTermination(max_turns),
        )
        
        try:
            # Run the scenario
            _ = await standard_team.run(task=scenario.task_prompt)
            
            # Basic completion tracking
            metrics.decision_reached = True
            metrics.consensus_type = "sequential"
            
            # Mark completion
            metrics.complete_benchmark()
            
        except Exception as e:
            print(f"Error in standard scenario: {e}")
            metrics.complete_benchmark()
        
        finally:
            self.metrics_collector.stop_collection()
        
        return metrics
    
    async def run_comparison(
        self,
        scenario: BenchmarkScenario,
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        save_results: bool = True
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
            scenario_description=scenario.description
        )
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.results_dir / f"{scenario.name}_{voting_method.value}_{timestamp}.json"
            results.save_to_file(str(filename))
            print(f"  Results saved to: {filename}")
        
        return results
    
    async def run_all_scenarios(
        self,
        scenario_type: Optional[ScenarioType] = None,
        voting_methods: Optional[List[VotingMethod]] = None
    ) -> List[ComparisonResults]:
        """Run all scenarios with specified parameters."""
        
        if voting_methods is None:
            voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]
        
        scenarios = get_all_scenarios()
        if scenario_type:
            scenarios = [s for s in scenarios if s.scenario_type == scenario_type]
        
        all_results: List[ComparisonResults] = []
        
        for scenario in scenarios:
            for voting_method in voting_methods:
                try:
                    result = await self.run_comparison(scenario, voting_method)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error running scenario {scenario.name} with {voting_method.value}: {e}")
        
        # Save summary results
        summary_filename = self.results_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data: Dict[str, Any] = {
            "total_scenarios": len(all_results),
            "timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in all_results]
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Summary results saved to: {summary_filename}")
        return all_results
    
    def analyze_results(self, results: List[ComparisonResults]) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights."""
        
        analysis: Dict[str, Any] = {
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
            }
        }
        
        # Group by scenario type
        scenario_types: Dict[str, List[ComparisonResults]] = {}
        for result in results:
            scenario_type = result.scenario_name.split('_')[0]  # Simple grouping
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