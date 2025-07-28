"""Example demonstrating the benchmark system for comparing voting vs standard approaches."""

import asyncio
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.runner import BenchmarkRunner
from benchmarks.scenarios import ScenarioType, get_scenario_by_name, VotingMethod


async def run_single_benchmark():
    """Run a single benchmark comparison."""
    print("=== Single Benchmark Example ===")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    runner = BenchmarkRunner(model_name="gpt-4o-mini")
    
    # Get a specific scenario
    scenario = get_scenario_by_name("bug_detection_security")
    if not scenario:
        print("Scenario not found!")
        return
    
    # Run comparison
    result = await runner.run_comparison(
        scenario=scenario,
        voting_method=VotingMethod.MAJORITY,
        save_results=True
    )
    
    print(f"\nResults for {scenario.name}:")
    print(f"Voting duration: {result.voting_metrics.duration_seconds:.2f}s")
    print(f"Standard duration: {result.standard_metrics.duration_seconds:.2f}s")
    print(f"Voting messages: {result.voting_metrics.total_messages}")
    print(f"Standard messages: {result.standard_metrics.total_messages}")
    print(f"Efficiency comparison: {result.efficiency_comparison}")


async def run_code_review_benchmarks():
    """Run all code review benchmarks."""
    print("=== Code Review Benchmarks ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    runner = BenchmarkRunner(model_name="gpt-4o-mini")
    
    # Run all code review scenarios
    results = await runner.run_all_scenarios(
        scenario_type=ScenarioType.CODE_REVIEW,
        voting_methods=[VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY]
    )
    
    # Analyze results
    analysis = runner.analyze_results(results)
    
    print(f"\nCompleted {len(results)} benchmark comparisons")
    print(f"Average time ratio (voting/standard): {analysis['efficiency_metrics']['avg_time_ratio']:.2f}")
    print(f"Average message ratio: {analysis['efficiency_metrics']['avg_message_ratio']:.2f}")
    print(f"Voting success rate: {analysis['decision_quality']['voting_success_rate']:.2%}")
    print(f"Standard success rate: {analysis['decision_quality']['standard_success_rate']:.2%}")


async def compare_voting_methods():
    """Compare different voting methods on the same scenario."""
    print("=== Voting Method Comparison ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    runner = BenchmarkRunner(model_name="gpt-4o-mini")
    scenario = get_scenario_by_name("performance_optimization")
    
    if not scenario:
        print("Scenario not found!")
        return
    
    voting_methods = [
        VotingMethod.MAJORITY,
        VotingMethod.QUALIFIED_MAJORITY,
        VotingMethod.UNANIMOUS
    ]
    
    results = {}
    
    for method in voting_methods:
        print(f"\nRunning with {method.value}...")
        result = await runner.run_comparison(
            scenario=scenario,
            voting_method=method,
            save_results=False
        )
        results[method.value] = result
    
    print("\n=== Voting Method Comparison Results ===")
    for method_name, result in results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Duration: {result.voting_metrics.duration_seconds:.2f}s")
        print(f"  Messages: {result.voting_metrics.total_messages}")
        print(f"  Decision reached: {result.voting_metrics.decision_reached}")
        print(f"  Final votes: {result.voting_metrics.final_vote_counts}")


async def scalability_test():
    """Test how voting scales with different numbers of agents."""
    print("=== Scalability Test ===")
    print("Note: This is a simulation - actual implementation would vary agent counts")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # This is a conceptual example - you'd need to modify scenarios to support variable agent counts
    runner = BenchmarkRunner(model_name="gpt-4o-mini")
    scenario = get_scenario_by_name("code_quality_readability")
    
    if scenario:
        result = await runner.run_comparison(scenario, VotingMethod.MAJORITY)
        print(f"3-agent scenario completed in {result.voting_metrics.duration_seconds:.2f}s")
        print("For true scalability testing, modify scenarios to support 5, 7, 10+ agents")


def main():
    """Run benchmark examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark examples")
    parser.add_argument(
        "--example",
        choices=["single", "code-review", "voting-methods", "scalability", "all"],
        default="single",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "single" or args.example == "all":
        asyncio.run(run_single_benchmark())
    
    if args.example == "code-review" or args.example == "all":
        asyncio.run(run_code_review_benchmarks())
    
    if args.example == "voting-methods" or args.example == "all":
        asyncio.run(compare_voting_methods())
    
    if args.example == "scalability" or args.example == "all":
        asyncio.run(scalability_test())


if __name__ == "__main__":
    main()