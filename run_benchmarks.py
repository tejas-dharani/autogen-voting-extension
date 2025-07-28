#!/usr/bin/env python3
"""Script to run comprehensive benchmarks comparing voting vs standard group chat approaches."""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmarks.runner import BenchmarkRunner
from benchmarks.scenarios import ScenarioType, get_all_scenarios
from autogen_voting import VotingMethod


async def run_quick_test():
    """Run a quick test to verify everything works."""
    print("=== Quick Benchmark Test ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return False
    
    try:
        runner = BenchmarkRunner(model_name="gpt-4o-mini")
        
        # Create a simple test scenario
        from benchmarks.scenarios import BenchmarkScenario
        test_scenario = BenchmarkScenario(
            name="quick_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Quick test scenario",
            task_prompt="Should we approve this simple bug fix: change `if x = 5` to `if x == 5`?",
            agent_personas=[
                {"name": "Reviewer1", "role": "Code reviewer focused on correctness"},
                {"name": "Reviewer2", "role": "Senior developer reviewing changes"},
                {"name": "Reviewer3", "role": "Team lead making final decisions"}
            ]
        )
        
        print("Running quick comparison...")
        result = await runner.run_comparison(
            scenario=test_scenario,
            voting_method=VotingMethod.MAJORITY,
            save_results=False
        )
        
        print(f"✅ Quick test completed successfully!")
        print(f"   Voting approach: {result.voting_metrics.duration_seconds:.1f}s")
        print(f"   Standard approach: {result.standard_metrics.duration_seconds:.1f}s")
        print(f"   Voting messages: {result.voting_metrics.total_messages}")
        print(f"   Standard messages: {result.standard_metrics.total_messages}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


async def run_full_benchmarks(scenario_types=None, voting_methods=None):
    """Run comprehensive benchmarks."""
    print("=== Full Benchmark Suite ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        return
    
    if voting_methods is None:
        voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]
    
    runner = BenchmarkRunner(model_name="gpt-4o-mini")
    
    all_results = []
    
    # Run scenarios by type if specified, otherwise run all
    if scenario_types:
        for scenario_type in scenario_types:
            print(f"\n--- Running {scenario_type.value} scenarios ---")
            results = await runner.run_all_scenarios(
                scenario_type=scenario_type,
                voting_methods=voting_methods
            )
            all_results.extend(results)
    else:
        print("--- Running all scenarios ---")
        all_results = await runner.run_all_scenarios(voting_methods=voting_methods)
    
    # Analyze and display results
    analysis = runner.analyze_results(all_results)
    
    print(f"\n=== Benchmark Results Summary ===")
    print(f"Total comparisons completed: {analysis['total_comparisons']}")
    print(f"Average time ratio (voting/standard): {analysis['efficiency_metrics']['avg_time_ratio']:.2f}")
    print(f"Average message ratio: {analysis['efficiency_metrics']['avg_message_ratio']:.2f}")
    print(f"Average token ratio: {analysis['efficiency_metrics']['avg_token_ratio']:.2f}")
    print(f"Voting success rate: {analysis['decision_quality']['voting_success_rate']:.1%}")
    print(f"Standard success rate: {analysis['decision_quality']['standard_success_rate']:.1%}")
    
    # Show scenario type breakdown
    print(f"\n=== Results by Scenario Type ===")
    for scenario_type, results in analysis['scenario_types'].items():
        print(f"{scenario_type}: {len(results)} comparisons")
        if results:
            avg_time_ratio = sum(r.efficiency_comparison['time_ratio'] for r in results) / len(results)
            avg_msg_ratio = sum(r.efficiency_comparison['message_ratio'] for r in results) / len(results)
            print(f"  Avg time ratio: {avg_time_ratio:.2f}")
            print(f"  Avg message ratio: {avg_msg_ratio:.2f}")
    
    return all_results


async def run_scalability_test():
    """Run scalability tests with different agent counts."""
    print("=== Scalability Test ===")
    
    # This would need the scalability example to be properly integrated
    print("Running scalability analysis...")
    print("Note: For detailed scalability testing, run: python examples/scalability_example.py")
    
    # Run a basic scalability comparison
    from benchmarks.scenarios import get_scenario_by_name
    runner = BenchmarkRunner(model_name="gpt-4o-mini")
    
    scenario = get_scenario_by_name("bug_detection_security")
    if scenario:
        result = await runner.run_comparison(scenario, VotingMethod.MAJORITY)
        print(f"3-agent scenario: {result.voting_metrics.duration_seconds:.1f}s")
    
    print("For comprehensive scalability testing with 5, 7, 10+ agents,")
    print("modify scenarios to support variable agent counts.")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run AutoGen Voting Extension benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --quick                    # Quick test
  python run_benchmarks.py --full                     # All scenarios
  python run_benchmarks.py --code-review              # Code review only
  python run_benchmarks.py --architecture             # Architecture only
  python run_benchmarks.py --moderation              # Content moderation only
  python run_benchmarks.py --scalability             # Scalability test
  python run_benchmarks.py --majority-only           # Only majority voting
        """
    )
    
    # Test options
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--scalability", action="store_true", help="Run scalability tests")
    
    # Scenario type filters
    parser.add_argument("--code-review", action="store_true", help="Run code review scenarios only")
    parser.add_argument("--architecture", action="store_true", help="Run architecture scenarios only")
    parser.add_argument("--moderation", action="store_true", help="Run content moderation scenarios only")
    
    # Voting method filters
    parser.add_argument("--majority-only", action="store_true", help="Test majority voting only")
    parser.add_argument("--unanimous-only", action="store_true", help="Test unanimous voting only")
    parser.add_argument("--qualified-only", action="store_true", help="Test qualified majority only")
    
    args = parser.parse_args()
    
    # If no specific arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Determine scenario types
    scenario_types = []
    if args.code_review:
        scenario_types.append(ScenarioType.CODE_REVIEW)
    if args.architecture:
        scenario_types.append(ScenarioType.ARCHITECTURE_DECISION)
    if args.moderation:
        scenario_types.append(ScenarioType.CONTENT_MODERATION)
    
    # Determine voting methods
    voting_methods = []
    if args.majority_only:
        voting_methods.append(VotingMethod.MAJORITY)
    if args.unanimous_only:
        voting_methods.append(VotingMethod.UNANIMOUS)
    if args.qualified_only:
        voting_methods.append(VotingMethod.QUALIFIED_MAJORITY)
    
    # Default to all methods if none specified
    if not voting_methods and not args.quick:
        voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]
    
    # Run requested benchmarks
    if args.quick:
        success = asyncio.run(run_quick_test())
        if not success:
            sys.exit(1)
    
    if args.full or any([args.code_review, args.architecture, args.moderation]):
        asyncio.run(run_full_benchmarks(
            scenario_types=scenario_types if scenario_types else None,
            voting_methods=voting_methods
        ))
    
    if args.scalability:
        asyncio.run(run_scalability_test())


if __name__ == "__main__":
    main()