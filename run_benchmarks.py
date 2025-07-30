#!/usr/bin/env python3
"""Script to run comprehensive benchmarks comparing voting vs standard group chat approaches."""

import asyncio
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_voting import VotingMethod
from benchmarks.metrics import ComparisonResults
from benchmarks.runner import BenchmarkRunner
from benchmarks.scenarios import BenchmarkScenario, ScenarioType, get_scenario_by_name


async def run_validation_test() -> bool:
    """Run quick validation tests for all scenario types."""
    print("=== System Validation Test ===")
    print("Testing all scenario types with simple prompts...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        return False

    try:
        runner = BenchmarkRunner(model_name="gpt-4o-mini", rate_limit_delay=0.5, max_retries=3)
        
        # Quick scenarios for each type
        test_scenarios = [
            BenchmarkScenario(
                name="validate_code_review",
                scenario_type=ScenarioType.CODE_REVIEW,
                description="Quick code review validation",
                task_prompt="Should we approve this fix: change `x = 5` to `x == 5`? Yes/No.",
                agent_personas=[
                    {"name": "Dev1", "role": "Developer"},
                    {"name": "Dev2", "role": "Reviewer"},
                    {"name": "Lead", "role": "Lead"},
                ],
            ),
            BenchmarkScenario(
                name="validate_architecture",
                scenario_type=ScenarioType.ARCHITECTURE_DECISION,
                description="Quick architecture validation",
                task_prompt="For a simple blog, choose: SQL database or NoSQL? One sentence reasoning.",
                agent_personas=[
                    {"name": "Architect", "role": "Solutions Architect"},
                    {"name": "Backend", "role": "Backend Developer"},
                    {"name": "DBA", "role": "Database Admin"},
                ],
            ),
            BenchmarkScenario(
                name="validate_moderation",
                scenario_type=ScenarioType.CONTENT_MODERATION,
                description="Quick moderation validation",
                task_prompt="Should this be approved: 'Great product, highly recommend!'? Approve/Reject.",
                agent_personas=[
                    {"name": "Mod1", "role": "Content Moderator"},
                    {"name": "Mod2", "role": "Senior Moderator"},
                    {"name": "Manager", "role": "Community Manager"},
                ],
            ),
        ]
        
        print(f"Running {len(test_scenarios)} validation tests...")
        all_passed = True
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n[{i}/{len(test_scenarios)}] Testing {scenario.scenario_type.value}...")
            try:
                result = await runner.run_comparison(scenario, VotingMethod.MAJORITY, save_results=False)
                voting_ok = result.voting_metrics.decision_reached
                standard_ok = result.standard_metrics.decision_reached
                
                if voting_ok and standard_ok:
                    print(f"   ✅ {scenario.scenario_type.value}: PASSED")
                    print(f"      Voting: {result.voting_metrics.duration_seconds:.1f}s, {result.voting_metrics.total_messages} msgs")
                    print(f"      Standard: {result.standard_metrics.duration_seconds:.1f}s, {result.standard_metrics.total_messages} msgs")
                else:
                    print(f"   ❌ {scenario.scenario_type.value}: FAILED")
                    print(f"      Voting decision: {voting_ok}, Standard decision: {standard_ok}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {scenario.scenario_type.value}: ERROR - {e}")
                all_passed = False
        
        if all_passed:
            print(f"\n🎉 System Validation: ALL TESTS PASSED!")
            print("Your voting extension is working correctly across all scenario types.")
        else:
            print(f"\n⚠️  System Validation: SOME TESTS FAILED")
            print("Check the errors above to diagnose issues.")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


async def run_quick_test() -> bool:
    """Run a quick test to verify everything works."""
    print("=== Quick Benchmark Test ===")
    print("debug: Starting quick test function")

    if not os.getenv("OPENAI_API_KEY"):
        print("debug: OPENAI_API_KEY not found in environment")
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return False

    try:
        print("debug: Creating BenchmarkRunner with model gpt-4o-mini")
        runner = BenchmarkRunner(model_name="gpt-4o-mini", rate_limit_delay=2.0, max_retries=5)

        # Create a simple test scenario
        print("debug: Creating test scenario for code review")
        test_scenario = BenchmarkScenario(
            name="quick_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Quick test scenario",
            task_prompt="Should we approve this simple bug fix: change `if x = 5` to `if x == 5`?",
            agent_personas=[
                {"name": "Reviewer1", "role": "Code reviewer focused on correctness"},
                {"name": "Reviewer2", "role": "Senior developer reviewing changes"},
                {"name": "Reviewer3", "role": "Team lead making final decisions"},
            ],
        )
        print(f"debug: Test scenario created: {test_scenario.name} ({test_scenario.scenario_type.value})")

        print("Running quick comparison...")
        print("debug: Starting comparison with MAJORITY voting method")
        result = await runner.run_comparison(
            scenario=test_scenario, voting_method=VotingMethod.MAJORITY, save_results=True
        )
        print("debug: Comparison completed successfully")

        print("✅ Quick test completed successfully!")
        print(f"   Voting approach: {result.voting_metrics.duration_seconds:.1f}s")
        print(f"   Standard approach: {result.standard_metrics.duration_seconds:.1f}s")
        print(f"   Voting messages: {result.voting_metrics.total_messages}")
        print(f"   Standard messages: {result.standard_metrics.total_messages}")
        print(f"debug: Quick test results - voting decision: {result.voting_metrics.decision_reached}, standard decision: {result.standard_metrics.decision_reached}")

        return True

    except Exception as e:
        print(f"debug: Exception occurred in quick test: {type(e).__name__}")
        print(f"❌ Quick test failed: {e}")
        return False


async def run_full_benchmarks(
    scenario_types: list[ScenarioType] | None = None, voting_methods: list[VotingMethod] | None = None
) -> list[ComparisonResults]:
    """Run comprehensive benchmarks."""
    print("=== Full Benchmark Suite ===")
    print(f"debug: Starting full benchmarks with scenario_types={scenario_types}, voting_methods={voting_methods}")

    if not os.getenv("OPENAI_API_KEY"):
        print("debug: OPENAI_API_KEY not found in environment")
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        return []

    if voting_methods is None:
        voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]
        print("debug: Using default voting methods: MAJORITY, QUALIFIED_MAJORITY, UNANIMOUS")

    print("debug: Creating BenchmarkRunner with model gpt-4o-mini")
    runner = BenchmarkRunner(model_name="gpt-4o-mini", rate_limit_delay=2.0, max_retries=5)

    all_results: list[ComparisonResults] = []

    # Run scenarios by type if specified, otherwise run all
    if scenario_types:
        print(f"debug: Running specific scenario types: {[st.value for st in scenario_types]}")
        for scenario_type in scenario_types:
            print(f"\n--- Running {scenario_type.value} scenarios ---")
            print(f"debug: Starting scenarios for {scenario_type.value}")
            results = await runner.run_all_scenarios(scenario_type=scenario_type, voting_methods=voting_methods)
            print(f"debug: Completed {len(results)} scenarios for {scenario_type.value}")
            all_results.extend(results)
    else:
        print("--- Running all scenarios ---")
        print("debug: Running all available scenarios")
        all_results = await runner.run_all_scenarios(voting_methods=voting_methods)
        print(f"debug: Completed all scenarios, total results: {len(all_results)}")

    # Analyze and display results
    print(f"debug: Analyzing {len(all_results)} results")
    analysis = runner.analyze_results(all_results)
    print("debug: Analysis completed")

    print("\n=== Benchmark Results Summary ===")
    print(f"Total comparisons completed: {analysis['total_comparisons']}")
    print(f"Average time ratio (voting/standard): {analysis['efficiency_metrics']['avg_time_ratio']:.2f}")
    print(f"Average message ratio: {analysis['efficiency_metrics']['avg_message_ratio']:.2f}")
    print(f"Average token ratio: {analysis['efficiency_metrics']['avg_token_ratio']:.2f}")
    print(f"Voting success rate: {analysis['decision_quality']['voting_success_rate']:.1%}")
    print(f"Standard success rate: {analysis['decision_quality']['standard_success_rate']:.1%}")

    # Show scenario type breakdown
    print("\n=== Results by Scenario Type ===")
    for scenario_type, results in analysis["scenario_types"].items():
        print(f"{scenario_type}: {len(results)} comparisons")
        if results:
            avg_time_ratio = sum(r.efficiency_comparison["time_ratio"] for r in results) / len(results)
            avg_msg_ratio = sum(r.efficiency_comparison["message_ratio"] for r in results) / len(results)
            print(f"  Avg time ratio: {avg_time_ratio:.2f}")
            print(f"  Avg message ratio: {avg_msg_ratio:.2f}")

    return all_results


async def run_scalability_test() -> None:
    """Run scalability tests with different agent counts."""
    print("=== Scalability Test ===")
    print("debug: Starting scalability test")

    # This would need the scalability example to be properly integrated
    print("Running scalability analysis...")
    print("Note: For detailed scalability testing, run: python examples/scalability_example.py")

    # Run a basic scalability comparison
    print("debug: Creating BenchmarkRunner for scalability test")
    runner = BenchmarkRunner(model_name="gpt-4o-mini", rate_limit_delay=2.0, max_retries=5)

    print("debug: Getting bug_detection_security scenario")
    scenario = get_scenario_by_name("bug_detection_security")
    if scenario:
        print("debug: Running scalability comparison with MAJORITY voting")
        result = await runner.run_comparison(scenario, VotingMethod.MAJORITY)
        print(f"3-agent scenario: {result.voting_metrics.duration_seconds:.1f}s")
        print(f"debug: Scalability test completed in {result.voting_metrics.duration_seconds:.1f}s")
    else:
        print("debug: bug_detection_security scenario not found")

    print("For comprehensive scalability testing with 5, 7, 10+ agents,")
    print("modify scenarios to support variable agent counts.")


def main() -> None:
    """Main entry point for benchmark runner."""
    parser = ArgumentParser(
        description="Run AutoGen Voting Extension benchmarks",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --quick                    # Quick test (code review only)
  python run_benchmarks.py --validate                 # Validate all scenario types (2-3 minutes)
  python run_benchmarks.py --full                     # All scenarios (30+ minutes)
  python run_benchmarks.py --code-review              # Code review only
  python run_benchmarks.py --architecture             # Architecture only
  python run_benchmarks.py --moderation              # Content moderation only
  python run_benchmarks.py --scalability             # Scalability test
  python run_benchmarks.py --majority-only           # Only majority voting
        """,
    )

    # Test options
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--validate", action="store_true", help="Run validation tests for all scenario types")
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
        print("debug: No arguments provided, showing help")
        parser.print_help()
        return

    # Determine scenario types
    scenario_types: list[ScenarioType] = []
    if args.code_review:
        scenario_types.append(ScenarioType.CODE_REVIEW)
    if args.architecture:
        scenario_types.append(ScenarioType.ARCHITECTURE_DECISION)
    if args.moderation:
        scenario_types.append(ScenarioType.CONTENT_MODERATION)
    print(f"debug: Selected scenario types: {[st.value for st in scenario_types] if scenario_types else 'all'}")

    # Determine voting methods
    voting_methods: list[VotingMethod] = []
    if args.majority_only:
        voting_methods.append(VotingMethod.MAJORITY)
    if args.unanimous_only:
        voting_methods.append(VotingMethod.UNANIMOUS)
    if args.qualified_only:
        voting_methods.append(VotingMethod.QUALIFIED_MAJORITY)

    # Default to all methods if none specified
    if not voting_methods and not args.quick:
        voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]
    print(f"debug: Selected voting methods: {[vm.value for vm in voting_methods] if voting_methods else 'none (quick test)'}")

    # Run requested benchmarks
    if args.quick:
        print("debug: Running quick test")
        success = asyncio.run(run_quick_test())
        if not success:
            print("debug: Quick test failed, exiting with code 1")
            sys.exit(1)

    if args.validate:
        print("debug: Running validation tests")
        success = asyncio.run(run_validation_test())
        if not success:
            print("debug: Validation tests failed, exiting with code 1")
            sys.exit(1)

    if args.full or any([args.code_review, args.architecture, args.moderation]):
        print("debug: Running full benchmarks")
        asyncio.run(
            run_full_benchmarks(
                scenario_types=scenario_types if scenario_types else None, voting_methods=voting_methods
            )
        )

    if args.scalability:
        print("debug: Running scalability test")
        asyncio.run(run_scalability_test())


if __name__ == "__main__":
    main()
