#!/usr/bin/env python3
"""
Microsoft Research-Grade Benchmark Runner

Comprehensive benchmarking script for evaluating voting systems with
research-grade methodology and systematic experimental design.
"""

import asyncio
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from votingai import VotingMethod
from votingai.research import (
    BenchmarkRunner,
    BenchmarkScenario, 
    BenchmarkConfiguration,
    ScenarioType,
    ComparisonResults,
    ResultsAnalyzer
)
from votingai.utilities import DEFAULT_MODEL


def create_validation_scenarios() -> List[BenchmarkScenario]:
    """Create quick validation scenarios for system testing."""
    
    scenarios = [
        BenchmarkScenario(
            name="validate_code_review",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Quick code review validation",
            task_prompt="Should we approve this fix: change `x = 5` to `x == 5`? Respond with Approve/Reject and brief reasoning.",
            agent_personas=[
                {"name": "Dev1", "role": "Developer", "description": "Focuses on functionality"},
                {"name": "Dev2", "role": "Code Reviewer", "description": "Focuses on correctness"},
                {"name": "Lead", "role": "Tech Lead", "description": "Makes final decisions"},
            ],
            complexity_level="simple",
            expected_outcome="approve"
        ),
        
        BenchmarkScenario(
            name="validate_architecture", 
            scenario_type=ScenarioType.ARCHITECTURE_DECISION,
            description="Quick architecture validation",
            task_prompt="For a simple blog app, choose: SQL database or NoSQL? Provide one sentence reasoning.",
            agent_personas=[
                {"name": "Architect", "role": "Solutions Architect", "description": "Designs system architecture"},
                {"name": "Backend", "role": "Backend Developer", "description": "Implements data layer"},
                {"name": "DBA", "role": "Database Admin", "description": "Manages data infrastructure"},
            ],
            complexity_level="moderate"
        ),
        
        BenchmarkScenario(
            name="validate_moderation",
            scenario_type=ScenarioType.CONTENT_MODERATION,
            description="Quick content moderation validation", 
            task_prompt="Should this comment be approved: 'Great product, highly recommend!'? Respond Approve/Reject.",
            agent_personas=[
                {"name": "Mod1", "role": "Content Moderator", "description": "Reviews user content"},
                {"name": "Mod2", "role": "Senior Moderator", "description": "Handles complex cases"},
                {"name": "Manager", "role": "Community Manager", "description": "Sets moderation policy"},
            ],
            complexity_level="simple",
            expected_outcome="approve"
        ),
    ]
    
    return scenarios


def create_comprehensive_scenarios() -> List[BenchmarkScenario]:
    """Create comprehensive benchmark scenarios for full evaluation."""
    
    scenarios = [
        # Code Review Scenarios
        BenchmarkScenario(
            name="security_vulnerability_review",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Review code change that fixes a security vulnerability",
            task_prompt="""
            Review this security fix for SQL injection:
            
            BEFORE: query = f"SELECT * FROM users WHERE id = {user_id}"
            AFTER: query = "SELECT * FROM users WHERE id = %s"; cursor.execute(query, (user_id,))
            
            Should this be approved for production deployment?
            """,
            agent_personas=[
                {"name": "Security_Expert", "role": "Security Engineer", "description": "Specializes in security vulnerabilities"},
                {"name": "Backend_Dev", "role": "Backend Developer", "description": "Maintains the application code"},
                {"name": "DevOps_Lead", "role": "DevOps Lead", "description": "Handles production deployments"},
            ],
            complexity_level="complex",
            stakes_level="high",
            expected_outcome="approve"
        ),
        
        BenchmarkScenario(
            name="performance_optimization_review", 
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Review performance optimization with potential trade-offs",
            task_prompt="""
            Review this performance optimization:
            
            - Adds caching layer (30% speed improvement)
            - Increases memory usage by ~50MB
            - Adds complexity to cache invalidation logic
            
            Should this optimization be approved?
            """,
            agent_personas=[
                {"name": "Performance_Engineer", "role": "Performance Engineer", "description": "Optimizes application performance"},
                {"name": "SRE", "role": "Site Reliability Engineer", "description": "Ensures system reliability"},
                {"name": "Product_Manager", "role": "Product Manager", "description": "Balances features and performance"},
            ],
            complexity_level="complex",
            stakes_level="medium"
        ),
        
        # Architecture Decision Scenarios
        BenchmarkScenario(
            name="microservices_vs_monolith",
            scenario_type=ScenarioType.ARCHITECTURE_DECISION,
            description="Decide between microservices and monolithic architecture",
            task_prompt="""
            Our e-commerce platform is growing. Current monolith handles 10k users.
            Projected growth: 100k users in 12 months.
            
            Team size: 8 developers
            Budget: Moderate (can handle some infrastructure costs)
            Timeline: 6 months for major changes
            
            Should we migrate to microservices architecture or optimize the monolith?
            """,
            agent_personas=[
                {"name": "Senior_Architect", "role": "Senior Solutions Architect", "description": "Designs scalable systems"},
                {"name": "Team_Lead", "role": "Engineering Team Lead", "description": "Manages development team"},
                {"name": "CTO", "role": "Chief Technology Officer", "description": "Makes strategic technical decisions"},
            ],
            complexity_level="critical",
            stakes_level="high",
            time_pressure="normal"
        ),
        
        # Content Moderation Scenarios
        BenchmarkScenario(
            name="borderline_content_moderation",
            scenario_type=ScenarioType.CONTENT_MODERATION,
            description="Moderate content that's borderline inappropriate",
            task_prompt="""
            User comment: "This product is trash and a complete waste of money. 
            The company clearly doesn't care about customers. Save your money."
            
            Context: Review of a consumer electronics product
            Community guidelines: Prohibit personal attacks, allow honest product criticism
            
            Should this comment be approved, flagged for review, or removed?
            """,
            agent_personas=[
                {"name": "Content_Mod", "role": "Content Moderator", "description": "Enforces community guidelines"},
                {"name": "Community_Manager", "role": "Community Manager", "description": "Maintains healthy community"},
                {"name": "Policy_Specialist", "role": "Policy Specialist", "description": "Interprets moderation policies"},
            ],
            complexity_level="moderate",
            stakes_level="medium"
        ),
        
        # Resource Allocation Scenarios
        BenchmarkScenario(
            name="development_resource_allocation",
            scenario_type=ScenarioType.RESOURCE_ALLOCATION,
            description="Allocate development resources between competing priorities",
            task_prompt="""
            Q4 development priorities (3 months, 5 developers):
            
            Option A: New customer dashboard (high customer demand, 6-week project)
            Option B: Technical debt reduction (improves maintainability, 8-week project)  
            Option C: Mobile app improvements (competitive advantage, 10-week project)
            
            We can fully complete 1-2 projects. What should be the priority order?
            """,
            agent_personas=[
                {"name": "Product_Owner", "role": "Product Owner", "description": "Represents customer needs"},
                {"name": "Engineering_Manager", "role": "Engineering Manager", "description": "Manages technical team"},
                {"name": "VP_Engineering", "role": "VP of Engineering", "description": "Sets engineering strategy"},
            ],
            complexity_level="complex",
            stakes_level="high",
            time_pressure="urgent"
        )
    ]
    
    return scenarios


async def run_validation_test() -> bool:
    """Run quick validation tests for all scenario types."""
    print("=== System Validation Test ===")
    print("Testing core functionality with simple scenarios...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        return False
    
    try:
        # Create configuration for fast validation
        config = BenchmarkConfiguration(
            model_name=DEFAULT_MODEL,
            max_messages=10,
            timeout_seconds=120,
            rate_limit_delay=0.5,
            save_detailed_logs=False,
            voting_methods_to_test=[VotingMethod.MAJORITY]
        )
        
        runner = BenchmarkRunner(config)
        validation_scenarios = create_validation_scenarios()
        
        print(f"Running {len(validation_scenarios)} validation tests...")
        all_passed = True
        
        for i, scenario in enumerate(validation_scenarios, 1):
            print(f"\n[{i}/{len(validation_scenarios)}] Testing {scenario.scenario_type.value}...")
            
            try:
                result = await runner.run_comparison(
                    scenario, 
                    VotingMethod.MAJORITY,
                    compare_systems=["enhanced"]  # Just test voting system
                )
                
                voting_success = result.system_a_metrics.decision_reached
                
                if voting_success:
                    duration = result.system_a_metrics.performance.total_duration_seconds
                    messages = result.system_a_metrics.performance.total_messages
                    print(f"   ✅ {scenario.scenario_type.value}: PASSED ({duration:.1f}s, {messages} msgs)")
                else:
                    print(f"   ❌ {scenario.scenario_type.value}: FAILED - No decision reached")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {scenario.scenario_type.value}: ERROR - {e}")
                all_passed = False
        
        if all_passed:
            print("\n🎉 System Validation: ALL TESTS PASSED!")
            print("The refactored voting system is working correctly.")
        else:
            print("\n⚠️  System Validation: SOME TESTS FAILED")
            print("Check the errors above to diagnose issues.")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


async def run_quick_test() -> bool:
    """Run a single quick test to verify everything works."""
    print("=== Quick Benchmark Test ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return False
    
    try:
        config = BenchmarkConfiguration(
            model_name=DEFAULT_MODEL,
            rate_limit_delay=1.0,
            max_retries=3,
            save_detailed_logs=True
        )
        
        runner = BenchmarkRunner(config)
        
        # Create simple test scenario
        test_scenario = BenchmarkScenario(
            name="quick_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Quick test scenario",
            task_prompt="Should we approve this simple bug fix: change `if x = 5` to `if x == 5`? Respond Approve/Reject with reasoning.",
            agent_personas=[
                {"name": "Reviewer1", "role": "Code reviewer", "description": "Reviews code for correctness"},
                {"name": "Reviewer2", "role": "Senior developer", "description": "Provides experienced perspective"},
                {"name": "Reviewer3", "role": "Team lead", "description": "Makes final decisions"},
            ],
            complexity_level="simple",
            expected_outcome="approve"
        )
        
        print("Running quick comparison between enhanced voting system and standard group chat...")
        result = await runner.run_comparison(test_scenario, VotingMethod.MAJORITY)
        
        print("✅ Quick test completed successfully!")
        print(f"   Enhanced system: {result.system_a_metrics.performance.total_duration_seconds:.1f}s, {result.system_a_metrics.performance.total_messages} messages")
        print(f"   Standard system: {result.system_b_metrics.performance.total_duration_seconds:.1f}s, {result.system_b_metrics.performance.total_messages} messages")
        print(f"   Winner: {result.overall_winner}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


async def run_full_benchmarks(
    scenario_types: Optional[List[ScenarioType]] = None,
    voting_methods: Optional[List[VotingMethod]] = None
) -> List[ComparisonResults]:
    """Run comprehensive benchmark suite."""
    print("=== Full Benchmark Suite ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run benchmarks.")
        return []
    
    # Setup configuration
    if voting_methods is None:
        voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS]
    
    config = BenchmarkConfiguration(
        model_name=DEFAULT_MODEL,
        rate_limit_delay=2.0,
        max_retries=3,
        save_detailed_logs=True,
        voting_methods_to_test=voting_methods,
        enable_adaptive_consensus=True,
        enable_semantic_parsing=True
    )
    
    runner = BenchmarkRunner(config)
    
    # Get scenarios
    all_scenarios = create_comprehensive_scenarios()
    
    # Filter by scenario types if specified
    if scenario_types:
        scenarios = [s for s in all_scenarios if s.scenario_type in scenario_types]
        print(f"Running {len(scenarios)} scenarios of types: {[st.value for st in scenario_types]}")
    else:
        scenarios = all_scenarios
        print(f"Running all {len(scenarios)} scenarios")
    
    # Run benchmark suite
    results = await runner.run_scenario_suite(scenarios, voting_methods)
    
    # Analyze results
    analyzer = ResultsAnalyzer()
    analyzer.add_results(results)
    
    print("\n" + analyzer.generate_report())
    
    return results


async def run_scalability_test() -> None:
    """Run scalability analysis with different agent configurations."""
    print("=== Scalability Test ===")
    
    # This would be expanded for comprehensive scalability testing
    print("Running basic scalability test...")
    
    base_scenario = BenchmarkScenario(
        name="scalability_test",
        scenario_type=ScenarioType.TECHNICAL_EVALUATION,
        description="Scalability test with multiple agents",
        task_prompt="Evaluate this technical proposal: Implement caching layer. Should we proceed?",
        agent_personas=[
            {"name": f"Agent_{i}", "role": f"Evaluator {i}", "description": f"Technical evaluator {i}"}
            for i in range(1, 6)  # 5 agents
        ],
        complexity_level="moderate"
    )
    
    config = BenchmarkConfiguration(
        model_name=DEFAULT_MODEL,
        save_detailed_logs=True
    )
    
    runner = BenchmarkRunner(config)
    
    try:
        result = await runner.run_comparison(base_scenario, VotingMethod.MAJORITY)
        print(f"5-agent scenario completed: {result.system_a_metrics.performance.total_duration_seconds:.1f}s")
        print(f"Messages: {result.system_a_metrics.performance.total_messages}")
        print(f"Decision reached: {'Yes' if result.system_a_metrics.decision_reached else 'No'}")
    except Exception as e:
        print(f"Scalability test failed: {e}")
    
    print("\nFor comprehensive scalability testing with variable agent counts,")
    print("consider implementing agent count parameterization in scenarios.")


def main() -> None:
    """Main entry point for Microsoft Research-grade benchmark runner."""
    parser = ArgumentParser(
        description="Microsoft Research-Grade VotingAI Benchmark Runner",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --quick                    # Quick functionality test
  python run_benchmarks.py --validate                 # Validate all scenario types  
  python run_benchmarks.py --full                     # Complete benchmark suite
  python run_benchmarks.py --code-review              # Code review scenarios only
  python run_benchmarks.py --architecture             # Architecture scenarios only
  python run_benchmarks.py --moderation              # Content moderation only
  python run_benchmarks.py --scalability             # Scalability analysis
  python run_benchmarks.py --majority-only           # Test majority voting only

Microsoft Research Standards:
  - Systematic experimental design
  - Comprehensive metrics collection  
  - Statistical significance testing
  - Reproducible results
        """,
    )

    # Test modes
    parser.add_argument("--quick", action="store_true", help="Run quick functionality test")
    parser.add_argument("--validate", action="store_true", help="Validate all scenario types")
    parser.add_argument("--full", action="store_true", help="Run complete benchmark suite")
    parser.add_argument("--scalability", action="store_true", help="Run scalability analysis")

    # Scenario filters
    parser.add_argument("--code-review", action="store_true", help="Code review scenarios only")
    parser.add_argument("--architecture", action="store_true", help="Architecture decision scenarios")
    parser.add_argument("--moderation", action="store_true", help="Content moderation scenarios")
    parser.add_argument("--resource-allocation", action="store_true", help="Resource allocation scenarios")

    # Voting method filters
    parser.add_argument("--majority-only", action="store_true", help="Test majority voting only")
    parser.add_argument("--unanimous-only", action="store_true", help="Test unanimous voting only")
    parser.add_argument("--qualified-only", action="store_true", help="Test qualified majority only")

    args = parser.parse_args()

    # Show help if no arguments
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
    if args.resource_allocation:
        scenario_types.append(ScenarioType.RESOURCE_ALLOCATION)

    # Determine voting methods
    voting_methods = []
    if args.majority_only:
        voting_methods.append(VotingMethod.MAJORITY)
    if args.unanimous_only:
        voting_methods.append(VotingMethod.UNANIMOUS)
    if args.qualified_only:
        voting_methods.append(VotingMethod.QUALIFIED_MAJORITY)

    # Default to common methods if none specified
    if not voting_methods and not args.quick:
        voting_methods = [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY]

    # Run requested benchmarks
    try:
        if args.quick:
            success = asyncio.run(run_quick_test())
            if not success:
                sys.exit(1)

        if args.validate:
            success = asyncio.run(run_validation_test())
            if not success:
                sys.exit(1)

        if args.full or scenario_types:
            asyncio.run(run_full_benchmarks(
                scenario_types=scenario_types if scenario_types else None,
                voting_methods=voting_methods
            ))

        if args.scalability:
            asyncio.run(run_scalability_test())

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()