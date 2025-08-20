#!/usr/bin/env python3
"""
VotingAI Methods Testing Demo
============================

Comprehensive testing of all voting methods with real AI agents.
Tests each method individually to see how they behave with different scenarios.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from votingai import (
    # Core voting components
    VotingMethod,
    VoteType,
    VoteContent,
    ProposalContent,
    VoteMessage,
    ProposalMessage,
    VotingResult,
    DEFAULT_MODEL,
)

# Import research components for simplified testing
from votingai.research.benchmarking_suite import BenchmarkRunner, BenchmarkConfiguration, ScenarioType, BenchmarkScenario


class VotingMethodTester:
    """Test each voting method systematically."""
    
    def __init__(self):
        self.config = BenchmarkConfiguration(
            model_name=DEFAULT_MODEL,
            max_messages=15,
            timeout_seconds=120,
            rate_limit_delay=1.0,
            save_detailed_logs=True,
            enable_adaptive_consensus=False,  # Test core methods first
            enable_semantic_parsing=False
        )
        self.runner = BenchmarkRunner(self.config)
    
    async def test_majority_voting(self):
        """Test simple majority voting (>50%)."""
        
        print("🗳️  Testing MAJORITY Voting (>50%)")
        print("=" * 50)
        
        # Create test scenario
        scenario = BenchmarkScenario(
            name="majority_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Test majority voting with clear fix",
            task_prompt="""
Code Review: Bug Fix

ISSUE: Function returns None instead of empty list
BEFORE: return None
AFTER:  return []

This fixes a potential NoneType error in calling code.
Should this fix be approved?

Vote with: Approve, Reject, or Abstain
Provide brief reasoning.
            """,
            agent_personas=[
                {"name": "Dev1", "role": "Junior Developer", "description": "Focuses on code correctness"},
                {"name": "Dev2", "role": "Senior Developer", "description": "Considers broader implications"},  
                {"name": "Dev3", "role": "Code Reviewer", "description": "Ensures quality standards"},
            ],
            complexity_level="simple",
            expected_outcome="approve"
        )
        
        print("🤖 Running majority vote with 3 agents...")
        print("📋 Proposal: Simple bug fix (None → empty list)")
        print("🎯 Expected: Should pass with majority support")
        
        try:
            result = await self.runner.run_comparison(
                scenario, 
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            self._print_voting_result("MAJORITY", metrics)
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Majority voting test failed: {e}")
            return False
    
    async def test_qualified_majority_voting(self):
        """Test qualified majority voting (67% threshold)."""
        
        print("\n🗳️  Testing QUALIFIED MAJORITY Voting (67%)")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="qualified_majority_test",
            scenario_type=ScenarioType.ARCHITECTURE_DECISION,
            description="Test qualified majority for architectural change",
            task_prompt="""
Architecture Decision: Database Migration

Current: SQLite (file-based)
Proposed: PostgreSQL (server-based)

Benefits:
+ Better performance for concurrent users
+ Advanced features (JSON, full-text search)
+ Better backup/recovery

Concerns:
- Migration complexity 
- Infrastructure overhead
- Team learning curve

Should we migrate to PostgreSQL?

Vote: Approve, Reject, or Abstain
Provide reasoning considering tradeoffs.
            """,
            agent_personas=[
                {"name": "DBA", "role": "Database Administrator", "description": "Database expert focused on performance"},
                {"name": "DevOps", "role": "DevOps Engineer", "description": "Infrastructure and deployment focused"},
                {"name": "TeamLead", "role": "Tech Lead", "description": "Balances technical and business needs"},
            ],
            complexity_level="complex",
            stakes_level="high"
        )
        
        print("🤖 Running qualified majority vote with 3 agents...")
        print("📋 Proposal: Major database migration decision")
        print("🎯 Expected: Requires 67% agreement (2/3 agents)")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.QUALIFIED_MAJORITY, 
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            self._print_voting_result("QUALIFIED MAJORITY", metrics)
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Qualified majority test failed: {e}")
            return False
    
    async def test_unanimous_voting(self):
        """Test unanimous voting (100% agreement)."""
        
        print("\n🗳️  Testing UNANIMOUS Voting (100%)")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="unanimous_test", 
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Test unanimous voting for critical security fix",
            task_prompt="""
SECURITY CRITICAL: Authentication Bypass Fix

VULNERABILITY: Users can login without password verification
SEVERITY: Critical (allows unauthorized access)

BEFORE:
```python
if user_exists(username):
    return login_success(username)
```

AFTER:
```python
if user_exists(username) and verify_password(username, password):
    return login_success(username)
```

This is a CRITICAL security fix that MUST be deployed immediately.
Should this security fix be approved?

Vote: Approve, Reject, or Abstain
Consider: This is a security emergency requiring unanimous approval.
            """,
            agent_personas=[
                {"name": "SecurityExpert", "role": "Security Engineer", "description": "Security vulnerability specialist"},
                {"name": "BackendDev", "role": "Backend Developer", "description": "Authentication system maintainer"}, 
                {"name": "TechLead", "role": "Technical Lead", "description": "Makes critical deployment decisions"},
            ],
            complexity_level="simple",
            stakes_level="critical",
            expected_outcome="approve"
        )
        
        print("🤖 Running unanimous vote with 3 agents...")
        print("📋 Proposal: Critical security vulnerability fix")
        print("🎯 Expected: Should get unanimous approval (3/3 agents)")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.UNANIMOUS,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            self._print_voting_result("UNANIMOUS", metrics)
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Unanimous voting test failed: {e}")
            return False
    
    async def test_plurality_voting(self):
        """Test plurality voting (most votes wins)."""
        
        print("\n🗳️  Testing PLURALITY Voting (Most Votes Wins)")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="plurality_test",
            scenario_type=ScenarioType.TECHNICAL_EVALUATION,
            description="Test plurality voting with multiple options",
            task_prompt="""
Technology Choice: Frontend Framework

We need to choose a frontend framework for the new project:

Option A: React
- Pros: Large ecosystem, team experience, job market
- Cons: Complex build setup, learning curve for new features

Option B: Vue.js  
- Pros: Gentle learning curve, good documentation, smaller bundle
- Cons: Smaller ecosystem, less job market demand

Option C: Svelte
- Pros: No virtual DOM, smaller bundles, modern approach  
- Cons: Small ecosystem, limited tooling, newer technology

Vote for your preferred option: React, Vue, or Svelte
Provide reasoning for your choice considering project needs.
            """,
            agent_personas=[
                {"name": "FrontendDev", "role": "Frontend Developer", "description": "Frontend technology specialist"},
                {"name": "FullStackDev", "role": "Full-Stack Developer", "description": "Considers both frontend and backend integration"},
                {"name": "Architect", "role": "Software Architect", "description": "Focuses on long-term maintainability"},
                {"name": "ProductManager", "role": "Product Manager", "description": "Considers business and user needs"}
            ],
            complexity_level="moderate"
        )
        
        print("🤖 Running plurality vote with 4 agents...")
        print("📋 Proposal: Framework choice with 3 options")
        print("🎯 Expected: Option with most votes wins")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.PLURALITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            self._print_voting_result("PLURALITY", metrics)
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Plurality voting test failed: {e}")
            return False
    
    async def test_edge_cases(self):
        """Test edge cases and tie scenarios."""
        
        print("\n🗳️  Testing EDGE CASES (Ties & Conflicts)")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="edge_case_test",
            scenario_type=ScenarioType.TECHNICAL_EVALUATION, 
            description="Test controversial decision causing potential tie",
            task_prompt="""
Controversial Decision: Code Style Enforcement

Proposal: Enforce strict linting rules with automatic fixes
- Enforces consistent code style across team
- Automatically fixes style issues on commit
- May override personal style preferences
- Could slow down development initially

This is a divisive topic that developers feel strongly about.
Some love automatic formatting, others prefer manual control.

Should we implement strict automatic code formatting?

Vote: Approve, Reject, or Abstain
Be true to your role's perspective on this controversial topic.
            """,
            agent_personas=[
                {"name": "StrictDev", "role": "Senior Developer (Pro-Standards)", "description": "Strongly believes in code consistency and automation"},
                {"name": "FreestyleDev", "role": "Creative Developer (Pro-Freedom)", "description": "Values developer autonomy and flexibility"},
            ],
            complexity_level="moderate"
        )
        
        print("🤖 Running edge case test with 2 agents (potential tie)...")
        print("📋 Proposal: Controversial code style enforcement") 
        print("🎯 Expected: May result in tie/no consensus")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            self._print_voting_result("TIE SCENARIO", metrics)
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Edge case test failed: {e}")
            return False
    
    def _print_voting_result(self, method_name: str, metrics: Any):
        """Print formatted voting results."""
        
        print(f"\n📊 {method_name} RESULTS:")
        print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
        print(f"   Duration: {metrics.performance.total_duration_seconds:.1f} seconds")
        print(f"   Messages: {metrics.performance.total_messages}")
        print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
        
        if hasattr(metrics, 'voting_outcome') and metrics.voting_outcome:
            outcome = metrics.voting_outcome
            print(f"   Final Result: {outcome.get('result', 'Unknown').upper()}")
            if 'votes_summary' in outcome:
                for vote_type, count in outcome['votes_summary'].items():
                    print(f"   {vote_type.title()}: {count} votes")


async def main():
    """Run comprehensive voting method tests."""
    
    print("🚀 VotingAI Methods Testing Suite")
    print("=================================")
    print("Testing each voting method with real AI agents\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return
    
    tester = VotingMethodTester()
    results = {}
    
    try:
        # Test each voting method
        print("🎯 Testing Core Voting Methods:")
        print("Each test uses different scenarios to showcase method behavior\n")
        
        # Test 1: Majority Voting
        results['majority'] = await tester.test_majority_voting()
        
        # Test 2: Qualified Majority
        results['qualified_majority'] = await tester.test_qualified_majority_voting()
        
        # Test 3: Unanimous Voting  
        results['unanimous'] = await tester.test_unanimous_voting()
        
        # Test 4: Plurality Voting
        results['plurality'] = await tester.test_plurality_voting()
        
        # Test 5: Edge Cases
        results['edge_cases'] = await tester.test_edge_cases()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 VOTING METHODS TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for success in results.values() if success)
        
        for method, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {method.replace('_', ' ').title()}: {status}")
        
        print(f"\n📊 Overall: {successful_tests}/{total_tests} tests successful")
        
        if successful_tests == total_tests:
            print("🎉 All voting methods working perfectly!")
        else:
            print("⚠️  Some methods need attention - check logs above")
        
        print(f"\n💡 Key Insights:")
        print(f"• Each voting method behaves differently for different scenarios")
        print(f"• Majority: Good for routine decisions")
        print(f"• Qualified Majority: Better for important changes")  
        print(f"• Unanimous: Best for critical/security decisions")
        print(f"• Plurality: Useful for multi-option choices")
        print(f"• Your system handles ties and edge cases gracefully")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())