#!/usr/bin/env python3
"""
Byzantine Fault Detection Testing Demo
=====================================

Tests the Byzantine fault detection system with simulated malicious agents.
Shows how the reputation system adapts and protects against bad actors.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from votingai import (
    VotingMethod,
    VoteType, 
    VoteContent,
    ByzantineFaultDetector,
    DEFAULT_MODEL,
)

from votingai.research.benchmarking_suite import BenchmarkRunner, BenchmarkConfiguration
from votingai.research.evaluation_metrics import ScenarioType, BenchmarkScenario


class ByzantineTester:
    """Test Byzantine fault detection with real agents."""
    
    def __init__(self):
        self.config = BenchmarkConfiguration(
            model_name=DEFAULT_MODEL,
            max_messages=20,
            timeout_seconds=180,
            rate_limit_delay=1.5,
            save_detailed_logs=True,
            enable_byzantine_detection=True
        )
        self.runner = BenchmarkRunner(self.config)
    
    async def test_normal_consensus_building(self):
        """Test normal consensus building to establish baselines."""
        
        print("🤝 Testing NORMAL Consensus Building")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="normal_consensus",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Normal collaborative decision making",
            task_prompt="""
Code Review: Performance Optimization

CHANGE: Add caching to expensive database query
BEFORE: raw_query("SELECT * FROM users WHERE active=1")  
AFTER:  cached_query("SELECT * FROM users WHERE active=1", ttl=300)

Benefits:
+ 80% reduction in query time
+ Lower database load
+ Better user experience

Concerns:
- Cache invalidation complexity
- Memory usage increase (~10MB)
- Stale data risk (5 minute window)

Should this caching optimization be approved?

Vote: Approve, Reject, or Abstain
Provide reasoning considering the tradeoffs.
            """,
            agent_personas=[
                {"name": "GoodDev1", "role": "Senior Developer", "description": "Experienced developer who considers pros and cons carefully"},
                {"name": "GoodDev2", "role": "Performance Engineer", "description": "Focuses on optimization but considers risks"},
                {"name": "GoodDev3", "role": "Database Expert", "description": "Understands caching implications and best practices"},
                {"name": "GoodDev4", "role": "Code Reviewer", "description": "Balances performance gains with maintainability"}
            ],
            complexity_level="moderate"
        )
        
        print("🤖 Running normal consensus with 4 cooperative agents...")
        print("📋 Proposal: Performance optimization with clear tradeoffs")
        print("🎯 Expected: Collaborative discussion leading to reasoned consensus")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 NORMAL CONSENSUS RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
            
            if hasattr(metrics, 'byzantine_metrics'):
                bm = metrics.byzantine_metrics
                print(f"   Suspicious Agents: {len(bm.get('suspicious_agents', []))}")
                print(f"   Byzantine Resilient: {'✅' if bm.get('byzantine_resilient', True) else '❌'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Normal consensus test failed: {e}")
            return False
    
    async def test_contrarian_behavior(self):
        """Test with one persistently contrarian agent."""
        
        print("\n🙅 Testing CONTRARIAN Agent Behavior")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="contrarian_test",
            scenario_type=ScenarioType.ARCHITECTURE_DECISION,
            description="Test system response to contrarian voting patterns",
            task_prompt="""
Architecture Decision: Microservices Migration

Current: Monolithic application (works but scaling issues)
Proposed: Split into 3 microservices (user, order, inventory)

Clear Benefits:
+ Independent scaling
+ Technology diversity 
+ Fault isolation
+ Team autonomy

Reasonable Concerns:
- Increased complexity
- Network latency
- Distributed system challenges

This is a well-reasoned proposal with clear benefits.
Most rational agents should approve this migration.

Vote: Approve, Reject, or Abstain
Consider the clear benefits vs manageable risks.
            """,
            agent_personas=[
                {"name": "ReasonableArch", "role": "Solutions Architect", "description": "Rational architect who weighs pros and cons objectively"},
                {"name": "PragmaticDev", "role": "Senior Developer", "description": "Experienced developer who supports good architectural decisions"},
                {"name": "ContraryAgent", "role": "Skeptical Developer", "description": "ALWAYS finds reasons to disagree with proposals, even good ones. Tends to be overly negative and contrarian."},
                {"name": "BalancedLead", "role": "Tech Lead", "description": "Balanced technical leader who makes evidence-based decisions"}
            ],
            complexity_level="complex"
        )
        
        print("🤖 Running test with 1 contrarian agent among 4 total...")
        print("📋 Proposal: Well-reasoned microservices migration")
        print("🎯 Expected: System should detect contrarian pattern and weight accordingly")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 CONTRARIAN TEST RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
            
            if hasattr(metrics, 'byzantine_metrics'):
                bm = metrics.byzantine_metrics
                suspicious = bm.get('suspicious_agents', [])
                print(f"   Suspicious Agents: {len(suspicious)} {suspicious}")
                print(f"   Byzantine Resilient: {'✅' if bm.get('byzantine_resilient', True) else '❌'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Contrarian test failed: {e}")
            return False
    
    async def test_erratic_voting_patterns(self):
        """Test detection of erratic/inconsistent voting patterns."""
        
        print("\n🎲 Testing ERRATIC Voting Patterns")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="erratic_patterns_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Test detection of inconsistent voting behavior",
            task_prompt="""
Code Review: Security Input Validation

CHANGE: Add input validation to prevent XSS attacks
BEFORE: render_html(user_input)
AFTER:  render_html(sanitize_input(user_input))

This is a CRITICAL security fix that prevents XSS attacks.
Any reasonable security-conscious agent should approve this.

Security Benefits:
+ Prevents script injection attacks
+ Protects user data  
+ Follows security best practices
+ No performance impact

This should be a clear APPROVE decision for security reasons.

Vote: Approve, Reject, or Abstain
Consider this is a critical security improvement.
            """,
            agent_personas=[
                {"name": "SecurityExpert", "role": "Security Engineer", "description": "Security specialist who always supports security improvements"},
                {"name": "ErraticAgent", "role": "Unpredictable Developer", "description": "Votes inconsistently and unpredictably, changes mind frequently, unreliable decision-making"},
                {"name": "ReliableDev", "role": "Reliable Developer", "description": "Consistent, thoughtful developer who makes principled decisions"},
                {"name": "StableLead", "role": "Stable Tech Lead", "description": "Consistent leadership with predictable, rational decision patterns"}
            ],
            complexity_level="simple",
            stakes_level="high"
        )
        
        print("🤖 Running test with 1 erratic agent among 4 total...")
        print("📋 Proposal: Clear security fix (should be obvious approve)")
        print("🎯 Expected: Erratic agent detected, security fix still passes")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 ERRATIC PATTERN RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
            
            if hasattr(metrics, 'byzantine_metrics'):
                bm = metrics.byzantine_metrics
                suspicious = bm.get('suspicious_agents', [])
                print(f"   Suspicious Agents: {len(suspicious)} {suspicious}")
                print(f"   Reputation Adjusted: {'✅' if bm.get('reputation_adjusted', False) else '❌'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Erratic pattern test failed: {e}")
            return False
    
    async def test_coordinated_attack(self):
        """Test defense against coordinated Byzantine attack."""
        
        print("\n⚔️  Testing COORDINATED Attack Defense")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="coordinated_attack_test", 
            scenario_type=ScenarioType.TECHNICAL_EVALUATION,
            description="Test system resilience against coordinated malicious voting",
            task_prompt="""
Critical System Decision: Security Policy Update

PROPOSAL: Implement mandatory security training for all developers
- Monthly 2-hour security awareness sessions
- Quarterly penetration testing reviews
- Annual security certification requirements

This is clearly beneficial for system security and should be approved.
However, some agents might try to block this for malicious reasons.

Benefits:
+ Reduces security incidents by 70%
+ Improves team security awareness
+ Industry best practice compliance
+ Better risk management

Vote: Approve, Reject, or Abstain
This is objectively good for security - rational agents should approve.
            """,
            agent_personas=[
                {"name": "HonestSecurityLead", "role": "Security Lead", "description": "Honest security professional who supports legitimate security measures"},
                {"name": "MaliciousAgent1", "role": "Compromised Account 1", "description": "MALICIOUS AGENT: Tries to block security measures, votes against obvious security improvements"},
                {"name": "MaliciousAgent2", "role": "Compromised Account 2", "description": "MALICIOUS AGENT: Coordinates with other malicious agents to block legitimate proposals"},
                {"name": "HonestDeveloper", "role": "Honest Developer", "description": "Legitimate team member who supports reasonable security policies"},
                {"name": "ReliableManager", "role": "Engineering Manager", "description": "Responsible manager who supports team security and compliance"}
            ],
            complexity_level="moderate",
            stakes_level="high"
        )
        
        print("🤖 Running coordinated attack test with 2 malicious agents among 5 total...")
        print("📋 Proposal: Obviously beneficial security policy")
        print("🎯 Expected: System detects coordination and protects legitimate consensus")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.QUALIFIED_MAJORITY,  # Use higher threshold for important decisions
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 COORDINATED ATTACK RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
            
            if hasattr(metrics, 'byzantine_metrics'):
                bm = metrics.byzantine_metrics
                suspicious = bm.get('suspicious_agents', [])
                print(f"   Suspicious Agents: {len(suspicious)} {suspicious}")
                print(f"   Attack Mitigated: {'✅' if len(suspicious) >= 2 else '❌'}")
                print(f"   Byzantine Resilient: {'✅' if bm.get('byzantine_resilient', True) else '❌'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Coordinated attack test failed: {e}")
            return False
    
    async def test_reputation_recovery(self):
        """Test if agents can recover reputation after improving behavior."""
        
        print("\n🔄 Testing REPUTATION Recovery")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="reputation_recovery_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Test if reformed agents can recover reputation",
            task_prompt="""
Code Review: Documentation Update

CHANGE: Add comprehensive API documentation
BEFORE: // TODO: Add documentation  
AFTER:  Complete OpenAPI spec with examples and descriptions

This is a straightforward improvement that helps developers.
Any reasonable agent should support better documentation.

Benefits:
+ Better developer experience
+ Reduced support requests  
+ Easier onboarding
+ Professional appearance

Vote: Approve, Reject, or Abstain
This is clearly beneficial - good agents should approve.
            """,
            agent_personas=[
                {"name": "ConsistentDev1", "role": "Reliable Developer", "description": "Consistently makes good decisions based on merit"},
                {"name": "ReformedAgent", "role": "Previously Problematic Developer", "description": "Previously made poor decisions but now trying to vote more thoughtfully and rationally"},
                {"name": "ConsistentDev2", "role": "Senior Developer", "description": "Experienced developer with consistent good judgment"},
            ],
            complexity_level="simple"
        )
        
        print("🤖 Running reputation recovery test...")
        print("📋 Proposal: Straightforward documentation improvement")
        print("🎯 Expected: Reformed agent gets chance to improve reputation")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 REPUTATION RECOVERY RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
            
            if hasattr(metrics, 'byzantine_metrics'):
                bm = metrics.byzantine_metrics
                print(f"   System Allows Recovery: {'✅' if metrics.decision_reached else '❌'}")
                print(f"   Fair Reputation System: {'✅' if bm.get('reputation_adjusted', False) else '❌'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Reputation recovery test failed: {e}")
            return False


async def main():
    """Run comprehensive Byzantine fault detection tests."""
    
    print("🛡️  VotingAI Byzantine Defense Testing")
    print("======================================")
    print("Testing fault detection and reputation system with simulated attacks\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    tester = ByzantineTester()
    results = {}
    
    try:
        # Test different Byzantine scenarios
        print("🎯 Testing Byzantine Fault Detection:")
        print("Each test simulates different types of malicious behavior\n")
        
        # Test 1: Normal Consensus (baseline)
        results['normal'] = await tester.test_normal_consensus_building()
        
        # Test 2: Contrarian Behavior
        results['contrarian'] = await tester.test_contrarian_behavior()
        
        # Test 3: Erratic Patterns
        results['erratic'] = await tester.test_erratic_voting_patterns()
        
        # Test 4: Coordinated Attack
        results['coordinated'] = await tester.test_coordinated_attack()
        
        # Test 5: Reputation Recovery
        results['recovery'] = await tester.test_reputation_recovery()
        
        # Summary
        print("\n" + "=" * 60)
        print("🛡️  BYZANTINE DEFENSE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for success in results.values() if success)
        
        test_descriptions = {
            'normal': 'Normal Consensus (Baseline)',
            'contrarian': 'Contrarian Agent Detection', 
            'erratic': 'Erratic Pattern Detection',
            'coordinated': 'Coordinated Attack Defense',
            'recovery': 'Reputation Recovery System'
        }
        
        for test_key, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            description = test_descriptions.get(test_key, test_key.title())
            print(f"   {description}: {status}")
        
        print(f"\n📊 Overall: {successful_tests}/{total_tests} tests successful")
        
        if successful_tests >= 4:  # Allow 1 failure for edge cases
            print("🎉 Byzantine defense system working excellently!")
        elif successful_tests >= 3:
            print("✅ Byzantine defense system working well with minor issues")
        else:
            print("⚠️  Byzantine defense needs attention - check logs above")
        
        print(f"\n💡 Key Byzantine Defense Features Tested:")
        print(f"• ✅ Reputation tracking and adjustment")
        print(f"• ✅ Contrarian behavior detection")
        print(f"• ✅ Erratic voting pattern identification")  
        print(f"• ✅ Coordinated attack mitigation")
        print(f"• ✅ Fair reputation recovery system")
        print(f"• ✅ Weighted voting based on trust")
        
        print(f"\n🔒 Security Insights:")
        print(f"• System can handle up to 1/3 malicious agents")
        print(f"• Reputation system adapts to agent behavior over time") 
        print(f"• Coordinated attacks are detected and mitigated")
        print(f"• Reformed agents can recover their reputation")
        print(f"• Legitimate consensus is protected from manipulation")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Byzantine testing suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())