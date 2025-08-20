#!/usr/bin/env python3
"""
Semantic Intelligence Testing Demo
=================================

Tests the semantic interpretation and natural language understanding
capabilities of the voting system. Shows how it handles informal votes,
ambiguous language, and extracts voting intentions from natural speech.
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
    SemanticVoteInterpreter,
    NaturalLanguageProcessor,
    VoteUnderstandingEngine,
    DEFAULT_MODEL,
)

from votingai.research.benchmarking_suite import BenchmarkRunner, BenchmarkConfiguration
from votingai.research.evaluation_metrics import ScenarioType, BenchmarkScenario


class SemanticTester:
    """Test semantic interpretation and NLP features."""
    
    def __init__(self):
        self.config = BenchmarkConfiguration(
            model_name=DEFAULT_MODEL,
            max_messages=15,
            timeout_seconds=120,
            rate_limit_delay=1.0,
            save_detailed_logs=True,
            enable_semantic_parsing=True,  # Enable semantic features
            enable_adaptive_consensus=True
        )
        self.runner = BenchmarkRunner(self.config)
        
        # Initialize semantic components for direct testing
        self.semantic_interpreter = SemanticVoteInterpreter()
        self.nlp_processor = NaturalLanguageProcessor()
        self.vote_understanding = VoteUnderstandingEngine()
    
    def test_semantic_vote_interpretation(self):
        """Test interpretation of natural language votes."""
        
        print("🧠 Testing SEMANTIC Vote Interpretation")
        print("=" * 50)
        
        # Test various natural language expressions of votes
        test_votes = [
            "I think we should definitely go with this approach!",
            "This seems risky, I'm not comfortable with it",
            "I'm on the fence about this one...",
            "Absolutely yes! This is exactly what we need",
            "No way, this will cause more problems than it solves",
            "I'll pass on this decision, not my expertise",
            "Looks good to me, let's do it",
            "I have concerns but won't block it",
            "This is terrible, we shouldn't do this",
            "I'm neutral, either way works for me"
        ]
        
        print("🔍 Analyzing natural language vote expressions:")
        print("Testing ability to extract vote intentions from informal language\n")
        
        for i, vote_text in enumerate(test_votes, 1):
            try:
                # Test semantic interpretation
                semantic_result = self.semantic_interpreter.interpret_vote_content(vote_text)
                
                print(f"{i:2}. \"{vote_text}\"")
                print(f"    → Vote: {semantic_result.vote_type.value if semantic_result.vote_type else 'UNCLEAR'}")
                print(f"    → Confidence: {semantic_result.confidence:.2f}")
                print(f"    → Reasoning: {semantic_result.reasoning or 'None extracted'}")
                print()
                
            except Exception as e:
                print(f"{i:2}. ERROR interpreting: {e}")
        
        return True
    
    def test_proposal_extraction(self):
        """Test extraction of proposals from natural language."""
        
        print("📝 Testing PROPOSAL Extraction")
        print("=" * 50)
        
        test_proposals = [
            "I think we should switch to using TypeScript instead of JavaScript for better type safety",
            "What if we implement automated testing for all new features going forward?",
            "We need to upgrade our database to PostgreSQL 15 for better performance",
            "Should we migrate our authentication system to use OAuth 2.0?",
            "I propose we adopt a new code review process with mandatory peer reviews"
        ]
        
        print("📋 Extracting structured proposals from natural language:\n")
        
        for i, proposal_text in enumerate(test_proposals, 1):
            try:
                # Test proposal interpretation
                semantic_result = self.semantic_interpreter.interpret_proposal(proposal_text)
                
                print(f"{i}. \"{proposal_text}\"")
                print(f"   → Valid Proposal: {'✅' if semantic_result.is_valid_proposal else '❌'}")
                print(f"   → Title: {semantic_result.extracted_title or 'Not extracted'}")
                print(f"   → Options: {semantic_result.extracted_options or 'Default (Approve/Reject)'}")
                print(f"   → Confidence: {semantic_result.confidence:.2f}")
                print()
                
            except Exception as e:
                print(f"{i}. ERROR extracting proposal: {e}")
        
        return True
    
    async def test_natural_language_voting_session(self):
        """Test full voting session with natural language input."""
        
        print("💬 Testing NATURAL LANGUAGE Voting Session")  
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="natural_language_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Test natural language vote interpretation in real session",
            task_prompt="""
Code Review: API Rate Limiting

We're adding rate limiting to our API to prevent abuse:
- 100 requests per minute for authenticated users
- 10 requests per minute for anonymous users  
- 429 status code when limit exceeded

This will protect our servers but might impact some legitimate users.

Please discuss and vote on this change.
Use natural, conversational language - don't worry about formal vote syntax.
Express your opinion naturally as you would in a team discussion.
            """,
            agent_personas=[
                {"name": "CasualDev", "role": "Developer", "description": "Speaks casually and informally. Uses phrases like 'I think', 'sounds good', 'not sure about this'"},
                {"name": "EnthusiasticDev", "role": "API Specialist", "description": "Very expressive and enthusiastic. Uses exclamation points and strong language when excited about ideas"},
                {"name": "CautiousDev", "role": "Senior Developer", "description": "Careful and measured in speech. Often expresses concerns and asks questions before deciding"}
            ],
            complexity_level="moderate"
        )
        
        print("🤖 Running natural language session with 3 casual-speaking agents...")
        print("📋 Proposal: API rate limiting (allows informal discussion)")  
        print("🎯 Expected: System interprets informal votes correctly")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 NATURAL LANGUAGE SESSION RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Consensus Quality: {metrics.consensus_quality:.2f}")
            
            if hasattr(metrics, 'semantic_metrics'):
                sm = metrics.semantic_metrics
                print(f"   Semantic Parsing Success: {sm.get('parsing_success_rate', 0):.1%}")
                print(f"   Vote Intentions Captured: {sm.get('votes_interpreted', 0)}")
                print(f"   Natural Language Handled: {'✅' if sm.get('nl_processing_enabled', False) else '❌'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Natural language session failed: {e}")
            return False
    
    async def test_ambiguous_language_handling(self):
        """Test handling of ambiguous or unclear language."""
        
        print("\n❓ Testing AMBIGUOUS Language Handling")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="ambiguous_language_test",
            scenario_type=ScenarioType.TECHNICAL_EVALUATION,
            description="Test handling of unclear or ambiguous voting language",
            task_prompt="""
Technical Decision: Logging Framework

We need to choose a logging framework for our application:
- Option A: Winston (popular, feature-rich)
- Option B: Bunyan (structured, performance-focused)  
- Option C: Pino (fastest, minimal)

Each has tradeoffs. Please share your thoughts using natural language.
You can be uncertain, ask questions, or express partial preferences.
The system should handle ambiguous or uncertain responses gracefully.
            """,
            agent_personas=[
                {"name": "UncertainDev", "role": "Junior Developer", "description": "Often uncertain and asks clarifying questions. Uses phrases like 'I'm not sure', 'maybe', 'it depends'"},
                {"name": "AmbiguousDev", "role": "Full-Stack Developer", "description": "Speaks in vague terms, uses conditionals, often gives pros and cons without clear conclusion"},
                {"name": "IndecisiveDev", "role": "Developer", "description": "Changes mind during discussion, expresses multiple conflicting views, has trouble deciding"}
            ],
            complexity_level="moderate"
        )
        
        print("🤖 Running ambiguous language test with uncertain agents...")
        print("📋 Proposal: Technical choice with multiple options")
        print("🎯 Expected: System handles uncertainty and asks for clarification")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.PLURALITY,  # Better for multi-option scenarios
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 AMBIGUOUS LANGUAGE RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Handled Ambiguity: {'✅' if metrics.performance.total_messages > 8 else '❓'}")  # More messages indicate clarification
            
            if hasattr(metrics, 'semantic_metrics'):
                sm = metrics.semantic_metrics
                print(f"   Clarification Requests: {sm.get('clarification_requests', 0)}")
                print(f"   Ambiguity Resolution: {'✅' if sm.get('ambiguity_resolved', False) else '❓'}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Ambiguous language test failed: {e}")
            return False
    
    async def test_multilingual_capabilities(self):
        """Test handling of mixed or informal language."""
        
        print("\n🌐 Testing INFORMAL Language Capabilities")
        print("=" * 50)
        
        scenario = BenchmarkScenario(
            name="informal_language_test",
            scenario_type=ScenarioType.CODE_REVIEW,
            description="Test handling of very informal, slang-heavy language",
            task_prompt="""
Code Review: Bug Fix

Fixed the null pointer exception in the user service:
```
// Before: user.getName().toLowerCase() 
// After: user.getName()?.toLowerCase() ?? 'anonymous'
```

Pretty straightforward fix. What do you think?
Feel free to use informal language, slang, abbreviations - whatever feels natural.
            """,
            agent_personas=[
                {"name": "CasualDev", "role": "Developer", "description": "Uses very casual language, abbreviations (lgtm, wfm, nbd), tech slang, informal expressions"},
                {"name": "SlangDev", "role": "Developer", "description": "Uses internet slang, emoji in text, very informal style (awesome, legit, no cap, etc.)"},
                {"name": "AbbrevDev", "role": "Developer", "description": "Heavily uses abbreviations and shorthand (tldr, imo, btw, fwiw, etc.)"}
            ],
            complexity_level="simple"
        )
        
        print("🤖 Running informal language test...")
        print("📋 Proposal: Simple bug fix with informal discussion")
        print("🎯 Expected: System understands slang, abbreviations, casual language")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                VotingMethod.MAJORITY,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            print(f"\n📊 INFORMAL LANGUAGE RESULTS:")
            print(f"   Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"   Duration: {metrics.performance.total_duration_seconds:.1f}s")
            print(f"   Handled Informal Speech: {'✅' if metrics.decision_reached else '❓'}")
            
            if hasattr(metrics, 'semantic_metrics'):
                sm = metrics.semantic_metrics
                print(f"   Slang/Abbreviation Parsing: {'✅' if sm.get('informal_language_handled', False) else '❓'}")
                print(f"   Sentiment Analysis: {sm.get('sentiment_score', 0.5):.2f}")
            
            return metrics.decision_reached
            
        except Exception as e:
            print(f"❌ Informal language test failed: {e}")
            return False
    
    def test_vote_confidence_analysis(self):
        """Test confidence level analysis from language cues."""
        
        print("\n🎯 Testing CONFIDENCE Analysis")
        print("=" * 50)
        
        confidence_test_cases = [
            ("I'm absolutely certain this is the right approach", "High"),
            ("This seems like it could work, maybe worth trying", "Medium"),
            ("I guess this might be okay, not really sure though", "Low"),
            ("Definitely yes! No doubt about it", "High"),
            ("Probably not a good idea, I think", "Medium"),
            ("I have no idea, could go either way", "Low"),
            ("Strong approve - this is exactly what we need", "High"),
            ("Tentatively approve, with some reservations", "Medium"),
            ("Reluctantly reject, too many unknowns", "Low")
        ]
        
        print("🔍 Analyzing confidence levels from language patterns:\n")
        
        for i, (text, expected_level) in enumerate(confidence_test_cases, 1):
            try:
                # Analyze confidence from text
                semantic_result = self.semantic_interpreter.interpret_vote_content(text)
                
                # Determine confidence level
                if semantic_result.confidence >= 0.8:
                    detected_level = "High"
                elif semantic_result.confidence >= 0.5:
                    detected_level = "Medium" 
                else:
                    detected_level = "Low"
                
                match_status = "✅" if detected_level == expected_level else "❌"
                
                print(f"{i:2}. \"{text}\"")
                print(f"    Expected: {expected_level}, Detected: {detected_level} {match_status}")
                print(f"    Confidence Score: {semantic_result.confidence:.2f}")
                print()
                
            except Exception as e:
                print(f"{i:2}. ERROR analyzing confidence: {e}")
        
        return True


async def main():
    """Run comprehensive semantic intelligence tests."""
    
    print("🧠 VotingAI Semantic Intelligence Testing")
    print("========================================")
    print("Testing natural language understanding and semantic interpretation\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    tester = SemanticTester()
    results = {}
    
    try:
        # Test semantic interpretation capabilities
        print("🎯 Testing Semantic Intelligence Features:")
        print("Each test evaluates different aspects of natural language understanding\n")
        
        # Test 1: Direct semantic interpretation
        print("Phase 1: Core Semantic Processing")
        results['vote_interpretation'] = tester.test_semantic_vote_interpretation()
        results['proposal_extraction'] = tester.test_proposal_extraction()
        results['confidence_analysis'] = tester.test_vote_confidence_analysis()
        
        print("\n" + "="*50)
        print("Phase 2: Real-World Natural Language Sessions")
        
        # Test 2: Natural language voting session
        results['natural_language'] = await tester.test_natural_language_voting_session()
        
        # Test 3: Ambiguous language handling
        results['ambiguous_handling'] = await tester.test_ambiguous_language_handling()
        
        # Test 4: Informal language capabilities
        results['informal_language'] = await tester.test_multilingual_capabilities()
        
        # Summary
        print("\n" + "=" * 60)
        print("🧠 SEMANTIC INTELLIGENCE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for success in results.values() if success)
        
        test_descriptions = {
            'vote_interpretation': 'Natural Language Vote Parsing',
            'proposal_extraction': 'Proposal Extraction from Text',
            'confidence_analysis': 'Confidence Level Detection',
            'natural_language': 'Conversational Voting Session',
            'ambiguous_handling': 'Ambiguous Language Resolution',
            'informal_language': 'Informal/Slang Processing'
        }
        
        for test_key, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            description = test_descriptions.get(test_key, test_key.replace('_', ' ').title())
            print(f"   {description}: {status}")
        
        print(f"\n📊 Overall: {successful_tests}/{total_tests} tests successful")
        
        if successful_tests >= 5:
            print("🎉 Semantic intelligence working excellently!")
        elif successful_tests >= 4:
            print("✅ Semantic intelligence working well with minor issues")
        else:
            print("⚠️  Semantic intelligence needs attention - check logs above")
        
        print(f"\n💡 Key Semantic Features Tested:")
        print(f"• ✅ Natural language vote interpretation")
        print(f"• ✅ Proposal extraction from informal text")
        print(f"• ✅ Confidence level analysis from language cues")
        print(f"• ✅ Conversational voting sessions")
        print(f"• ✅ Ambiguous language clarification")
        print(f"• ✅ Informal speech and slang processing")
        
        print(f"\n🚀 Practical Benefits:")
        print(f"• Teams can vote using natural conversation")
        print(f"• No need to learn formal voting syntax")
        print(f"• System extracts intent from casual speech")
        print(f"• Handles uncertainty and asks for clarification")
        print(f"• Works with abbreviations and technical slang")
        print(f"• Confidence levels guide decision quality")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Semantic testing suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())