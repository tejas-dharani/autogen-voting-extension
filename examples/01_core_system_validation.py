#!/usr/bin/env python3
"""
Simple VotingAI Demo
===================

Basic demonstration of the voting system core components without full agent setup.
Shows the voting protocols, Byzantine fault detection, and decision-making logic.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from votingai import (
    # Core voting protocols
    VotingMethod,
    VoteType,
    VoteContent,
    ProposalContent,
    VotingResult,
    
    # Byzantine fault detection
    ByzantineFaultDetector,
    
    # Security
    SecurityValidator,
)


def demo_voting_protocols():
    """Demonstrate core voting protocols and methods."""
    
    print("🗳️  VotingAI Core Protocols Demo")
    print("=" * 40)
    
    # Show available voting methods
    print("\n📊 Available Voting Methods:")
    for method in VotingMethod:
        print(f"   • {method.name}: {method.value}")
    
    # Show vote types
    print("\n🔘 Available Vote Types:")
    for vote_type in VoteType:
        print(f"   • {vote_type.name}: {vote_type.value}")
    
    # Create sample votes
    proposal_id = "proposal_001"
    votes = [
        VoteContent(
            vote=VoteType.APPROVE,
            proposal_id=proposal_id,
            reasoning="This security fix addresses critical SQL injection vulnerability",
            confidence=0.95
        ),
        VoteContent(
            vote=VoteType.APPROVE, 
            proposal_id=proposal_id,
            reasoning="Good fix, though we should add integration tests",
            confidence=0.8
        ),
        VoteContent(
            vote=VoteType.REJECT,
            proposal_id=proposal_id,
            reasoning="Need more thorough testing before production deployment",
            confidence=0.7
        )
    ]
    
    print(f"\n📝 Sample Votes ({len(votes)} total):")
    for i, vote in enumerate(votes, 1):
        print(f"   {i}. {vote.vote.value} (confidence: {vote.confidence:.2f})")
        print(f"      Reasoning: {vote.reasoning}")
    
    # Demonstrate voting result calculation
    approve_count = sum(1 for v in votes if v.vote == VoteType.APPROVE)
    reject_count = sum(1 for v in votes if v.vote == VoteType.REJECT)
    abstain_count = sum(1 for v in votes if v.vote == VoteType.ABSTAIN)
    
    print(f"\n📈 Vote Summary:")
    print(f"   Approve: {approve_count}")
    print(f"   Reject:  {reject_count}")
    print(f"   Abstain: {abstain_count}")
    
    # Apply different voting methods
    print(f"\n⚖️  Results by Voting Method:")
    
    # Majority (>50%)
    if approve_count > len(votes) / 2:
        majority_result = "APPROVED"
    elif reject_count > len(votes) / 2:
        majority_result = "REJECTED" 
    else:
        majority_result = "NO CONSENSUS"
    print(f"   Majority (>50%):           {majority_result}")
    
    # Qualified majority (67%)
    if approve_count >= len(votes) * 0.67:
        qualified_result = "APPROVED"
    elif reject_count >= len(votes) * 0.67:
        qualified_result = "REJECTED"
    else:
        qualified_result = "NO CONSENSUS"
    print(f"   Qualified Majority (67%):  {qualified_result}")
    
    # Unanimous
    if approve_count == len(votes):
        unanimous_result = "APPROVED"
    elif reject_count == len(votes):
        unanimous_result = "REJECTED"
    else:
        unanimous_result = "NO CONSENSUS"
    print(f"   Unanimous:                 {unanimous_result}")


def demo_byzantine_fault_detection():
    """Demonstrate Byzantine fault detection and reputation system."""
    
    print("\n🛡️  Byzantine Fault Detection Demo")
    print("=" * 40)
    
    # Create fault detector for 5 agents
    detector = ByzantineFaultDetector(total_agents=5, detection_threshold=0.3)
    
    # Simulate agent participation
    agents = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    
    print(f"\n👥 Agents: {', '.join(agents)}")
    
    # Initialize reputations
    for agent in agents:
        detector.initialize_agent_reputation(agent)
        print(f"   {agent}: reputation = {detector.reputation_scores[agent]:.2f}")
    
    # Simulate voting rounds with some Byzantine behavior
    print(f"\n🔄 Simulating voting rounds...")
    
    # Round 1: Normal voting
    votes_round1 = {
        "Alice": {"vote": VoteType.APPROVE},
        "Bob": {"vote": VoteType.APPROVE},
        "Charlie": {"vote": VoteType.REJECT},
        "Diana": {"vote": VoteType.APPROVE},
        "Eve": {"vote": VoteType.APPROVE}
    }
    
    consensus_outcome = "approved"  # Majority voted approve
    
    print(f"\n   Round 1 (Consensus: {consensus_outcome}):")
    for agent, vote_data in votes_round1.items():
        detector.update_reputation(agent, vote_data["vote"], consensus_outcome)
        is_byzantine = detector.detect_byzantine_behavior(agent)
        print(f"     {agent}: {vote_data['vote'].value} → reputation = {detector.reputation_scores[agent]:.2f} {'⚠️ SUSPICIOUS' if is_byzantine else '✅'}")
    
    # Round 2: Eve starts acting suspiciously
    votes_round2 = {
        "Alice": {"vote": VoteType.APPROVE},
        "Bob": {"vote": VoteType.APPROVE},
        "Charlie": {"vote": VoteType.APPROVE},
        "Diana": {"vote": VoteType.APPROVE},
        "Eve": {"vote": VoteType.REJECT}  # Against consensus
    }
    
    consensus_outcome = "approved"
    
    print(f"\n   Round 2 (Consensus: {consensus_outcome}):")
    for agent, vote_data in votes_round2.items():
        detector.update_reputation(agent, vote_data["vote"], consensus_outcome)
        is_byzantine = detector.detect_byzantine_behavior(agent)
        print(f"     {agent}: {vote_data['vote'].value} → reputation = {detector.reputation_scores[agent]:.2f} {'⚠️ SUSPICIOUS' if is_byzantine else '✅'}")
    
    # Show weighted vote calculation
    weighted_votes = detector.get_weighted_vote_count(votes_round2)
    print(f"\n   Weighted Vote Counts:")
    for vote_type, weight in weighted_votes.items():
        print(f"     {vote_type}: {weight:.2f}")
    
    # Check Byzantine resilience
    is_resilient = detector.is_byzantine_resilient(votes_round2)
    print(f"\n   Byzantine Resilient: {'✅ YES' if is_resilient else '❌ NO'}")
    
    if detector.suspicious_agents:
        print(f"   Suspicious Agents: {', '.join(detector.suspicious_agents)} 🚨")


def demo_security_features():
    """Demonstrate security validation features."""
    
    print("\n🔐 Security Features Demo") 
    print("=" * 40)
    
    # Test input sanitization
    test_inputs = [
        "Normal proposal title",
        "Title with <script>alert('xss')</script>",
        "Very long title " + "x" * 500,
        "Title with special chars: @#$%^&*()",
        ""  # Empty input
    ]
    
    print(f"\n🧹 Input Sanitization:")
    for i, test_input in enumerate(test_inputs, 1):
        try:
            sanitized = SecurityValidator().sanitize_text(test_input, max_length=100)
            print(f"   {i}. '{test_input[:50]}{'...' if len(test_input) > 50 else ''}'")
            print(f"      → '{sanitized}'")
        except Exception as e:
            print(f"   {i}. ERROR: {e}")
    
    # Test agent name validation
    test_names = [
        "ValidAgent",
        "agent-123",
        "<script>alert('hack')</script>",
        "Very_Long_Agent_Name_That_Exceeds_Limits_" + "x" * 100,
        "",
        "Normal Agent"
    ]
    
    print(f"\n👤 Agent Name Validation:")
    for i, name in enumerate(test_names, 1):
        try:
            validated = SecurityValidator.validate_agent_name(name)
            print(f"   {i}. '{name}' → ✅ '{validated}'")
        except Exception as e:
            print(f"   {i}. '{name}' → ❌ {e}")
    
    # Test proposal ID generation
    print(f"\n🆔 Secure Proposal ID Generation:")
    for i in range(3):
        proposal_id = SecurityValidator().generate_proposal_id()
        print(f"   {i+1}. {proposal_id}")


def main():
    """Run all demos."""
    
    print("🚀 VotingAI System Core Demo")
    print("============================")
    print("Demonstrating core voting system functionality without external dependencies.\n")
    
    try:
        # Run all demo sections
        demo_voting_protocols()
        demo_byzantine_fault_detection()
        demo_security_features()
        
        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("\n📋 System Capabilities Demonstrated:")
        print("✅ Multiple voting methods (Majority, Qualified, Unanimous)")
        print("✅ Byzantine fault detection with reputation tracking")
        print("✅ Security input validation and sanitization")
        print("✅ Structured vote and proposal handling")
        print("✅ Weighted voting with reputation adjustment")
        print("✅ Consensus outcome tracking")
        
        print(f"\n💡 Next Steps:")
        print("• Set OPENAI_API_KEY to test with real AI agents")
        print("• Use run_benchmarks.py for comprehensive testing")
        print("• Integrate with your multi-agent applications")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()