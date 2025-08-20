"""Basic tests for core voting functionality."""

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.replay import ReplayChatCompletionClient

from src.votingai import (
    BaseVotingGroupChat,
    VoteType,
    VotingMethod,
    VotingPhase,
    ProposalContent,
    ProposalMessage,
    VoteContent,
    VoteMessage,
    VotingResult,
    VotingResultMessage,
    CoreVotingManager,
    ByzantineFaultDetector,
    SecurityValidator,
    AuditLogger,
)


class TestBaseVotingGroupChat:
    """Test BaseVotingGroupChat functionality."""

    @pytest.fixture
    def mock_model_client(self) -> ReplayChatCompletionClient:
        """Create a replay model client for testing."""
        return ReplayChatCompletionClient(
            chat_completions=[
                "I propose we implement the new feature.",
                "I vote APPROVE - this looks good.",
                "I vote APPROVE - agreed.", 
                "I vote REJECT - needs more work.",
            ]
        )

    @pytest.fixture
    def voting_agents(self, mock_model_client):
        """Create test agents for voting."""
        return [
            AssistantAgent("Agent1", model_client=mock_model_client),
            AssistantAgent("Agent2", model_client=mock_model_client),
            AssistantAgent("Agent3", model_client=mock_model_client),
        ]

    def test_base_voting_group_chat_creation(self, voting_agents):
        """Test basic BaseVotingGroupChat creation."""
        voting_team = BaseVotingGroupChat(
            participants=voting_agents,
            voting_method=VotingMethod.MAJORITY,
            max_turns=10,
            termination_condition=MaxMessageTermination(10),
        )

        assert voting_team is not None
        assert isinstance(voting_team, BaseVotingGroupChat)

    def test_voting_method_configurations(self, voting_agents):
        """Test different voting method configurations."""
        # Test majority voting
        majority_team = BaseVotingGroupChat(
            participants=voting_agents, 
            voting_method=VotingMethod.MAJORITY
        )
        assert majority_team is not None

        # Test qualified majority
        qualified_team = BaseVotingGroupChat(
            participants=voting_agents,
            voting_method=VotingMethod.QUALIFIED_MAJORITY,
            qualified_majority_threshold=0.75
        )
        assert qualified_team is not None

        # Test unanimous voting  
        unanimous_team = BaseVotingGroupChat(
            participants=voting_agents,
            voting_method=VotingMethod.UNANIMOUS
        )
        assert unanimous_team is not None

    def test_voting_validation(self, voting_agents):
        """Test validation of BaseVotingGroupChat parameters."""
        # Test minimum participants requirement
        with pytest.raises(ValueError, match="at least 2 participants"):
            BaseVotingGroupChat(participants=[voting_agents[0]])

        # Test invalid threshold
        with pytest.raises(ValueError, match="must be between 0.5 and 1.0"):
            BaseVotingGroupChat(
                participants=voting_agents,
                qualified_majority_threshold=0.3
            )


class TestVotingMessages:
    """Test voting message types."""

    def test_vote_message_creation(self):
        """Test VoteMessage creation and serialization."""
        vote = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE,
                proposal_id="test-proposal", 
                reasoning="This looks good to me",
                confidence=0.9
            ),
            source="TestAgent",
        )

        assert vote.content.vote == VoteType.APPROVE
        assert vote.content.proposal_id == "test-proposal"
        assert vote.content.reasoning == "This looks good to me"
        assert vote.content.confidence == 0.9
        assert "Vote: approve" in vote.to_model_text()

    def test_proposal_message_creation(self):
        """Test ProposalMessage creation and serialization."""
        proposal = ProposalMessage(
            content=ProposalContent(
                proposal_id="test-proposal",
                title="Test Proposal",
                description="This is a test proposal",
                options=["Option A", "Option B"],
            ),
            source="ProposerAgent",
        )

        assert proposal.content.proposal_id == "test-proposal"
        assert proposal.content.title == "Test Proposal"
        assert len(proposal.content.options) == 2
        assert "Proposal: Test Proposal" in proposal.to_model_text()

    def test_voting_result_message_creation(self):
        """Test VotingResultMessage creation and formatting."""
        result = VotingResult(
            proposal_id="test-123",
            result="approved",
            votes_summary={"approve": 2, "reject": 1},
            winning_option="approve",
            total_voters=3,
            participation_rate=1.0,
            confidence_average=0.85,
            detailed_votes={},
        )

        message = VotingResultMessage(content=result, source="VotingManager")
        text = message.to_model_text()
        
        assert "Voting Result: APPROVED" in text
        assert "Participation: 100.0%" in text
        assert "Average Confidence: 0.85" in text
        assert "approve: 2 votes" in text
        assert "Winning Option: approve" in text


class TestSecurityComponents:
    """Test security components."""

    @pytest.fixture
    def security_validator(self):
        """Create a SecurityValidator instance for testing."""
        return SecurityValidator()

    @pytest.fixture  
    def byzantine_detector(self):
        """Create a ByzantineFaultDetector instance for testing."""
        return ByzantineFaultDetector(total_agents=3)

    @pytest.fixture
    def audit_logger(self):
        """Create an AuditLogger instance for testing.""" 
        return AuditLogger()

    def test_security_validator_sanitize_text(self, security_validator):
        """Test text sanitization."""
        # Test valid text
        valid_text = "This is a normal proposal"
        sanitized = security_validator.sanitize_text(valid_text, 1000)
        assert sanitized == valid_text

        # Test dangerous characters get sanitized
        dangerous_text = "Text with <script>alert('XSS')</script>"
        sanitized = security_validator.sanitize_text(dangerous_text, 1000)
        assert "<" not in sanitized
        assert ">" not in sanitized

        # Test length limiting
        long_text = "A" * 1001
        with pytest.raises(ValueError, match="exceeds maximum length"):
            security_validator.sanitize_text(long_text, 1000)

    def test_byzantine_detector_initialization(self, byzantine_detector):
        """Test Byzantine fault detector initialization."""
        assert byzantine_detector.total_agents == 3
        assert byzantine_detector.detection_threshold == 0.3
        assert len(byzantine_detector.reputation_scores) == 0

    def test_audit_logging(self, audit_logger):
        """Test audit logging for security events.""" 
        from src.votingai.security.audit_framework import AuditEvent, AuditEventType
        import datetime
        
        initial_entries = len(audit_logger.events)

        # Create and log a test event
        test_event = AuditEvent(
            event_type=AuditEventType.PROPOSAL_CREATED,
            timestamp=datetime.datetime.now(),
            event_id="test-event-1",
            agent_name="Agent1",
            proposal_id="test-id"
        )
        
        audit_logger.log_event(test_event)

        assert len(audit_logger.events) == initial_entries + 1
        assert audit_logger.events[-1].event_type == AuditEventType.PROPOSAL_CREATED
        assert audit_logger.events[-1].agent_name == "Agent1"


class TestVotingEnums:
    """Test voting enums and constants."""

    def test_voting_method_enum(self):
        """Test VotingMethod enum values."""
        assert VotingMethod.MAJORITY.value == "majority"
        assert VotingMethod.PLURALITY.value == "plurality"  
        assert VotingMethod.UNANIMOUS.value == "unanimous"
        assert VotingMethod.QUALIFIED_MAJORITY.value == "qualified_majority"
        assert VotingMethod.RANKED_CHOICE.value == "ranked_choice"

    def test_vote_type_enum(self):
        """Test VoteType enum values."""
        assert VoteType.APPROVE.value == "approve"
        assert VoteType.REJECT.value == "reject"
        assert VoteType.ABSTAIN.value == "abstain"

    def test_voting_phase_enum(self):
        """Test VotingPhase enum values."""
        assert VotingPhase.PROPOSAL.value == "proposal"
        assert VotingPhase.VOTING.value == "voting"
        assert VotingPhase.DISCUSSION.value == "discussion"
        assert VotingPhase.CONSENSUS.value == "consensus"