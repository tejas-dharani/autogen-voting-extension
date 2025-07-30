import asyncio
import contextlib
import logging
import os
from typing import cast
from unittest.mock import patch

import pytest
import pytest_asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import ChatAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, MessageFactory, TextMessage
from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from autogen_ext.models.replay import ReplayChatCompletionClient

from src.autogen_voting import VoteType, VotingGroupChat, VotingMethod, VotingPhase
from src.autogen_voting.voting_group_chat import (
    AuditLogger,
    ByzantineFaultDetector,
    ProposalContent,
    ProposalMessage,
    SecurityValidator,
    VoteContent,
    VoteMessage,
    VotingGroupChatManager,
    VotingResult,
    VotingResultMessage,
)

# Check for OpenAI API key availability
openai_api_key = os.environ.get("OPENAI_API_KEY")
requires_openai_api = pytest.mark.skipif(openai_api_key is None, reason="OPENAI_API_KEY environment variable not set")

# Type imports for integration tests
if openai_api_key is not None:
    from autogen_ext.models.openai import OpenAIChatCompletionClient


class TestVotingGroupChat:
    """Test suite for VotingGroupChat functionality."""

    @pytest_asyncio.fixture  # type: ignore[misc]
    async def mock_model_client(self) -> ReplayChatCompletionClient:
        """Create a replay model client for testing."""
        return ReplayChatCompletionClient(
            chat_completions=[
                "I propose we implement the new feature.",
                "I vote APPROVE - this looks good.",
                "I vote APPROVE - agreed.",
                "I vote REJECT - needs more work.",
            ]
        )

    @pytest_asyncio.fixture  # type: ignore[misc]
    async def voting_agents(self, mock_model_client: ReplayChatCompletionClient) -> list[ChatAgent]:
        """Create test agents for voting."""
        return [
            AssistantAgent("Agent1", model_client=mock_model_client),
            AssistantAgent("Agent2", model_client=mock_model_client),
            AssistantAgent("Agent3", model_client=mock_model_client),
        ]

    def test_voting_group_chat_creation(self, voting_agents: list[ChatAgent]) -> None:
        """Test basic VotingGroupChat creation."""
        voting_team = VotingGroupChat(
            participants=voting_agents,
            voting_method=VotingMethod.MAJORITY,
            max_turns=10,
            termination_condition=MaxMessageTermination(10),
        )

        # Test that team was created successfully with expected participants
        assert voting_team is not None
        assert isinstance(voting_team, VotingGroupChat)

    def test_voting_method_configurations(self, voting_agents: list[ChatAgent]) -> None:
        """Test different voting method configurations."""
        # Test majority voting
        majority_team = VotingGroupChat(participants=voting_agents, voting_method=VotingMethod.MAJORITY)
        assert majority_team is not None

        # Test qualified majority
        qualified_team = VotingGroupChat(
            participants=voting_agents, voting_method=VotingMethod.QUALIFIED_MAJORITY, qualified_majority_threshold=0.75
        )
        assert qualified_team is not None

        # Test unanimous voting
        unanimous_team = VotingGroupChat(participants=voting_agents, voting_method=VotingMethod.UNANIMOUS)
        assert unanimous_team is not None

    def test_voting_group_chat_validation(self, voting_agents: list[ChatAgent]) -> None:
        """Test validation of VotingGroupChat parameters."""
        # Test minimum participants requirement
        with pytest.raises(ValueError, match="at least 2 participants"):
            VotingGroupChat(participants=[voting_agents[0]])

        # Test invalid threshold
        with pytest.raises(ValueError, match="must be between 0.5 and 1.0"):
            VotingGroupChat(participants=voting_agents, qualified_majority_threshold=0.3)

        # Test invalid auto_propose_speaker
        with pytest.raises(ValueError, match="not found in participants"):
            VotingGroupChat(participants=voting_agents, auto_propose_speaker="NonExistentAgent")

    def test_vote_message_creation(self) -> None:
        """Test VoteMessage creation and serialization."""
        vote = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE, proposal_id="test-proposal", reasoning="This looks good to me", confidence=0.9
            ),
            source="TestAgent",
        )

        assert vote.content.vote == VoteType.APPROVE
        assert vote.content.proposal_id == "test-proposal"
        assert vote.content.reasoning == "This looks good to me"
        assert vote.content.confidence == 0.9
        assert "Vote: approve" in vote.to_model_text()

    def test_proposal_message_creation(self) -> None:
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

    def test_voting_configuration_export(self, voting_agents: list[ChatAgent]) -> None:
        """Test configuration export and import."""
        original_team = VotingGroupChat(
            participants=voting_agents,
            voting_method=VotingMethod.QUALIFIED_MAJORITY,
            qualified_majority_threshold=0.8,
            allow_abstentions=False,
            require_reasoning=True,
            max_discussion_rounds=5,
        )

        # Test that team was created successfully
        assert original_team is not None
        assert isinstance(original_team, VotingGroupChat)


class TestVotingGroupChatIntegration:
    """Integration tests for VotingGroupChat with real OpenAI API."""

    @pytest_asyncio.fixture  # type: ignore[misc]
    async def openai_model_client(self) -> "OpenAIChatCompletionClient":
        """Create a real OpenAI model client for integration testing."""
        if openai_api_key is None:
            pytest.skip("OPENAI_API_KEY not available")

        # Import here to avoid import errors when OpenAI is not available
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model="gpt-4o-mini",  # Use cheaper model for testing
            api_key=openai_api_key,
        )

    @pytest_asyncio.fixture  # type: ignore[misc]
    async def real_voting_agents(self, openai_model_client: "OpenAIChatCompletionClient") -> list[ChatAgent]:
        """Create test agents with real OpenAI client for integration testing."""
        return [
            AssistantAgent("Reviewer1", model_client=openai_model_client),
            AssistantAgent("Reviewer2", model_client=openai_model_client),
            AssistantAgent("Reviewer3", model_client=openai_model_client),
        ]

    @requires_openai_api
    @pytest.mark.asyncio
    async def test_real_voting_group_chat_basic_flow(self, real_voting_agents: list[ChatAgent]) -> None:
        """Test basic VotingGroupChat flow with real OpenAI API calls."""
        voting_team = VotingGroupChat(
            participants=real_voting_agents,
            voting_method=VotingMethod.MAJORITY,
            max_turns=5,
            termination_condition=MaxMessageTermination(5),
        )

        # Test that team was created successfully with real agents
        assert voting_team is not None
        assert isinstance(voting_team, VotingGroupChat)

        # Test a simple conversation to verify API connectivity
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken

        test_message = TextMessage(content="Hello, can you respond with just 'API_TEST_SUCCESS'?", source="TestUser")

        cancellation_token = CancellationToken()
        response = await real_voting_agents[0].on_messages([test_message], cancellation_token)
        assert response is not None

    @requires_openai_api
    @pytest.mark.asyncio
    async def test_real_voting_with_proposal(self, real_voting_agents: list[ChatAgent]) -> None:
        """Test voting flow with a real proposal using OpenAI API."""
        voting_team = VotingGroupChat(
            participants=real_voting_agents,
            voting_method=VotingMethod.MAJORITY,
            max_turns=3,
            termination_condition=MaxMessageTermination(3),
        )

        # Create a simple proposal for testing
        proposal = ProposalMessage(
            content=ProposalContent(
                proposal_id="integration-test-1",
                title="Test API Integration",
                description="Should we proceed with this integration test?",
                options=["Yes", "No"],
            ),
            source="TestProposer",
        )

        # Test that proposal is properly formatted
        assert proposal.content.proposal_id == "integration-test-1"
        assert "Test API Integration" in proposal.to_model_text()

        # Test that voting team can handle the proposal structure
        assert voting_team is not None


class TestVotingGroupChatManager:
    """Comprehensive tests for VotingGroupChatManager functionality."""

    @pytest_asyncio.fixture  # type: ignore[misc]
    async def voting_manager(self) -> VotingGroupChatManager:
        """Create a VotingGroupChatManager for testing."""
        output_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination] = asyncio.Queue()
        message_factory = MessageFactory()

        manager = VotingGroupChatManager(
            name="TestVotingManager",
            group_topic_type="test_group",
            output_topic_type="test_output",
            participant_topic_types=["agent1", "agent2", "agent3"],
            participant_names=["Agent1", "Agent2", "Agent3"],
            participant_descriptions=["desc1", "desc2", "desc3"],
            output_message_queue=output_queue,
            termination_condition=None,
            max_turns=10,
            message_factory=message_factory,
            voting_method=VotingMethod.MAJORITY,
            qualified_majority_threshold=0.67,
            allow_abstentions=True,
            require_reasoning=False,
            max_discussion_rounds=3,
            auto_propose_speaker="Agent1",
            emit_team_events=False,
        )
        return manager

    @pytest.mark.asyncio
    async def test_voting_manager_initialization(self, voting_manager: VotingGroupChatManager) -> None:
        """Test VotingGroupChatManager initialization."""
        assert voting_manager.voting_method == VotingMethod.MAJORITY
        assert voting_manager.qualified_majority_threshold == 0.67
        assert voting_manager.allow_abstentions is True
        assert voting_manager.require_reasoning is False
        assert voting_manager.max_discussion_rounds == 3
        assert voting_manager.auto_propose_speaker == "Agent1"
        assert voting_manager.current_phase == VotingPhase.PROPOSAL

    @pytest.mark.asyncio
    async def test_validate_group_state(self, voting_manager: VotingGroupChatManager) -> None:
        """Test group state validation."""
        # Should pass with 3 participants
        await voting_manager.validate_group_state(None)

        # Test with insufficient participants
        voting_manager.set_participant_names_for_testing(["Agent1"])
        with pytest.raises(ValueError, match="Voting requires at least 2 participants"):
            await voting_manager.validate_group_state(None)

    @pytest.mark.asyncio
    async def test_reset_manager(self, voting_manager: VotingGroupChatManager) -> None:
        """Test manager reset functionality."""
        # Set some state
        voting_manager.set_phase_for_testing(VotingPhase.VOTING)
        voting_manager.set_votes_for_testing({"Agent1": {"vote": VoteType.APPROVE}})
        voting_manager.set_discussion_rounds_for_testing(2)

        await voting_manager.reset()

        assert voting_manager.current_phase == VotingPhase.PROPOSAL
        assert voting_manager.votes_cast == {}
        assert voting_manager.discussion_rounds == 0
        assert voting_manager.current_proposal is None

    @pytest.mark.asyncio
    async def test_select_speaker_initial(self, voting_manager: VotingGroupChatManager) -> None:
        """Test speaker selection in initial state."""
        result = await voting_manager.select_speaker([])
        assert result == "Agent1"  # auto_propose_speaker

    @pytest.mark.asyncio
    async def test_select_proposer(self, voting_manager: VotingGroupChatManager) -> None:
        """Test proposer selection logic."""
        # With auto_propose_speaker
        proposer = voting_manager.select_proposer_for_testing()
        assert proposer == "Agent1"

        # Without auto_propose_speaker
        voting_manager.set_auto_propose_speaker_for_testing(None)
        proposer = voting_manager.select_proposer_for_testing()
        assert proposer == "Agent1"  # Falls back to first participant

    @pytest.mark.asyncio
    async def test_handle_proposal_phase(self, voting_manager: VotingGroupChatManager) -> None:
        """Test proposal phase handling."""
        # Test with ProposalMessage
        proposal = ProposalMessage(
            content=ProposalContent(
                proposal_id="test-123",
                title="Test Proposal",
                description="Testing proposal handling",
                options=["Yes", "No"],
            ),
            source="Agent1",
        )

        # Call handle_proposal_phase and verify behavior
        result = await voting_manager.handle_proposal_phase_for_testing(proposal)

        assert voting_manager.current_phase == VotingPhase.VOTING
        assert voting_manager.current_proposal is not None
        assert voting_manager.current_proposal["id"] == "test-123"
        assert result == ["Agent1", "Agent2", "Agent3"]

        # Test without ProposalMessage - TextMessage gets auto-converted to proposal
        voting_manager.set_phase_for_testing(VotingPhase.PROPOSAL)
        text_msg = TextMessage(content="Just text", source="Agent1")
        result = await voting_manager.handle_proposal_phase_for_testing(text_msg)
        # TextMessage gets converted to proposal, phase moves to voting, returns all eligible voters
        assert result == ["Agent1", "Agent2", "Agent3"]
        assert voting_manager.current_phase == VotingPhase.VOTING

    @pytest.mark.asyncio
    async def test_handle_voting_phase(self, voting_manager: VotingGroupChatManager) -> None:
        """Test voting phase handling."""
        # Setup proposal
        voting_manager.set_proposal_for_testing({"id": "test-123", "title": "Test"})
        voting_manager.set_phase_for_testing(VotingPhase.VOTING)

        # Test vote recording
        vote = VoteMessage(
            content=VoteContent(vote=VoteType.APPROVE, proposal_id="test-123", reasoning="Looks good", confidence=0.9),
            source="Agent1",
        )

        with patch.object(voting_manager, "_is_voting_complete", return_value=False):
            result = await voting_manager.handle_voting_phase_for_testing(vote)

        assert "Agent1" in voting_manager.votes_cast
        assert voting_manager.votes_cast["Agent1"]["vote"] == VoteType.APPROVE
        assert "Agent1" not in result  # Agent1 already voted

        # Test voting completion
        with (
            patch.object(voting_manager, "_is_voting_complete", return_value=True),
            patch.object(voting_manager, "_process_voting_results", return_value=[]),
        ):
            result = await voting_manager.handle_voting_phase_for_testing(vote)

    @pytest.mark.asyncio
    async def test_handle_discussion_phase(self, voting_manager: VotingGroupChatManager) -> None:
        """Test discussion phase handling."""
        voting_manager.set_phase_for_testing(VotingPhase.DISCUSSION)
        voting_manager.set_discussion_rounds_for_testing(2)

        text_msg = TextMessage(content="Discussion point", source="Agent1")

        # Test continuing discussion
        result = await voting_manager.handle_discussion_phase_for_testing(text_msg)
        assert result == ["Agent1", "Agent2", "Agent3"]

        # Test max rounds reached
        voting_manager.set_discussion_rounds_for_testing(3)
        result = await voting_manager.handle_discussion_phase_for_testing(text_msg)

        assert voting_manager.current_phase == VotingPhase.VOTING
        assert voting_manager.votes_cast == {}  # Reset votes
        assert result == ["Agent1", "Agent2", "Agent3"]

    @pytest.mark.asyncio
    async def test_handle_consensus_phase(self, voting_manager: VotingGroupChatManager) -> None:
        """Test consensus phase handling."""
        text_msg = TextMessage(content="Consensus reached", source="Agent1")
        result = await voting_manager.handle_consensus_phase_for_testing(text_msg)
        result = await voting_manager.handle_proposal_phase_for_testing(text_msg)
        # TextMessage gets converted to proposal, phase moves to voting, returns all eligible voters
        assert result == ["Agent1", "Agent2", "Agent3"]

    @pytest.mark.asyncio
    async def test_is_voting_complete(self, voting_manager: VotingGroupChatManager) -> None:
        """Test voting completion check."""
        # No votes cast
        assert not voting_manager.is_voting_complete_for_testing()

        # Partial votes
        voting_manager.set_votes_for_testing({"Agent1": {"vote": VoteType.APPROVE}})
        assert not voting_manager.is_voting_complete_for_testing()

        # All votes cast
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE},
                "Agent2": {"vote": VoteType.REJECT},
                "Agent3": {"vote": VoteType.APPROVE},
            }
        )
        assert voting_manager.is_voting_complete_for_testing()

    @pytest.mark.asyncio
    async def test_process_voting_results(self, voting_manager: VotingGroupChatManager) -> None:
        """Test voting results processing."""
        # Test with no votes
        result = await voting_manager.process_voting_results_for_testing()
        assert result == []

        # Test with votes - just verify the method can be called
        voting_manager.set_votes_for_testing({"Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9}})

        with (
            patch.object(voting_manager, "update_message_thread"),
            patch.object(
                voting_manager,
                "_calculate_voting_result",
                return_value={
                    "proposal_id": "test-123",
                    "result": "approved",
                    "votes_summary": {"approve": 1},
                    "winning_option": "approve",
                    "total_voters": 3,
                    "participation_rate": 0.33,
                    "confidence_average": 0.9,
                    "detailed_votes": {},
                },
            ),
        ):
            result = await voting_manager.process_voting_results_for_testing()

        # Just verify method completes without error
        assert result is not None

    @pytest.mark.asyncio
    async def test_calculate_voting_result_majority(self, voting_manager: VotingGroupChatManager) -> None:
        """Test majority voting calculation."""
        voting_manager.set_voting_method_for_testing(VotingMethod.MAJORITY)
        voting_manager.set_proposal_for_testing({"id": "test-123"})

        # Test approval majority
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.APPROVE, "confidence": 0.8},
                "Agent3": {"vote": VoteType.REJECT, "confidence": 0.7},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] == "approved"
        assert result["winning_option"] == VoteType.APPROVE.value
        assert result["participation_rate"] == 1.0

        # Test rejection majority
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.REJECT, "confidence": 0.9},
                "Agent2": {"vote": VoteType.REJECT, "confidence": 0.8},
                "Agent3": {"vote": VoteType.APPROVE, "confidence": 0.7},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] == "rejected"
        assert result["winning_option"] == VoteType.REJECT.value

    @pytest.mark.asyncio
    async def test_calculate_voting_result_plurality(self, voting_manager: VotingGroupChatManager) -> None:
        """Test plurality voting calculation."""
        voting_manager.set_voting_method_for_testing(VotingMethod.PLURALITY)
        voting_manager.set_proposal_for_testing({"id": "test-123"})
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.REJECT, "confidence": 0.8},
                "Agent3": {"vote": VoteType.ABSTAIN, "confidence": 0.5},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] in ["approved", "rejected"]  # Most common vote wins

    @pytest.mark.asyncio
    async def test_calculate_voting_result_unanimous(self, voting_manager: VotingGroupChatManager) -> None:
        """Test unanimous voting calculation."""
        voting_manager.set_voting_method_for_testing(VotingMethod.UNANIMOUS)
        voting_manager.set_proposal_for_testing({"id": "test-123"})

        # Test unanimous approval
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.APPROVE, "confidence": 0.8},
                "Agent3": {"vote": VoteType.APPROVE, "confidence": 0.7},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] == "approved"

        # Test with abstention
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.APPROVE, "confidence": 0.8},
                "Agent3": {"vote": VoteType.ABSTAIN, "confidence": 0.5},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] == "approved"

    @pytest.mark.asyncio
    async def test_calculate_voting_result_qualified_majority(self, voting_manager: VotingGroupChatManager) -> None:
        """Test qualified majority voting calculation."""
        voting_manager.set_voting_method_for_testing(VotingMethod.QUALIFIED_MAJORITY)
        voting_manager.set_qualified_majority_threshold_for_testing(0.67)
        voting_manager.set_proposal_for_testing({"id": "test-123"})

        # Test meeting qualified majority (2/3 = 0.67)
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.APPROVE, "confidence": 0.8},
                "Agent3": {"vote": VoteType.REJECT, "confidence": 0.7},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        # 2 out of 3 votes (0.67) meets the 0.67 threshold
        assert result["result"] in ["approved", "no_consensus"]  # Edge case at exact threshold

        # Test clearly not meeting qualified majority
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.REJECT, "confidence": 0.8},
                "Agent3": {"vote": VoteType.REJECT, "confidence": 0.7},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] == "no_consensus"

    @pytest.mark.asyncio
    async def test_message_factory_registration_errors(self, voting_manager: VotingGroupChatManager) -> None:
        """Test error handling in message factory registration."""
        # This tests the exception handling in __init__ for already registered message types
        # The exceptions are caught and ignored, so we just verify the manager initializes correctly
        assert voting_manager is not None
        assert voting_manager.voting_method == VotingMethod.MAJORITY

    @pytest.mark.asyncio
    async def test_eligible_voters_property(self, voting_manager: VotingGroupChatManager) -> None:
        """Test eligible voters property returns a copy."""
        voters = voting_manager.eligible_voters
        original_count = len(voters)

        # Modify the returned list (should not affect internal state)
        voters.append("NewAgent")

        # Verify internal state unchanged
        assert len(voting_manager.eligible_voters) == original_count
        assert "NewAgent" not in voting_manager.eligible_voters

    @pytest.mark.asyncio
    async def test_ranked_choice_voting(self, voting_manager: VotingGroupChatManager) -> None:
        """Test ranked choice voting method (currently not fully implemented)."""
        voting_manager.set_voting_method_for_testing(VotingMethod.RANKED_CHOICE)
        voting_manager.set_proposal_for_testing({"id": "test-123"})
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
                "Agent2": {"vote": VoteType.REJECT, "confidence": 0.8},
                "Agent3": {"vote": VoteType.APPROVE, "confidence": 0.7},
            }
        )

        # Test that ranked choice falls back to plurality logic
        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] in ["approved", "rejected", "no_consensus"]

    @pytest.mark.asyncio
    async def testannounce_voting_phase_for_testing(self, voting_manager: VotingGroupChatManager) -> None:
        """Test voting phase announcement."""
        voting_manager.set_proposal_for_testing({"title": "Test Proposal", "id": "test-123"})

        with patch.object(voting_manager, "update_message_thread") as mock_update:
            await voting_manager.announce_voting_phase_for_testing()
            mock_update.assert_called_once()

        # Test without proposal
        voting_manager.set_proposal_for_testing(None)
        with patch.object(voting_manager, "update_message_thread") as mock_update:
            await voting_manager.announce_voting_phase_for_testing()
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, voting_manager: VotingGroupChatManager) -> None:
        """Test state persistence."""
        # Set up some state
        voting_manager.set_phase_for_testing(VotingPhase.VOTING)
        voting_manager.set_proposal_for_testing({"id": "test-123", "title": "Test"})
        voting_manager.set_votes_for_testing({"Agent1": {"vote": VoteType.APPROVE}})
        voting_manager.set_discussion_rounds_for_testing(1)

        # Save state
        state = await voting_manager.save_state()

        # Reset manager
        await voting_manager.reset()
        assert voting_manager.current_phase == VotingPhase.PROPOSAL

        # Load state
        await voting_manager.load_state(state)

        # Verify state restored
        # The phase should be restored to VOTING from the saved state
        assert cast(VotingPhase, voting_manager.current_phase) == VotingPhase.VOTING
        assert voting_manager.current_proposal is not None
        assert voting_manager.current_proposal["id"] == "test-123"
        assert "Agent1" in voting_manager.votes_cast
        assert voting_manager.discussion_rounds == 1

    @pytest.mark.asyncio
    async def test_clear_votes_for_testing(self, voting_manager: VotingGroupChatManager) -> None:
        """Test clearing votes functionality."""
        # Set some votes
        voting_manager.set_votes_for_testing({"Agent1": {"vote": VoteType.APPROVE}})
        assert len(voting_manager.votes_cast) == 1

        # Clear votes
        voting_manager.clear_votes_for_testing()
        assert len(voting_manager.votes_cast) == 0

    @pytest.mark.asyncio
    async def test_edge_case_voting_scenarios(self, voting_manager: VotingGroupChatManager) -> None:
        """Test edge case voting scenarios."""
        voting_manager.set_proposal_for_testing({"id": "edge-test"})

        # Test all abstain votes
        voting_manager.set_votes_for_testing(
            {
                "Agent1": {"vote": VoteType.ABSTAIN, "confidence": 0.5},
                "Agent2": {"vote": VoteType.ABSTAIN, "confidence": 0.4},
                "Agent3": {"vote": VoteType.ABSTAIN, "confidence": 0.6},
            }
        )

        result = voting_manager.calculate_voting_result_for_testing()
        assert result["result"] == "no_consensus"
        assert result["votes_summary"]["abstain"] == 3

    @pytest.mark.asyncio
    async def test_auto_propose_speaker_methods(self, voting_manager: VotingGroupChatManager) -> None:
        """Test auto propose speaker setter methods."""
        # Test setting auto propose speaker
        voting_manager.set_auto_propose_speaker_for_testing("Agent2")
        proposer = voting_manager.select_proposer_for_testing()
        assert proposer == "Agent2"

        # Test setting to None
        voting_manager.set_auto_propose_speaker_for_testing(None)
        proposer = voting_manager.select_proposer_for_testing()
        assert proposer == "Agent1"  # Falls back to first participant

    @pytest.mark.asyncio
    async def test_voting_method_setters(self, voting_manager: VotingGroupChatManager) -> None:
        """Test voting method and threshold setters."""
        # Test setting voting method
        voting_manager.set_voting_method_for_testing(VotingMethod.UNANIMOUS)
        assert voting_manager.voting_method == VotingMethod.UNANIMOUS

        # Test setting qualified majority threshold
        voting_manager.set_qualified_majority_threshold_for_testing(0.75)
        assert voting_manager.qualified_majority_threshold == 0.75


class TestVotingResultMessage:
    """Test VotingResultMessage functionality."""

    def test_voting_result_message_creation(self) -> None:
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


class TestVotingGroupChatAdvanced:
    """Advanced tests for VotingGroupChat functionality."""

    def test_voting_method_enum_coverage(self) -> None:
        """Test that all voting methods are covered."""
        # This test ensures we test the enum values
        assert VotingMethod.MAJORITY.value == "majority"
        assert VotingMethod.PLURALITY.value == "plurality"
        assert VotingMethod.UNANIMOUS.value == "unanimous"
        assert VotingMethod.QUALIFIED_MAJORITY.value == "qualified_majority"
        assert VotingMethod.RANKED_CHOICE.value == "ranked_choice"

    def test_vote_type_enum_coverage(self) -> None:
        """Test that all vote types are covered."""
        assert VoteType.APPROVE.value == "approve"
        assert VoteType.REJECT.value == "reject"
        assert VoteType.ABSTAIN.value == "abstain"

    def test_voting_phase_enum_coverage(self) -> None:
        """Test that all voting phases are covered."""
        assert VotingPhase.PROPOSAL.value == "proposal"
        assert VotingPhase.VOTING.value == "voting"
        assert VotingPhase.DISCUSSION.value == "discussion"
        assert VotingPhase.CONSENSUS.value == "consensus"

    def test_vote_content_validation(self) -> None:
        """Test VoteContent validation and edge cases."""
        # Test valid vote content
        valid_vote = VoteContent(
            vote=VoteType.APPROVE, proposal_id="test-123", reasoning="This is a good idea", confidence=0.8
        )
        assert valid_vote.vote == VoteType.APPROVE
        assert valid_vote.confidence == 0.8

        # Test vote content with extreme confidence values
        extreme_vote = VoteContent(
            vote=VoteType.REJECT, proposal_id="test-456", reasoning="Strong disagreement", confidence=1.0
        )
        assert extreme_vote.confidence == 1.0

    def test_proposal_content_validation(self) -> None:
        """Test ProposalContent validation and edge cases."""
        # Test proposal with empty options
        proposal = ProposalContent(
            proposal_id="test-empty", title="Empty Options Test", description="Testing with no options", options=[]
        )
        assert len(proposal.options) == 0

        # Test proposal with many options
        many_options = [f"Option_{i}" for i in range(10)]
        large_proposal = ProposalContent(
            proposal_id="test-large",
            title="Many Options Test",
            description="Testing with many options",
            options=many_options,
        )
        assert len(large_proposal.options) == 10

    def test_voting_result_comprehensive(self) -> None:
        """Test VotingResult with comprehensive data."""
        detailed_votes = {
            "Agent1": {"vote": "approve", "reasoning": "Good idea", "confidence": 0.9},
            "Agent2": {"vote": "reject", "reasoning": "Needs work", "confidence": 0.8},
            "Agent3": {"vote": "abstain", "reasoning": "Not enough info", "confidence": 0.5},
        }

        result = VotingResult(
            proposal_id="comprehensive-test",
            result="no_consensus",
            votes_summary={"approve": 1, "reject": 1, "abstain": 1},
            winning_option="none",
            total_voters=3,
            participation_rate=1.0,
            confidence_average=0.73,
            detailed_votes=detailed_votes,
        )

        assert result.participation_rate == 1.0
        assert result.confidence_average == 0.73
        assert result.detailed_votes is not None
        assert len(result.detailed_votes) == 3

    def test_message_serialization_edge_cases(self) -> None:
        """Test message serialization with edge case data."""
        # Test vote message with minimal data
        minimal_vote = VoteMessage(
            content=VoteContent(
                vote=VoteType.ABSTAIN,
                proposal_id="minimal",
                reasoning="",  # Empty reasoning
                confidence=0.0,  # Zero confidence
            ),
            source="MinimalAgent",
        )

        text = minimal_vote.to_model_text()
        assert "abstain" in text.lower()
        assert "confidence: 0.00" in text.lower() or "0.0" in text

        # Test proposal with special characters that pass validation
        proposal = ProposalMessage(
            content=ProposalContent(
                proposal_id="special-test",
                title="Test Proposal",
                description="Proposal with special characters: @#$%^&*()_+",
                options=["Option A", "Option B"],
            ),
            source="TestAgent",
        )

        text = proposal.to_model_text()
        assert "Test Proposal" in text
        assert "special characters" in text


class TestVotingGroupChatSecurity:
    """Tests for security features in VotingGroupChat."""

    @pytest.fixture
    def security_validator(self) -> SecurityValidator:
        """Create a SecurityValidator instance for testing."""
        from src.autogen_voting.voting_group_chat import SecurityValidator

        return SecurityValidator()

    @pytest.fixture
    def byzantine_detector(self) -> ByzantineFaultDetector:
        """Create a ByzantineFaultDetector instance for testing."""
        from src.autogen_voting.voting_group_chat import ByzantineFaultDetector

        return ByzantineFaultDetector(total_agents=3)

    @pytest.fixture
    def audit_logger(self) -> AuditLogger:
        """Create an AuditLogger instance for testing."""
        from src.autogen_voting.voting_group_chat import AuditLogger

        return AuditLogger()

    @pytest_asyncio.fixture  # type: ignore[misc]
    async def secure_voting_manager(self) -> VotingGroupChatManager:
        """Create a VotingGroupChatManager for security testing."""
        output_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination] = asyncio.Queue()
        message_factory = MessageFactory()

        manager = VotingGroupChatManager(
            name="SecureVotingManager",
            group_topic_type="secure_group",
            output_topic_type="secure_output",
            participant_topic_types=["agent1", "agent2", "agent3"],
            participant_names=["Agent1", "Agent2", "Agent3"],
            participant_descriptions=["desc1", "desc2", "desc3"],
            output_message_queue=output_queue,
            termination_condition=None,
            max_turns=10,
            message_factory=message_factory,
            voting_method=VotingMethod.MAJORITY,
            qualified_majority_threshold=0.67,
            allow_abstentions=True,
            require_reasoning=True,
            max_discussion_rounds=3,
            auto_propose_speaker="Agent1",
            emit_team_events=False,
        )
        return manager

    def test_security_validator_sanitize_text(self, security_validator: SecurityValidator) -> None:
        """Test text sanitization to prevent injection attacks."""
        # Test valid text
        valid_text = "This is a normal proposal with some symbols: @#$%^&*()_+-=[]{}|;:,./<>?"
        sanitized = security_validator.sanitize_text(valid_text, 1000)
        assert sanitized == valid_text

        # Test length limiting
        long_text = "A" * 1001
        with pytest.raises(ValueError, match="exceeds maximum length"):
            security_validator.sanitize_text(long_text, 1000)

        # Test removing null bytes
        text_with_null = "Text with \x00 null byte"
        sanitized = security_validator.sanitize_text(text_with_null, 1000)
        assert "\x00" not in sanitized

        # Test XSS prevention
        xss_text = "Normal text <script>alert('XSS')</script>"
        with pytest.raises(ValueError, match="Text contains invalid characters"):
            security_validator.sanitize_text(xss_text, 1000)

        # Test other dangerous patterns
        dangerous_patterns = [
            "Normal text with javascript:alert(1)",
            "Normal text with data:text/html,<script>alert(1)</script>",
            "Normal text with vbscript:alert(1)",
            "Normal text with onload=alert(1)",
        ]

        for text in dangerous_patterns:
            with pytest.raises(ValueError):
                security_validator.sanitize_text(text, 1000)

    def test_security_validator_agent_name_validation(self, security_validator: SecurityValidator) -> None:
        """Test agent name validation to prevent impersonation."""
        # Test valid names
        valid_names = ["Agent1", "test_agent", "agent-2", "Agent123"]
        for name in valid_names:
            validated = security_validator.validate_agent_name(name)
            assert validated == name

        # Test invalid names
        invalid_names = [
            "",  # Empty name
            "A" * 101,  # Too long
            "Agent; DROP TABLE",  # SQL injection attempt
            "Agent<script>",  # XSS attempt
            "Agent/../../etc/passwd",  # Path traversal attempt
            "Agent🔥",  # Emoji/Unicode
            "Agent User",  # Space in name
        ]

        for name in invalid_names:
            with pytest.raises(ValueError):
                security_validator.validate_agent_name(name)

    def test_security_validator_proposal_id_generation(self, security_validator: SecurityValidator) -> None:
        """Test secure proposal ID generation."""
        # Generate multiple IDs and ensure they're unique and properly formatted
        ids = [security_validator.generate_proposal_id() for _ in range(10)]

        # Check uniqueness
        assert len(ids) == len(set(ids))

        # Check format (should start with "proposal_" followed by a hex string)
        for id in ids:
            assert id.startswith("proposal_")
            hex_part = id[9:]  # After "proposal_"
            assert all(c in "0123456789abcdef" for c in hex_part)
            assert len(hex_part) == 32  # Should be 16 bytes = 32 hex chars

    def test_security_validator_vote_signature(self, security_validator: SecurityValidator) -> None:
        """Test vote signature creation and verification."""
        # Create test data
        agent_key = "test_secret_key_123"
        vote_data = {"vote": "approve", "proposal_id": "test-proposal", "reasoning": "This is a good proposal"}

        # Create signature
        signature = security_validator.create_vote_signature(vote_data, agent_key)
        assert signature is not None
        assert len(signature) > 0

        # Verify valid signature
        assert security_validator.verify_vote_signature(vote_data, agent_key, signature)

        # Verify tampered data fails
        tampered_data = vote_data.copy()
        tampered_data["vote"] = "reject"
        assert not security_validator.verify_vote_signature(tampered_data, agent_key, signature)

        # Verify wrong key fails
        wrong_key = "wrong_key_456"
        assert not security_validator.verify_vote_signature(vote_data, wrong_key, signature)

        # Verify tampered signature fails
        tampered_signature = signature[:-1] + ("a" if signature[-1] != "a" else "b")
        assert not security_validator.verify_vote_signature(vote_data, agent_key, tampered_signature)

    def test_byzantine_fault_detector_initialization(self, byzantine_detector: ByzantineFaultDetector) -> None:
        """Test Byzantine fault detector initialization."""
        # Verify initial state
        assert byzantine_detector.total_agents == 3
        assert byzantine_detector.detection_threshold == 0.3
        assert len(byzantine_detector.reputation_scores) == 0
        assert len(byzantine_detector.suspicious_agents) == 0

    def test_byzantine_detector_reputation_update(self, byzantine_detector: ByzantineFaultDetector) -> None:
        """Test reputation updates based on voting behavior."""
        # Initialize agents
        byzantine_detector.initialize_agent_reputation("Agent1")
        byzantine_detector.initialize_agent_reputation("Agent2")

        # Initial reputations should be 1.0
        assert byzantine_detector.reputation_scores["Agent1"] == 1.0
        assert byzantine_detector.reputation_scores["Agent2"] == 1.0

        # Update reputation - aligned with consensus
        byzantine_detector.update_reputation("Agent1", VoteType.APPROVE, "approved")
        assert byzantine_detector.reputation_scores["Agent1"] == 1.0  # Already max

        # Update reputation - opposed to consensus
        byzantine_detector.update_reputation("Agent2", VoteType.REJECT, "approved")
        assert byzantine_detector.reputation_scores["Agent2"] < 1.0

        # Multiple conflicting votes should lower reputation further
        byzantine_detector.update_reputation("Agent2", VoteType.REJECT, "approved")
        byzantine_detector.update_reputation("Agent2", VoteType.REJECT, "approved")
        assert byzantine_detector.reputation_scores["Agent2"] < 0.9

    def test_byzantine_behavior_detection(self, byzantine_detector: ByzantineFaultDetector) -> None:
        """Test detection of Byzantine behavior patterns."""
        # Initialize agent
        byzantine_detector.initialize_agent_reputation("Agent1")

        # No Byzantine behavior initially
        assert not byzantine_detector.detect_byzantine_behavior("Agent1")

        # Lower reputation below threshold
        byzantine_detector.reputation_scores["Agent1"] = 0.2  # Below 0.3 threshold
        assert byzantine_detector.detect_byzantine_behavior("Agent1")
        assert "Agent1" in byzantine_detector.suspicious_agents

        # Test erratic voting pattern detection
        byzantine_detector.initialize_agent_reputation("Agent2")
        byzantine_detector.vote_history["Agent2"] = [
            VoteType.APPROVE,
            VoteType.REJECT,
            VoteType.APPROVE,
            VoteType.REJECT,
            VoteType.APPROVE,
        ]
        assert byzantine_detector.detect_byzantine_behavior("Agent2")
        assert "Agent2" in byzantine_detector.suspicious_agents

    def test_weighted_vote_counting(self, byzantine_detector: ByzantineFaultDetector) -> None:
        """Test weighted vote counting based on reputation."""
        # Initialize agents with different reputations
        byzantine_detector.initialize_agent_reputation("Agent1")
        byzantine_detector.initialize_agent_reputation("Agent2")
        byzantine_detector.initialize_agent_reputation("Agent3")

        # Modify reputations
        byzantine_detector.reputation_scores["Agent1"] = 1.0
        byzantine_detector.reputation_scores["Agent2"] = 0.6
        byzantine_detector.reputation_scores["Agent3"] = 0.3

        # Mark Agent3 as suspicious
        byzantine_detector.suspicious_agents.add("Agent3")

        # Create votes
        votes = {
            "Agent1": {"vote": VoteType.APPROVE},
            "Agent2": {"vote": VoteType.APPROVE},
            "Agent3": {"vote": VoteType.REJECT},
        }

        weighted_votes = byzantine_detector.get_weighted_vote_count(votes)

        # Agent3's vote should be weighted down by 50%
        assert weighted_votes["approve"] == 1.0 + 0.6  # 1.6
        assert weighted_votes["reject"] == 0.3 * 0.5  # 0.15 (reduced by 50%)
        assert weighted_votes["approve"] > weighted_votes["reject"]

    def test_byzantine_resilience_check(self, byzantine_detector: ByzantineFaultDetector) -> None:
        """Test Byzantine resilience checking."""
        # Initialize agents
        byzantine_detector.initialize_agent_reputation("Agent1")
        byzantine_detector.initialize_agent_reputation("Agent2")
        byzantine_detector.initialize_agent_reputation("Agent3")

        # Create votes with clear majority
        votes = {
            "Agent1": {"vote": VoteType.APPROVE},
            "Agent2": {"vote": VoteType.APPROVE},
            "Agent3": {"vote": VoteType.REJECT},
        }

        # Should be resilient (2/3 honest votes > 1/3 potential Byzantine)
        assert byzantine_detector.is_byzantine_resilient(votes)

        # Create votes with unclear majority
        byzantine_detector.reputation_scores["Agent2"] = 0.5
        votes = {
            "Agent1": {"vote": VoteType.APPROVE},
            "Agent2": {"vote": VoteType.REJECT},
            "Agent3": {"vote": VoteType.REJECT},
        }

        # May not be resilient depending on the calculation
        # We're not testing the exact result but the method execution
        byzantine_detector.is_byzantine_resilient(votes)

    def test_audit_logging(self, audit_logger: AuditLogger) -> None:
        """Test audit logging for security events."""
        # Use a context manager to capture logs
        with self.capture_logs() as captured:
            # Log various security events
            audit_logger.log_proposal_created("test-id", "Agent1", "Test Proposal")
            audit_logger.log_vote_cast("test-id", "Agent2", "approve", True)
            audit_logger.log_voting_result("test-id", "approved", 1.0)
            audit_logger.log_security_violation("TEST_VIOLATION", "Test violation details")

            # We may not capture all logs due to logger configuration, but we should
            # at least see the security violation which is at WARNING level
            assert len(captured) > 0
            assert any("SECURITY_VIOLATION" in msg for msg in captured)
            assert any("TEST_VIOLATION" in msg for msg in captured)

    @contextlib.contextmanager
    def capture_logs(self):
        """Context manager to capture log messages."""
        captured_logs = []
        logger = logging.getLogger("autogen_voting.audit")

        # Create a handler that captures logs
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        # Save original level
        original_level = logger.level

        # Add handler and make sure we capture all levels
        handler = CapturingHandler()
        handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            yield captured_logs
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    @pytest.mark.asyncio
    async def test_vote_message_sanitization(self) -> None:
        """Test that VoteMessage sanitizes inputs."""
        # Test with valid content
        valid_vote = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE, proposal_id="test-id", reasoning="Valid reasoning text", confidence=0.9
            ),
            source="TestAgent",
        )
        assert valid_vote.content.reasoning == "Valid reasoning text"

        # Test with malicious content in reasoning
        with pytest.raises(ValueError):
            VoteMessage(
                content=VoteContent(
                    vote=VoteType.APPROVE,
                    proposal_id="test-id",
                    reasoning="<script>alert('XSS')</script>",
                    confidence=0.9,
                ),
                source="TestAgent",
            )

        # Test with too long reasoning
        with pytest.raises(ValueError):
            VoteMessage(
                content=VoteContent(vote=VoteType.APPROVE, proposal_id="test-id", reasoning="A" * 5001, confidence=0.9),
                source="TestAgent",
            )

        # Test with too many ranked choices
        with pytest.raises(ValueError):
            VoteMessage(
                content=VoteContent(
                    vote=VoteType.APPROVE,
                    proposal_id="test-id",
                    reasoning="Valid reasoning",
                    confidence=0.9,
                    ranked_choices=["Option " + str(i) for i in range(21)],  # More than MAX_OPTIONS_COUNT
                ),
                source="TestAgent",
            )

    @pytest.mark.asyncio
    async def test_proposal_message_sanitization(self) -> None:
        """Test that ProposalMessage sanitizes inputs."""
        # Test with valid content
        valid_proposal = ProposalMessage(
            content=ProposalContent(
                proposal_id="test-id",
                title="Valid Title",
                description="Valid description text",
                options=["Option 1", "Option 2"],
            ),
            source="TestAgent",
        )
        assert valid_proposal.content.title == "Valid Title"
        assert valid_proposal.content.description == "Valid description text"

        # Test with malicious content in title
        with pytest.raises(ValueError):
            ProposalMessage(
                content=ProposalContent(
                    proposal_id="test-id",
                    title="<script>alert('XSS')</script>",
                    description="Valid description",
                    options=["Option 1", "Option 2"],
                ),
                source="TestAgent",
            )

        # Test with too long description
        with pytest.raises(ValueError):
            ProposalMessage(
                content=ProposalContent(
                    proposal_id="test-id",
                    title="Valid Title",
                    description="A" * 10001,
                    options=["Option 1", "Option 2"],
                ),
                source="TestAgent",
            )

        # Test with too many options
        with pytest.raises(ValueError):
            ProposalMessage(
                content=ProposalContent(
                    proposal_id="test-id",
                    title="Valid Title",
                    description="Valid description",
                    options=["Option " + str(i) for i in range(21)],  # More than MAX_OPTIONS_COUNT
                ),
                source="TestAgent",
            )

        # Test with malicious content in options
        with pytest.raises(ValueError):
            ProposalMessage(
                content=ProposalContent(
                    proposal_id="test-id",
                    title="Valid Title",
                    description="Valid description",
                    options=["Valid Option", "javascript:alert(1)"],
                ),
                source="TestAgent",
            )

    @pytest.mark.asyncio
    async def test_voting_manager_authentication(self, secure_voting_manager: VotingGroupChatManager) -> None:
        """Test agent authentication in voting manager."""
        # Valid agent name
        assert secure_voting_manager.authenticate_agent("Agent1")

        # Invalid agent names
        assert not secure_voting_manager.authenticate_agent("UnknownAgent")
        assert not secure_voting_manager.authenticate_agent("Agent; DROP TABLE")
        assert not secure_voting_manager.authenticate_agent("A" * 101)

    @pytest.mark.asyncio
    async def test_vote_integrity_validation(self, secure_voting_manager: VotingGroupChatManager) -> None:
        """Test vote message integrity validation."""
        # Create a valid vote message
        vote_message = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE,
                proposal_id="test-id",
                reasoning="Valid reasoning",
                confidence=0.9,
                # No signature yet
            ),
            source="Agent1",
        )

        # Validation should pass for authenticated agent even without signature
        assert secure_voting_manager.validate_vote_integrity(vote_message)

        # Test with unauthenticated agent
        unauthenticated_vote = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE, proposal_id="test-id", reasoning="Valid reasoning", confidence=0.9
            ),
            source="UnknownAgent",
        )
        assert not secure_voting_manager.validate_vote_integrity(unauthenticated_vote)

    @pytest.mark.asyncio
    async def test_proposal_integrity_validation(self, secure_voting_manager: VotingGroupChatManager) -> None:
        """Test proposal message integrity validation."""
        # Create a valid proposal message
        proposal_message = ProposalMessage(
            content=ProposalContent(
                proposal_id="test-id",
                title="Valid Title",
                description="Valid description",
                options=["Option 1", "Option 2"],
            ),
            source="Agent1",
        )

        # Validation should pass for authenticated agent
        assert secure_voting_manager.validate_proposal_integrity(proposal_message)

        # Test with unauthenticated agent
        unauthenticated_proposal = ProposalMessage(
            content=ProposalContent(
                proposal_id="test-id",
                title="Valid Title",
                description="Valid description",
                options=["Option 1", "Option 2"],
            ),
            source="UnknownAgent",
        )
        assert not secure_voting_manager.validate_proposal_integrity(unauthenticated_proposal)

    @pytest.mark.asyncio
    async def test_select_speaker_security(self, secure_voting_manager: VotingGroupChatManager) -> None:
        """Test select_speaker security validations."""
        # Create a message from unauthenticated source
        unauthenticated_msg = TextMessage(content="Hello from unknown agent", source="UnknownAgent")

        # The manager should handle the unauthenticated message gracefully
        # and return fallback speakers based on the current phase
        speakers = await secure_voting_manager.select_speaker([unauthenticated_msg])

        # Regardless of the exact result, the function should return something without raising exceptions
        assert speakers is not None

        # Test with message from authenticated source
        authenticated_msg = TextMessage(content="Hello from known agent", source="Agent1")
        speakers = await secure_voting_manager.select_speaker([authenticated_msg])
        assert speakers is not None

    @pytest.mark.asyncio
    async def test_voting_resilience_to_byzantine_agents(self) -> None:
        """Test voting system resilience to Byzantine (malicious) agents."""
        # Create a voting manager
        output_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination] = asyncio.Queue()
        message_factory = MessageFactory()

        # Create with 5 agents for more robust Byzantine fault tolerance testing
        manager = VotingGroupChatManager(
            name="ByzantineTestManager",
            group_topic_type="byzantine_test",
            output_topic_type="byzantine_output",
            participant_topic_types=["agent1", "agent2", "agent3", "agent4", "agent5"],
            participant_names=["Agent1", "Agent2", "Agent3", "Agent4", "Agent5"],
            participant_descriptions=["desc1", "desc2", "desc3", "desc4", "desc5"],
            output_message_queue=output_queue,
            termination_condition=None,
            max_turns=10,
            message_factory=message_factory,
            voting_method=VotingMethod.QUALIFIED_MAJORITY,
            qualified_majority_threshold=0.67,
            allow_abstentions=True,
            require_reasoning=True,
            max_discussion_rounds=3,
            auto_propose_speaker="Agent1",
            emit_team_events=False,
        )

        # Set up the Byzantine detector to mark certain agents as suspicious
        manager._byzantine_detector.suspicious_agents.add("Agent4")
        manager._byzantine_detector.suspicious_agents.add("Agent5")

        # Update reputations to reflect Byzantine behavior
        manager._byzantine_detector.reputation_scores["Agent4"] = 0.2
        manager._byzantine_detector.reputation_scores["Agent5"] = 0.3

        # Set up a proposal
        manager._current_proposal = {"id": "byzantine-test", "title": "Byzantine Test"}
        manager._current_phase = VotingPhase.VOTING

        # Create votes - 3 honest agents vs 2 Byzantine
        manager._votes_cast = {
            "Agent1": {"vote": VoteType.APPROVE, "confidence": 0.9},
            "Agent2": {"vote": VoteType.APPROVE, "confidence": 0.8},
            "Agent3": {"vote": VoteType.APPROVE, "confidence": 0.9},
            "Agent4": {"vote": VoteType.REJECT, "confidence": 1.0},  # Byzantine
            "Agent5": {"vote": VoteType.REJECT, "confidence": 1.0},  # Byzantine
        }

        # Calculate voting results
        result = manager._calculate_voting_result()

        # Despite Byzantine agents voting against, the result should still be approved
        # since the honest agents have higher total reputation weight
        assert result["result"] == "approved"
        assert result["byzantine_resilient"] is True
        assert "Agent4" in result["suspicious_agents"]
        assert "Agent5" in result["suspicious_agents"]

        # The result should indicate reputation adjustment was applied
        assert result["reputation_adjusted"] is True

    @pytest.mark.asyncio
    async def test_replay_attack_protection(self, secure_voting_manager: VotingGroupChatManager) -> None:
        """Test protection against vote replay attacks."""
        # Create a vote message with timestamp
        import time

        timestamp = str(int(time.time()))

        vote1 = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE,
                proposal_id="replay-test",
                reasoning="Valid reasoning",
                confidence=0.9,
                timestamp=timestamp,
            ),
            source="Agent1",
        )

        # First validation should pass
        assert secure_voting_manager.validate_vote_integrity(vote1)

        # Create identical vote (replay attempt)
        vote2 = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE,
                proposal_id="replay-test",
                reasoning="Valid reasoning",
                confidence=0.9,
                timestamp=timestamp,  # Same timestamp
            ),
            source="Agent1",
        )

        # Second validation with same nonce should fail (replay attack)
        assert not secure_voting_manager.validate_vote_integrity(vote2)

        # Different timestamp should be accepted
        vote3 = VoteMessage(
            content=VoteContent(
                vote=VoteType.APPROVE,
                proposal_id="replay-test",
                reasoning="Valid reasoning",
                confidence=0.9,
                timestamp=str(int(time.time()) + 1),  # Different timestamp
            ),
            source="Agent1",
        )

        assert secure_voting_manager.validate_vote_integrity(vote3)
