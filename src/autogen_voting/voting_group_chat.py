import asyncio
import logging
import secrets
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, cast

from autogen_agentchat.base import ChatAgent, Team, TerminationCondition
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MessageFactory,
    StructuredMessage,
    TextMessage,
)
from autogen_agentchat.state import BaseGroupChatManagerState
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.teams._group_chat._base_group_chat_manager import BaseGroupChatManager
from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from autogen_core import AgentRuntime, Component, ComponentModel
from pydantic import BaseModel, Field
from typing_extensions import Self

from .security import AuditLogger, CryptographicIntegrity, SecurityValidator

TRACE_LOGGER_NAME = "autogen_agentchat.trace"

# Set up logging for this module
logger = logging.getLogger(__name__)


class VotingMethod(str, Enum):
    """Supported voting methods for consensus building."""

    MAJORITY = "majority"  # >50% of votes required
    PLURALITY = "plurality"  # Most votes wins (simple)
    UNANIMOUS = "unanimous"  # All voters must agree
    QUALIFIED_MAJORITY = "qualified_majority"  # Configurable threshold (e.g., 2/3)
    RANKED_CHOICE = "ranked_choice"  # Ranked choice voting with elimination


class VoteType(str, Enum):
    """Types of votes that can be cast."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class VotingPhase(str, Enum):
    """Current phase of the voting process."""

    PROPOSAL = "proposal"  # Initial proposal or discussion
    VOTING = "voting"  # Collecting votes
    CONSENSUS = "consensus"  # Consensus reached
    DISCUSSION = "discussion"  # Additional discussion needed


class ByzantineFaultDetector:
    """Detects and mitigates Byzantine faults in voting systems."""

    def __init__(self, total_agents: int, detection_threshold: float = 0.3):
        self.total_agents = total_agents
        self.detection_threshold = detection_threshold
        self.reputation_scores: dict[str, float] = {}
        self.vote_history: dict[str, list[VoteType]] = {}
        self.suspicious_agents: set[str] = set()
        self.consensus_history: list[dict[str, Any]] = []

    def initialize_agent_reputation(self, agent_name: str) -> None:
        """Initialize reputation for a new agent."""
        self.reputation_scores[agent_name] = 1.0
        self.vote_history[agent_name] = []

    def update_reputation(self, agent_name: str, vote: VoteType, consensus_outcome: str) -> None:
        """Update agent reputation based on voting behavior."""
        if agent_name not in self.reputation_scores:
            self.initialize_agent_reputation(agent_name)

        # Track vote history
        self.vote_history[agent_name].append(vote)

        # Simple reputation update: reward consistency with consensus
        if consensus_outcome == "approved" and vote == VoteType.APPROVE:
            self.reputation_scores[agent_name] = min(1.0, self.reputation_scores[agent_name] + 0.1)
        elif consensus_outcome == "rejected" and vote == VoteType.REJECT:
            self.reputation_scores[agent_name] = min(1.0, self.reputation_scores[agent_name] + 0.1)
        else:
            # Penalize inconsistent voting (but not too harshly for legitimate disagreement)
            self.reputation_scores[agent_name] = max(0.1, self.reputation_scores[agent_name] - 0.05)

    def detect_byzantine_behavior(self, agent_name: str) -> bool:
        """Detect potential Byzantine behavior patterns."""
        if agent_name not in self.reputation_scores:
            return False

        reputation = self.reputation_scores[agent_name]
        vote_history = self.vote_history[agent_name]

        # Check reputation threshold
        if reputation < self.detection_threshold:
            self.suspicious_agents.add(agent_name)
            return True

        # Check for erratic voting patterns (too many rapid changes)
        if len(vote_history) >= 5:
            recent_votes = vote_history[-5:]
            # Count vote type changes
            changes = sum(1 for i in range(1, len(recent_votes)) if recent_votes[i] != recent_votes[i - 1])
            if changes >= 4:  # Changed vote type 4+ times in last 5 votes
                self.suspicious_agents.add(agent_name)
                return True

        return False

    def get_weighted_vote_count(self, votes: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate weighted vote counts based on reputation."""
        weighted_counts = {"approve": 0.0, "reject": 0.0, "abstain": 0.0}

        for agent_name, vote_data in votes.items():
            reputation = self.reputation_scores.get(agent_name, 1.0)
            vote_type = vote_data["vote"].value if hasattr(vote_data["vote"], "value") else str(vote_data["vote"])

            # Reduce weight for suspicious agents
            if agent_name in self.suspicious_agents:
                reputation *= 0.5

            weighted_counts[vote_type] += reputation

        return weighted_counts

    def is_byzantine_resilient(self, votes: dict[str, dict[str, Any]]) -> bool:
        """Check if voting result is resilient to Byzantine faults."""
        total_weight = sum(self.reputation_scores.get(name, 1.0) for name in votes.keys())

        # Estimate maximum Byzantine weight (assume up to 1/3 could be Byzantine)
        max_byzantine_weight = total_weight / 3

        weighted_counts = self.get_weighted_vote_count(votes)
        max_honest_votes = max(weighted_counts.values())

        # Result is resilient if honest majority exceeds potential Byzantine influence
        return max_honest_votes > max_byzantine_weight


trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

# Security constants
MAX_PROPOSAL_LENGTH = 10000
MAX_REASONING_LENGTH = 5000
MAX_OPTION_LENGTH = 500
MAX_OPTIONS_COUNT = 20


class VoteContent(BaseModel):
    vote: VoteType
    proposal_id: str
    reasoning: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ranked_choices: list[str] | None = None  # For ranked choice voting
    signature: str | None = None  # Cryptographic signature for integrity
    timestamp: str | None = None  # Vote timestamp for audit trail

    def model_post_init(self, __context: Any) -> None:
        """Validate and sanitize vote content after initialization."""
        if self.reasoning:
            self.reasoning = SecurityValidator.sanitize_text(self.reasoning, MAX_REASONING_LENGTH)

        if self.ranked_choices:
            if len(self.ranked_choices) > MAX_OPTIONS_COUNT:
                raise ValueError(f"Too many ranked choices (max {MAX_OPTIONS_COUNT})")
            self.ranked_choices = [
                SecurityValidator.sanitize_text(choice, MAX_OPTION_LENGTH) for choice in self.ranked_choices
            ]


class VoteMessage(StructuredMessage[VoteContent]):
    """Message containing a vote from an agent."""

    content: VoteContent

    def to_model_text(self) -> str:
        text = f"Vote: {self.content.vote.value}"
        if self.content.reasoning:
            text += f" - Reasoning: {self.content.reasoning}"
        if self.content.confidence < 1.0:
            text += f" (Confidence: {self.content.confidence:.2f})"
        return text


class ProposalContent(BaseModel):
    proposal_id: str
    title: str
    description: str
    options: list[str] = Field(default_factory=list)  # For multiple choice proposals
    requires_discussion: bool = False
    deadline: str | None = None
    created_timestamp: str | None = None  # Proposal creation timestamp
    proposer_signature: str | None = None  # Cryptographic signature from proposer

    def model_post_init(self, __context: Any) -> None:
        """Validate and sanitize proposal content after initialization."""
        # Auto-generate secure ID if not provided
        if not self.proposal_id:
            self.proposal_id = SecurityValidator.generate_proposal_id()

        # Validate and sanitize text fields
        self.title = SecurityValidator.sanitize_text(self.title, MAX_OPTION_LENGTH)
        self.description = SecurityValidator.sanitize_text(self.description, MAX_PROPOSAL_LENGTH)

        # Validate options
        if len(self.options) > MAX_OPTIONS_COUNT:
            raise ValueError(f"Too many options (max {MAX_OPTIONS_COUNT})")
        self.options = [SecurityValidator.sanitize_text(option, MAX_OPTION_LENGTH) for option in self.options]


class ProposalMessage(StructuredMessage[ProposalContent]):
    """Message containing a proposal for voting."""

    content: ProposalContent

    def to_model_text(self) -> str:
        text = f"Proposal: {self.content.title}\n{self.content.description}"
        if self.content.options:
            text += f"\nOptions: {', '.join(self.content.options)}"
        return text


class VotingResult(BaseModel):
    proposal_id: str
    result: Literal["approved", "rejected", "no_consensus"]
    votes_summary: dict[str, int]  # vote_type -> count
    winning_option: str | None = None
    total_voters: int
    participation_rate: float
    confidence_average: float
    detailed_votes: dict[str, dict[str, Any]] | None = None


class VotingResultMessage(StructuredMessage[VotingResult]):
    """Message containing voting results."""

    content: VotingResult

    def to_model_text(self) -> str:
        result = self.content
        text = f"Voting Result: {result.result.upper()}\n"
        text += f"Participation: {result.participation_rate:.1%} ({result.total_voters} voters)\n"
        text += f"Average Confidence: {result.confidence_average:.2f}\n"

        for vote_type, count in result.votes_summary.items():
            text += f"{vote_type}: {count} votes\n"

        if result.winning_option:
            text += f"Winning Option: {result.winning_option}"

        return text


class VotingManagerState(BaseGroupChatManagerState):
    """State for the voting group chat manager."""

    type: str = "VotingManagerState"
    current_phase: VotingPhase = VotingPhase.PROPOSAL
    current_proposal: dict[str, Any] | None = None
    votes_cast: dict[str, dict[str, Any]] = Field(default_factory=dict)  # agent_name -> vote_data
    eligible_voters: list[str] = Field(default_factory=list)
    discussion_rounds: int = 0
    max_discussion_rounds: int = 3


class VotingGroupChatManager(BaseGroupChatManager):
    """A group chat manager that enables democratic consensus through configurable voting mechanisms."""

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: list[str],
        participant_names: list[str],
        participant_descriptions: list[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
        voting_method: VotingMethod,
        qualified_majority_threshold: float,
        allow_abstentions: bool,
        require_reasoning: bool,
        max_discussion_rounds: int,
        auto_propose_speaker: str | None,
        emit_team_events: bool,
        metrics_collector: Any = None,  # Add metrics collector parameter
    ) -> None:
        super().__init__(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
            message_factory,
            emit_team_events,
        )

        # Voting configuration
        self._voting_method = voting_method
        self._qualified_majority_threshold = qualified_majority_threshold
        self._allow_abstentions = allow_abstentions
        self._require_reasoning = require_reasoning
        self._max_discussion_rounds = max_discussion_rounds
        self._auto_propose_speaker = auto_propose_speaker

        # Voting state
        self._current_phase = VotingPhase.PROPOSAL
        self._current_proposal: dict[str, Any] | None = None
        self._votes_cast: dict[str, dict[str, Any]] = {}
        self._eligible_voters = list(participant_names)
        self._discussion_rounds = 0

        # Security features
        self._agent_keys: dict[str, str] = {}  # Agent name -> secret key for authentication
        self._audit_logger: AuditLogger = AuditLogger()
        self._crypto_integrity = CryptographicIntegrity()
        self._vote_nonces: set[str] = set()  # Prevent replay attacks
        self._byzantine_detector = ByzantineFaultDetector(len(participant_names))

        # Initialize agent keys and reputation (in production, these should be provided securely)
        for name in participant_names:
            validated_name = SecurityValidator.validate_agent_name(name)
            self._agent_keys[validated_name] = secrets.token_hex(32)
            self._crypto_integrity.register_agent(validated_name)
            self._byzantine_detector.initialize_agent_reputation(validated_name)

        # Metrics collection
        self._metrics_collector: Any = metrics_collector
        logger.debug(f"MANAGER_INIT - Metrics collector: {type(metrics_collector) if metrics_collector else 'None'}")

        # Register custom message types (check if already registered)
        try:
            message_factory.register(VoteMessage)
        except ValueError:
            pass  # Already registered
        try:
            message_factory.register(ProposalMessage)
        except ValueError:
            pass  # Already registered
        try:
            message_factory.register(VotingResultMessage)
        except ValueError:
            pass  # Already registered

    # Public properties for testing and inspection
    @property
    def voting_method(self) -> VotingMethod:
        """Get the current voting method."""
        return self._voting_method

    @property
    def qualified_majority_threshold(self) -> float:
        """Get the qualified majority threshold."""
        return self._qualified_majority_threshold

    @property
    def allow_abstentions(self) -> bool:
        """Get whether abstentions are allowed."""
        return self._allow_abstentions

    @property
    def require_reasoning(self) -> bool:
        """Get whether reasoning is required for votes."""
        return self._require_reasoning

    @property
    def max_discussion_rounds(self) -> int:
        """Get the maximum number of discussion rounds."""
        return self._max_discussion_rounds

    @property
    def auto_propose_speaker(self) -> str | None:
        """Get the auto-propose speaker setting."""
        return self._auto_propose_speaker

    @property
    def current_phase(self) -> VotingPhase:
        """Get the current voting phase."""
        return self._current_phase

    @property
    def current_proposal(self) -> dict[str, Any] | None:
        """Get the current proposal."""
        return self._current_proposal

    @property
    def votes_cast(self) -> dict[str, dict[str, Any]]:
        """Get the votes that have been cast."""
        return self._votes_cast.copy()  # Return a copy to prevent external modification

    @property
    def discussion_rounds(self) -> int:
        """Get the current number of discussion rounds."""
        return self._discussion_rounds

    @property
    def eligible_voters(self) -> list[str]:
        """Get the list of eligible voters."""
        return self._eligible_voters.copy()  # Return a copy to prevent external modification

    # Test-specific accessors for Byzantine fault testing
    @property
    def byzantine_detector(self) -> ByzantineFaultDetector:
        """Get the Byzantine fault detector for testing."""
        return self._byzantine_detector

    def set_current_proposal_for_testing(self, proposal: dict[str, Any] | None) -> None:
        """Set current proposal for testing purposes."""
        self._current_proposal = proposal

    def set_current_phase_for_testing(self, phase: VotingPhase) -> None:
        """Set current phase for testing purposes."""
        self._current_phase = phase

    def set_votes_cast_for_testing(self, votes: dict[str, dict[str, Any]]) -> None:
        """Set votes cast for testing purposes."""
        self._votes_cast = votes

    def calculate_voting_result_for_testing(self) -> dict[str, Any]:
        """Calculate voting result for testing purposes."""
        return self._calculate_voting_result()

    def authenticate_agent(self, agent_name: str) -> bool:
        """Authenticate an agent for voting operations."""
        try:
            validated_name = SecurityValidator.validate_agent_name(agent_name)
            return validated_name in self._agent_keys
        except ValueError as e:
            self._audit_logger.log_security_violation("INVALID_AGENT_NAME", str(e))
            return False

    def validate_vote_integrity(self, vote_message: VoteMessage) -> bool:
        """Validate vote message integrity and authenticity."""
        try:
            voter_name = vote_message.source
            if not self.authenticate_agent(voter_name):
                return False

            # Check for signature if present
            if vote_message.content.signature:
                vote_data = {
                    "vote": vote_message.content.vote.value,
                    "proposal_id": vote_message.content.proposal_id,
                    "reasoning": vote_message.content.reasoning or "",
                }

                agent_key = self._agent_keys[voter_name]
                if not SecurityValidator.verify_vote_signature(vote_data, agent_key, vote_message.content.signature):
                    self._audit_logger.log_security_violation("INVALID_VOTE_SIGNATURE", f"voter={voter_name}")
                    return False

            # Check for replay attacks using nonce
            if vote_message.content.timestamp:
                nonce = f"{voter_name}:{vote_message.content.proposal_id}:{vote_message.content.timestamp}"
                if nonce in self._vote_nonces:
                    self._audit_logger.log_security_violation("REPLAY_ATTACK", f"nonce={nonce}")
                    return False
                self._vote_nonces.add(nonce)

            return True
        except Exception as e:
            self._audit_logger.log_security_violation("VOTE_VALIDATION_ERROR", str(e))
            return False

    def validate_proposal_integrity(self, proposal_message: ProposalMessage) -> bool:
        """Validate proposal message integrity and authenticity."""
        try:
            proposer_name = proposal_message.source
            if not self.authenticate_agent(proposer_name):
                return False

            # Additional proposal-specific validation could go here
            return True
        except Exception as e:
            self._audit_logger.log_security_violation("PROPOSAL_VALIDATION_ERROR", str(e))
            return False

    async def validate_group_state(self, messages: list[BaseChatMessage] | None) -> None:
        """Validate the group state for voting."""
        if len(self._participant_names) < 2:
            raise ValueError("Voting requires at least 2 participants.")

    async def reset(self) -> None:
        """Reset the voting manager state."""
        self._current_turn = 0
        self._message_thread.clear()
        if self._termination_condition is not None:
            await self._termination_condition.reset()

        # Reset voting state
        self._current_phase = VotingPhase.PROPOSAL
        self._current_proposal = None
        self._votes_cast = {}
        self._eligible_voters = list(self._participant_names)
        self._discussion_rounds = 0

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> list[str] | str:
        """Select speakers based on current voting phase and state with comprehensive error handling."""
        try:
            logger.debug(f"SELECT_SPEAKER - Phase: {self._current_phase.value}, Thread: {len(thread)} messages")

            if not thread:
                # Initial state - select proposer
                proposer = self._select_proposer()
                logger.debug(f"SELECT_SPEAKER - No thread, selected proposer: {proposer}")
                return proposer

            last_message = thread[-1]
            message_source = getattr(last_message, "source", "unknown")
            logger.debug(f"SELECT_SPEAKER - Last message: {type(last_message).__name__} from {message_source}")

            # Validate message source
            if message_source != "user" and not self.authenticate_agent(message_source):
                self._audit_logger.log_security_violation("UNAUTHENTICATED_MESSAGE", f"source={message_source}")
                # Return current phase appropriate speakers as fallback
                return self._get_fallback_speakers()

            # Track message for metrics (if not from user)
            if self._metrics_collector and message_source != "user":
                try:
                    logger.debug(f"SELECT_SPEAKER - Recording message from {message_source}")
                    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    message_content = getattr(last_message, "content", "")
                    estimated_tokens = len(str(message_content)) // 4
                    self._metrics_collector.record_message(message_source, estimated_tokens)
                    # Record API call (each agent message likely represents an API call)
                    self._metrics_collector.record_api_call(estimated_tokens)
                    logger.debug(f"SELECT_SPEAKER - Estimated {estimated_tokens} tokens, recorded API call")
                except Exception as e:
                    logger.debug(f"SELECT_SPEAKER - Metrics recording failed: {e}")

            # Handle different voting phases with error recovery
            try:
                if self._current_phase == VotingPhase.PROPOSAL:
                    result = await self._handle_proposal_phase(last_message)
                    logger.debug(f"SELECT_SPEAKER - Proposal phase result: {result}")
                    return result
                elif self._current_phase == VotingPhase.VOTING:
                    result = await self._handle_voting_phase(last_message)
                    logger.debug(f"SELECT_SPEAKER - Voting phase result: {result}")
                    return result
                elif self._current_phase == VotingPhase.DISCUSSION:
                    result = await self._handle_discussion_phase(last_message)
                    logger.debug(f"SELECT_SPEAKER - Discussion phase result: {result}")
                    return result
                else:  # VotingPhase.CONSENSUS
                    result = await self._handle_consensus_phase(last_message)
                    logger.debug(f"SELECT_SPEAKER - Consensus phase result: {result}")
                    return result
            except Exception as phase_error:
                self._audit_logger.log_security_violation(
                    "PHASE_HANDLING_ERROR", f"phase={self._current_phase.value}, error={str(phase_error)}"
                )
                logger.error(f"SELECT_SPEAKER - Phase handling failed: {phase_error}")
                return self._get_fallback_speakers()

        except Exception as e:
            # Critical error recovery
            self._audit_logger.log_security_violation("CRITICAL_SELECT_SPEAKER_ERROR", str(e))
            logger.critical(f"SELECT_SPEAKER - {e}")
            return self._get_fallback_speakers()

    def _get_fallback_speakers(self) -> list[str]:
        """Get fallback speakers based on current phase when errors occur."""
        try:
            if self._current_phase == VotingPhase.PROPOSAL:
                return [self._select_proposer()]
            elif self._current_phase == VotingPhase.VOTING:
                # Return voters who haven't voted yet
                remaining = [name for name in self._eligible_voters if name not in self._votes_cast]
                return remaining if remaining else []
            elif self._current_phase == VotingPhase.DISCUSSION:
                return self._participant_names
            else:  # VotingPhase.CONSENSUS
                return []
        except Exception:
            # Ultimate fallback
            return [self._participant_names[0]] if self._participant_names else []

    def _select_proposer(self) -> str:
        """Select who should make the initial proposal."""
        if self._auto_propose_speaker and self._auto_propose_speaker in self._participant_names:
            return self._auto_propose_speaker
        return self._participant_names[0]

    async def _handle_proposal_phase(self, last_message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Handle speaker selection during proposal phase."""
        logger.debug(
            f"PROPOSAL_PHASE - Message type: {type(last_message).__name__}, source: {getattr(last_message, 'source', 'unknown')}"
        )

        if isinstance(last_message, ProposalMessage):
            logger.debug("PROPOSAL_PHASE - Received ProposalMessage, validating integrity")

            # Validate proposal integrity and authenticity
            if not self.validate_proposal_integrity(last_message):
                logger.debug("PROPOSAL_PHASE - Proposal validation failed")
                self._audit_logger.log_security_violation("INVALID_PROPOSAL", f"proposer={last_message.source}")
                return [self._select_proposer()]  # Request new proposal

            logger.debug("PROPOSAL_PHASE - Proposal validated, transitioning to voting")

            # Proposal received, transition to voting
            self._current_proposal = {
                "id": last_message.content.proposal_id,
                "title": last_message.content.title,
                "description": last_message.content.description,
                "options": last_message.content.options,
            }
            self._current_phase = VotingPhase.VOTING
            logger.debug(f"PROPOSAL_PHASE - Set proposal: {self._current_proposal['title']}")

            # Log proposal creation
            self._audit_logger.log_proposal_created(
                last_message.content.proposal_id, last_message.source, last_message.content.title
            )

            # Announce voting phase
            await self._announce_voting_phase()

            # Return all eligible voters
            logger.debug(f"PROPOSAL_PHASE - Returning eligible voters: {self._eligible_voters}")
            return self._eligible_voters

        # Accept TextMessage as proposal and auto-convert
        elif isinstance(last_message, TextMessage) and last_message.source != "user":
            logger.debug("PROPOSAL_PHASE - Converting TextMessage to proposal")
            # Auto-convert TextMessage to proposal
            self._current_proposal = {
                "id": f"proposal_{len(self._message_thread)}",
                "title": "Decision Required",
                "description": last_message.content,
                "options": ["APPROVE", "REJECT"],
            }
            self._current_phase = VotingPhase.VOTING
            logger.debug(f"PROPOSAL_PHASE - Created proposal ID: {self._current_proposal['id']}")

            # Announce voting phase
            await self._announce_voting_phase()

            # Return all eligible voters
            logger.debug(f"PROPOSAL_PHASE - Returning eligible voters: {self._eligible_voters}")
            return self._eligible_voters

        # Still waiting for proposal
        proposer = self._select_proposer()
        logger.debug(f"PROPOSAL_PHASE - Still waiting, selecting proposer: {proposer}")
        return [proposer]

    async def _handle_voting_phase(self, last_message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Handle speaker selection during voting phase."""
        logger.debug(
            f"VOTING_PHASE - Message type: {type(last_message).__name__}, source: {getattr(last_message, 'source', 'unknown')}"
        )

        if isinstance(last_message, VoteMessage):
            # Validate vote integrity and authenticity
            voter_name = last_message.source
            logger.debug(f"VOTING_PHASE - Received VoteMessage from {voter_name}, validating integrity")

            if not self.validate_vote_integrity(last_message):
                logger.debug(f"VOTING_PHASE - Vote validation failed for {voter_name}")
                # Continue without recording invalid vote
                remaining_voters = [name for name in self._eligible_voters if name not in self._votes_cast]
                return remaining_voters if remaining_voters else []

            logger.debug(f"VOTING_PHASE - Vote validated for {voter_name}")

            # Record the vote
            if voter_name in self._eligible_voters:
                self._votes_cast[voter_name] = {
                    "vote": last_message.content.vote,
                    "reasoning": last_message.content.reasoning,
                    "confidence": last_message.content.confidence,
                    "ranked_choices": last_message.content.ranked_choices,
                    "signature": last_message.content.signature,
                    "timestamp": last_message.content.timestamp,
                }

                trace_logger.debug(f"Vote recorded from {voter_name}: {last_message.content.vote}")
                logger.debug(f"VOTING_PHASE - Recorded vote: {voter_name} -> {last_message.content.vote}")

                # Log vote casting
                self._audit_logger.log_vote_cast(
                    last_message.content.proposal_id,
                    voter_name,
                    last_message.content.vote.value,
                    last_message.content.signature is not None,
                )

                # Track vote for metrics
                if self._metrics_collector:
                    vote_value = (
                        last_message.content.vote.value
                        if hasattr(last_message.content.vote, "value")
                        else str(last_message.content.vote)
                    )
                    logger.debug(f"VOTING_PHASE - Recording VoteMessage in metrics: {voter_name} -> {vote_value}")
                    self._metrics_collector.record_vote(voter_name, vote_value)

                    # Track abstentions and reasoning
                    if last_message.content.vote == VoteType.ABSTAIN:
                        self._metrics_collector.current_metrics.abstentions += 1
                    reasoning = last_message.content.reasoning
                    if reasoning and len(reasoning.strip()) > 10:  # Has meaningful reasoning
                        self._metrics_collector.current_metrics.reasoning_provided = True

        # Accept TextMessage as vote and auto-convert
        elif isinstance(last_message, TextMessage) and last_message.source in self._eligible_voters:
            voter_name = last_message.source
            logger.debug(f"VOTING_PHASE - Processing TextMessage from voter {voter_name}")

            # Parse vote from text content
            content = last_message.content.lower()
            vote_type = None
            reasoning = last_message.content

            if any(word in content for word in ["approve", "accept", "yes", "support", "agree"]):
                vote_type = VoteType.APPROVE
            elif any(word in content for word in ["reject", "deny", "no", "oppose", "disagree"]):
                vote_type = VoteType.REJECT
            elif any(word in content for word in ["abstain", "neutral", "pass"]):
                vote_type = VoteType.ABSTAIN

            logger.debug(f"VOTING_PHASE - Parsed vote type: {vote_type}")
            if vote_type and voter_name not in self._votes_cast:
                self._votes_cast[voter_name] = {
                    "vote": vote_type,
                    "reasoning": reasoning,
                    "confidence": 1.0,
                    "ranked_choices": None,
                }
                logger.debug(f"VOTING_PHASE - Recorded parsed vote: {voter_name} -> {vote_type.value}")

                # Track vote for metrics
                if self._metrics_collector:
                    logger.debug(f"VOTING_PHASE - Recording vote in metrics: {voter_name} -> {vote_type.value}")
                    self._metrics_collector.record_vote(voter_name, vote_type.value)

                    # Track abstentions and reasoning
                    if vote_type == VoteType.ABSTAIN:
                        self._metrics_collector.current_metrics.abstentions += 1
                    if reasoning and len(reasoning.strip()) > 10:  # Has meaningful reasoning
                        self._metrics_collector.current_metrics.reasoning_provided = True

            elif voter_name in self._votes_cast:
                logger.debug(f"VOTING_PHASE - {voter_name} already voted")
            else:
                logger.debug(f"VOTING_PHASE - Could not parse vote from {voter_name}")

        # Check if voting is complete
        votes_cast_count = len(self._votes_cast)
        eligible_count = len(self._eligible_voters)
        logger.debug(f"VOTING_PHASE - Vote progress: {votes_cast_count}/{eligible_count}")
        logger.debug(f"VOTING_PHASE - Votes cast: {list(self._votes_cast.keys())}")

        if self._is_voting_complete():
            logger.debug("VOTING_PHASE - Voting complete, processing results")
            return await self._process_voting_results()

        # Return voters who haven't voted yet
        remaining_voters = [name for name in self._eligible_voters if name not in self._votes_cast]
        logger.debug(f"VOTING_PHASE - Remaining voters: {remaining_voters}")
        return remaining_voters if remaining_voters else []

    async def _handle_discussion_phase(self, last_message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Handle speaker selection during discussion phase."""

        # Allow open discussion among all participants
        # After sufficient discussion, transition back to voting

        if self._discussion_rounds >= self._max_discussion_rounds:
            # Reset votes and start new voting round
            self._votes_cast = {}
            self._current_phase = VotingPhase.VOTING
            await self._announce_voting_phase()
            return self._eligible_voters

        # Continue discussion
        return self._participant_names

    async def _handle_consensus_phase(self, last_message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Handle speaker selection after consensus is reached."""
        # Consensus reached, no more speakers needed
        return []

    def _is_voting_complete(self) -> bool:
        """Check if all eligible voters have cast their votes."""
        return len(self._votes_cast) >= len(self._eligible_voters)

    async def _process_voting_results(self) -> list[str]:
        """Process voting results and determine outcome."""
        logger.debug(f"PROCESS_RESULTS - Processing {len(self._votes_cast)} votes")

        if not self._votes_cast:
            logger.debug("PROCESS_RESULTS - No votes cast, returning empty")
            return []

        # Calculate results based on voting method
        result = self._calculate_voting_result()
        logger.debug(f"PROCESS_RESULTS - Calculated result: {result['result']}")
        logger.debug(f"PROCESS_RESULTS - Vote summary: {result['votes_summary']}")

        # Create and send result message
        result_message = VotingResultMessage(content=VotingResult(**result), source=self._name)
        logger.debug("PROCESS_RESULTS - Created result message")

        # Log voting result
        self._audit_logger.log_voting_result(result["proposal_id"], result["result"], result["participation_rate"])

        await self.update_message_thread([result_message])
        logger.debug("PROCESS_RESULTS - Updated message thread")

        # Determine next phase
        if result["result"] == "no_consensus" and self._discussion_rounds < self._max_discussion_rounds:
            self._current_phase = VotingPhase.DISCUSSION
            self._discussion_rounds += 1
            logger.debug(f"PROCESS_RESULTS - Moving to discussion phase, round {self._discussion_rounds}")

            # Track discussion rounds in metrics
            if self._metrics_collector:
                self._metrics_collector.current_metrics.discussion_rounds = self._discussion_rounds
                logger.debug(f"PROCESS_RESULTS - Updated metrics discussion rounds: {self._discussion_rounds}")

            return self._participant_names  # Open discussion
        else:
            self._current_phase = VotingPhase.CONSENSUS
            logger.debug("PROCESS_RESULTS - Moving to consensus phase, voting complete")
            return []  # End voting process

    def _calculate_voting_result(self) -> dict[str, Any]:
        """Calculate voting results based on the configured method with Byzantine fault tolerance and error handling."""
        try:
            # Check for Byzantine behavior and update reputations
            suspicious_detected: list[str] = []
            for agent_name in self._votes_cast.keys():
                try:
                    if self._byzantine_detector.detect_byzantine_behavior(agent_name):
                        self._audit_logger.log_security_violation("BYZANTINE_BEHAVIOR_DETECTED", f"agent={agent_name}")
                        suspicious_detected.append(agent_name)
                except Exception as e:
                    self._audit_logger.log_security_violation(
                        "BYZANTINE_DETECTION_ERROR", f"agent={agent_name}, error={str(e)}"
                    )

            # Get both regular and weighted vote counts with error handling
            vote_counts: Counter[str] = Counter[str]()
            weighted_counts: dict[str, float] = {"approve": 0.0, "reject": 0.0, "abstain": 0.0}

            try:
                vote_counts = Counter[str](vote_data["vote"].value for vote_data in self._votes_cast.values())
                weighted_counts = self._byzantine_detector.get_weighted_vote_count(self._votes_cast)
            except Exception as e:
                self._audit_logger.log_security_violation("VOTE_COUNT_ERROR", str(e))
                # Fallback to simple counting - use default values initialized above
                for vote_data in self._votes_cast.values():
                    try:
                        vote_value = (
                            vote_data["vote"].value if hasattr(vote_data["vote"], "value") else str(vote_data["vote"])
                        )
                        vote_counts[vote_value] += 1
                        weighted_counts[vote_value] += 1.0
                    except Exception as ex:
                        self._audit_logger.log_security_violation("VOTE_VALUE_ERROR", str(ex))
                        # Continue processing other votes

            total_votes = len(self._votes_cast)

            # Calculate confidence scores with error handling
            try:
                confidence_scores = [
                    float(vote_data.get("confidence", 1.0))
                    for vote_data in self._votes_cast.values()
                    if isinstance(vote_data.get("confidence"), int | float)
                ]
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            except Exception:
                avg_confidence = 1.0  # Default confidence

            # Check Byzantine resilience with error handling
            try:
                is_resilient = self._byzantine_detector.is_byzantine_resilient(self._votes_cast)
            except Exception as e:
                self._audit_logger.log_security_violation("BYZANTINE_RESILIENCE_CHECK_ERROR", str(e))
                is_resilient = False  # Conservative assumption

            # Use weighted votes for calculation if Byzantine threats detected
            use_weighted = len(self._byzantine_detector.suspicious_agents) > 0
            try:
                effective_counts: dict[str, float] = (
                    weighted_counts if use_weighted else {k: float(v) for k, v in vote_counts.items()}
                )
            except Exception:
                effective_counts = {k: float(v) for k, v in vote_counts.items()}
                use_weighted = False

            # Determine result based on voting method with error handling
            result = "no_consensus"
            winning_option: str | None = None

            try:
                if self._voting_method == VotingMethod.MAJORITY:
                    approve_count = effective_counts.get(VoteType.APPROVE.value, 0.0)
                    total_effective_votes = sum(effective_counts.values())
                    if total_effective_votes > 0:
                        if approve_count > total_effective_votes / 2:
                            result = "approved"
                            winning_option = VoteType.APPROVE.value
                        elif effective_counts.get(VoteType.REJECT.value, 0.0) > total_effective_votes / 2:
                            result = "rejected"
                            winning_option = VoteType.REJECT.value

                elif self._voting_method == VotingMethod.PLURALITY:
                    if effective_counts and sum(effective_counts.values()) > 0:
                        max_count = max(effective_counts.values())
                        winning_option = next(
                            vote_type for vote_type, count in effective_counts.items() if count == max_count
                        )
                        result = "approved" if winning_option == VoteType.APPROVE.value else "rejected"

                elif self._voting_method == VotingMethod.UNANIMOUS:
                    # For unanimous, we need all non-abstaining agents to agree (considering reputation)
                    non_abstain_votes = {
                        agent: vote_data["vote"]
                        for agent, vote_data in self._votes_cast.items()
                        if vote_data["vote"] != VoteType.ABSTAIN
                        and self._byzantine_detector.reputation_scores.get(agent, 1.0) > 0.5
                    }

                    if non_abstain_votes and len(set(non_abstain_votes.values())) == 1:
                        winning_vote = next(iter(non_abstain_votes.values()))
                        result = "approved" if winning_vote == VoteType.APPROVE else "rejected"
                        winning_option = winning_vote.value

                elif self._voting_method == VotingMethod.QUALIFIED_MAJORITY:
                    approve_count = effective_counts.get(VoteType.APPROVE.value, 0.0)
                    total_effective_votes = sum(effective_counts.values())
                    if total_effective_votes > 0:
                        if approve_count >= total_effective_votes * self._qualified_majority_threshold:
                            result = "approved"
                            winning_option = VoteType.APPROVE.value
                        elif (
                            effective_counts.get(VoteType.REJECT.value, 0.0)
                            >= total_effective_votes * self._qualified_majority_threshold
                        ):
                            result = "rejected"
                            winning_option = VoteType.REJECT.value

            except Exception as e:
                self._audit_logger.log_security_violation(
                    "VOTING_METHOD_CALCULATION_ERROR", f"method={self._voting_method.value}, error={str(e)}"
                )
                result = "no_consensus"  # Safe fallback
                winning_option = None

            # Update agent reputations based on consensus outcome with error handling
            try:
                for agent_name, vote_data in self._votes_cast.items():
                    try:
                        self._byzantine_detector.update_reputation(agent_name, vote_data["vote"], result)
                    except Exception as e:
                        self._audit_logger.log_security_violation(
                            "REPUTATION_UPDATE_ERROR", f"agent={agent_name}, error={str(e)}"
                        )
            except Exception as e:
                self._audit_logger.log_security_violation("REPUTATION_UPDATE_BATCH_ERROR", str(e))

            # Prepare result with error handling
            try:
                proposal_id = self._current_proposal["id"] if self._current_proposal else "unknown"
                participation_rate = total_votes / len(self._eligible_voters) if self._eligible_voters else 0.0

                return {
                    "proposal_id": proposal_id,
                    "result": result,
                    "votes_summary": dict(vote_counts),
                    "weighted_votes_summary": dict(weighted_counts) if use_weighted else None,
                    "winning_option": winning_option,
                    "total_voters": len(self._eligible_voters),
                    "participation_rate": participation_rate,
                    "confidence_average": avg_confidence,
                    "detailed_votes": dict(self._votes_cast.items()),
                    "byzantine_resilient": is_resilient,
                    "suspicious_agents": list(self._byzantine_detector.suspicious_agents),
                    "reputation_adjusted": use_weighted,
                }
            except Exception as e:
                self._audit_logger.log_security_violation("RESULT_PREPARATION_ERROR", str(e))
                # Return minimal safe result
                return {
                    "proposal_id": "error",
                    "result": "no_consensus",
                    "votes_summary": {},
                    "winning_option": None,
                    "total_voters": len(self._eligible_voters) if hasattr(self, "_eligible_voters") else 0,
                    "participation_rate": 0.0,
                    "confidence_average": 0.0,
                    "detailed_votes": {},
                    "byzantine_resilient": False,
                    "suspicious_agents": [],
                    "reputation_adjusted": False,
                }

        except Exception as e:
            # Critical error fallback
            self._audit_logger.log_security_violation("CRITICAL_CALCULATION_ERROR", str(e))
            return {
                "proposal_id": "critical_error",
                "result": "no_consensus",
                "votes_summary": {},
                "winning_option": None,
                "total_voters": 0,
                "participation_rate": 0.0,
                "confidence_average": 0.0,
                "detailed_votes": {},
                "byzantine_resilient": False,
                "suspicious_agents": [],
                "reputation_adjusted": False,
            }

    async def _announce_voting_phase(self) -> None:
        """Announce the start of voting phase."""
        if self._current_proposal:
            announcement = TextMessage(
                content=f"Voting has begun for proposal: {self._current_proposal['title']}\n"
                f"Voting method: {self._voting_method.value}\n"
                f"Please cast your votes using VoteMessage.",
                source=self._name,
            )
            await self.update_message_thread([announcement])

    async def save_state(self) -> Mapping[str, Any]:
        """Save the voting manager state."""
        state = VotingManagerState(
            message_thread=[msg.dump() for msg in self._message_thread],
            current_turn=self._current_turn,
            current_phase=self._current_phase,
            current_proposal=self._current_proposal,
            votes_cast=self._votes_cast,
            eligible_voters=self._eligible_voters,
            discussion_rounds=self._discussion_rounds,
            max_discussion_rounds=self._max_discussion_rounds,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the voting manager state."""
        voting_state = VotingManagerState.model_validate(state)
        self._message_thread = [self._message_factory.create(msg) for msg in voting_state.message_thread]
        self._current_turn = voting_state.current_turn
        self._current_phase = voting_state.current_phase
        self._current_proposal = voting_state.current_proposal
        self._votes_cast = voting_state.votes_cast or {}
        self._eligible_voters = voting_state.eligible_voters or list(self._participant_names)
        self._discussion_rounds = voting_state.discussion_rounds
        self._max_discussion_rounds = voting_state.max_discussion_rounds

    # Methods below are used only for testing
    def select_proposer_for_testing(self) -> str:
        """Wrapper for _select_proposer for testing."""
        return self._select_proposer()

    def set_phase_for_testing(self, phase: VotingPhase) -> None:
        """Set the voting phase for testing."""
        self._current_phase = phase

    def set_proposal_for_testing(self, proposal: dict[str, Any] | None) -> None:
        """Set the current proposal for testing."""
        self._current_proposal = proposal

    def set_votes_for_testing(self, votes: dict[str, dict[str, Any]]) -> None:
        """Set the votes cast for testing."""
        self._votes_cast = votes

    def set_discussion_rounds_for_testing(self, rounds: int) -> None:
        """Set the discussion rounds for testing."""
        self._discussion_rounds = rounds

    def set_participant_names_for_testing(self, names: list[str]) -> None:
        """Set the participant names for testing."""
        self._participant_names = names

    def set_auto_propose_speaker_for_testing(self, speaker: str | None) -> None:
        """Set the auto propose speaker for testing."""
        self._auto_propose_speaker = speaker

    def set_voting_method_for_testing(self, method: VotingMethod) -> None:
        """Set the voting method for testing."""
        self._voting_method = method

    def set_qualified_majority_threshold_for_testing(self, threshold: float) -> None:
        """Set the qualified majority threshold for testing."""
        self._qualified_majority_threshold = threshold

    async def handle_proposal_phase_for_testing(self, message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Wrapper for _handle_proposal_phase for testing."""
        return await self._handle_proposal_phase(message)

    async def handle_voting_phase_for_testing(self, message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Wrapper for _handle_voting_phase for testing."""
        return await self._handle_voting_phase(message)

    async def handle_discussion_phase_for_testing(self, message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Wrapper for _handle_discussion_phase for testing."""
        return await self._handle_discussion_phase(message)

    async def handle_consensus_phase_for_testing(self, message: BaseAgentEvent | BaseChatMessage) -> list[str]:
        """Wrapper for _handle_consensus_phase for testing."""
        return await self._handle_consensus_phase(message)

    def is_voting_complete_for_testing(self) -> bool:
        """Wrapper for _is_voting_complete for testing."""
        return self._is_voting_complete()

    async def process_voting_results_for_testing(self) -> list[str]:
        """Wrapper for _process_voting_results for testing."""
        return await self._process_voting_results()

    async def announce_voting_phase_for_testing(self) -> None:
        """Wrapper for _announce_voting_phase for testing."""
        await self._announce_voting_phase()

    def clear_votes_for_testing(self) -> None:
        """Clear votes for testing."""
        self._votes_cast = {}


class VotingGroupChatConfig(BaseModel):
    """Configuration for VotingGroupChat."""

    participants: list[ComponentModel]
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    voting_method: VotingMethod = VotingMethod.MAJORITY
    qualified_majority_threshold: float = Field(default=0.67, ge=0.5, le=1.0)
    allow_abstentions: bool = True
    require_reasoning: bool = False
    max_discussion_rounds: int = 3
    auto_propose_speaker: str | None = None
    emit_team_events: bool = False


class VotingGroupChat(BaseGroupChat, Component[VotingGroupChatConfig]):
    """A group chat team that enables democratic consensus through configurable voting mechanisms.

    Perfect for code reviews, architecture decisions, content moderation, and any scenario
    requiring group consensus with transparent decision-making processes.

    Args:
        participants (List[ChatAgent]): The agents participating in the voting process.
        voting_method (VotingMethod, optional): Method used for determining consensus. Defaults to VotingMethod.MAJORITY.
        qualified_majority_threshold (float, optional): Threshold for qualified majority voting (0.5-1.0). Defaults to 0.67.
        allow_abstentions (bool, optional): Whether agents can abstain from voting. Defaults to True.
        require_reasoning (bool, optional): Whether votes must include reasoning. Defaults to False.
        max_discussion_rounds (int, optional): Maximum rounds of discussion before final decision. Defaults to 3.
        auto_propose_speaker (str, optional): Agent name to automatically select as proposer. Defaults to None.
        termination_condition (TerminationCondition, optional): Condition for ending the chat. Defaults to None.
        max_turns (int, optional): Maximum number of turns before forcing termination. Defaults to None.
        runtime (AgentRuntime, optional): The agent runtime to use. Defaults to None.
        custom_message_types (List[type[BaseAgentEvent | BaseChatMessage]], optional): Additional message types for the chat. Defaults to None.
        emit_team_events (bool, optional): Whether to emit team events for UI integration. Defaults to False.

    Raises:
        ValueError: If fewer than 2 participants, invalid thresholds, or missing auto_propose_speaker.

    Examples:

    Code review voting with qualified majority:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_voting import VotingGroupChat, VotingMethod
            from autogen_agentchat.conditions import MaxMessageTermination


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # Create reviewers with different expertise
                senior_dev = AssistantAgent(
                    "SeniorDev",
                    model_client,
                    system_message="Senior developer focused on architecture and best practices.",
                )
                security_expert = AssistantAgent(
                    "SecurityExpert", model_client, system_message="Security specialist reviewing for vulnerabilities."
                )
                performance_engineer = AssistantAgent(
                    "PerformanceEngineer",
                    model_client,
                    system_message="Performance engineer optimizing for speed and efficiency.",
                )

                # Create voting team for code review
                voting_team = VotingGroupChat(
                    participants=[senior_dev, security_expert, performance_engineer],
                    voting_method=VotingMethod.QUALIFIED_MAJORITY,
                    qualified_majority_threshold=0.67,
                    require_reasoning=True,
                    max_discussion_rounds=2,
                    termination_condition=MaxMessageTermination(20),
                )

                # Review code changes
                task = "Proposal: Approve code change for merge with caching implementation"

                result = await voting_team.run(task=task)
                print(result)


            asyncio.run(main())

    Architecture decision with unanimous consensus:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_voting import VotingGroupChat, VotingMethod
            from autogen_agentchat.conditions import MaxMessageTermination


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # Create architecture team
                tech_lead = AssistantAgent(
                    "TechLead", model_client, system_message="Technical lead with expertise in distributed systems."
                )
                solution_architect = AssistantAgent(
                    "SolutionArchitect",
                    model_client,
                    system_message="Solution architect focused on enterprise patterns.",
                )
                devops_engineer = AssistantAgent(
                    "DevOpsEngineer",
                    model_client,
                    system_message="DevOps engineer focused on deployment and operations.",
                )

                # Create voting team requiring unanimous consensus
                voting_team = VotingGroupChat(
                    participants=[tech_lead, solution_architect, devops_engineer],
                    voting_method=VotingMethod.UNANIMOUS,
                    max_discussion_rounds=3,
                    auto_propose_speaker="TechLead",
                    termination_condition=MaxMessageTermination(30),
                )

                task = "Proposal: Choose microservices communication pattern from available options"

                result = await voting_team.run(task=task)
                print(result)


            asyncio.run(main())

    Content moderation with simple majority:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_voting import VotingGroupChat, VotingMethod
            from autogen_agentchat.conditions import MaxMessageTermination


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # Create moderation team
                community_manager = AssistantAgent(
                    "CommunityManager",
                    model_client,
                    system_message="Community manager maintaining positive environment.",
                )
                safety_specialist = AssistantAgent(
                    "SafetySpecialist",
                    model_client,
                    system_message="Safety specialist focused on harmful content detection.",
                )
                legal_advisor = AssistantAgent(
                    "LegalAdvisor", model_client, system_message="Legal advisor focused on compliance and risk."
                )

                # Create voting team for content moderation
                voting_team = VotingGroupChat(
                    participants=[community_manager, safety_specialist, legal_advisor],
                    voting_method=VotingMethod.MAJORITY,
                    allow_abstentions=True,
                    max_discussion_rounds=1,
                    termination_condition=MaxMessageTermination(15),
                )

                task = "Proposal: Moderate user forum post about platform feedback"

                result = await voting_team.run(task=task)
                print(result)


            asyncio.run(main())
    """

    component_config_schema = VotingGroupChatConfig
    component_provider_override = "autogen_agentchat.teams.VotingGroupChat"

    def __init__(
        self,
        participants: list[ChatAgent],
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        qualified_majority_threshold: float = 0.67,
        allow_abstentions: bool = True,
        require_reasoning: bool = False,
        max_discussion_rounds: int = 3,
        auto_propose_speaker: str | None = None,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        custom_message_types: list[type[BaseAgentEvent | BaseChatMessage]] | None = None,
        emit_team_events: bool = False,
    ) -> None:
        # Validate participants
        if len(participants) < 2:
            raise ValueError("Voting requires at least 2 participants.")

        if auto_propose_speaker and auto_propose_speaker not in [p.name for p in participants]:
            raise ValueError(f"auto_propose_speaker '{auto_propose_speaker}' not found in participants.")

        if not (0.5 <= qualified_majority_threshold <= 1.0):
            raise ValueError("qualified_majority_threshold must be between 0.5 and 1.0")

        # Add voting message types to custom types
        voting_message_types: list[type[BaseAgentEvent | BaseChatMessage]] = [
            VoteMessage,
            ProposalMessage,
            VotingResultMessage,
        ]
        if custom_message_types:
            custom_message_types.extend(voting_message_types)
        else:
            custom_message_types = voting_message_types

        super().__init__(
            name="VotingGroupChat",
            description="A group chat team that enables democratic consensus through configurable voting mechanisms",
            participants=cast(list[ChatAgent | Team], participants),
            group_chat_manager_name="VotingGroupChatManager",
            group_chat_manager_class=VotingGroupChatManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types,
            emit_team_events=emit_team_events,
        )

        # Store voting configuration
        self._voting_method = voting_method
        self._qualified_majority_threshold = qualified_majority_threshold
        self._allow_abstentions = allow_abstentions
        self._require_reasoning = require_reasoning
        self._max_discussion_rounds = max_discussion_rounds
        self._auto_propose_speaker = auto_propose_speaker

        # Initialize metrics collector (will be set by benchmark runner)
        self._metrics_collector = None

    def set_metrics_collector(self, metrics_collector: Any) -> None:
        """Set the metrics collector for tracking performance data."""
        self._metrics_collector = metrics_collector
        logger.debug(f"VOTING_CHAT - Set metrics collector: {type(metrics_collector)}")

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: list[str],
        participant_names: list[str],
        participant_descriptions: list[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
    ) -> Callable[[], VotingGroupChatManager]:
        def _factory() -> VotingGroupChatManager:
            return VotingGroupChatManager(
                name=name,
                group_topic_type=group_topic_type,
                output_topic_type=output_topic_type,
                participant_topic_types=participant_topic_types,
                participant_names=participant_names,
                participant_descriptions=participant_descriptions,
                output_message_queue=output_message_queue,
                termination_condition=termination_condition,
                max_turns=max_turns,
                message_factory=message_factory,
                voting_method=self._voting_method,
                qualified_majority_threshold=self._qualified_majority_threshold,
                allow_abstentions=self._allow_abstentions,
                require_reasoning=self._require_reasoning,
                max_discussion_rounds=self._max_discussion_rounds,
                auto_propose_speaker=self._auto_propose_speaker,
                emit_team_events=self._emit_team_events,
                metrics_collector=getattr(self, "_metrics_collector", None),  # Pass metrics collector
            )

        return _factory

    def _to_config(self) -> VotingGroupChatConfig:
        """Convert to configuration object."""
        return VotingGroupChatConfig(
            participants=[participant.dump_component() for participant in self._participants],
            termination_condition=self._termination_condition.dump_component() if self._termination_condition else None,
            max_turns=self._max_turns,
            voting_method=self._voting_method,
            qualified_majority_threshold=self._qualified_majority_threshold,
            allow_abstentions=self._allow_abstentions,
            require_reasoning=self._require_reasoning,
            max_discussion_rounds=self._max_discussion_rounds,
            auto_propose_speaker=self._auto_propose_speaker,
            emit_team_events=self._emit_team_events,
        )

    @classmethod
    def _from_config(cls, config: VotingGroupChatConfig) -> Self:
        """Create from configuration object."""
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )

        return cls(
            participants=participants,
            voting_method=config.voting_method,
            qualified_majority_threshold=config.qualified_majority_threshold,
            allow_abstentions=config.allow_abstentions,
            require_reasoning=config.require_reasoning,
            max_discussion_rounds=config.max_discussion_rounds,
            auto_propose_speaker=config.auto_propose_speaker,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
            emit_team_events=config.emit_team_events,
        )
