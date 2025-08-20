"""
Core Voting Manager Implementation

Manages the voting process flow, participant coordination, and result calculation.
Refactored from VotingGroupChatManager with improved naming and structure.
"""

import asyncio
import logging
import secrets
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_voting_system import VoteMessage, ProposalMessage, VotingResultMessage

from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MessageFactory,
    TextMessage,
)
from autogen_agentchat.state import BaseGroupChatManagerState
from autogen_agentchat.teams._group_chat._base_group_chat_manager import BaseGroupChatManager
from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from pydantic import BaseModel, Field

from .voting_protocols import (
    VotingMethod, VoteType, VotingPhase, VotingResult
)
from ..security.cryptographic_services import (
    CryptographicIntegrity, SecurityValidator
)
from ..security.audit_framework import AuditLogger

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger("votingai.trace")


class ByzantineFaultDetector:
    """
    Detects and mitigates Byzantine faults in voting systems.
    
    Implements reputation-based detection algorithms to identify
    potentially malicious or faulty agents in the voting process.
    """

    def __init__(self, total_agents: int, detection_threshold: float = 0.3):
        self.total_agents = total_agents
        self.detection_threshold = detection_threshold
        self.reputation_scores: Dict[str, float] = {}
        self.vote_history: Dict[str, List[VoteType]] = {}
        self.suspicious_agents: set[str] = set()
        self.consensus_history: List[Dict[str, Any]] = []

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

    def get_weighted_vote_count(self, votes: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
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

    def is_byzantine_resilient(self, votes: Dict[str, Dict[str, Any]]) -> bool:
        """Check if voting result is resilient to Byzantine faults."""
        total_weight = sum(self.reputation_scores.get(name, 1.0) for name in votes.keys())

        # Estimate maximum Byzantine weight (assume up to 1/3 could be Byzantine)
        max_byzantine_weight = total_weight / 3

        weighted_counts = self.get_weighted_vote_count(votes)
        max_honest_votes = max(weighted_counts.values())

        # Result is resilient if honest majority exceeds potential Byzantine influence
        return max_honest_votes > max_byzantine_weight


class VotingManagerState(BaseGroupChatManagerState):
    """State management for the core voting manager."""

    type: str = "VotingManagerState"
    current_phase: VotingPhase = VotingPhase.PROPOSAL
    current_proposal: Optional[Dict[str, Any]] = None
    votes_cast: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # agent_name -> vote_data
    eligible_voters: List[str] = Field(default_factory=list)
    discussion_rounds: int = 0
    max_discussion_rounds: int = 3


class CoreVotingManager(BaseGroupChatManager):
    """
    Core voting manager that orchestrates the democratic consensus process.
    
    This class manages the voting workflow, from proposal creation through
    vote collection to consensus determination. It includes comprehensive
    security features and Byzantine fault tolerance.
    """

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: Optional[TerminationCondition],
        max_turns: Optional[int],
        message_factory: MessageFactory,
        voting_method: VotingMethod,
        qualified_majority_threshold: float,
        allow_abstentions: bool,
        require_reasoning: bool,
        max_discussion_rounds: int,
        auto_propose_speaker: Optional[str],
        emit_team_events: bool,
        metrics_collector: Optional[Any] = None,
        enable_audit_logging: bool = True,
        enable_file_logging: bool = False,
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
        self._current_proposal: Optional[Dict[str, Any]] = None
        self._votes_cast: Dict[str, Dict[str, Any]] = {}
        self._eligible_voters = list(participant_names)
        self._discussion_rounds = 0

        # Security features
        self._agent_keys: Dict[str, str] = {}  # Agent name -> secret key for authentication
        if enable_audit_logging:
            log_dir = "audit_logs" if enable_file_logging else None
            self._audit_logger: Optional[AuditLogger] = AuditLogger(log_directory=log_dir)
        else:
            self._audit_logger = None
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
        self._metrics_collector: Optional[Any] = metrics_collector
        logger.debug(f"CoreVotingManager initialized with {len(participant_names)} participants")

        # Register custom message types (check if already registered)
        from .base_voting_system import VoteMessage, ProposalMessage, VotingResultMessage
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

    # Public properties for inspection and testing
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
    def auto_propose_speaker(self) -> Optional[str]:
        """Get the auto-propose speaker setting."""
        return self._auto_propose_speaker

    @property
    def current_phase(self) -> VotingPhase:
        """Get the current voting phase."""
        return self._current_phase

    @property
    def current_proposal(self) -> Optional[Dict[str, Any]]:
        """Get the current proposal."""
        return self._current_proposal

    @property
    def votes_cast(self) -> Dict[str, Dict[str, Any]]:
        """Get the votes that have been cast."""
        return self._votes_cast.copy()  # Return a copy to prevent external modification

    @property
    def discussion_rounds(self) -> int:
        """Get the current number of discussion rounds."""
        return self._discussion_rounds

    @property
    def eligible_voters(self) -> List[str]:
        """Get the list of eligible voters."""
        return self._eligible_voters.copy()  # Return a copy to prevent external modification

    @property
    def byzantine_detector(self) -> ByzantineFaultDetector:
        """Get the Byzantine fault detector for testing."""
        return self._byzantine_detector

    def _log_security_violation(self, violation_type: str, details: str) -> None:
        """Helper method to log security violations if audit logging is enabled."""
        if self._audit_logger:
            self._audit_logger.log_security_violation(violation_type, details)

    def _log_proposal_created(self, proposal_id: str, agent_name: str, title: str) -> None:
        """Helper method to log proposal creation if audit logging is enabled."""
        if self._audit_logger:
            self._audit_logger.log_proposal_created(proposal_id, agent_name, title)

    def _log_vote_cast(self, proposal_id: str, agent_name: str, vote: str, is_valid: bool) -> None:
        """Helper method to log vote casting if audit logging is enabled."""
        if self._audit_logger:
            self._audit_logger.log_vote_cast(proposal_id, agent_name, vote, is_valid)

    def _log_voting_result(self, proposal_id: str, result: str, participation_rate: float) -> None:
        """Helper method to log voting result if audit logging is enabled."""
        if self._audit_logger:
            self._audit_logger.log_voting_result(proposal_id, result, participation_rate)

    async def validate_group_state(self, messages: Optional[List[BaseChatMessage]]) -> None:
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

    def _select_proposer(self) -> str:
        """Select who should make the initial proposal."""
        if self._auto_propose_speaker and self._auto_propose_speaker in self._participant_names:
            return self._auto_propose_speaker
        return self._participant_names[0]

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

    # ========================================================================================
    # CORE VOTING LOGIC IMPLEMENTATION
    # Adaptive consensus implementation with quality controls and fault tolerance
    # ========================================================================================

    async def select_speaker(
        self, 
        messages: List[BaseChatMessage],
        cancellation_token: Optional[Any] = None
    ) -> str:
        """
        Select the next speaker based on current voting phase and adaptive strategy.
        
        This method implements intelligent speaker selection that adapts to decision
        complexity and ensures balanced participation while maintaining quality.
        """
        if not messages:
            return self._select_proposer()

        last_message = messages[-1]
        current_speaker = last_message.source

        # Security validation - ensure speaker authentication
        if not self._authenticate_agent(current_speaker):
            logger.warning(f"Unauthenticated agent {current_speaker} attempted to participate")
            self._log_security_violation("UNAUTHORIZED_PARTICIPATION", f"Agent {current_speaker}")
            return self._select_proposer()

        # Phase-based speaker selection with adaptive intelligence
        if self._current_phase == VotingPhase.PROPOSAL:
            return await self._handle_proposal_phase(last_message)
        
        elif self._current_phase == VotingPhase.VOTING:
            return await self._handle_voting_phase(last_message)
        
        elif self._current_phase == VotingPhase.DISCUSSION:
            return await self._handle_discussion_phase(last_message)
        
        elif self._current_phase == VotingPhase.CONSENSUS:
            return await self._handle_consensus_phase(last_message)
        
        return self._select_proposer()

    async def _handle_proposal_phase(self, message: BaseChatMessage) -> str:
        """Handle proposal phase with semantic understanding and complexity analysis."""
        from ..intelligence.semantic_interpreter import SemanticVoteInterpreter
        from ..consensus.adaptive_strategies import AdaptiveStrategySelector
        
        from .base_voting_system import ProposalMessage
        
        if isinstance(message, ProposalMessage):
            # Structured proposal received - analyze complexity and select strategy
            proposal = message.content
            
            # Store proposal with security validation
            validated_proposal = {
                "id": proposal.proposal_id,
                "title": SecurityValidator().sanitize_text(proposal.title, 200),
                "description": SecurityValidator().sanitize_text(proposal.description, 2000),
                "options": [SecurityValidator().sanitize_text(opt, 100) for opt in proposal.options[:20]],
                "proposer": message.source,
                "timestamp": message.timestamp or "",
            }
            
            self._current_proposal = validated_proposal
            self._log_proposal_created(proposal.proposal_id, message.source, proposal.title)
            
            # Analyze decision complexity for adaptive strategy selection
            strategy_selector = AdaptiveStrategySelector()
            complexity = strategy_selector.classify_decision_complexity(
                proposal_text=f"{proposal.title} {proposal.description}",
                options_count=len(proposal.options),
                participant_count=len(self._eligible_voters)
            )
            
            # Select appropriate consensus strategy based on complexity
            consensus_strategy = strategy_selector.select_consensus_strategy(complexity)
            
            if consensus_strategy.needs_deliberation():
                # Complex decisions require discussion before voting
                self._current_phase = VotingPhase.DISCUSSION
                self._discussion_rounds = 0
                return self._select_discussion_facilitator()
            else:
                # Simple decisions can proceed directly to voting
                self._current_phase = VotingPhase.VOTING
                return self._select_next_voter()
        
        elif isinstance(message, TextMessage):
            # Convert unstructured text to proposal using semantic interpretation
            interpreter = SemanticVoteInterpreter()
            semantic_result = interpreter.interpret_proposal(message.content)
            
            if semantic_result.is_valid_proposal:
                # Auto-generate structured proposal from semantic understanding
                auto_proposal = {
                    "id": SecurityValidator().generate_proposal_id(),
                    "title": semantic_result.extracted_title or "Proposal",
                    "description": SecurityValidator().sanitize_text(message.content, 2000),
                    "options": semantic_result.extracted_options or ["Approve", "Reject"],
                    "proposer": message.source,
                    "timestamp": message.timestamp or "",
                }
                
                self._current_proposal = auto_proposal
                self._log_proposal_created(auto_proposal["id"], message.source, auto_proposal["title"])
                
                # Proceed with complexity analysis for auto-generated proposal
                self._current_phase = VotingPhase.VOTING
                return self._select_next_voter()
        
        # Invalid proposal - request clarification
        return message.source

    async def _handle_voting_phase(self, message: BaseChatMessage) -> str:
        """Handle voting phase with Byzantine fault tolerance and quality validation."""
        from .base_voting_system import VoteMessage
        
        if isinstance(message, VoteMessage):
            vote_content = message.content
            voter = message.source
            
            # Comprehensive vote validation
            if not self._validate_vote_integrity(message):
                self._log_security_violation("INVALID_VOTE", f"Vote from {voter}")
                return self._select_next_voter()
            
            # Record vote with Byzantine fault detection
            vote_record = {
                "vote": vote_content.vote,
                "reasoning": vote_content.reasoning,
                "confidence": vote_content.confidence,
                "timestamp": message.timestamp,
                "ranked_choices": getattr(vote_content, 'ranked_choices', None)
            }
            
            self._votes_cast[voter] = vote_record
            self._log_vote_cast(
                self._current_proposal["id"] if self._current_proposal else "unknown",
                voter,
                vote_content.vote.value,
                True
            )
            
            # Update Byzantine fault detection
            if self._current_proposal and "result" in self._current_proposal:
                self._byzantine_detector.update_reputation(
                    voter, vote_content.vote, self._current_proposal["result"]
                )
            
            # Check for Byzantine behavior
            if self._byzantine_detector.detect_byzantine_behavior(voter):
                logger.warning(f"Byzantine behavior detected from agent {voter}")
                self._log_security_violation("BYZANTINE_BEHAVIOR", f"Agent {voter}")
            
            # Check if voting is complete
            if self._is_voting_complete():
                return await self._process_voting_results()
            
            return self._select_next_voter()
        
        elif isinstance(message, TextMessage):
            # Attempt semantic interpretation of vote
            from ..intelligence.semantic_interpreter import SemanticVoteInterpreter
            
            interpreter = SemanticVoteInterpreter()
            semantic_result = interpreter.interpret_vote_content(message.content)
            
            if semantic_result.vote_type and semantic_result.confidence > 0.7:
                # High-confidence semantic vote interpretation
                auto_vote_record = {
                    "vote": semantic_result.vote_type,
                    "reasoning": semantic_result.reasoning or message.content,
                    "confidence": semantic_result.confidence,
                    "timestamp": message.timestamp,
                    "semantic_interpretation": True
                }
                
                self._votes_cast[message.source] = auto_vote_record
                self._log_vote_cast(
                    self._current_proposal["id"] if self._current_proposal else "unknown",
                    message.source,
                    semantic_result.vote_type.value,
                    True
                )
                
                if self._is_voting_complete():
                    return await self._process_voting_results()
                
                return self._select_next_voter()
        
        # Invalid vote - request clarification
        return message.source

    async def _handle_discussion_phase(self, message: BaseChatMessage) -> str:
        """Handle discussion phase with structured deliberation and convergence tracking."""
        from ..consensus.deliberation_engine import StructuredDeliberationEngine
        
        if not hasattr(self, '_deliberation_engine'):
            self._deliberation_engine = StructuredDeliberationEngine(
                participant_names=self._eligible_voters,
                max_rounds=self._max_discussion_rounds
            )
        
        # Add message to current deliberation round
        self._deliberation_engine.add_message_to_current_round(
            speaker=message.source,
            content=message.content if isinstance(message, TextMessage) else str(message.content)
        )
        
        # Check if we need to continue discussion or move to voting
        should_continue = self._deliberation_engine.should_continue_discussion()
        convergence_metrics = self._deliberation_engine.get_convergence_metrics()
        
        if should_continue and self._discussion_rounds < self._max_discussion_rounds:
            self._discussion_rounds += 1
            return self._select_discussion_participant()
        else:
            # Discussion complete - transition to voting with insights
            discussion_insights = self._deliberation_engine.get_discussion_insights()
            
            if self._current_proposal:
                self._current_proposal["discussion_insights"] = discussion_insights
                self._current_proposal["convergence_metrics"] = convergence_metrics
            
            self._current_phase = VotingPhase.VOTING
            self._votes_cast.clear()  # Clear any preliminary votes
            return self._select_next_voter()

    async def _handle_consensus_phase(self, message: BaseChatMessage) -> str:
        """Handle consensus phase and potentially restart process."""
        # Consensus reached or process complete
        # This could trigger a new proposal if needed
        self._current_phase = VotingPhase.PROPOSAL
        return self._select_proposer()

    def _is_voting_complete(self) -> bool:
        """
        Determine if voting is complete based on participation and quality thresholds.
        
        Uses Byzantine fault tolerance and adaptive quality controls rather than
        simple vote count thresholds that compromised quality in previous versions.
        """
        if not self._current_proposal:
            return False
        
        # Check basic participation threshold
        votes_received = len(self._votes_cast)
        eligible_voters = len(self._eligible_voters)
        
        # Minimum participation requirement (adaptive based on voting method)
        min_participation = self._get_minimum_participation_threshold()
        
        if votes_received < min_participation:
            return False
        
        # Quality threshold - ensure sufficient reasoning if required
        if self._require_reasoning:
            reasoned_votes = sum(1 for vote in self._votes_cast.values() 
                               if vote.get("reasoning") and len(vote["reasoning"].strip()) > 10)
            
            if reasoned_votes < min_participation * 0.8:  # 80% of minimum participants must provide reasoning
                return False
        
        # Byzantine resilience check
        if not self._byzantine_detector.is_byzantine_resilient(self._votes_cast):
            logger.warning("Voting not Byzantine resilient - requiring more votes")
            return votes_received >= eligible_voters  # Require all votes if not resilient
        
        return True

    def _get_minimum_participation_threshold(self) -> int:
        """Get minimum participation threshold based on voting method."""
        total_voters = len(self._eligible_voters)
        
        if self._voting_method == VotingMethod.UNANIMOUS:
            return total_voters  # All must vote
        elif self._voting_method == VotingMethod.QUALIFIED_MAJORITY:
            return max(3, int(total_voters * self._qualified_majority_threshold))
        else:  # MAJORITY or PLURALITY
            return max(2, int(total_voters * 0.5) + 1)

    async def _process_voting_results(self) -> str:
        """Process voting results with comprehensive analysis and quality assessment."""
        if not self._current_proposal:
            return self._select_proposer()
        
        # Calculate results using Byzantine fault tolerance
        weighted_votes = self._byzantine_detector.get_weighted_vote_count(self._votes_cast)
        result = self._calculate_voting_result(weighted_votes)
        
        # Create comprehensive voting result
        voting_result = VotingResult(
            proposal_id=self._current_proposal["id"],
            result=result["result"],
            votes_summary=result["votes_summary"],
            winning_option=result["winning_option"],
            total_voters=len(self._eligible_voters),
            participation_rate=len(self._votes_cast) / len(self._eligible_voters),
            confidence_average=result["confidence_average"],
            detailed_votes=result.get("detailed_votes", {}),
            byzantine_resilient=result.get("byzantine_resilient", True),
            reputation_adjusted=result.get("reputation_adjusted", False),
            suspicious_agents=list(self._byzantine_detector.suspicious_agents)
        )
        
        # Log result
        self._log_voting_result(
            self._current_proposal["id"],
            result["result"],
            voting_result.participation_rate
        )
        
        # Update Byzantine detector with consensus outcome
        for voter, vote_data in self._votes_cast.items():
            self._byzantine_detector.update_reputation(
                voter, vote_data["vote"], result["result"]
            )
        
        # Create result message
        from .base_voting_system import VotingResultMessage
        result_message = VotingResultMessage(
            content=voting_result,
            source=self.name
        )
        
        # Add to message thread
        self._message_thread.append(result_message)
        
        # Transition to consensus phase
        self._current_phase = VotingPhase.CONSENSUS
        
        # Determine if consensus was reached or if we need discussion
        if result["result"] == "no_consensus" and self._discussion_rounds < self._max_discussion_rounds:
            self._current_phase = VotingPhase.DISCUSSION
            return self._select_discussion_facilitator()
        
        return ""  # Process complete

    def _calculate_voting_result(self, weighted_votes: Dict[str, float]) -> Dict[str, Any]:
        """Calculate voting result using specified method with Byzantine fault tolerance."""
        if not weighted_votes:
            return {
                "result": "no_consensus",
                "votes_summary": {"approve": 0, "reject": 0, "abstain": 0},
                "winning_option": "none",
                "confidence_average": 0.0,
                "byzantine_resilient": False,
                "reputation_adjusted": True
            }
        
        total_weight = sum(weighted_votes.values())
        approve_weight = weighted_votes.get("approve", 0)
        reject_weight = weighted_votes.get("reject", 0)
        abstain_weight = weighted_votes.get("abstain", 0)
        
        # Calculate confidence average
        confidence_sum = sum(vote.get("confidence", 1.0) for vote in self._votes_cast.values())
        confidence_avg = confidence_sum / len(self._votes_cast) if self._votes_cast else 0.0
        
        result = {
            "votes_summary": {
                "approve": int(approve_weight),
                "reject": int(reject_weight), 
                "abstain": int(abstain_weight)
            },
            "confidence_average": confidence_avg,
            "byzantine_resilient": self._byzantine_detector.is_byzantine_resilient(self._votes_cast),
            "reputation_adjusted": any(self._byzantine_detector.reputation_scores[name] < 1.0 
                                     for name in self._votes_cast.keys()
                                     if name in self._byzantine_detector.reputation_scores)
        }
        
        # Apply voting method logic
        if self._voting_method == VotingMethod.UNANIMOUS:
            if abstain_weight == 0 and (approve_weight == total_weight or reject_weight == total_weight):
                result["result"] = "approved" if approve_weight > reject_weight else "rejected"
                result["winning_option"] = "approve" if approve_weight > reject_weight else "reject"
            else:
                result["result"] = "no_consensus"
                result["winning_option"] = "none"
        
        elif self._voting_method == VotingMethod.QUALIFIED_MAJORITY:
            threshold = self._qualified_majority_threshold
            if approve_weight / total_weight >= threshold:
                result["result"] = "approved"
                result["winning_option"] = "approve"
            elif reject_weight / total_weight >= threshold:
                result["result"] = "rejected"
                result["winning_option"] = "reject"
            else:
                result["result"] = "no_consensus"
                result["winning_option"] = "none"
        
        elif self._voting_method == VotingMethod.MAJORITY:
            if approve_weight > total_weight / 2:
                result["result"] = "approved"
                result["winning_option"] = "approve"
            elif reject_weight > total_weight / 2:
                result["result"] = "rejected"
                result["winning_option"] = "reject"
            else:
                result["result"] = "no_consensus"
                result["winning_option"] = "none"
        
        else:  # PLURALITY or RANKED_CHOICE (simplified)
            if approve_weight > reject_weight and approve_weight > abstain_weight:
                result["result"] = "approved"
                result["winning_option"] = "approve"
            elif reject_weight > approve_weight and reject_weight > abstain_weight:
                result["result"] = "rejected"
                result["winning_option"] = "reject"
            else:
                result["result"] = "no_consensus"
                result["winning_option"] = "none"
        
        return result

    def _select_next_voter(self) -> str:
        """Select next voter ensuring balanced participation."""
        # Exclude voters who have already cast votes
        remaining_voters = [v for v in self._eligible_voters if v not in self._votes_cast]
        
        if not remaining_voters:
            return self._eligible_voters[0]  # Fallback
        
        # Prioritize voters with higher reputation (Byzantine fault tolerance)
        if hasattr(self, '_byzantine_detector') and self._byzantine_detector.reputation_scores:
            remaining_with_reputation = [
                (voter, self._byzantine_detector.reputation_scores.get(voter, 1.0))
                for voter in remaining_voters
            ]
            remaining_with_reputation.sort(key=lambda x: x[1], reverse=True)
            return remaining_with_reputation[0][0]
        
        return remaining_voters[0]

    def _select_discussion_facilitator(self) -> str:
        """Select discussion facilitator based on expertise and reputation."""
        if self._auto_propose_speaker and self._auto_propose_speaker in self._eligible_voters:
            return self._auto_propose_speaker
        
        # Select based on reputation scores
        if hasattr(self, '_byzantine_detector') and self._byzantine_detector.reputation_scores:
            voters_with_reputation = [
                (voter, self._byzantine_detector.reputation_scores.get(voter, 1.0))
                for voter in self._eligible_voters
            ]
            voters_with_reputation.sort(key=lambda x: x[1], reverse=True)
            return voters_with_reputation[0][0]
        
        return self._eligible_voters[0]

    def _select_discussion_participant(self) -> str:
        """Select next discussion participant ensuring balanced participation."""
        if not hasattr(self, '_deliberation_engine'):
            return self._eligible_voters[0]
        
        # Use deliberation engine to select next participant
        return self._deliberation_engine.select_next_speaker(self._eligible_voters)

    def _authenticate_agent(self, agent_name: str) -> bool:
        """Authenticate agent participation."""
        try:
            validated_name = SecurityValidator.validate_agent_name(agent_name)
            return validated_name in self._agent_keys
        except ValueError:
            return False

    def _validate_vote_integrity(self, vote_message: "VoteMessage") -> bool:
        """Validate vote message integrity and prevent replay attacks."""
        voter = vote_message.source
        vote_content = vote_message.content
        
        # Basic validation
        if not self._authenticate_agent(voter):
            return False
        
        if voter in self._votes_cast:
            logger.warning(f"Agent {voter} attempted to vote twice")
            return False
        
        # Prevent replay attacks using nonce
        if hasattr(vote_content, 'timestamp') and vote_content.timestamp:
            nonce = f"{voter}:{vote_content.timestamp}"
            if nonce in self._vote_nonces:
                logger.warning(f"Replay attack detected from {voter}")
                return False
            self._vote_nonces.add(nonce)
        
        return True