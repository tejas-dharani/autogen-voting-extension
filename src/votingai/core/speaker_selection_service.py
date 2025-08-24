"""
Speaker Selection Service

Handles intelligent speaker selection with:
- Strategy pattern for different selection algorithms
- Reputation-based selection
- Participation tracking and balancing
- Comprehensive logging and monitoring
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .voting_protocols import VotingPhase

logger = logging.getLogger(__name__)


@dataclass
class ParticipationMetrics:
    """Tracks participation metrics for an agent."""

    agent_name: str
    total_speaking_turns: int = 0
    recent_activity_score: float = 0.0
    last_spoke_timestamp: Optional[datetime] = None
    expertise_areas: List[str] = field(default_factory=lambda: [])
    effectiveness_score: float = 1.0  # How effective their contributions are

    @property
    def participation_balance_score(self) -> float:
        """Calculate how balanced this agent's participation is."""
        # Lower score = needs more participation opportunities
        base_score = 1.0 / max(1, self.total_speaking_turns * 0.1)
        return min(1.0, base_score * self.effectiveness_score)


@dataclass
class SpeakerSelectionContext:
    """Context information for speaker selection decisions."""

    current_phase: VotingPhase
    participant_names: List[str]
    current_speaker: Optional[str] = None
    discussion_round: int = 0
    remaining_voters: List[str] = field(default_factory=lambda: [])
    proposal_complexity: str = "moderate"  # simple, moderate, complex
    requires_expertise: Optional[str] = None  # domain expertise needed

    def __post_init__(self) -> None:
        """Initialize remaining voters if not provided."""
        if not self.remaining_voters:
            object.__setattr__(self, "remaining_voters", self.participant_names.copy())


class ISpeakerSelectionStrategy(ABC):
    """Interface for speaker selection strategies."""

    @abstractmethod
    def select_speaker(
        self,
        context: SpeakerSelectionContext,
        participation_metrics: Dict[str, ParticipationMetrics],
        reputation_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Select the next speaker based on strategy logic."""
        pass


class ReputationBasedSelectionStrategy(ISpeakerSelectionStrategy):
    """
    Select speakers based on reputation scores and participation balance.

    """

    def __init__(self, reputation_weight: float = 0.6, participation_weight: float = 0.4):
        """
        Initialize reputation-based selection.

        Args:
            reputation_weight: Weight given to reputation scores (0.0-1.0)
            participation_weight: Weight given to participation balance (0.0-1.0)
        """
        if reputation_weight + participation_weight != 1.0:
            raise ValueError("Reputation and participation weights must sum to 1.0")

        self.reputation_weight = reputation_weight
        self.participation_weight = participation_weight

    def select_speaker(
        self,
        context: SpeakerSelectionContext,
        participation_metrics: Dict[str, ParticipationMetrics],
        reputation_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Select speaker based on weighted reputation and participation scores."""

        if not context.participant_names:
            raise ValueError("No participants available for selection")

        reputation_scores = reputation_scores or {}
        speaker_scores: List[Tuple[str, float]] = []

        for agent_name in context.participant_names:
            # Get reputation score (default to 1.0 if not available)
            reputation = reputation_scores.get(agent_name, 1.0)

            # Get participation metrics
            participation = participation_metrics.get(agent_name, ParticipationMetrics(agent_name))

            # Calculate weighted score
            combined_score = (
                reputation * self.reputation_weight
                + participation.participation_balance_score * self.participation_weight
            )

            speaker_scores.append((agent_name, combined_score))

        # Sort by score (descending) and select the highest
        speaker_scores.sort(key=lambda x: x[1], reverse=True)
        selected_speaker = speaker_scores[0][0]

        logger.debug(
            f"Selected speaker using reputation strategy: {selected_speaker}",
            extra={
                "selection_scores": dict(speaker_scores),
                "reputation_weight": self.reputation_weight,
                "participation_weight": self.participation_weight,
            },
        )

        return selected_speaker


class RoundRobinSelectionStrategy(ISpeakerSelectionStrategy):
    """
    Round-robin speaker selection ensuring equal participation.

    """

    def __init__(self) -> None:
        """Initialize round-robin selection."""
        self._last_selected_index = -1

    def select_speaker(
        self,
        context: SpeakerSelectionContext,
        participation_metrics: Dict[str, ParticipationMetrics],
        reputation_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Select next speaker in round-robin fashion."""

        if not context.participant_names:
            raise ValueError("No participants available for selection")

        # Select next agent in sequence
        self._last_selected_index = (self._last_selected_index + 1) % len(context.participant_names)
        selected_speaker = context.participant_names[self._last_selected_index]

        logger.debug(
            f"Selected speaker using round-robin: {selected_speaker}",
            extra={"selected_index": self._last_selected_index, "total_participants": len(context.participant_names)},
        )

        return selected_speaker


class ExpertiseBasedSelectionStrategy(ISpeakerSelectionStrategy):
    """
    Select speakers based on domain expertise for the current discussion.

    """

    def select_speaker(
        self,
        context: SpeakerSelectionContext,
        participation_metrics: Dict[str, ParticipationMetrics],
        reputation_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Select speaker based on relevant expertise."""

        if not context.participant_names:
            raise ValueError("No participants available for selection")

        # If no specific expertise is required, fall back to reputation-based
        if not context.requires_expertise:
            fallback_strategy = ReputationBasedSelectionStrategy()
            return fallback_strategy.select_speaker(context, participation_metrics, reputation_scores)

        # Find agents with relevant expertise
        expert_candidates: List[Tuple[str, float]] = []
        for agent_name in context.participant_names:
            participation = participation_metrics.get(agent_name, ParticipationMetrics(agent_name))
            if context.requires_expertise in participation.expertise_areas:
                reputation = reputation_scores.get(agent_name, 1.0) if reputation_scores else 1.0
                expert_candidates.append((agent_name, reputation))

        if expert_candidates:
            # Select the expert with highest reputation
            expert_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_speaker = expert_candidates[0][0]

            logger.debug(
                f"Selected expert speaker: {selected_speaker}",
                extra={
                    "required_expertise": context.requires_expertise,
                    "expert_candidates": [name for name, _ in expert_candidates],
                },
            )

            return selected_speaker
        else:
            # No experts available, fall back to reputation-based
            logger.debug(f"No experts found for {context.requires_expertise}, using fallback")
            fallback_strategy = ReputationBasedSelectionStrategy()
            return fallback_strategy.select_speaker(context, participation_metrics, reputation_scores)


class SpeakerSelectionService:
    """
    Enterprise speaker selection service.

    - Strategy pattern for selection algorithms
    - Comprehensive tracking and metrics
    - Dependency injection ready
    - Logging and monitoring integration
    """

    def __init__(
        self, default_strategy: Optional[ISpeakerSelectionStrategy] = None, auto_propose_speaker: Optional[str] = None
    ):
        """
        Initialize speaker selection service.

        Args:
            default_strategy: Default selection strategy to use
            auto_propose_speaker: Preferred speaker for proposals
        """
        self._default_strategy = default_strategy or ReputationBasedSelectionStrategy()
        self._auto_propose_speaker = auto_propose_speaker

        # Participation tracking
        self._participation_metrics: Dict[str, ParticipationMetrics] = {}

        # Strategy mapping by phase
        self._phase_strategies: Dict[VotingPhase, ISpeakerSelectionStrategy] = {}

        logger.info("SpeakerSelectionService initialized")

    def register_participant(self, agent_name: str, expertise_areas: Optional[List[str]] = None) -> None:
        """Register a participant for tracking."""
        if agent_name not in self._participation_metrics:
            self._participation_metrics[agent_name] = ParticipationMetrics(
                agent_name=agent_name, expertise_areas=expertise_areas or []
            )
            logger.debug(f"Registered participant: {agent_name}")

    def set_strategy_for_phase(self, phase: VotingPhase, strategy: ISpeakerSelectionStrategy) -> None:
        """Set a specific strategy for a voting phase."""
        self._phase_strategies[phase] = strategy
        logger.debug(f"Set strategy for {phase}: {type(strategy).__name__}")

    def select_next_speaker(
        self, context: SpeakerSelectionContext, reputation_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Select the next speaker using appropriate strategy.

        Args:
            context: Current selection context
            reputation_scores: Current reputation scores for agents

        Returns:
            Name of selected speaker
        """

        # Ensure all participants are registered
        for participant in context.participant_names:
            self.register_participant(participant)

        # Handle special cases first
        if context.current_phase == VotingPhase.PROPOSAL and self._auto_propose_speaker:
            if self._auto_propose_speaker in context.participant_names:
                logger.debug(f"Using auto-propose speaker: {self._auto_propose_speaker}")
                return self._auto_propose_speaker

        # Select strategy based on phase or use default
        strategy = self._phase_strategies.get(context.current_phase, self._default_strategy)

        # Make selection
        selected_speaker = strategy.select_speaker(context, self._participation_metrics, reputation_scores)

        # Update participation tracking
        self._update_participation_metrics(selected_speaker)

        logger.info(
            f"Selected speaker for {context.current_phase}: {selected_speaker}",
            extra={
                "phase": context.current_phase.value
                if hasattr(context.current_phase, "value")
                else str(context.current_phase),
                "strategy": type(strategy).__name__,
                "total_participants": len(context.participant_names),
            },
        )

        return selected_speaker

    def select_next_voter(
        self, remaining_voters: List[str], reputation_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Select the next voter from remaining eligible voters.

        Prioritizes voters with higher reputation scores.
        """
        if not remaining_voters:
            raise ValueError("No remaining voters to select from")

        if not reputation_scores:
            # No reputation data, use first available
            return remaining_voters[0]

        # Select voter with highest reputation
        voter_scores = [(voter, reputation_scores.get(voter, 1.0)) for voter in remaining_voters]
        voter_scores.sort(key=lambda x: x[1], reverse=True)

        selected_voter = voter_scores[0][0]

        logger.debug(
            f"Selected next voter: {selected_voter}",
            extra={"remaining_voters": len(remaining_voters), "voter_scores": dict(voter_scores)},
        )

        return selected_voter

    def select_discussion_facilitator(
        self, participants: List[str], reputation_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Select a facilitator for discussion phase.

        Chooses the participant with highest reputation and discussion experience.
        """
        if not participants:
            raise ValueError("No participants available for facilitation")

        # Use auto-propose speaker if available and suitable
        if self._auto_propose_speaker and self._auto_propose_speaker in participants:
            return self._auto_propose_speaker

        # Select based on reputation and discussion effectiveness
        facilitator_scores: List[Tuple[str, float]] = []
        for participant in participants:
            reputation = reputation_scores.get(participant, 1.0) if reputation_scores else 1.0
            metrics = self._participation_metrics.get(participant, ParticipationMetrics(participant))

            # Weight reputation more heavily for facilitation
            facilitator_score = reputation * 0.8 + metrics.effectiveness_score * 0.2
            facilitator_scores.append((participant, facilitator_score))

        facilitator_scores.sort(key=lambda x: x[1], reverse=True)
        selected_facilitator = facilitator_scores[0][0]

        logger.debug(
            f"Selected discussion facilitator: {selected_facilitator}",
            extra={"facilitator_scores": dict(facilitator_scores)},
        )

        return selected_facilitator

    def get_participation_metrics(self) -> Dict[str, ParticipationMetrics]:
        """Get current participation metrics for all tracked participants."""
        return self._participation_metrics.copy()

    def update_effectiveness_score(self, agent_name: str, effectiveness: float) -> None:
        """
        Update effectiveness score for an agent.

        Args:
            agent_name: Name of the agent
            effectiveness: New effectiveness score (0.0-1.0)
        """
        if agent_name in self._participation_metrics:
            self._participation_metrics[agent_name].effectiveness_score = max(0.0, min(1.0, effectiveness))
            logger.debug(f"Updated effectiveness score for {agent_name}: {effectiveness}")

    def _update_participation_metrics(self, speaker: str) -> None:
        """Update participation metrics when a speaker is selected."""
        if speaker in self._participation_metrics:
            metrics = self._participation_metrics[speaker]
            metrics.total_speaking_turns += 1
            metrics.last_spoke_timestamp = datetime.now(timezone.utc)

            # Update recent activity score (decaying average)
            metrics.recent_activity_score = metrics.recent_activity_score * 0.9 + 0.1

            logger.debug(f"Updated participation metrics for {speaker}")


# Utility functions for common selection scenarios
def create_voting_phase_context(
    phase: VotingPhase, participants: List[str], current_speaker: Optional[str] = None, **kwargs: Any
) -> SpeakerSelectionContext:
    """Create a speaker selection context for a voting phase."""
    return SpeakerSelectionContext(
        current_phase=phase, participant_names=participants, current_speaker=current_speaker, **kwargs
    )


def create_balanced_selection_service(
    participants: List[str], auto_propose_speaker: Optional[str] = None
) -> SpeakerSelectionService:
    """
    Create a speaker selection service with balanced strategies for different phases.

    """
    service = SpeakerSelectionService(
        default_strategy=ReputationBasedSelectionStrategy(reputation_weight=0.7, participation_weight=0.3),
        auto_propose_speaker=auto_propose_speaker,
    )

    # Set phase-specific strategies
    service.set_strategy_for_phase(VotingPhase.DISCUSSION, ExpertiseBasedSelectionStrategy())

    # Register all participants
    for participant in participants:
        service.register_participant(participant)

    return service
