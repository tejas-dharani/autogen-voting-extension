"""
Voting Calculation Strategies

Implements Strategy Pattern for different voting methods:
- Single Responsibility: Each strategy handles one voting method
- Open/Closed Principle: Easy to add new voting methods
- Dependency Inversion: Abstractions not depending on concretions
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Removed unused enum import
from typing import Any, Dict, List, Optional, Tuple, Type

from .voting_protocols import VotingMethod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VotingResult:
    """

    Contains all information needed about a voting outcome.
    """

    result: str  # "approved", "rejected", "no_consensus"
    votes_summary: Dict[str, int]
    winning_option: str
    confidence_average: float
    total_voters: int
    participation_rate: float

    # Additional metadata
    byzantine_resilient: bool = True
    reputation_adjusted: bool = False
    detailed_votes: Optional[Dict[str, Any]] = None
    suspicious_agents: Optional[List[str]] = None

    @property
    def is_approved(self) -> bool:
        """Check if the proposal was approved."""
        return self.result == "approved"

    @property
    def is_rejected(self) -> bool:
        """Check if the proposal was rejected."""
        return self.result == "rejected"

    @property
    def has_consensus(self) -> bool:
        """Check if consensus was reached."""
        return self.result != "no_consensus"


class IVotingStrategy(ABC):
    """
    Interface for voting calculation strategies.

    """

    @abstractmethod
    def calculate_result(
        self,
        weighted_votes: Dict[str, float],
        total_eligible_voters: int,
        confidence_scores: List[float],
        **kwargs: Any,
    ) -> VotingResult:
        """
        Calculate voting result using this strategy.

        Args:
            weighted_votes: Vote counts by type (approve, reject, abstain)
            total_eligible_voters: Total number of eligible voters
            confidence_scores: Confidence scores for each vote
            **kwargs: Any: Strategy-specific parameters

        Returns:
            VotingResult with complete outcome information
        """
        pass

    def _no_consensus_result(self, total_voters: int, confidence_scores: List[float]) -> VotingResult:
        """Create a no-consensus result."""
        return VotingResult(
            result="no_consensus",
            votes_summary={"approve": 0, "reject": 0, "abstain": 0},
            winning_option="none",
            confidence_average=sum(confidence_scores) / max(1, len(confidence_scores)),
            total_voters=total_voters,
            participation_rate=0.0,
            byzantine_resilient=False,
            reputation_adjusted=False,
        )

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Get the name of this voting method."""
        pass

    @property
    @abstractmethod
    def requires_threshold(self) -> bool:
        """Whether this method requires a threshold parameter."""
        pass


class MajorityVotingStrategy(IVotingStrategy):
    """
    Majority voting strategy (>50% wins).

    """

    @property
    def method_name(self) -> str:
        return "majority"

    @property
    def requires_threshold(self) -> bool:
        return False

    def calculate_result(
        self,
        weighted_votes: Dict[str, float],
        total_eligible_voters: int,
        confidence_scores: List[float],
        **kwargs: Any,
    ) -> VotingResult:
        """Calculate majority voting result (>50%)."""

        logger.debug(
            "Calculating majority voting result",
            extra={"weighted_votes": weighted_votes, "total_eligible_voters": total_eligible_voters},
        )

        if not weighted_votes:
            return self._no_consensus_result(total_eligible_voters, confidence_scores)

        total_weight = sum(weighted_votes.values())
        approve_weight = weighted_votes.get("approve", 0)
        reject_weight = weighted_votes.get("reject", 0)
        abstain_weight = weighted_votes.get("abstain", 0)

        # Majority calculation: >50% of total votes
        if total_weight > 0 and approve_weight > total_weight / 2:
            result = "approved"
            winning_option = "approve"
        elif total_weight > 0 and reject_weight > total_weight / 2:
            result = "rejected"
            winning_option = "reject"
        else:
            result = "no_consensus"
            winning_option = "none"

        return VotingResult(
            result=result,
            votes_summary={
                "approve": int(approve_weight),
                "reject": int(reject_weight),
                "abstain": int(abstain_weight),
            },
            winning_option=winning_option,
            confidence_average=sum(confidence_scores) / max(1, len(confidence_scores)),
            total_voters=total_eligible_voters,
            participation_rate=total_weight / max(1, total_eligible_voters),
            byzantine_resilient=kwargs.get("byzantine_resilient", True),
            reputation_adjusted=kwargs.get("reputation_adjusted", False),
        )


class QualifiedMajorityStrategy(IVotingStrategy):
    """
    Qualified majority voting strategy (configurable threshold, typically 67%).

    """

    def __init__(self, threshold: float = 0.67):
        """
        Initialize qualified majority strategy.

        Args:
            threshold: Required threshold (0.0 to 1.0)
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("Qualified majority threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        logger.debug(f"QualifiedMajorityStrategy initialized with threshold {threshold}")

    @property
    def method_name(self) -> str:
        return f"qualified_majority_{int(self.threshold * 100)}%"

    @property
    def requires_threshold(self) -> bool:
        return True

    def calculate_result(
        self,
        weighted_votes: Dict[str, float],
        total_eligible_voters: int,
        confidence_scores: List[float],
        **kwargs: Any,
    ) -> VotingResult:
        """Calculate qualified majority result with configurable threshold."""

        logger.debug(
            f"Calculating qualified majority result with threshold {self.threshold}",
            extra={"weighted_votes": weighted_votes, "threshold": self.threshold},
        )

        if not weighted_votes:
            return self._no_consensus_result(total_eligible_voters, confidence_scores)

        total_weight = sum(weighted_votes.values())
        approve_weight = weighted_votes.get("approve", 0)
        reject_weight = weighted_votes.get("reject", 0)
        abstain_weight = weighted_votes.get("abstain", 0)

        # Qualified majority calculation
        if total_weight > 0:
            approve_ratio = approve_weight / total_weight
            reject_ratio = reject_weight / total_weight

            if approve_ratio >= self.threshold:
                result = "approved"
                winning_option = "approve"
            elif reject_ratio >= self.threshold:
                result = "rejected"
                winning_option = "reject"
            else:
                result = "no_consensus"
                winning_option = "none"
        else:
            result = "no_consensus"
            winning_option = "none"

        return VotingResult(
            result=result,
            votes_summary={
                "approve": int(approve_weight),
                "reject": int(reject_weight),
                "abstain": int(abstain_weight),
            },
            winning_option=winning_option,
            confidence_average=sum(confidence_scores) / max(1, len(confidence_scores)),
            total_voters=total_eligible_voters,
            participation_rate=total_weight / max(1, total_eligible_voters),
            byzantine_resilient=kwargs.get("byzantine_resilient", True),
            reputation_adjusted=kwargs.get("reputation_adjusted", False),
        )


class UnanimousVotingStrategy(IVotingStrategy):
    """
    Unanimous voting strategy (100% agreement required).

    """

    @property
    def method_name(self) -> str:
        return "unanimous"

    @property
    def requires_threshold(self) -> bool:
        return False

    def calculate_result(
        self,
        weighted_votes: Dict[str, float],
        total_eligible_voters: int,
        confidence_scores: List[float],
        **kwargs: Any,
    ) -> VotingResult:
        """Calculate unanimous voting result (100% agreement)."""

        logger.debug("Calculating unanimous voting result", extra={"weighted_votes": weighted_votes})

        if not weighted_votes:
            return self._no_consensus_result(total_eligible_voters, confidence_scores)

        total_weight = sum(weighted_votes.values())
        approve_weight = weighted_votes.get("approve", 0)
        reject_weight = weighted_votes.get("reject", 0)
        abstain_weight = weighted_votes.get("abstain", 0)

        # Unanimous requires no abstentions and all votes in same direction
        if abstain_weight == 0:
            if approve_weight == total_weight and approve_weight > 0:
                result = "approved"
                winning_option = "approve"
            elif reject_weight == total_weight and reject_weight > 0:
                result = "rejected"
                winning_option = "reject"
            else:
                result = "no_consensus"
                winning_option = "none"
        else:
            # Any abstention prevents unanimous decision
            result = "no_consensus"
            winning_option = "none"

        return VotingResult(
            result=result,
            votes_summary={
                "approve": int(approve_weight),
                "reject": int(reject_weight),
                "abstain": int(abstain_weight),
            },
            winning_option=winning_option,
            confidence_average=sum(confidence_scores) / max(1, len(confidence_scores)),
            total_voters=total_eligible_voters,
            participation_rate=total_weight / max(1, total_eligible_voters),
            byzantine_resilient=kwargs.get("byzantine_resilient", True),
            reputation_adjusted=kwargs.get("reputation_adjusted", False),
        )


class PluralityVotingStrategy(IVotingStrategy):
    """
    Plurality voting strategy (most votes wins, no majority required).

    """

    @property
    def method_name(self) -> str:
        return "plurality"

    @property
    def requires_threshold(self) -> bool:
        return False

    def calculate_result(
        self,
        weighted_votes: Dict[str, float],
        total_eligible_voters: int,
        confidence_scores: List[float],
        **kwargs: Any,
    ) -> VotingResult:
        """Calculate plurality result (highest vote count wins)."""

        logger.debug("Calculating plurality voting result", extra={"weighted_votes": weighted_votes})

        if not weighted_votes:
            return self._no_consensus_result(total_eligible_voters, confidence_scores)

        approve_weight = weighted_votes.get("approve", 0)
        reject_weight = weighted_votes.get("reject", 0)
        abstain_weight = weighted_votes.get("abstain", 0)

        # Find the option with the most votes
        if approve_weight > reject_weight and approve_weight > abstain_weight:
            result = "approved"
            winning_option = "approve"
        elif reject_weight > approve_weight and reject_weight > abstain_weight:
            result = "rejected"
            winning_option = "reject"
        else:
            # Tie or abstain wins - no clear consensus
            result = "no_consensus"
            winning_option = "none"

        return VotingResult(
            result=result,
            votes_summary={
                "approve": int(approve_weight),
                "reject": int(reject_weight),
                "abstain": int(abstain_weight),
            },
            winning_option=winning_option,
            confidence_average=sum(confidence_scores) / max(1, len(confidence_scores)),
            total_voters=total_eligible_voters,
            participation_rate=sum(weighted_votes.values()) / max(1, total_eligible_voters),
            byzantine_resilient=kwargs.get("byzantine_resilient", True),
            reputation_adjusted=kwargs.get("reputation_adjusted", False),
        )


class VotingStrategyFactory:
    """
    Factory for creating voting strategies.

    """

    def __init__(self) -> None:
        """Initialize the strategy factory."""
        self._strategies: Dict[VotingMethod, Tuple[Type[IVotingStrategy], Dict[str, Any]]] = {}
        logger.debug("VotingStrategyFactory initialized")

    def create_strategy(
        self, voting_method: VotingMethod, threshold: Optional[float] = None, **kwargs: Any
    ) -> IVotingStrategy:
        """
        Create a voting strategy for the specified method.

        Args:
            voting_method: The voting method to use
            threshold: Threshold for qualified majority (if applicable)
            **kwargs: Any: Additional strategy-specific parameters

        Returns:
            Configured voting strategy

        Raises:
            ValueError: If voting method is unsupported
        """

        logger.debug(
            f"Creating strategy for {voting_method}",
            extra={
                "voting_method": voting_method.value if hasattr(voting_method, "value") else str(voting_method),
                "threshold": threshold,
            },
        )

        if voting_method == VotingMethod.MAJORITY:
            return MajorityVotingStrategy()

        elif voting_method == VotingMethod.QUALIFIED_MAJORITY:
            if threshold is None:
                threshold = 0.67  # Default qualified majority
            return QualifiedMajorityStrategy(threshold)

        elif voting_method == VotingMethod.UNANIMOUS:
            return UnanimousVotingStrategy()

        elif voting_method == VotingMethod.PLURALITY:
            return PluralityVotingStrategy()

        else:
            raise ValueError(f"Unsupported voting method: {voting_method}")

    def get_available_methods(self) -> List[VotingMethod]:
        """Get list of supported voting methods."""
        return [VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY, VotingMethod.UNANIMOUS, VotingMethod.PLURALITY]

    def register_custom_strategy(self, method: VotingMethod, strategy_class: type, **default_kwargs: Any) -> None:
        """
        Register a custom voting strategy.

        """
        self._strategies[method] = (strategy_class, default_kwargs)
        logger.info(f"Registered custom strategy for {method}: {strategy_class.__name__}")


# Utility functions for common operations
def extract_confidence_scores(votes: Dict[str, Dict[str, Any]]) -> List[float]:
    """Extract confidence scores from vote data."""
    scores: List[float] = []
    for vote_data in votes.values():
        confidence = vote_data.get("confidence", 1.0)
        if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
            scores.append(float(confidence))
        else:
            scores.append(1.0)  # Default confidence
    return scores


def validate_weighted_votes(weighted_votes: Dict[str, float]) -> None:
    """Validate weighted vote data structure."""
    required_keys = {"approve", "reject", "abstain"}

    # Type is already enforced by function annotation

    missing_keys = required_keys - set(weighted_votes.keys())
    if missing_keys:
        raise ValueError(f"Missing required vote types: {missing_keys}")

    for vote_type, weight in weighted_votes.items():
        # Type is enforced by annotation, but check for negative values
        if weight < 0:
            raise ValueError(f"Invalid weight for {vote_type}: {weight}")
