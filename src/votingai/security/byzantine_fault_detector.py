"""
Byzantine Fault Detection Service

Extracted from voting_manager.py for:
- Single Responsibility Principle
- Better testability
- Dependency injection compatibility
- Enterprise security standards
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class VoteType(Enum):
    """Vote type enumeration - duplicated to avoid circular imports."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


logger = logging.getLogger(__name__)


@dataclass
class AgentReputation:
    """Represents an agent's reputation metrics."""

    agent_name: str
    reputation_score: float = 1.0
    vote_history: List[VoteType] = field(default_factory=lambda: [])
    consensus_alignment_rate: float = 0.0
    suspicious_activity_count: int = 0
    last_updated: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate reputation data."""
        if not (0.0 <= self.reputation_score <= 1.0):
            raise ValueError("Reputation score must be between 0.0 and 1.0")


@dataclass
class ByzantineDetectionResult:
    """Result of Byzantine behavior detection."""

    is_byzantine: bool
    confidence: float
    reason: str
    agent_name: str
    evidence: Dict[str, Any] = field(default_factory=lambda: {})


class IByzantineDetectionStrategy(ABC):
    """Interface for Byzantine detection strategies."""

    @abstractmethod
    def detect_byzantine_behavior(
        self, agent_reputation: AgentReputation, recent_votes: List[VoteType], consensus_history: List[str]
    ) -> ByzantineDetectionResult:
        """Detect Byzantine behavior patterns."""
        pass


class ReputationBasedDetectionStrategy(IByzantineDetectionStrategy):
    """Detection strategy based on reputation scoring."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def detect_byzantine_behavior(
        self, agent_reputation: AgentReputation, recent_votes: List[VoteType], consensus_history: List[str]
    ) -> ByzantineDetectionResult:
        """Detect based on reputation threshold."""

        is_byzantine = agent_reputation.reputation_score < self.threshold
        confidence = 1.0 - agent_reputation.reputation_score if is_byzantine else 0.0

        reason = f"Reputation score {agent_reputation.reputation_score:.2f} below threshold {self.threshold}"

        return ByzantineDetectionResult(
            is_byzantine=is_byzantine,
            confidence=confidence,
            reason=reason,
            agent_name=agent_reputation.agent_name,
            evidence={
                "reputation_score": agent_reputation.reputation_score,
                "threshold": self.threshold,
                "vote_history_length": len(agent_reputation.vote_history),
            },
        )


class ErraticVotingDetectionStrategy(IByzantineDetectionStrategy):
    """Detection strategy based on erratic voting patterns."""

    def __init__(self, max_changes_threshold: int = 4, window_size: int = 5):
        self.max_changes_threshold = max_changes_threshold
        self.window_size = window_size

    def detect_byzantine_behavior(
        self, agent_reputation: AgentReputation, recent_votes: List[VoteType], consensus_history: List[str]
    ) -> ByzantineDetectionResult:
        """Detect based on rapid vote changes."""

        if len(recent_votes) < self.window_size:
            return ByzantineDetectionResult(
                is_byzantine=False,
                confidence=0.0,
                reason="Insufficient vote history for analysis",
                agent_name=agent_reputation.agent_name,
            )

        # Count vote type changes in recent history
        recent_window = recent_votes[-self.window_size :]
        changes = sum(1 for i in range(1, len(recent_window)) if recent_window[i] != recent_window[i - 1])

        is_byzantine = changes >= self.max_changes_threshold
        confidence = min(1.0, changes / self.max_changes_threshold) if is_byzantine else 0.0

        reason = f"Vote changes ({changes}) exceed threshold ({self.max_changes_threshold})"

        return ByzantineDetectionResult(
            is_byzantine=is_byzantine,
            confidence=confidence,
            reason=reason,
            agent_name=agent_reputation.agent_name,
            evidence={
                "vote_changes": changes,
                "threshold": self.max_changes_threshold,
                "recent_votes": [vote.value for vote in recent_window],
            },
        )


class ByzantineFaultDetector:
    """
    Enterprise-grade Byzantine fault detection service.

    - Strategy pattern for detection algorithms
    - Dependency injection ready
    - Comprehensive logging
    - Immutable data structures
    - Result pattern for error handling
    """

    def __init__(
        self,
        total_agents: int,
        detection_strategies: Optional[List[IByzantineDetectionStrategy]] = None,
        detection_threshold: float = 0.3,
    ):
        """
        Initialize Byzantine fault detector.

        Args:
            total_agents: Total number of agents in the system
            detection_strategies: List of detection strategies to use
            detection_threshold: Default threshold for detection
        """
        self.total_agents = total_agents
        self.detection_threshold = detection_threshold

        # Agent reputation tracking
        self._agent_reputations: Dict[str, AgentReputation] = {}
        self._consensus_history: List[Dict[str, Any]] = []
        self._suspicious_agents: set[str] = set()

        # Detection strategies
        self._detection_strategies = detection_strategies or [
            ReputationBasedDetectionStrategy(detection_threshold),
            ErraticVotingDetectionStrategy(),
        ]

        logger.info(f"ByzantineFaultDetector initialized for {total_agents} agents")

    def register_agent(self, agent_name: str) -> None:
        """Register a new agent for reputation tracking."""
        if agent_name not in self._agent_reputations:
            self._agent_reputations[agent_name] = AgentReputation(agent_name=agent_name)
            logger.debug(f"Registered agent for Byzantine detection: {agent_name}")

    def update_reputation(self, agent_name: str, vote: VoteType, consensus_outcome: str) -> None:
        """
        Update agent reputation based on voting behavior.

        """
        if agent_name not in self._agent_reputations:
            self.register_agent(agent_name)

        reputation = self._agent_reputations[agent_name]

        # Update vote history (immutable pattern)
        new_vote_history = reputation.vote_history + [vote]

        # Calculate reputation adjustment
        reputation_delta = self._calculate_reputation_delta(vote, consensus_outcome)
        new_reputation_score = max(0.1, min(1.0, reputation.reputation_score + reputation_delta))

        # Create updated reputation (immutable)
        self._agent_reputations[agent_name] = AgentReputation(
            agent_name=agent_name,
            reputation_score=new_reputation_score,
            vote_history=new_vote_history,
            consensus_alignment_rate=self._calculate_alignment_rate(new_vote_history, consensus_outcome),
            suspicious_activity_count=reputation.suspicious_activity_count,
            last_updated=datetime.now().isoformat(),
        )

        logger.debug(
            f"Updated reputation for {agent_name}: {reputation.reputation_score:.2f} -> {new_reputation_score:.2f}",
            extra={
                "agent_name": agent_name,
                "old_reputation": reputation.reputation_score,
                "new_reputation": new_reputation_score,
                "vote": vote.value,
                "consensus_outcome": consensus_outcome,
            },
        )

    def detect_byzantine_behavior(self, agent_name: str) -> ByzantineDetectionResult:
        """
        Detect Byzantine behavior using all configured strategies.

        Returns the most confident detection result.
        """
        if agent_name not in self._agent_reputations:
            return ByzantineDetectionResult(
                is_byzantine=False,
                confidence=0.0,
                reason="Agent not registered for reputation tracking",
                agent_name=agent_name,
            )

        reputation = self._agent_reputations[agent_name]
        recent_votes = reputation.vote_history
        consensus_history = [str(entry.get("result", "")) for entry in self._consensus_history]

        # Run all detection strategies
        detection_results: List[ByzantineDetectionResult] = []
        for strategy in self._detection_strategies:
            try:
                result = strategy.detect_byzantine_behavior(reputation, recent_votes, consensus_history)
                detection_results.append(result)
            except Exception as ex:
                logger.warning(
                    f"Detection strategy failed for {agent_name}: {ex}",
                    extra={"agent_name": agent_name, "strategy": type(strategy).__name__},
                )

        # Return the most confident positive detection, or the highest confidence result
        positive_detections: List[ByzantineDetectionResult] = [r for r in detection_results if r.is_byzantine]
        if positive_detections:
            best_result = max(positive_detections, key=lambda r: r.confidence)
            if best_result.is_byzantine:
                self._suspicious_agents.add(agent_name)
            return best_result

        # Return the most confident negative result
        return (
            max(detection_results, key=lambda r: r.confidence)
            if detection_results
            else ByzantineDetectionResult(
                is_byzantine=False, confidence=0.0, reason="No detection strategies available", agent_name=agent_name
            )
        )

    def get_weighted_vote_count(self, votes: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weighted vote counts based on reputation."""
        weighted_counts = {"approve": 0.0, "reject": 0.0, "abstain": 0.0}

        for agent_name, vote_data in votes.items():
            reputation_score = self._agent_reputations.get(agent_name, AgentReputation(agent_name)).reputation_score

            # Extract vote type safely
            vote_type = self._extract_vote_type(vote_data.get("vote"))

            # Reduce weight for suspicious agents
            if agent_name in self._suspicious_agents:
                reputation_score *= 0.5
                logger.debug(f"Reduced weight for suspicious agent {agent_name}")

            weighted_counts[vote_type] += reputation_score

        return weighted_counts

    def is_byzantine_resilient(self, votes: Dict[str, Dict[str, Any]]) -> bool:
        """
        Check if voting result is resilient to Byzantine faults.

        Uses the standard Byzantine fault tolerance assumption that up to 1/3 of nodes may be faulty.
        """
        if not votes:
            return False

        total_weight = sum(
            self._agent_reputations.get(name, AgentReputation(name)).reputation_score for name in votes.keys()
        )

        # Standard Byzantine assumption: up to 1/3 could be Byzantine
        max_byzantine_weight = total_weight / 3 if total_weight > 0 else 0

        weighted_counts = self.get_weighted_vote_count(votes)
        max_honest_votes = max(weighted_counts.values()) if weighted_counts else 0

        is_resilient = max_honest_votes > max_byzantine_weight

        logger.debug(
            f"Byzantine resilience check: max_honest={max_honest_votes:.2f}, "
            f"max_byzantine={max_byzantine_weight:.2f}, resilient={is_resilient}",
            extra={
                "total_weight": total_weight,
                "max_honest_votes": max_honest_votes,
                "max_byzantine_weight": max_byzantine_weight,
                "is_resilient": is_resilient,
            },
        )

        return is_resilient

    @property
    def suspicious_agents(self) -> set[str]:
        """Get the set of agents flagged as suspicious."""
        return self._suspicious_agents.copy()

    @property
    def reputation_scores(self) -> Dict[str, float]:
        """Get current reputation scores for all agents."""
        return {name: reputation.reputation_score for name, reputation in self._agent_reputations.items()}

    def get_agent_reputation(self, agent_name: str) -> Optional[AgentReputation]:
        """Get detailed reputation information for an agent."""
        return self._agent_reputations.get(agent_name)

    def _calculate_reputation_delta(self, vote: VoteType, consensus_outcome: str) -> float:
        """Calculate reputation change based on vote alignment with consensus."""
        if consensus_outcome == "approved" and vote == VoteType.APPROVE:
            return 0.1
        elif consensus_outcome == "rejected" and vote == VoteType.REJECT:
            return 0.1
        else:
            # Small penalty for disagreement (allows for legitimate dissent)
            return -0.05

    def _calculate_alignment_rate(self, vote_history: List[VoteType], consensus_outcome: str) -> float:
        """Calculate how often the agent's votes align with consensus."""
        if not vote_history:
            return 0.0

        # This is a simplified calculation - in production you'd track historical consensus outcomes
        return 0.5  # Placeholder

    def _extract_vote_type(self, vote: Any) -> str:
        """Safely extract vote type string from vote data."""
        if hasattr(vote, "value"):
            return str(vote.value).lower()
        elif isinstance(vote, str):
            return vote.lower()
        else:
            logger.warning(f"Unknown vote type: {vote}")
            return "abstain"
