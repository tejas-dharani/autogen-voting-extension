"""
Adaptive Consensus Strategies

Intelligent strategy selection based on decision complexity and context analysis.
Refactored from adaptive_consensus.py with improved naming conventions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DecisionComplexity(str, Enum):
    """Classification of decision complexity based on contextual analysis."""

    TRIVIAL = "trivial"  # Clear-cut decisions, simple keyword matching sufficient
    SIMPLE = "simple"  # Straightforward with limited options
    MODERATE = "moderate"  # Multiple factors, some discussion beneficial
    COMPLEX = "complex"  # High stakes, multiple perspectives needed
    CRITICAL = "critical"  # Mission-critical, extensive deliberation required


class ConsensusStrategy(str, Enum):
    """Available consensus strategies based on decision context."""

    FAST_TRACK = "fast_track"  # Immediate keyword-based voting
    STRUCTURED_VOTING = "structured"  # Standard voting with minimal discussion
    DELIBERATIVE = "deliberative"  # Discussion rounds followed by voting
    COLLABORATIVE = "collaborative"  # Extended deliberation with refinement
    CRITICAL_CONSENSUS = "critical"  # Maximum deliberation for critical decisions


@dataclass
class ContextualMetrics:
    """
    Comprehensive metrics for analyzing decision context and complexity.

    This class encapsulates all relevant factors that influence the optimal
    consensus strategy selection for a given decision scenario.
    """

    # Content analysis metrics
    proposal_length: int = 0
    technical_complexity_score: int = 0
    sentiment_uncertainty: float = 0.0
    semantic_complexity_score: float = 0.0

    # Stakeholder analysis metrics
    participant_count: int = 0
    expertise_diversity: float = 0.0
    historical_agreement_rate: float = 1.0

    # Decision impact analysis
    available_options_count: int = 2
    decision_stakes_level: float = 0.5  # 0.0 = low stakes, 1.0 = critical
    time_pressure_level: float = 0.5  # 0.0 = no rush, 1.0 = urgent
    reversibility_factor: float = 0.5  # 0.0 = irreversible, 1.0 = easily reversed

    # Historical performance indicators
    similar_decisions_success_rate: float = 0.8
    previous_complexity_handling_score: float = 0.5

    def calculate_overall_complexity_score(self) -> float:
        """
        Calculate overall complexity score using weighted combination of factors.

        Returns:
            float: Complexity score between 0.0 (trivial) and 1.0 (critical)
        """
        # Content complexity component (30% weight)
        content_complexity = min(
            1.0,
            (self.proposal_length / 1000 + self.technical_complexity_score / 10 + self.semantic_complexity_score) / 3,
        )

        # Stakeholder complexity component (30% weight)
        stakeholder_complexity = min(
            1.0, (self.participant_count / 10 + self.expertise_diversity + (1.0 - self.historical_agreement_rate)) / 3
        )

        # Decision impact complexity component (30% weight)
        decision_complexity = (
            self.decision_stakes_level
            + (1.0 - self.reversibility_factor)
            + max(0, self.available_options_count - 2) / 5
        ) / 3

        # Historical adjustment component (10% weight)
        historical_factor = 1.0 - (self.similar_decisions_success_rate * self.previous_complexity_handling_score)

        # Weighted combination
        overall_complexity = (
            content_complexity * 0.3
            + stakeholder_complexity * 0.3
            + decision_complexity * 0.3
            + historical_factor * 0.1
        )

        return min(1.0, overall_complexity)


class ComplexityClassifier:
    """
    Sophisticated classifier for determining decision complexity using multiple heuristics.

    This class analyzes proposal text and context to determine the appropriate
    level of deliberation and consensus strategy required.
    """

    def __init__(self) -> None:
        self.technical_keywords = {
            "architecture",
            "algorithm",
            "performance",
            "security",
            "scalability",
            "implementation",
            "database",
            "api",
            "infrastructure",
            "deployment",
            "optimization",
            "refactoring",
            "integration",
            "protocol",
            "framework",
            "microservices",
            "distributed",
            "concurrent",
            "asynchronous",
            "threading",
        }

        self.high_stakes_keywords = {
            "critical",
            "production",
            "breaking",
            "irreversible",
            "permanent",
            "delete",
            "remove",
            "shutdown",
            "deprecated",
            "migration",
            "security",
            "vulnerability",
            "compliance",
            "legal",
            "budget",
            "customer",
            "revenue",
            "data-loss",
            "outage",
            "downtime",
        }

        self.complexity_indicators = {
            "multiple",
            "complex",
            "sophisticated",
            "intricate",
            "nuanced",
            "trade-off",
            "balance",
            "consider",
            "evaluate",
            "analyze",
            "implications",
            "consequences",
            "impact",
            "dependencies",
        }

    def analyze_proposal_content(self, text: str) -> ContextualMetrics:
        """
        Analyze proposal text to extract complexity metrics.

        Args:
            text: The proposal text to analyze

        Returns:
            ContextualMetrics: Extracted metrics for complexity assessment
        """
        words = text.lower().split()
        word_set = set(words)

        metrics = ContextualMetrics()
        metrics.proposal_length = len(text)
        metrics.technical_complexity_score = len(word_set.intersection(self.technical_keywords))

        # High stakes detection
        stakes_indicators = len(word_set.intersection(self.high_stakes_keywords))
        metrics.decision_stakes_level = min(1.0, stakes_indicators / 3)

        # Count complexity indicators (used for stake assessment)
        complexity_indicator_count = len(word_set.intersection(self.complexity_indicators))
        metrics.decision_stakes_level += complexity_indicator_count * 0.1

        # Semantic complexity using vocabulary diversity
        unique_words = len(word_set)
        total_words = len(words)
        metrics.semantic_complexity_score = min(1.0, unique_words / max(1, total_words))

        # Option count detection
        if any(word in text.lower() for word in ["option", "choice", "alternative"]):
            import re

            options = len(re.findall(r"[1-9]\.|•|\*|\-", text))
            metrics.available_options_count = max(2, options)

        # Time pressure detection
        time_pressure_words = {"urgent", "immediately", "asap", "deadline", "quickly"}
        if word_set.intersection(time_pressure_words):
            metrics.time_pressure_level = 0.8

        # Reversibility assessment
        irreversible_words = {"irreversible", "permanent", "delete", "remove", "final"}
        if word_set.intersection(irreversible_words):
            metrics.reversibility_factor = 0.2

        return metrics

    def classify_decision_complexity(self, metrics: ContextualMetrics) -> DecisionComplexity:
        """
        Classify decision complexity based on computed metrics.

        Args:
            metrics: The contextual metrics to analyze

        Returns:
            DecisionComplexity: The determined complexity level
        """
        complexity_score = metrics.calculate_overall_complexity_score()

        if complexity_score < 0.2:
            return DecisionComplexity.TRIVIAL
        elif complexity_score < 0.4:
            return DecisionComplexity.SIMPLE
        elif complexity_score < 0.6:
            return DecisionComplexity.MODERATE
        elif complexity_score < 0.8:
            return DecisionComplexity.COMPLEX
        else:
            return DecisionComplexity.CRITICAL


class AdaptiveStrategySelector:
    """
    Selects optimal consensus strategies based on decision complexity and context.

    This class implements the core logic for mapping decision characteristics
    to appropriate consensus strategies, with context-aware adjustments.
    """

    def __init__(self) -> None:
        self.base_strategy_mapping = {
            DecisionComplexity.TRIVIAL: ConsensusStrategy.FAST_TRACK,
            DecisionComplexity.SIMPLE: ConsensusStrategy.STRUCTURED_VOTING,
            DecisionComplexity.MODERATE: ConsensusStrategy.DELIBERATIVE,
            DecisionComplexity.COMPLEX: ConsensusStrategy.COLLABORATIVE,
            DecisionComplexity.CRITICAL: ConsensusStrategy.CRITICAL_CONSENSUS,
        }

        self.strategy_configurations = {
            ConsensusStrategy.FAST_TRACK: {
                "discussion_rounds": 0,
                "voting_method": "majority",
                "require_reasoning": False,
                "timeout_minutes": 2,
                "convergence_threshold": 0.5,
            },
            ConsensusStrategy.STRUCTURED_VOTING: {
                "discussion_rounds": 1,
                "voting_method": "majority",
                "require_reasoning": True,
                "timeout_minutes": 5,
                "convergence_threshold": 0.6,
            },
            ConsensusStrategy.DELIBERATIVE: {
                "discussion_rounds": 2,
                "voting_method": "qualified_majority",
                "require_reasoning": True,
                "timeout_minutes": 10,
                "convergence_threshold": 0.7,
            },
            ConsensusStrategy.COLLABORATIVE: {
                "discussion_rounds": 3,
                "voting_method": "qualified_majority",
                "require_reasoning": True,
                "timeout_minutes": 15,
                "convergence_threshold": 0.8,
            },
            ConsensusStrategy.CRITICAL_CONSENSUS: {
                "discussion_rounds": 5,
                "voting_method": "unanimous",
                "require_reasoning": True,
                "timeout_minutes": 30,
                "convergence_threshold": 0.9,
            },
        }

    def select_optimal_strategy(
        self, complexity: DecisionComplexity, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ConsensusStrategy, Dict[str, Any]]:
        """
        Select optimal consensus strategy with context-aware adjustments.

        Args:
            complexity: The determined decision complexity level
            context: Additional contextual information for strategy adjustment

        Returns:
            Tuple of (selected_strategy, strategy_configuration)
        """
        base_strategy = self.base_strategy_mapping[complexity]
        configuration = self.strategy_configurations[base_strategy].copy()

        # Apply context-based adjustments
        if context:
            configuration = self._apply_contextual_adjustments(base_strategy, configuration, context)

        return base_strategy, configuration

    def _apply_contextual_adjustments(
        self, strategy: ConsensusStrategy, config: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply context-specific adjustments to strategy configuration."""

        # Time pressure adjustments
        time_pressure = context.get("time_pressure_level", 0.5)
        if time_pressure > 0.7:
            if strategy != ConsensusStrategy.FAST_TRACK:
                # Reduce discussion rounds under time pressure
                config["discussion_rounds"] = max(0, config["discussion_rounds"] - 1)
                config["timeout_minutes"] = int(config["timeout_minutes"] * 0.7)

        # Stakes level adjustments
        stakes_level = context.get("decision_stakes_level", 0.5)
        if stakes_level > 0.8:
            # Higher stakes require more careful deliberation
            if strategy in [ConsensusStrategy.FAST_TRACK, ConsensusStrategy.STRUCTURED_VOTING]:
                config["voting_method"] = "qualified_majority"
                config["discussion_rounds"] += 1

        # Participant count adjustments
        participant_count = context.get("participant_count", 3)
        if participant_count > 5:
            # More participants need more structure
            config["discussion_rounds"] = min(config["discussion_rounds"] + 1, 5)
            config["timeout_minutes"] = int(config["timeout_minutes"] * 1.5)

        # Historical performance adjustments
        success_rate = context.get("similar_decisions_success_rate", 0.8)
        if success_rate < 0.6:
            # Poor historical performance needs more deliberation
            config["discussion_rounds"] += 1
            if config["voting_method"] == "majority":
                config["voting_method"] = "qualified_majority"

        return config

    def calculate_strategy_confidence(self, complexity: DecisionComplexity, metrics: ContextualMetrics) -> float:
        """
        Calculate confidence in the selected strategy.

        Args:
            complexity: The determined complexity level
            metrics: The contextual metrics used for classification

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        complexity_score = metrics.calculate_overall_complexity_score()

        # Confidence is higher when score is clearly in one complexity range
        if complexity == DecisionComplexity.TRIVIAL:
            confidence = 1.0 - (complexity_score / 0.2)
        elif complexity == DecisionComplexity.SIMPLE:
            mid_point = 0.3
            distance_from_mid = abs(complexity_score - mid_point)
            confidence = 0.8 + (distance_from_mid * 2)
        elif complexity == DecisionComplexity.MODERATE:
            mid_point = 0.5
            distance_from_mid = abs(complexity_score - mid_point)
            confidence = 0.7 + (distance_from_mid * 3)
        elif complexity == DecisionComplexity.COMPLEX:
            mid_point = 0.7
            distance_from_mid = abs(complexity_score - mid_point)
            confidence = 0.8 + (distance_from_mid * 2)
        else:  # CRITICAL
            confidence = min(1.0, (complexity_score - 0.8) / 0.2)

        # Adjust based on historical performance
        if metrics.similar_decisions_success_rate > 0.8:
            confidence *= 1.1
        elif metrics.similar_decisions_success_rate < 0.6:
            confidence *= 0.9

        return min(1.0, max(0.3, confidence))
