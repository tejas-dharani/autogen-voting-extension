"""
Smart Consensus Orchestration Algorithms

Main orchestrator that combines adaptive strategies with deliberation engines
to provide intelligent consensus recommendations and learning capabilities.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .adaptive_strategies import (
    AdaptiveStrategySelector,
    ComplexityClassifier,
    ConsensusStrategy,
    ContextualMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsensusRecommendation:
    """
    Comprehensive recommendation for consensus approach.

    Contains all the information needed to execute an optimal consensus
    strategy for a given decision scenario.
    """

    # Core recommendation
    complexity_classification: str
    complexity_score: float
    recommended_strategy: str
    strategy_configuration: Dict[str, Any]
    confidence_score: float

    # Analysis details
    contextual_analysis: Dict[str, Any]
    estimated_duration_minutes: float
    expected_participation_quality: float

    # Adaptive adjustments
    context_adjustments_applied: List[str]
    risk_factors_identified: List[str]
    success_probability: float

    # Metadata
    recommendation_timestamp: float
    analysis_version: str = "2.0"


@dataclass
class PerformanceInsights:
    """
    Insights about recommendation performance and system learning.

    Provides feedback on how well the consensus system is performing
    and identifies areas for improvement.
    """

    # Overall performance metrics
    overall_success_rate: float
    average_efficiency_ratio: float
    average_participant_satisfaction: float
    total_decisions_processed: int

    # Strategy-specific performance
    strategy_performance_breakdown: Dict[str, Dict[str, float]]

    # Trend analysis
    recent_performance_trend: str  # 'improving', 'stable', 'declining'
    performance_confidence_interval: Tuple[float, float]

    # Learning insights
    most_successful_strategy: str
    least_successful_strategy: str
    common_failure_patterns: List[str]
    recommended_system_adjustments: List[str]


class LearningFramework:
    """
    Machine learning framework for continuous system improvement.

    Tracks recommendation outcomes and adapts strategy selection
    based on historical performance patterns.
    """

    def __init__(self, max_history_size: int = 100):
        self.max_history_size = max_history_size
        self.performance_history: List[Dict[str, Any]] = []
        self.strategy_success_rates: defaultdict[str, List[float]] = defaultdict(list)
        self.complexity_prediction_accuracy: List[float] = []

    def record_decision_outcome(self, recommendation: ConsensusRecommendation, actual_outcome: Dict[str, Any]) -> None:
        """
        Record the outcome of a decision for learning purposes.

        Args:
            recommendation: The original recommendation made
            actual_outcome: What actually happened during the decision process
        """
        performance_record = {
            "timestamp": time.time(),
            "recommendation": self._serialize_recommendation(recommendation),
            "actual_outcome": actual_outcome,
            "success": actual_outcome.get("decision_reached", False),
            "efficiency_ratio": self._calculate_efficiency_ratio(recommendation, actual_outcome),
            "satisfaction_score": actual_outcome.get("participant_satisfaction", 0.5),
            "convergence_achieved": actual_outcome.get("convergence_score", 0.0) > 0.7,
        }

        self.performance_history.append(performance_record)

        # Update strategy-specific tracking
        strategy = recommendation.recommended_strategy
        success_score = self._calculate_success_score(performance_record)
        self.strategy_success_rates[strategy].append(success_score)

        # Maintain history size limit
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size :]

        # Update complexity prediction accuracy
        predicted_duration = recommendation.estimated_duration_minutes
        actual_duration = actual_outcome.get("actual_duration_minutes", predicted_duration)
        if actual_duration > 0:
            accuracy = 1.0 - abs(predicted_duration - actual_duration) / actual_duration
            self.complexity_prediction_accuracy.append(max(0.0, accuracy))

        logger.debug(f"Recorded outcome for strategy {strategy}: success={performance_record['success']}")

    def _serialize_recommendation(self, recommendation: ConsensusRecommendation) -> Dict[str, Any]:
        """Serialize recommendation for storage."""
        return {
            "complexity_classification": recommendation.complexity_classification,
            "recommended_strategy": recommendation.recommended_strategy,
            "confidence_score": recommendation.confidence_score,
            "estimated_duration": recommendation.estimated_duration_minutes,
        }

    def _calculate_efficiency_ratio(self, recommendation: ConsensusRecommendation, outcome: Dict[str, Any]) -> float:
        """Calculate how efficient the actual process was vs. predicted."""
        predicted_duration = recommendation.estimated_duration_minutes
        actual_duration = outcome.get("actual_duration_minutes", predicted_duration)

        if predicted_duration == 0:
            return 1.0

        return float(predicted_duration / max(actual_duration, 0.1))  # Avoid division by zero

    def _calculate_success_score(self, performance_record: Dict[str, Any]) -> float:
        """Calculate overall success score for a decision."""
        success = 1.0 if performance_record["success"] else 0.0
        efficiency = min(1.0, performance_record["efficiency_ratio"])
        satisfaction = performance_record["satisfaction_score"]
        convergence = 1.0 if performance_record["convergence_achieved"] else 0.0

        # Weighted combination
        return float(success * 0.4 + efficiency * 0.2 + satisfaction * 0.3 + convergence * 0.1)

    def get_strategy_performance_insights(self) -> Dict[str, Dict[str, float]]:
        """Get performance insights for each strategy."""
        insights: Dict[str, Dict[str, float]] = {}

        for strategy, success_scores in self.strategy_success_rates.items():
            if success_scores:
                insights[strategy] = {
                    "average_success_score": sum(success_scores) / len(success_scores),
                    "success_rate": sum(1 for score in success_scores if score > 0.7) / len(success_scores),
                    "consistency": 1.0 - (max(success_scores) - min(success_scores))
                    if len(success_scores) > 1
                    else 1.0,
                    "total_uses": len(success_scores),
                }

        return insights

    def suggest_strategy_adjustments(self) -> List[str]:
        """Suggest adjustments based on learning insights."""
        suggestions: List[str] = []
        performance_insights = self.get_strategy_performance_insights()

        # Identify underperforming strategies
        for strategy, metrics in performance_insights.items():
            if metrics["success_rate"] < 0.6:
                suggestions.append(
                    f"Consider adjusting {strategy} parameters - low success rate: {metrics['success_rate']:.1%}"
                )

            if metrics["consistency"] < 0.7:
                suggestions.append(f"Strategy {strategy} shows inconsistent performance - review context factors")

        # Check prediction accuracy
        if self.complexity_prediction_accuracy:
            avg_accuracy = sum(self.complexity_prediction_accuracy) / len(self.complexity_prediction_accuracy)
            if avg_accuracy < 0.7:
                suggestions.append(
                    f"Duration prediction accuracy low ({avg_accuracy:.1%}) - review complexity classification"
                )

        return suggestions


class SmartConsensusOrchestrator:
    """
    Main orchestrator for intelligent consensus processes.

    Combines complexity analysis, strategy selection, and learning capabilities
    to provide comprehensive consensus recommendations and continuous improvement.
    """

    def __init__(self) -> None:
        self.complexity_classifier = ComplexityClassifier()
        self.strategy_selector = AdaptiveStrategySelector()
        self.learning_framework = LearningFramework()

        # System configuration
        self.system_version = "2.0"
        self.analysis_confidence_threshold = 0.6

        logger.info("SmartConsensusOrchestrator initialized")

    async def analyze_and_recommend(
        self, proposal_text: str, participants: List[str], context: Optional[Dict[str, Any]] = None
    ) -> ConsensusRecommendation:
        """
        Analyze a decision scenario and provide comprehensive consensus recommendation.

        Args:
            proposal_text: The text of the proposal to analyze
            participants: List of participant names
            context: Additional context information

        Returns:
            ConsensusRecommendation: Comprehensive recommendation for the decision
        """
        logger.debug(f"Analyzing proposal with {len(participants)} participants")

        # Step 1: Analyze proposal complexity
        contextual_metrics = self.complexity_classifier.analyze_proposal_content(proposal_text)
        contextual_metrics.participant_count = len(participants)

        # Step 2: Add additional context if provided
        if context:
            self._enrich_metrics_with_context(contextual_metrics, context)

        # Step 3: Classify complexity
        complexity = self.complexity_classifier.classify_decision_complexity(contextual_metrics)

        # Step 4: Select optimal strategy
        strategy, configuration = self.strategy_selector.select_optimal_strategy(
            complexity, self._prepare_strategy_context(contextual_metrics, participants)
        )

        # Step 5: Calculate confidence and risk assessment
        confidence = self.strategy_selector.calculate_strategy_confidence(complexity, contextual_metrics)
        risk_factors = self._identify_risk_factors(contextual_metrics, strategy)

        # Step 6: Estimate success probability based on historical data
        success_probability = self._estimate_success_probability(strategy, contextual_metrics)

        # Step 7: Prepare comprehensive recommendation
        recommendation = ConsensusRecommendation(
            complexity_classification=complexity.value,
            complexity_score=contextual_metrics.calculate_overall_complexity_score(),
            recommended_strategy=strategy.value,
            strategy_configuration=configuration,
            confidence_score=confidence,
            contextual_analysis=self._prepare_contextual_analysis(contextual_metrics),
            estimated_duration_minutes=configuration.get("timeout_minutes", 10),
            expected_participation_quality=self._estimate_participation_quality(contextual_metrics),
            context_adjustments_applied=self._get_applied_adjustments(configuration),
            risk_factors_identified=risk_factors,
            success_probability=success_probability,
            recommendation_timestamp=time.time(),
        )

        logger.debug(f"Generated recommendation: {strategy.value} for {complexity.value} decision")
        return recommendation

    def _enrich_metrics_with_context(self, metrics: ContextualMetrics, context: Dict[str, Any]) -> None:
        """Enrich metrics with additional context information."""
        metrics.decision_stakes_level = context.get("stakes_level", metrics.decision_stakes_level)
        metrics.time_pressure_level = context.get("time_pressure", metrics.time_pressure_level)
        metrics.reversibility_factor = context.get("reversibility", metrics.reversibility_factor)
        metrics.historical_agreement_rate = context.get("historical_agreement", metrics.historical_agreement_rate)

    def _prepare_strategy_context(self, metrics: ContextualMetrics, participants: List[str]) -> Dict[str, Any]:
        """Prepare context for strategy selection."""
        return {
            "participant_count": len(participants),
            "decision_stakes_level": metrics.decision_stakes_level,
            "time_pressure_level": metrics.time_pressure_level,
            "similar_decisions_success_rate": metrics.similar_decisions_success_rate,
        }

    def _prepare_contextual_analysis(self, metrics: ContextualMetrics) -> Dict[str, Any]:
        """Prepare detailed contextual analysis for the recommendation."""
        return {
            "proposal_length": metrics.proposal_length,
            "technical_complexity": metrics.technical_complexity_score,
            "participant_count": metrics.participant_count,
            "stakes_level": metrics.decision_stakes_level,
            "time_pressure": metrics.time_pressure_level,
            "reversibility": metrics.reversibility_factor,
            "options_available": metrics.available_options_count,
        }

    def _identify_risk_factors(self, metrics: ContextualMetrics, strategy: ConsensusStrategy) -> List[str]:
        """Identify potential risk factors for the decision process."""
        risks: List[str] = []

        if metrics.time_pressure_level > 0.7:
            risks.append("High time pressure may reduce deliberation quality")

        if metrics.decision_stakes_level > 0.8:
            risks.append("High stakes decision requires careful process management")

        if metrics.participant_count > 7:
            risks.append("Large participant count may complicate coordination")

        if metrics.reversibility_factor < 0.3:
            risks.append("Low reversibility increases decision pressure")

        if strategy == ConsensusStrategy.FAST_TRACK and metrics.decision_stakes_level > 0.6:
            risks.append("Fast track strategy may be inappropriate for high-stakes decision")

        return risks

    def _estimate_success_probability(self, strategy: ConsensusStrategy, metrics: ContextualMetrics) -> float:
        """Estimate probability of successful consensus based on historical data."""
        # Get strategy performance from learning framework
        strategy_insights = self.learning_framework.get_strategy_performance_insights()

        if strategy.value in strategy_insights:
            base_probability = strategy_insights[strategy.value]["success_rate"]
        else:
            # Default probabilities based on strategy complexity
            strategy_defaults = {
                ConsensusStrategy.FAST_TRACK: 0.85,
                ConsensusStrategy.STRUCTURED_VOTING: 0.80,
                ConsensusStrategy.DELIBERATIVE: 0.75,
                ConsensusStrategy.COLLABORATIVE: 0.70,
                ConsensusStrategy.CRITICAL_CONSENSUS: 0.65,
            }
            base_probability = strategy_defaults.get(strategy, 0.75)

        # Adjust based on context factors
        adjustments = 0.0

        if metrics.time_pressure_level > 0.7:
            adjustments -= 0.1  # Time pressure reduces success probability

        if metrics.participant_count > 6:
            adjustments -= 0.05  # More participants = more complexity

        if metrics.historical_agreement_rate > 0.8:
            adjustments += 0.1  # Good historical agreement helps

        if metrics.decision_stakes_level > 0.8:
            adjustments -= 0.05  # High stakes add pressure

        return max(0.1, min(0.95, base_probability + adjustments))

    def _estimate_participation_quality(self, metrics: ContextualMetrics) -> float:
        """Estimate expected quality of participant engagement."""
        base_quality = 0.7

        # Technical complexity can improve quality (more expertise needed)
        if metrics.technical_complexity_score > 3:
            base_quality += 0.1

        # Time pressure reduces quality
        if metrics.time_pressure_level > 0.7:
            base_quality -= 0.2

        # Moderate participant count is optimal
        if 3 <= metrics.participant_count <= 5:
            base_quality += 0.1
        elif metrics.participant_count > 7:
            base_quality -= 0.1

        return max(0.3, min(1.0, base_quality))

    def _get_applied_adjustments(self, configuration: Dict[str, Any]) -> List[str]:
        """Identify what adjustments were applied to the base strategy."""
        # This would track what contextual adjustments were made
        # For now, return empty list - would be enhanced in full implementation
        return []

    def record_outcome(self, recommendation: ConsensusRecommendation, actual_outcome: Dict[str, Any]) -> None:
        """
        Record the outcome of a consensus process for learning.

        Args:
            recommendation: The original recommendation
            actual_outcome: The actual outcome data
        """
        self.learning_framework.record_decision_outcome(recommendation, actual_outcome)
        logger.debug("Recorded consensus outcome for learning")

    def get_performance_insights(self) -> PerformanceInsights:
        """Get comprehensive performance insights about the system."""
        if not self.learning_framework.performance_history:
            return PerformanceInsights(
                overall_success_rate=0.0,
                average_efficiency_ratio=0.0,
                average_participant_satisfaction=0.0,
                total_decisions_processed=0,
                strategy_performance_breakdown={},
                recent_performance_trend="insufficient_data",
                performance_confidence_interval=(0.0, 0.0),
                most_successful_strategy="unknown",
                least_successful_strategy="unknown",
                common_failure_patterns=[],
                recommended_system_adjustments=[],
            )

        recent_records = self.learning_framework.performance_history[-20:]  # Last 20 decisions

        # Calculate overall metrics
        success_rate = sum(1 for r in recent_records if r["success"]) / len(recent_records)
        avg_efficiency = sum(r["efficiency_ratio"] for r in recent_records) / len(recent_records)
        avg_satisfaction = sum(r["satisfaction_score"] for r in recent_records) / len(recent_records)

        # Get strategy performance breakdown
        strategy_performance = self.learning_framework.get_strategy_performance_insights()

        # Determine performance trend
        if len(recent_records) >= 10:
            first_half = recent_records[: len(recent_records) // 2]
            second_half = recent_records[len(recent_records) // 2 :]

            first_success = sum(1 for r in first_half if r["success"]) / len(first_half)
            second_success = sum(1 for r in second_half if r["success"]) / len(second_half)

            if second_success > first_success + 0.1:
                trend = "improving"
            elif second_success < first_success - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Find best and worst performing strategies
        best_strategy = "unknown"
        worst_strategy = "unknown"
        if strategy_performance:
            best_strategy = max(
                strategy_performance.keys(), key=lambda s: strategy_performance[s]["average_success_score"]
            )
            worst_strategy = min(
                strategy_performance.keys(), key=lambda s: strategy_performance[s]["average_success_score"]
            )

        # Get system adjustment suggestions
        adjustments = self.learning_framework.suggest_strategy_adjustments()

        return PerformanceInsights(
            overall_success_rate=success_rate,
            average_efficiency_ratio=avg_efficiency,
            average_participant_satisfaction=avg_satisfaction,
            total_decisions_processed=len(self.learning_framework.performance_history),
            strategy_performance_breakdown=strategy_performance,
            recent_performance_trend=trend,
            performance_confidence_interval=(max(0.0, success_rate - 0.1), min(1.0, success_rate + 0.1)),
            most_successful_strategy=best_strategy,
            least_successful_strategy=worst_strategy,
            common_failure_patterns=[],  # Would analyze patterns in full implementation
            recommended_system_adjustments=adjustments,
        )
