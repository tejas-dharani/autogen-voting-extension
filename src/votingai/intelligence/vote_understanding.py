"""
Vote Understanding Engine

High-level interface that combines semantic interpretation with natural language
processing to provide comprehensive vote understanding capabilities.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .natural_language_processor import NaturalLanguageProcessor
from .semantic_interpreter import SemanticVoteInterpreter, SemanticVoteResult

logger = logging.getLogger(__name__)


@dataclass
class ParsingStatistics:
    """Statistics about vote parsing performance."""

    total_votes_processed: int
    semantic_success_rate: float
    fallback_rate: float
    average_confidence: float
    average_processing_time_ms: float
    confidence_distribution: Dict[str, int]
    intention_distribution: Dict[str, int]
    contextual_elements_extracted: Dict[str, int]


class IntentionClassifier:
    """Classifies vote intentions with advanced heuristics."""

    def __init__(self) -> None:
        self.classification_rules = {
            "high_confidence_approval": {
                "sentiment_threshold": 0.6,
                "certainty_threshold": 0.7,
                "patterns": ["strong_approve", "conditional_approve"],
            },
            "high_confidence_rejection": {
                "sentiment_threshold": -0.6,
                "certainty_threshold": 0.7,
                "patterns": ["strong_reject", "conditional_reject"],
            },
            "uncertain_vote": {"certainty_threshold": 0.3, "patterns": ["abstain", "clarification"]},
        }

    def classify_intention_strength(self, result: SemanticVoteResult) -> str:
        """Classify the strength of vote intention."""
        if result.confidence >= 0.8 and result.certainty_level >= 0.7:
            return "high_confidence"
        elif result.confidence >= 0.6:
            return "moderate_confidence"
        elif result.confidence >= 0.4:
            return "low_confidence"
        else:
            return "very_uncertain"


class MessageInsightExtractor:
    """Extracts insights and metadata from vote messages."""

    def __init__(self) -> None:
        self.insight_patterns = {
            "expertise_indicators": [
                "experience",
                "background",
                "expertise",
                "knowledge",
                "familiar",
                "worked with",
                "seen this before",
            ],
            "time_constraints": ["deadline", "urgent", "quickly", "asap", "time", "schedule", "timeline", "rush"],
            "collaboration_signals": ["team", "together", "collaborate", "discuss", "work with", "coordinate", "align"],
        }

    def extract_insights(self, text: str) -> Dict[str, Any]:
        """Extract various insights from vote message."""
        insights = {
            "has_expertise_indicators": self._has_patterns(text, "expertise_indicators"),
            "mentions_time_constraints": self._has_patterns(text, "time_constraints"),
            "suggests_collaboration": self._has_patterns(text, "collaboration_signals"),
            "message_length_category": self._categorize_length(text),
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
        }
        return insights

    def _has_patterns(self, text: str, pattern_type: str) -> bool:
        """Check if text contains patterns of given type."""
        patterns = self.insight_patterns.get(pattern_type, [])
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in patterns)

    def _categorize_length(self, text: str) -> str:
        """Categorize message length."""
        length = len(text)
        if length < 50:
            return "short"
        elif length < 200:
            return "medium"
        else:
            return "long"


class VoteUnderstandingEngine:
    """
    Main engine that orchestrates vote understanding capabilities.

    Provides a high-level interface for comprehensive vote analysis
    that combines multiple NLP and semantic analysis techniques.
    """

    def __init__(self) -> None:
        self.semantic_interpreter = SemanticVoteInterpreter()
        self.nlp_processor = NaturalLanguageProcessor()
        self.intention_classifier = IntentionClassifier()
        self.insight_extractor = MessageInsightExtractor()

        # Performance tracking
        self.processing_history: List[Dict[str, Any]] = []

        logger.info("VoteUnderstandingEngine initialized")

    def understand_vote_message(
        self, voter_name: str, message_text: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive understanding of a vote message.

        Args:
            voter_name: Name of the voter
            message_text: The vote message text
            context: Optional contextual information

        Returns:
            Comprehensive understanding result with semantic interpretation,
            insights, and metadata
        """
        # Perform semantic interpretation
        semantic_result = self.semantic_interpreter.interpret_vote(message_text, context)

        # Extract additional insights
        message_insights = self.insight_extractor.extract_insights(message_text)

        # Classify intention strength
        intention_strength = self.intention_classifier.classify_intention_strength(semantic_result)

        # Perform content analysis
        content_analysis = self.nlp_processor.analyze_content(message_text)

        # Compile comprehensive result
        understanding_result = {
            "voter_name": voter_name,
            "semantic_interpretation": semantic_result,
            "intention_strength": intention_strength,
            "message_insights": message_insights,
            "content_analysis": content_analysis,
            "processing_metadata": {
                "timestamp": self._get_current_timestamp(),
                "context_provided": context is not None,
                "message_length": len(message_text),
            },
        }

        # Track for performance analysis
        self._track_processing_result(understanding_result)

        return understanding_result

    def understand_batch_votes(
        self, vote_messages: List[Tuple[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Understand multiple vote messages in batch.

        Args:
            vote_messages: List of (voter_name, message_text) tuples
            context: Optional contextual information

        Returns:
            Dictionary mapping voter names to understanding results
        """
        results: Dict[str, Dict[str, Any]] = {}

        for voter_name, message_text in vote_messages:
            try:
                results[voter_name] = self.understand_vote_message(voter_name, message_text, context)
            except Exception as e:
                logger.error(f"Failed to understand vote from {voter_name}: {e}")
                results[voter_name] = self._create_error_result(voter_name, str(e))

        return results

    def analyze_voting_patterns(self, understanding_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns across multiple vote understanding results.

        Args:
            understanding_results: List of understanding results to analyze

        Returns:
            Pattern analysis including trends, consensus indicators, etc.
        """
        if not understanding_results:
            return {}

        # Aggregate semantic interpretations
        vote_types: defaultdict[str, int] = defaultdict(int)
        intentions: defaultdict[str, int] = defaultdict(int)
        confidence_levels: defaultdict[str, int] = defaultdict(int)

        total_confidence = 0.0
        total_sentiment = 0.0
        total_certainty = 0.0

        conditions_found = 0
        concerns_found = 0
        questions_found = 0

        for result in understanding_results:
            semantic = result["semantic_interpretation"]

            vote_types[semantic.vote_type.value] += 1
            intentions[semantic.vote_intention.value] += 1
            confidence_levels[semantic.confidence_level.value] += 1

            total_confidence += semantic.confidence
            total_sentiment += semantic.sentiment_score
            total_certainty += semantic.certainty_level

            conditions_found += len(semantic.conditions_identified)
            concerns_found += len(semantic.concerns_expressed)
            questions_found += len(semantic.questions_raised)

        count = len(understanding_results)

        # Calculate consensus indicators
        dominant_vote_type = max(vote_types.items(), key=lambda x: x[1])
        consensus_strength = dominant_vote_type[1] / count

        # Identify potential issues
        potential_issues: List[str] = []
        if concerns_found > count * 0.3:
            potential_issues.append("High number of concerns expressed")
        if questions_found > count * 0.2:
            potential_issues.append("Many clarification requests")
        if total_certainty / count < 0.5:
            potential_issues.append("Low overall certainty")

        return {
            "vote_distribution": dict(vote_types),
            "intention_distribution": dict(intentions),
            "confidence_distribution": dict(confidence_levels),
            "consensus_indicators": {
                "dominant_vote_type": dominant_vote_type[0],
                "consensus_strength": consensus_strength,
                "average_confidence": total_confidence / count,
                "average_sentiment": total_sentiment / count,
                "average_certainty": total_certainty / count,
            },
            "contextual_elements": {
                "conditions_identified": conditions_found,
                "concerns_expressed": concerns_found,
                "questions_raised": questions_found,
            },
            "potential_issues": potential_issues,
            "analysis_metadata": {"total_votes_analyzed": count, "analysis_timestamp": self._get_current_timestamp()},
        }

    def get_parsing_statistics(self) -> ParsingStatistics:
        """Get comprehensive parsing statistics."""
        if not self.processing_history:
            return ParsingStatistics(
                total_votes_processed=0,
                semantic_success_rate=0.0,
                fallback_rate=0.0,
                average_confidence=0.0,
                average_processing_time_ms=0.0,
                confidence_distribution={},
                intention_distribution={},
                contextual_elements_extracted={},
            )

        total_processed = len(self.processing_history)
        semantic_successes = sum(
            1 for result in self.processing_history if not result["semantic_interpretation"].fallback_used
        )

        avg_confidence = (
            sum(result["semantic_interpretation"].confidence for result in self.processing_history) / total_processed
        )

        avg_processing_time = (
            sum(result["semantic_interpretation"].processing_time_ms for result in self.processing_history)
            / total_processed
        )

        # Build distributions
        confidence_dist: defaultdict[str, int] = defaultdict(int)
        intention_dist: defaultdict[str, int] = defaultdict(int)

        for result in self.processing_history:
            semantic = result["semantic_interpretation"]
            confidence_dist[semantic.confidence_level.value] += 1
            intention_dist[semantic.vote_intention.value] += 1

        contextual_elements = {
            "conditions": sum(
                len(result["semantic_interpretation"].conditions_identified) for result in self.processing_history
            ),
            "concerns": sum(
                len(result["semantic_interpretation"].concerns_expressed) for result in self.processing_history
            ),
            "questions": sum(
                len(result["semantic_interpretation"].questions_raised) for result in self.processing_history
            ),
        }

        return ParsingStatistics(
            total_votes_processed=total_processed,
            semantic_success_rate=semantic_successes / total_processed,
            fallback_rate=(total_processed - semantic_successes) / total_processed,
            average_confidence=avg_confidence,
            average_processing_time_ms=avg_processing_time,
            confidence_distribution=dict(confidence_dist),
            intention_distribution=dict(intention_dist),
            contextual_elements_extracted=contextual_elements,
        )

    def _track_processing_result(self, result: Dict[str, Any]) -> None:
        """Track processing result for performance analysis."""
        self.processing_history.append(result)

        # Keep only recent history to prevent memory issues
        max_history = 1000
        if len(self.processing_history) > max_history:
            self.processing_history = self.processing_history[-max_history:]

    def _create_error_result(self, voter_name: str, error_message: str) -> Dict[str, Any]:
        """Create error result for failed vote understanding."""
        from ..core.voting_protocols import VoteType
        from .semantic_interpreter import ConfidenceLevel, SemanticVoteResult, VoteIntention

        error_semantic_result = SemanticVoteResult(
            vote_type=VoteType.ABSTAIN,
            vote_intention=VoteIntention.ABSTAIN,
            confidence=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            reasoning_text=f"Processing error: {error_message}",
            parsing_method="error_handling",
        )

        return {
            "voter_name": voter_name,
            "semantic_interpretation": error_semantic_result,
            "intention_strength": "error",
            "message_insights": {},
            "content_analysis": None,
            "processing_metadata": {"timestamp": self._get_current_timestamp(), "error": error_message},
        }

    def _get_current_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()
