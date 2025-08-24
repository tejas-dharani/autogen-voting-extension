"""
Semantic Vote Interpretation Engine

Advanced natural language understanding for vote interpretation that goes beyond
simple keyword matching to understand context, intent, and nuanced expressions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..core.voting_protocols import VoteType
from .natural_language_processor import NaturalLanguageProcessor

logger = logging.getLogger(__name__)


class VoteIntention(str, Enum):
    """
    Refined categorization of vote intentions beyond simple approve/reject.

    This enum captures the nuanced ways people express their voting intentions,
    allowing for more sophisticated analysis and decision-making.
    """

    STRONG_APPROVE = "strong_approve"  # Enthusiastic support
    CONDITIONAL_APPROVE = "conditional_approve"  # Approval with conditions
    WEAK_APPROVE = "weak_approve"  # Hesitant approval
    NEUTRAL_APPROVE = "neutral_approve"  # Neutral leaning approve
    ABSTAIN = "abstain"  # Neutral/undecided
    NEUTRAL_REJECT = "neutral_reject"  # Neutral leaning reject
    WEAK_REJECT = "weak_reject"  # Hesitant rejection
    CONDITIONAL_REJECT = "conditional_reject"  # Rejection with conditions
    STRONG_REJECT = "strong_reject"  # Strong opposition
    REQUEST_CLARIFICATION = "request_clarification"  # Need more information
    DEFER_DECISION = "defer"  # Defer to others/delay decision


class ConfidenceLevel(str, Enum):
    """Confidence levels for semantic interpretation accuracy."""

    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"  # 0.7-0.9
    MEDIUM = "medium"  # 0.5-0.7
    LOW = "low"  # 0.3-0.5
    VERY_LOW = "very_low"  # <0.3


@dataclass
class SemanticVoteResult:
    """
    Comprehensive result of semantic vote interpretation.

    Contains both the interpreted vote and rich contextual information
    that can be used for analysis and decision-making.
    """

    # Core vote classification
    vote_type: VoteType
    vote_intention: VoteIntention
    confidence: float
    confidence_level: ConfidenceLevel

    # Extracted semantic information
    reasoning_text: Optional[str] = None
    extracted_title: Optional[str] = None
    extracted_options: Optional[List[str]] = None
    conditions_identified: List[str] = field(default_factory=lambda: [])
    concerns_expressed: List[str] = field(default_factory=lambda: [])
    alternatives_suggested: List[str] = field(default_factory=lambda: [])
    questions_raised: List[str] = field(default_factory=lambda: [])

    # Contextual analysis
    sentiment_score: float = 0.0  # -1 (negative) to 1 (positive)
    urgency_level: float = 0.0  # 0 (no urgency) to 1 (urgent)
    certainty_level: float = 0.0  # 0 (uncertain) to 1 (certain)
    emotional_intensity: float = 0.0  # 0 (neutral) to 1 (highly emotional)

    # Processing metadata
    parsing_method: str = "semantic"
    fallback_used: bool = False
    processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.extracted_options is None:
            object.__setattr__(self, "extracted_options", [])

    def get_summary(self) -> str:
        """Generate a human-readable summary of the vote interpretation."""
        summary_parts = [
            f"Vote: {self.vote_type.value} ({self.vote_intention.value})",
            f"Confidence: {self.confidence_level.value} ({self.confidence:.2f})",
        ]

        if self.sentiment_score != 0.0:
            sentiment_desc = "positive" if self.sentiment_score > 0 else "negative"
            summary_parts.append(f"Sentiment: {sentiment_desc} ({self.sentiment_score:+.2f})")

        if self.conditions_identified:
            summary_parts.append(f"Conditions: {len(self.conditions_identified)}")

        if self.concerns_expressed:
            summary_parts.append(f"Concerns: {len(self.concerns_expressed)}")

        return "; ".join(summary_parts)

    @property
    def is_valid_proposal(self) -> bool:
        """Check if this result represents a valid proposal."""
        # A valid proposal should have reasonable confidence and content
        return (
            self.confidence >= 0.3
            and bool(self.reasoning_text)
            and len(self.reasoning_text.strip() if self.reasoning_text else "") > 10
        )


class SemanticVoteInterpreter:
    """
    Main semantic vote interpretation engine.

    Uses advanced natural language processing to understand vote intentions,
    extract contextual information, and provide confidence assessments.
    """

    def __init__(self) -> None:
        self.nlp_processor = NaturalLanguageProcessor()

        # Intention-to-vote mapping
        self.approval_intentions = {
            VoteIntention.STRONG_APPROVE,
            VoteIntention.CONDITIONAL_APPROVE,
            VoteIntention.WEAK_APPROVE,
            VoteIntention.NEUTRAL_APPROVE,
        }

        self.rejection_intentions = {
            VoteIntention.STRONG_REJECT,
            VoteIntention.CONDITIONAL_REJECT,
            VoteIntention.WEAK_REJECT,
            VoteIntention.NEUTRAL_REJECT,
        }

        # Fallback keyword sets for simple interpretation
        self.fallback_approve_keywords = {
            "approve",
            "accept",
            "yes",
            "support",
            "agree",
            "okay",
            "ok",
            "endorse",
            "back",
            "favor",
            "like",
            "good",
            "fine",
        }

        self.fallback_reject_keywords = {
            "reject",
            "deny",
            "no",
            "oppose",
            "disagree",
            "against",
            "dislike",
            "bad",
            "wrong",
            "refuse",
            "decline",
        }

        self.fallback_abstain_keywords = {
            "abstain",
            "neutral",
            "pass",
            "skip",
            "unsure",
            "undecided",
            "unclear",
            "defer",
            "postpone",
            "table",
        }

        logger.debug("SemanticVoteInterpreter initialized")

    def interpret_vote(self, message_text: str, context: Optional[Dict[str, Any]] = None) -> SemanticVoteResult:
        """
        Interpret a vote message using semantic analysis.

        Args:
            message_text: The text message to interpret
            context: Optional contextual information to aid interpretation

        Returns:
            SemanticVoteResult: Comprehensive interpretation result
        """
        start_time = self._get_current_time_ms()

        try:
            # Attempt sophisticated semantic interpretation
            result = self._perform_semantic_interpretation(message_text, context)

            # Validate confidence threshold
            if result.confidence >= 0.5:
                result.processing_time_ms = self._get_current_time_ms() - start_time
                return result
            else:
                logger.debug("Semantic interpretation confidence too low, falling back to simple parsing")

        except Exception as e:
            logger.warning(f"Semantic interpretation failed: {e}")

        # Fallback to simple keyword-based interpretation
        fallback_result = self._perform_fallback_interpretation(message_text)
        fallback_result.processing_time_ms = self._get_current_time_ms() - start_time
        return fallback_result

    def _perform_semantic_interpretation(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> SemanticVoteResult:
        """Perform sophisticated semantic interpretation of the vote message."""

        # Analyze content using NLP processor
        content_analysis = self.nlp_processor.analyze_content(text)

        # Extract vote intention using pattern matching
        vote_intention, base_confidence = self._extract_vote_intention(text)

        # Extract contextual information
        contextual_info = self._extract_contextual_information(text, content_analysis)

        # Map intention to vote type
        vote_type = self._map_intention_to_vote_type(vote_intention)

        # Adjust confidence based on contextual factors
        adjusted_confidence = self._adjust_confidence_score(base_confidence, content_analysis, contextual_info)

        # Determine confidence level
        confidence_level = self._calculate_confidence_level(adjusted_confidence)

        # Prepare reasoning text (truncated for brevity)
        reasoning = text[:200] + "..." if len(text) > 200 else text

        return SemanticVoteResult(
            vote_type=vote_type,
            vote_intention=vote_intention,
            confidence=adjusted_confidence,
            confidence_level=confidence_level,
            reasoning_text=reasoning,
            conditions_identified=contextual_info.get("conditions", []),
            concerns_expressed=contextual_info.get("concerns", []),
            alternatives_suggested=contextual_info.get("alternatives", []),
            questions_raised=contextual_info.get("questions", []),
            sentiment_score=content_analysis.sentiment_score,
            urgency_level=content_analysis.urgency_level,
            certainty_level=content_analysis.certainty_level,
            emotional_intensity=content_analysis.emotional_intensity,
            parsing_method="semantic",
        )

    def _extract_vote_intention(self, text: str) -> Tuple[VoteIntention, float]:
        """Extract vote intention using pattern matching and analysis."""

        # Use NLP processor to match against semantic patterns
        pattern_matches = self.nlp_processor.find_pattern_matches(text)

        # Find the highest confidence pattern match
        best_intention = VoteIntention.ABSTAIN
        best_confidence = 0.0

        for pattern_type, confidence in pattern_matches.items():
            if confidence > best_confidence:
                # Map pattern type to vote intention
                intention = self._map_pattern_to_intention(pattern_type)
                if intention:
                    best_intention = intention
                    best_confidence = confidence

        # If no strong pattern match, analyze general sentiment
        if best_confidence < 0.6:
            return self._analyze_general_sentiment(text)

        return best_intention, best_confidence

    def _map_pattern_to_intention(self, pattern_type: str) -> Optional[VoteIntention]:
        """Map detected pattern type to vote intention."""
        pattern_mapping = {
            "strong_approve": VoteIntention.STRONG_APPROVE,
            "conditional_approve": VoteIntention.CONDITIONAL_APPROVE,
            "weak_approve": VoteIntention.WEAK_APPROVE,
            "strong_reject": VoteIntention.STRONG_REJECT,
            "conditional_reject": VoteIntention.CONDITIONAL_REJECT,
            "weak_reject": VoteIntention.WEAK_REJECT,
            "abstain": VoteIntention.ABSTAIN,
            "clarification": VoteIntention.REQUEST_CLARIFICATION,
            "defer": VoteIntention.DEFER_DECISION,
        }
        return pattern_mapping.get(pattern_type)

    def _analyze_general_sentiment(self, text: str) -> Tuple[VoteIntention, float]:
        """Analyze general sentiment when no specific patterns match."""

        content_analysis = self.nlp_processor.analyze_content(text)

        sentiment = content_analysis.sentiment_score
        certainty = content_analysis.certainty_level

        # Map sentiment and certainty to intention
        if sentiment > 0.3 and certainty > 0.6:
            return VoteIntention.NEUTRAL_APPROVE, 0.6
        elif sentiment < -0.3 and certainty > 0.6:
            return VoteIntention.NEUTRAL_REJECT, 0.6
        elif certainty < 0.4:
            return VoteIntention.ABSTAIN, 0.5
        else:
            return VoteIntention.ABSTAIN, 0.3

    def _extract_contextual_information(self, text: str, content_analysis: Any) -> Dict[str, List[str]]:
        """Extract contextual information like conditions, concerns, etc."""

        contextual_info = {
            "conditions": self.nlp_processor.extract_conditions(text),
            "concerns": self.nlp_processor.extract_concerns(text),
            "alternatives": self.nlp_processor.extract_alternatives(text),
            "questions": self.nlp_processor.extract_questions(text),
        }

        return contextual_info

    def _map_intention_to_vote_type(self, intention: VoteIntention) -> VoteType:
        """Map vote intention to standard vote type."""

        if intention in self.approval_intentions:
            return VoteType.APPROVE
        elif intention in self.rejection_intentions:
            return VoteType.REJECT
        else:
            return VoteType.ABSTAIN

    def _adjust_confidence_score(
        self, base_confidence: float, content_analysis: Any, contextual_info: Dict[str, List[str]]
    ) -> float:
        """Adjust confidence based on contextual factors."""

        adjusted_confidence = base_confidence

        # Certainty level adjustment
        adjusted_confidence *= 0.5 + content_analysis.certainty_level * 0.5

        # Conditions reduce confidence (indicates uncertainty)
        if contextual_info["conditions"]:
            adjusted_confidence *= 1.0 - len(contextual_info["conditions"]) * 0.1

        # Concerns reduce confidence
        if contextual_info["concerns"]:
            adjusted_confidence *= 1.0 - len(contextual_info["concerns"]) * 0.05

        # Strong sentiment increases confidence
        if abs(content_analysis.sentiment_score) > 0.5:
            adjusted_confidence *= 1.1

        # Questions reduce confidence (indicates seeking information)
        if contextual_info["questions"]:
            adjusted_confidence *= 1.0 - len(contextual_info["questions"]) * 0.08

        return float(max(0.0, min(1.0, adjusted_confidence)))

    def _calculate_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numerical confidence to confidence level enum."""

        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _perform_fallback_interpretation(self, text: str) -> SemanticVoteResult:
        """Perform simple keyword-based interpretation as fallback."""

        words = set(text.lower().split())

        # Count keyword occurrences
        approve_count = len(words.intersection(self.fallback_approve_keywords))
        reject_count = len(words.intersection(self.fallback_reject_keywords))
        abstain_count = len(words.intersection(self.fallback_abstain_keywords))

        # Determine vote based on counts
        if abstain_count > 0:
            vote_type = VoteType.ABSTAIN
            vote_intention = VoteIntention.ABSTAIN
            confidence = 0.6
        elif approve_count > reject_count:
            vote_type = VoteType.APPROVE
            vote_intention = VoteIntention.NEUTRAL_APPROVE
            confidence = 0.5
        elif reject_count > approve_count:
            vote_type = VoteType.REJECT
            vote_intention = VoteIntention.NEUTRAL_REJECT
            confidence = 0.5
        else:
            # No clear indication
            vote_type = VoteType.ABSTAIN
            vote_intention = VoteIntention.ABSTAIN
            confidence = 0.3

        # Basic sentiment analysis for fallback
        basic_sentiment = self._calculate_basic_sentiment(words)

        return SemanticVoteResult(
            vote_type=vote_type,
            vote_intention=vote_intention,
            confidence=confidence,
            confidence_level=self._calculate_confidence_level(confidence),
            reasoning_text=text[:200] + "..." if len(text) > 200 else text,
            sentiment_score=basic_sentiment,
            urgency_level=0.0,
            certainty_level=0.5,
            emotional_intensity=0.0,
            parsing_method="fallback_keywords",
            fallback_used=True,
        )

    def _calculate_basic_sentiment(self, words: set[str]) -> float:
        """Calculate basic sentiment for fallback interpretation."""
        positive_words = {"good", "great", "excellent", "perfect", "like", "love", "agree"}
        negative_words = {"bad", "terrible", "awful", "hate", "dislike", "wrong", "disagree"}

        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0

        return (positive_count - negative_count) / total_sentiment_words

    def _get_current_time_ms(self) -> float:
        """Get current time in milliseconds."""
        import time

        return time.time() * 1000

    def interpret_batch_votes(
        self, vote_messages: List[Tuple[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SemanticVoteResult]:
        """
        Interpret multiple vote messages in batch for efficiency.

        Args:
            vote_messages: List of (voter_name, message_text) tuples
            context: Optional contextual information

        Returns:
            Dictionary mapping voter names to interpretation results
        """
        results: Dict[str, SemanticVoteResult] = {}

        for voter_name, message_text in vote_messages:
            try:
                results[voter_name] = self.interpret_vote(message_text, context)
            except Exception as e:
                logger.error(f"Failed to interpret vote from {voter_name}: {e}")
                # Create error fallback result
                results[voter_name] = SemanticVoteResult(
                    vote_type=VoteType.ABSTAIN,
                    vote_intention=VoteIntention.ABSTAIN,
                    confidence=0.0,
                    confidence_level=ConfidenceLevel.VERY_LOW,
                    reasoning_text="Interpretation error",
                    parsing_method="error_fallback",
                )

        return results

    def interpret_proposal(self, proposal_text: str) -> SemanticVoteResult:
        """
        Interpret a proposal message to understand its content and intent.

        Args:
            proposal_text: The proposal text to interpret

        Returns:
            SemanticVoteResult: Analysis of the proposal content
        """
        # For proposals, we analyze the content but don't assign a vote type
        # since proposals don't contain votes, just the subject matter
        try:
            content_analysis = self.nlp_processor.analyze_content(proposal_text)

            # Extract contextual information from the proposal
            contextual_info = self._extract_contextual_information(proposal_text, content_analysis)

            # Extract a title from the proposal text (first line or first 50 chars)
            lines = proposal_text.split("\n")
            extracted_title = lines[0][:50] + "..." if len(lines[0]) > 50 else lines[0]
            if not extracted_title.strip():
                extracted_title = proposal_text[:50] + "..." if len(proposal_text) > 50 else proposal_text

            # Extract default voting options (can be overridden by specific proposals)
            extracted_options = ["Approve", "Reject", "Abstain"]

            # Proposals don't have votes, so we use abstain as neutral
            return SemanticVoteResult(
                vote_type=VoteType.ABSTAIN,
                vote_intention=VoteIntention.ABSTAIN,
                confidence=0.8,  # High confidence that this is a proposal
                confidence_level=ConfidenceLevel.HIGH,
                reasoning_text=proposal_text[:200] + "..." if len(proposal_text) > 200 else proposal_text,
                extracted_title=extracted_title,
                extracted_options=extracted_options,
                conditions_identified=contextual_info.get("conditions", []),
                concerns_expressed=contextual_info.get("concerns", []),
                alternatives_suggested=contextual_info.get("alternatives", []),
                questions_raised=contextual_info.get("questions", []),
                sentiment_score=content_analysis.sentiment_score,
                urgency_level=content_analysis.urgency_level,
                certainty_level=content_analysis.certainty_level,
                emotional_intensity=content_analysis.emotional_intensity,
                parsing_method="proposal_analysis",
            )
        except Exception as e:
            logger.warning(f"Proposal interpretation failed: {e}")
            # Fallback for proposals
            return SemanticVoteResult(
                vote_type=VoteType.ABSTAIN,
                vote_intention=VoteIntention.ABSTAIN,
                confidence=0.5,
                confidence_level=ConfidenceLevel.MEDIUM,
                reasoning_text="Proposal analysis fallback",
                extracted_title="Proposal",
                extracted_options=["Approve", "Reject"],
                parsing_method="proposal_fallback",
            )

    def interpret_vote_content(self, vote_content: str) -> SemanticVoteResult:
        """
        Interpret vote content to understand voting intention.

        This is an alias for interpret_vote to maintain compatibility.

        Args:
            vote_content: The vote content to interpret

        Returns:
            SemanticVoteResult: Vote interpretation result
        """
        return self.interpret_vote(vote_content)

    def get_interpretation_statistics(self, results: List[SemanticVoteResult]) -> Dict[str, Any]:
        """
        Calculate statistics about interpretation performance.

        Args:
            results: List of interpretation results to analyze

        Returns:
            Dictionary containing performance statistics
        """
        if not results:
            return {}

        total_results = len(results)
        semantic_results = sum(1 for r in results if not r.fallback_used)
        fallback_results = sum(1 for r in results if r.fallback_used)

        # Calculate confidence distribution
        confidence_distribution: Dict[str, int] = {}
        intention_distribution: Dict[str, int] = {}

        avg_confidence = sum(r.confidence for r in results) / total_results
        avg_processing_time = sum(r.processing_time_ms for r in results) / total_results

        for result in results:
            conf_level = result.confidence_level.value
            confidence_distribution[conf_level] = confidence_distribution.get(conf_level, 0) + 1

            intention = result.vote_intention.value
            intention_distribution[intention] = intention_distribution.get(intention, 0) + 1

        return {
            "total_votes_processed": total_results,
            "semantic_interpretation_rate": semantic_results / total_results,
            "fallback_rate": fallback_results / total_results,
            "average_confidence": avg_confidence,
            "average_processing_time_ms": avg_processing_time,
            "confidence_distribution": confidence_distribution,
            "intention_distribution": intention_distribution,
            "contextual_elements_found": {
                "conditions": sum(len(r.conditions_identified) for r in results),
                "concerns": sum(len(r.concerns_expressed) for r in results),
                "alternatives": sum(len(r.alternatives_suggested) for r in results),
                "questions": sum(len(r.questions_raised) for r in results),
            },
        }
