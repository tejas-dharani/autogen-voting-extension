"""
Structured Deliberation Engine

Manages structured deliberation rounds with convergence tracking and quality assessment.
Refactored from StructuredDeliberation class with enhanced quality standards.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeliberationRound:
    """
    Represents one round of structured deliberation.
    
    Tracks participant messages, extracted insights, and convergence metrics
    for a single deliberation round in the consensus process.
    """
    
    round_number: int
    participants: List[str]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    concerns_identified: List[str] = field(default_factory=list)
    alternatives_proposed: List[str] = field(default_factory=list)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    
    def add_message(self, speaker: str, content: str, timestamp: Optional[float] = None) -> None:
        """Add a message to this deliberation round."""
        message_entry = {
            'speaker': speaker,
            'content': content,
            'timestamp': timestamp or time.time(),
            'message_length': len(content),
            'word_count': len(content.split())
        }
        self.messages.append(message_entry)
    
    def calculate_participation_balance(self) -> float:
        """Calculate how balanced participation is among participants."""
        if not self.messages:
            return 0.0
        
        speaker_counts = defaultdict(int)
        for message in self.messages:
            speaker_counts[message['speaker']] += 1
        
        # Calculate coefficient of variation (lower = more balanced)
        counts = list(speaker_counts.values())
        if len(counts) <= 1:
            return 1.0
        
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
        
        cv = std_count / mean_count
        # Convert to balance score (higher = more balanced)
        balance_score = max(0.0, 1.0 - cv)
        return balance_score


@dataclass 
class DeliberationSummary:
    """
    Comprehensive summary of the deliberation process.
    
    Provides aggregated insights and metrics across all deliberation rounds.
    """
    
    total_rounds_completed: int
    total_messages_exchanged: int
    total_insights_extracted: int
    total_concerns_raised: int
    total_alternatives_proposed: int
    final_convergence_score: float
    participation_balance_score: float
    overall_quality_score: float
    participant_position_evolution: Dict[str, Dict[str, float]]
    key_decision_factors: List[str]
    unresolved_concerns: List[str]
    
    def generate_executive_summary(self) -> str:
        """Generate a concise executive summary of the deliberation."""
        summary_parts = [
            f"Completed {self.total_rounds_completed} deliberation rounds",
            f"with {self.total_messages_exchanged} messages exchanged.",
            f"Extracted {self.total_insights_extracted} key insights",
            f"and identified {self.total_concerns_raised} concerns.",
            f"Final convergence: {self.final_convergence_score:.1%}",
            f"(Quality score: {self.overall_quality_score:.2f})"
        ]
        
        if self.total_alternatives_proposed > 0:
            summary_parts.append(f"Generated {self.total_alternatives_proposed} alternatives.")
        
        if self.unresolved_concerns:
            summary_parts.append(f"Unresolved concerns: {len(self.unresolved_concerns)}")
        
        return " ".join(summary_parts)


class ConvergenceAnalyzer:
    """
    Analyzes convergence patterns in deliberation rounds.
    
    Tracks how participant positions evolve and determines when
    sufficient convergence has been achieved for effective voting.
    """
    
    def __init__(self, convergence_threshold: float = 0.8):
        self.convergence_threshold = convergence_threshold
        self.participant_positions: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.position_history: List[Dict[str, Dict[str, float]]] = []
        
        # Sentiment analysis keywords (simplified NLP)
        self.support_indicators = {
            'agree', 'support', 'approve', 'good', 'excellent', 'correct',
            'right', 'yes', 'endorse', 'back', 'favor', 'like'
        }
        
        self.opposition_indicators = {
            'disagree', 'oppose', 'reject', 'wrong', 'bad', 'concerned',
            'no', 'against', 'dislike', 'problematic', 'issue', 'worry'
        }
        
        self.neutral_indicators = {
            'question', 'clarify', 'understand', 'explain', 'what', 'how',
            'unclear', 'consider', 'think', 'maybe', 'perhaps', 'possibly'
        }
    
    def update_participant_position(self, speaker: str, message: str) -> None:
        """Update position tracking based on participant's message."""
        message_lower = message.lower()
        words = set(message_lower.split())
        
        # Count sentiment indicators
        support_score = len(words.intersection(self.support_indicators))
        opposition_score = len(words.intersection(self.opposition_indicators))
        neutral_score = len(words.intersection(self.neutral_indicators))
        
        total_indicators = support_score + opposition_score + neutral_score
        
        if total_indicators > 0:
            # Normalize scores
            support_ratio = support_score / total_indicators
            opposition_ratio = opposition_score / total_indicators
            neutral_ratio = neutral_score / total_indicators
            
            # Update position with weighted averaging (gives recent messages more weight)
            current_position = self.participant_positions[speaker]
            alpha = 0.7  # Weight for new information
            
            current_position['support'] = (
                alpha * support_ratio + 
                (1 - alpha) * current_position.get('support', 0.5)
            )
            current_position['opposition'] = (
                alpha * opposition_ratio + 
                (1 - alpha) * current_position.get('opposition', 0.5)
            )
            current_position['neutral'] = (
                alpha * neutral_ratio + 
                (1 - alpha) * current_position.get('neutral', 0.5)
            )
    
    def calculate_current_convergence(self) -> float:
        """Calculate current convergence level among participants."""
        if not self.participant_positions:
            return 0.0
        
        # Extract position vectors for all participants
        support_scores = [pos.get('support', 0.5) for pos in self.participant_positions.values()]
        opposition_scores = [pos.get('opposition', 0.5) for pos in self.participant_positions.values()]
        
        if len(support_scores) < 2:
            return 0.0
        
        # Calculate variance in positions (lower variance = higher convergence)
        support_variance = np.var(support_scores)
        opposition_variance = np.var(opposition_scores)
        
        # Combined variance (normalized)
        combined_variance = (support_variance + opposition_variance) / 2
        
        # Convert to convergence score (0 = no convergence, 1 = perfect convergence)
        convergence_score = 1.0 - min(1.0, combined_variance * 4)  # Scale factor of 4
        
        return max(0.0, convergence_score)
    
    def analyze_position_trends(self) -> Dict[str, str]:
        """Analyze how participant positions are trending."""
        if len(self.position_history) < 2:
            return {}
        
        trends = {}
        current_positions = self.participant_positions
        previous_positions = self.position_history[-1] if self.position_history else {}
        
        for participant, current_pos in current_positions.items():
            if participant in previous_positions:
                prev_pos = previous_positions[participant]
                
                # Calculate change in support level
                support_change = current_pos.get('support', 0.5) - prev_pos.get('support', 0.5)
                
                if support_change > 0.1:
                    trends[participant] = "increasing_support"
                elif support_change < -0.1:
                    trends[participant] = "decreasing_support"
                else:
                    trends[participant] = "stable_position"
            else:
                trends[participant] = "new_participant"
        
        return trends
    
    def should_continue_deliberation(self) -> bool:
        """Determine if deliberation should continue based on convergence."""
        current_convergence = self.calculate_current_convergence()
        return current_convergence < self.convergence_threshold
    
    def snapshot_current_positions(self) -> None:
        """Take a snapshot of current positions for trend analysis."""
        position_snapshot = {}
        for participant, position in self.participant_positions.items():
            position_snapshot[participant] = position.copy()
        self.position_history.append(position_snapshot)


class StructuredDeliberationEngine:
    """
    Main engine for managing structured deliberation processes.
    
    Orchestrates deliberation rounds, tracks convergence, and provides
    comprehensive analysis of the deliberation quality and outcomes.
    """
    
    def __init__(self, max_rounds: int = 3, convergence_threshold: float = 0.8):
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.rounds: List[DeliberationRound] = []
        self.convergence_analyzer = ConvergenceAnalyzer(convergence_threshold)
        self.content_analyzer = ContentInsightExtractor()
    
    def start_new_round(self, round_number: int, participants: List[str]) -> DeliberationRound:
        """Initialize and start a new deliberation round."""
        deliberation_round = DeliberationRound(
            round_number=round_number,
            participants=participants,
            start_timestamp=time.time()
        )
        
        self.rounds.append(deliberation_round)
        logger.debug(f"Started deliberation round {round_number} with {len(participants)} participants")
        
        return deliberation_round
    
    def process_deliberation_message(
        self, 
        round_obj: DeliberationRound, 
        speaker: str, 
        message: str
    ) -> None:
        """Process a message within a deliberation round."""
        # Add message to round
        round_obj.add_message(speaker, message)
        
        # Update convergence tracking
        self.convergence_analyzer.update_participant_position(speaker, message)
        
        # Extract insights from message
        insights = self.content_analyzer.extract_insights(message)
        round_obj.key_insights.extend(insights.key_points)
        round_obj.concerns_identified.extend(insights.concerns)
        round_obj.alternatives_proposed.extend(insights.alternatives)
        
        logger.debug(f"Processed message from {speaker} in round {round_obj.round_number}")
    
    def complete_current_round(self) -> Optional[DeliberationRound]:
        """Complete the current deliberation round and calculate metrics."""
        if not self.rounds:
            return None
        
        current_round = self.rounds[-1]
        current_round.end_timestamp = time.time()
        
        if current_round.start_timestamp:
            current_round.duration_seconds = (
                current_round.end_timestamp - current_round.start_timestamp
            )
        
        # Calculate round-specific metrics
        current_round.convergence_metrics = {
            'convergence_score': self.convergence_analyzer.calculate_current_convergence(),
            'participation_balance': current_round.calculate_participation_balance(),
            'message_count': len(current_round.messages),
            'insights_extracted': len(current_round.key_insights)
        }
        
        # Snapshot positions for trend analysis
        self.convergence_analyzer.snapshot_current_positions()
        
        logger.debug(f"Completed round {current_round.round_number}")
        return current_round
    
    def should_continue_deliberation(self) -> bool:
        """Determine if deliberation should continue."""
        # Check if we've reached maximum rounds
        if len(self.rounds) >= self.max_rounds:
            return False
        
        # Check convergence level
        return self.convergence_analyzer.should_continue_deliberation()
    
    def generate_comprehensive_summary(self) -> DeliberationSummary:
        """Generate comprehensive summary of all deliberation rounds."""
        total_messages = sum(len(round_obj.messages) for round_obj in self.rounds)
        total_insights = sum(len(round_obj.key_insights) for round_obj in self.rounds)
        total_concerns = sum(len(round_obj.concerns_identified) for round_obj in self.rounds)
        total_alternatives = sum(len(round_obj.alternatives_proposed) for round_obj in self.rounds)
        
        # Calculate overall participation balance
        all_messages = []
        for round_obj in self.rounds:
            all_messages.extend(round_obj.messages)
        
        participation_balance = self._calculate_overall_participation_balance(all_messages)
        
        # Calculate quality score
        quality_score = self._calculate_deliberation_quality()
        
        # Extract key decision factors and unresolved concerns
        key_factors = self._extract_key_decision_factors()
        unresolved_concerns = self._identify_unresolved_concerns()
        
        return DeliberationSummary(
            total_rounds_completed=len(self.rounds),
            total_messages_exchanged=total_messages,
            total_insights_extracted=total_insights,
            total_concerns_raised=total_concerns,
            total_alternatives_proposed=total_alternatives,
            final_convergence_score=self.convergence_analyzer.calculate_current_convergence(),
            participation_balance_score=participation_balance,
            overall_quality_score=quality_score,
            participant_position_evolution=dict(self.convergence_analyzer.participant_positions),
            key_decision_factors=key_factors,
            unresolved_concerns=unresolved_concerns
        )
    
    def _calculate_overall_participation_balance(self, all_messages: List[Dict[str, Any]]) -> float:
        """Calculate overall participation balance across all rounds."""
        if not all_messages:
            return 0.0
        
        speaker_counts = defaultdict(int)
        for message in all_messages:
            speaker_counts[message['speaker']] += 1
        
        counts = list(speaker_counts.values())
        if len(counts) <= 1:
            return 1.0
        
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
        
        cv = std_count / mean_count
        return max(0.0, 1.0 - cv)
    
    def _calculate_deliberation_quality(self) -> float:
        """Calculate overall quality score for the deliberation process."""
        if not self.rounds:
            return 0.0
        
        # Quality factors
        convergence_score = self.convergence_analyzer.calculate_current_convergence()
        participation_score = self._calculate_overall_participation_balance([
            msg for round_obj in self.rounds for msg in round_obj.messages
        ])
        
        # Content quality (based on insights extracted)
        total_messages = sum(len(round_obj.messages) for round_obj in self.rounds)
        total_insights = sum(len(round_obj.key_insights) for round_obj in self.rounds)
        content_quality = min(1.0, total_insights / max(1, total_messages * 0.3))
        
        # Weighted combination
        quality_score = (
            convergence_score * 0.4 +
            participation_score * 0.3 +
            content_quality * 0.3
        )
        
        return quality_score
    
    def _extract_key_decision_factors(self) -> List[str]:
        """Extract the most important decision factors from deliberation."""
        all_insights = []
        for round_obj in self.rounds:
            all_insights.extend(round_obj.key_insights)
        
        # Simple frequency-based extraction (could be enhanced with NLP)
        return list(set(all_insights))[:5]  # Top 5 unique insights
    
    def _identify_unresolved_concerns(self) -> List[str]:
        """Identify concerns that were raised but not adequately addressed."""
        all_concerns = []
        for round_obj in self.rounds:
            all_concerns.extend(round_obj.concerns_identified)
        
        # Simple approach: return unique concerns (could be enhanced to check resolution)
        return list(set(all_concerns))


class ContentInsightExtractor:
    """
    Extracts insights, concerns, and alternatives from deliberation messages.
    
    Uses keyword-based analysis to identify important content elements
    during deliberation rounds.
    """
    
    def __init__(self):
        self.insight_indicators = {
            'important', 'key', 'critical', 'main', 'primary', 'essential',
            'crucial', 'significant', 'notable', 'insight', 'realize', 'understand'
        }
        
        self.concern_indicators = {
            'concern', 'worry', 'issue', 'problem', 'risk', 'challenge',
            'difficulty', 'obstacle', 'limitation', 'drawback', 'disadvantage'
        }
        
        self.alternative_indicators = {
            'alternative', 'option', 'instead', 'rather', 'alternatively',
            'suggestion', 'propose', 'recommend', 'consider', 'what if'
        }
    
    def extract_insights(self, message: str) -> 'MessageInsights':
        """Extract key insights from a deliberation message."""
        message_lower = message.lower()
        words = set(message_lower.split())
        
        insights = MessageInsights()
        
        # Extract key points
        if words.intersection(self.insight_indicators):
            insights.key_points.append(self._extract_content_snippet(message, self.insight_indicators))
        
        # Extract concerns
        if words.intersection(self.concern_indicators):
            insights.concerns.append(self._extract_content_snippet(message, self.concern_indicators))
        
        # Extract alternatives
        if words.intersection(self.alternative_indicators):
            insights.alternatives.append(self._extract_content_snippet(message, self.alternative_indicators))
        
        return insights
    
    def _extract_content_snippet(self, message: str, indicators: set) -> str:
        """Extract relevant content snippet around indicator words."""
        # Simple implementation: return first 100 characters
        # In practice, would use more sophisticated NLP to extract relevant sentences
        return message[:100] + "..." if len(message) > 100 else message


@dataclass
class MessageInsights:
    """Container for insights extracted from a single message."""
    key_points: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)