"""
Evaluation Metrics Framework

Comprehensive metrics collection and analysis for systematic evaluation
of voting systems performance, quality, and effectiveness.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Research-grade quality metrics for decision evaluation."""
    
    # Decision quality
    decision_accuracy: float = 0.0          # Accuracy vs ground truth (if available)
    consensus_strength: float = 0.0         # How strong the agreement was
    participant_satisfaction: float = 0.0   # Average satisfaction with process
    
    # Process quality
    deliberation_depth: float = 0.0         # Quality of discussion
    argument_quality: float = 0.0           # Quality of reasoning provided
    information_utilization: float = 0.0    # How well available info was used
    
    # Outcome quality
    solution_completeness: float = 0.0      # How complete the solution is
    implementation_feasibility: float = 0.0 # How feasible to implement
    stakeholder_alignment: float = 0.0      # Alignment with stakeholder needs
    
    def calculate_overall_quality(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'decision': 0.3,    # Decision-related metrics
            'process': 0.4,     # Process-related metrics  
            'outcome': 0.3      # Outcome-related metrics
        }
        
        decision_score = (self.decision_accuracy + self.consensus_strength + 
                         self.participant_satisfaction) / 3
        
        process_score = (self.deliberation_depth + self.argument_quality + 
                        self.information_utilization) / 3
        
        outcome_score = (self.solution_completeness + self.implementation_feasibility + 
                        self.stakeholder_alignment) / 3
        
        overall = (decision_score * weights['decision'] + 
                  process_score * weights['process'] + 
                  outcome_score * weights['outcome'])
        
        return min(1.0, max(0.0, overall))


@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics."""
    
    # Timing metrics
    total_duration_seconds: float = 0.0
    deliberation_time_seconds: float = 0.0
    voting_time_seconds: float = 0.0
    setup_time_seconds: float = 0.0
    
    # Communication metrics
    total_messages: int = 0
    total_tokens: int = 0
    average_message_length: float = 0.0
    deliberation_messages: int = 0
    voting_messages: int = 0
    
    # Resource utilization
    api_calls_made: int = 0
    api_tokens_consumed: int = 0
    estimated_cost_usd: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Efficiency ratios
    time_per_decision: float = 0.0
    tokens_per_decision: float = 0.0
    messages_per_participant: float = 0.0
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        # Lower values are better for efficiency
        time_factor = max(0.0, 1.0 - (self.total_duration_seconds / 300))  # 5 min baseline
        message_factor = max(0.0, 1.0 - (self.total_messages / 50))        # 50 msg baseline
        token_factor = max(0.0, 1.0 - (self.total_tokens / 10000))         # 10k token baseline
        
        return (time_factor + message_factor + token_factor) / 3


@dataclass
class ConsensusMetrics:
    """Metrics specific to consensus formation and quality."""
    
    # Consensus formation
    rounds_to_consensus: int = 0
    convergence_rate: float = 0.0           # How quickly positions converged
    final_agreement_level: float = 0.0      # Final level of agreement
    decision_stability: float = 0.0         # How stable the decision is
    
    # Participation metrics
    participation_balance: float = 0.0      # How balanced participation was
    expertise_utilization: float = 0.0     # How well expertise was utilized
    minority_voice_inclusion: float = 0.0  # Inclusion of minority perspectives
    
    # Deliberation quality
    perspective_diversity: float = 0.0      # Diversity of perspectives shared
    constructive_discourse: float = 0.0    # Quality of discourse
    conflict_resolution: float = 0.0       # How well conflicts were resolved
    
    # Adaptive features (if applicable)
    strategy_appropriateness: float = 0.0  # How appropriate the chosen strategy was
    complexity_handling: float = 0.0       # How well complexity was handled
    learning_effectiveness: float = 0.0    # Effectiveness of learning mechanisms


@dataclass
class BenchmarkMetrics:
    """Complete metrics collection for benchmarking."""
    
    # Core metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    consensus: ConsensusMetrics = field(default_factory=ConsensusMetrics)
    
    # Meta information
    scenario_name: str = ""
    voting_method: str = ""
    participant_count: int = 0
    decision_reached: bool = False
    timestamp: float = field(default_factory=time.time)
    
    # Additional context
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_summary_scores(self) -> Dict[str, float]:
        """Get summary scores for all metric categories."""
        return {
            'overall_quality': self.quality.calculate_overall_quality(),
            'efficiency_score': self.performance.calculate_efficiency_score(),
            'consensus_strength': self.consensus.final_agreement_level,
            'time_efficiency': 1.0 / max(0.1, self.performance.total_duration_seconds / 60),
            'communication_efficiency': 1.0 / max(1, self.performance.total_messages / self.participant_count),
            'decision_success': 1.0 if self.decision_reached else 0.0
        }


class MetricsCollector:
    """
    Collector for gathering metrics during voting system execution.
    
    Provides real-time metrics collection with minimal performance impact
    on the voting process itself.
    """
    
    def __init__(self):
        self.current_metrics = BenchmarkMetrics()
        self.start_time: Optional[float] = None
        self.phase_start_times: Dict[str, float] = {}
        self.message_log: List[Dict[str, Any]] = []
        self.token_usage: Dict[str, int] = defaultdict(int)
        
    def start_collection(self, scenario_name: str, voting_method: str, participant_count: int):
        """Initialize metrics collection for a benchmark run."""
        self.current_metrics = BenchmarkMetrics(
            scenario_name=scenario_name,
            voting_method=voting_method,
            participant_count=participant_count
        )
        self.start_time = time.time()
        self.phase_start_times = {'total': self.start_time}
        self.message_log = []
        self.token_usage = defaultdict(int)
        
        logger.debug(f"Started metrics collection for {scenario_name}")
    
    def start_phase(self, phase_name: str):
        """Mark the start of a specific phase (setup, deliberation, voting, etc.)."""
        self.phase_start_times[phase_name] = time.time()
        logger.debug(f"Started phase: {phase_name}")
    
    def end_phase(self, phase_name: str):
        """Mark the end of a specific phase and record duration."""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            
            # Record phase-specific durations
            if phase_name == 'deliberation':
                self.current_metrics.performance.deliberation_time_seconds = duration
            elif phase_name == 'voting':
                self.current_metrics.performance.voting_time_seconds = duration
            elif phase_name == 'setup':
                self.current_metrics.performance.setup_time_seconds = duration
            
            logger.debug(f"Ended phase {phase_name}: {duration:.2f}s")
    
    def record_message(self, sender: str, content: str, message_type: str = "standard"):
        """Record a message sent during the process."""
        message_entry = {
            'timestamp': time.time(),
            'sender': sender,
            'content': content,
            'message_type': message_type,
            'length': len(content),
            'word_count': len(content.split())
        }
        
        self.message_log.append(message_entry)
        
        # Update performance metrics
        self.current_metrics.performance.total_messages += 1
        if message_type == 'deliberation':
            self.current_metrics.performance.deliberation_messages += 1
        elif message_type == 'vote':
            self.current_metrics.performance.voting_messages += 1
        
        logger.debug(f"Recorded message from {sender}: {len(content)} chars")
    
    def record_api_call(self, tokens_used: int, estimated_cost: float = 0.0):
        """Record API usage for cost tracking."""
        self.current_metrics.performance.api_calls_made += 1
        self.current_metrics.performance.api_tokens_consumed += tokens_used
        self.current_metrics.performance.estimated_cost_usd += estimated_cost
        self.current_metrics.performance.total_tokens += tokens_used
        
        logger.debug(f"Recorded API call: {tokens_used} tokens, ${estimated_cost:.4f}")
    
    def record_decision_outcome(self, decision_reached: bool, final_result: Any = None):
        """Record the final decision outcome."""
        self.current_metrics.decision_reached = decision_reached
        
        if self.start_time:
            self.current_metrics.performance.total_duration_seconds = time.time() - self.start_time
        
        # Calculate derived metrics
        if self.message_log:
            total_length = sum(msg['length'] for msg in self.message_log)
            self.current_metrics.performance.average_message_length = total_length / len(self.message_log)
            
            self.current_metrics.performance.messages_per_participant = (
                len(self.message_log) / max(1, self.current_metrics.participant_count)
            )
        
        if decision_reached and self.current_metrics.performance.total_duration_seconds > 0:
            self.current_metrics.performance.time_per_decision = self.current_metrics.performance.total_duration_seconds
            self.current_metrics.performance.tokens_per_decision = self.current_metrics.performance.total_tokens
        
        logger.debug(f"Recorded decision outcome: {decision_reached}")
    
    def add_quality_assessment(self, quality_metrics: QualityMetrics):
        """Add quality assessment from external evaluators."""
        self.current_metrics.quality = quality_metrics
        logger.debug("Added quality assessment")
    
    def add_consensus_assessment(self, consensus_metrics: ConsensusMetrics):
        """Add consensus-specific assessment."""
        self.current_metrics.consensus = consensus_metrics
        logger.debug("Added consensus assessment")
    
    def get_current_metrics(self) -> BenchmarkMetrics:
        """Get the current metrics state."""
        return self.current_metrics
    
    def finalize_collection(self) -> BenchmarkMetrics:
        """Finalize and return complete metrics."""
        if not self.current_metrics.decision_reached:
            # Record as incomplete if not already recorded
            self.record_decision_outcome(False)
        
        logger.debug("Finalized metrics collection")
        return self.current_metrics


@dataclass
class ComparisonResults:
    """Results comparing two different approaches (e.g., voting vs standard)."""
    
    # Compared systems
    system_a_name: str
    system_b_name: str
    system_a_metrics: BenchmarkMetrics
    system_b_metrics: BenchmarkMetrics
    
    # Comparison analysis
    quality_comparison: Dict[str, float] = field(default_factory=dict)
    performance_comparison: Dict[str, float] = field(default_factory=dict)
    consensus_comparison: Dict[str, float] = field(default_factory=dict)
    overall_winner: str = ""
    
    # Statistical significance
    significance_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate comparisons after initialization."""
        self._calculate_comparisons()
        self._determine_winner()
    
    def _calculate_comparisons(self):
        """Calculate detailed comparisons between the two systems."""
        # Quality comparison ratios
        a_quality = self.system_a_metrics.quality.calculate_overall_quality()
        b_quality = self.system_b_metrics.quality.calculate_overall_quality()
        self.quality_comparison = {
            'overall_quality_ratio': a_quality / max(0.001, b_quality),
            'consensus_strength_ratio': (self.system_a_metrics.consensus.final_agreement_level / 
                                       max(0.001, self.system_b_metrics.consensus.final_agreement_level)),
            'satisfaction_ratio': (self.system_a_metrics.quality.participant_satisfaction /
                                 max(0.001, self.system_b_metrics.quality.participant_satisfaction))
        }
        
        # Performance comparison ratios  
        self.performance_comparison = {
            'time_ratio': (self.system_a_metrics.performance.total_duration_seconds /
                          max(0.1, self.system_b_metrics.performance.total_duration_seconds)),
            'message_ratio': (self.system_a_metrics.performance.total_messages /
                            max(1, self.system_b_metrics.performance.total_messages)),
            'token_ratio': (self.system_a_metrics.performance.total_tokens /
                          max(1, self.system_b_metrics.performance.total_tokens)),
            'cost_ratio': (self.system_a_metrics.performance.estimated_cost_usd /
                         max(0.001, self.system_b_metrics.performance.estimated_cost_usd))
        }
        
        # Consensus comparison
        self.consensus_comparison = {
            'convergence_ratio': (self.system_a_metrics.consensus.convergence_rate /
                                max(0.001, self.system_b_metrics.consensus.convergence_rate)),
            'participation_balance_ratio': (self.system_a_metrics.consensus.participation_balance /
                                          max(0.001, self.system_b_metrics.consensus.participation_balance)),
            'rounds_ratio': (self.system_b_metrics.consensus.rounds_to_consensus /  # Inverted: fewer rounds is better
                           max(1, self.system_a_metrics.consensus.rounds_to_consensus))
        }
    
    def _determine_winner(self):
        """Determine overall winner based on weighted scoring."""
        # Calculate weighted scores for each system
        weights = {
            'quality': 0.4,
            'performance': 0.3,
            'consensus': 0.3
        }
        
        a_scores = self.system_a_metrics.get_summary_scores()
        b_scores = self.system_b_metrics.get_summary_scores()
        
        a_weighted = sum(a_scores[key] * weights.get(key.split('_')[0], 0.1) 
                        for key in a_scores.keys())
        b_weighted = sum(b_scores[key] * weights.get(key.split('_')[0], 0.1) 
                        for key in b_scores.keys())
        
        if abs(a_weighted - b_weighted) < 0.05:  # Close tie
            self.overall_winner = "tie"
        elif a_weighted > b_weighted:
            self.overall_winner = self.system_a_name
        else:
            self.overall_winner = self.system_b_name
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the comparison results."""
        return {
            'winner': self.overall_winner,
            'quality_advantage': self.system_a_name if self.quality_comparison['overall_quality_ratio'] > 1.1 else 
                                self.system_b_name if self.quality_comparison['overall_quality_ratio'] < 0.9 else 'tie',
            'efficiency_advantage': self.system_a_name if self.performance_comparison['time_ratio'] < 0.9 else
                                  self.system_b_name if self.performance_comparison['time_ratio'] > 1.1 else 'tie',
            'consensus_advantage': self.system_a_name if self.consensus_comparison['convergence_ratio'] > 1.1 else
                                 self.system_b_name if self.consensus_comparison['convergence_ratio'] < 0.9 else 'tie',
            'key_metrics': {
                'quality_ratio': self.quality_comparison['overall_quality_ratio'],
                'time_ratio': self.performance_comparison['time_ratio'],
                'message_ratio': self.performance_comparison['message_ratio'],
                'consensus_strength_ratio': self.quality_comparison['consensus_strength_ratio']
            }
        }