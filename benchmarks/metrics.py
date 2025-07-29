"""Metrics collection and analysis for voting vs. standard group chat comparison."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for group chat performance evaluation."""
    
    # Basic metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    # Decision quality metrics
    decision_reached: bool = False
    consensus_type: Optional[str] = None  # "unanimous", "majority", "qualified_majority"
    final_vote_counts: Dict[str, int] = field(default_factory=dict)  # type: ignore
    
    # Efficiency metrics
    total_messages: int = 0
    discussion_rounds: int = 0
    token_usage: int = 0
    api_calls: int = 0
    
    # Agent participation metrics
    agent_participation: Dict[str, int] = field(default_factory=dict)  # type: ignore  # message count per agent
    voting_patterns: Dict[str, List[str]] = field(default_factory=dict)  # type: ignore  # votes per agent
    
    # Process metrics
    abstentions: int = 0
    reasoning_provided: bool = False
    tie_breaks: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)  # type: ignore
    
    def complete_benchmark(self) -> None:
        """Mark benchmark as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
    
    def add_message(self, agent_name: str) -> None:
        """Track message from an agent."""
        self.total_messages += 1
        self.agent_participation[agent_name] = self.agent_participation.get(agent_name, 0) + 1
    
    def add_vote(self, agent_name: str, vote: str) -> None:
        """Track vote from an agent."""
        if agent_name not in self.voting_patterns:
            self.voting_patterns[agent_name] = []
        self.voting_patterns[agent_name].append(vote)
        
        # Update vote counts
        self.final_vote_counts[vote] = self.final_vote_counts.get(vote, 0) + 1
    
    def add_tokens(self, count: int) -> None:
        """Add to token usage count."""
        self.token_usage += count
    
    def add_api_call(self) -> None:
        """Increment API call counter."""
        self.api_calls += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": self.duration_seconds,
            "decision_reached": self.decision_reached,
            "consensus_type": self.consensus_type,
            "final_vote_counts": self.final_vote_counts,
            "total_messages": self.total_messages,
            "discussion_rounds": self.discussion_rounds,
            "token_usage": self.token_usage,
            "api_calls": self.api_calls,
            "agent_participation": self.agent_participation,
            "voting_patterns": self.voting_patterns,
            "abstentions": self.abstentions,
            "reasoning_provided": self.reasoning_provided,
            "tie_breaks": self.tie_breaks,
            "custom_metrics": self.custom_metrics,
        }
    
    def save_to_file(self, filename: str) -> None:
        """Save metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ComparisonResults:
    """Results comparing VotingGroupChat vs standard GroupChat."""
    
    voting_metrics: BenchmarkMetrics
    standard_metrics: BenchmarkMetrics
    scenario_name: str
    scenario_description: str
    
    @property
    def efficiency_comparison(self) -> Dict[str, float]:
        """Compare efficiency metrics between approaches."""
        return {
            "time_ratio": (self.voting_metrics.duration_seconds or 0) / (self.standard_metrics.duration_seconds or 1),
            "message_ratio": self.voting_metrics.total_messages / max(self.standard_metrics.total_messages, 1),
            "token_ratio": self.voting_metrics.token_usage / max(self.standard_metrics.token_usage, 1),
            "api_call_ratio": self.voting_metrics.api_calls / max(self.standard_metrics.api_calls, 1),
        }
    
    @property
    def participation_comparison(self) -> Dict[str, Any]:
        """Compare agent participation patterns."""
        voting_participation = len([v for v in self.voting_metrics.agent_participation.values() if v > 0])
        standard_participation = len([v for v in self.standard_metrics.agent_participation.values() if v > 0])
        
        return {
            "voting_active_agents": voting_participation,
            "standard_active_agents": standard_participation,
            "voting_avg_messages_per_agent": sum(self.voting_metrics.agent_participation.values()) / max(voting_participation, 1),
            "standard_avg_messages_per_agent": sum(self.standard_metrics.agent_participation.values()) / max(standard_participation, 1),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison results to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "voting_metrics": self.voting_metrics.to_dict(),
            "standard_metrics": self.standard_metrics.to_dict(),
            "efficiency_comparison": self.efficiency_comparison,
            "participation_comparison": self.participation_comparison,
        }
    
    def save_to_file(self, filename: str) -> None:
        """Save comparison results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsCollector:
    """Collects metrics during group chat execution."""
    
    def __init__(self):
        self.current_metrics: Optional[BenchmarkMetrics] = None
        self.active = False
    
    def start_collection(self) -> BenchmarkMetrics:
        """Start collecting metrics for a new benchmark."""
        self.current_metrics = BenchmarkMetrics()
        self.active = True
        return self.current_metrics
    
    def stop_collection(self) -> Optional[BenchmarkMetrics]:
        """Stop collecting metrics and return results."""
        if self.current_metrics:
            self.current_metrics.complete_benchmark()
        self.active = False
        return self.current_metrics
    
    def record_message(self, agent_name: str, token_count: int = 0) -> None:
        """Record a message sent by an agent."""
        if self.active and self.current_metrics:
            self.current_metrics.add_message(agent_name)
            if token_count > 0:
                self.current_metrics.add_tokens(token_count)
    
    def record_vote(self, agent_name: str, vote: str) -> None:
        """Record a vote cast by an agent."""
        if self.active and self.current_metrics:
            self.current_metrics.add_vote(agent_name, vote)
    
    def record_api_call(self, token_count: int = 0) -> None:
        """Record an API call made during execution."""
        if self.active and self.current_metrics:
            self.current_metrics.add_api_call()
            if token_count > 0:
                self.current_metrics.add_tokens(token_count)