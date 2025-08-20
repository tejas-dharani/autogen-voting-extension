"""
Benchmarking Suite

Comprehensive benchmarking framework for systematic evaluation of voting systems
with standardized scenarios, configurations, and analysis capabilities.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..core import BaseVotingGroupChat, VotingMethod
from ..system import EnhancedVotingGroupChat, create_adaptive_voting_system
from ..utilities import VotingSystemConfig
from .evaluation_metrics import BenchmarkMetrics, ComparisonResults, MetricsCollector

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Types of benchmark scenarios for systematic evaluation."""
    
    CODE_REVIEW = "code_review"
    ARCHITECTURE_DECISION = "architecture_decision"
    CONTENT_MODERATION = "content_moderation"
    RESOURCE_ALLOCATION = "resource_allocation"
    POLICY_DECISION = "policy_decision"
    TECHNICAL_EVALUATION = "technical_evaluation"
    STRATEGIC_PLANNING = "strategic_planning"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class BenchmarkScenario:
    """
    Definition of a benchmark scenario for systematic evaluation.
    
    Provides standardized test cases across different decision-making contexts
    with controlled variables for reliable comparison.
    """
    
    name: str
    scenario_type: ScenarioType
    description: str
    task_prompt: str
    
    # Agent configuration
    agent_personas: List[Dict[str, str]]
    expected_outcome: Optional[str] = None
    ground_truth: Optional[Dict[str, Any]] = None
    
    # Scenario parameters
    complexity_level: str = "moderate"  # trivial, simple, moderate, complex, critical
    time_pressure: str = "normal"       # relaxed, normal, urgent, critical
    stakes_level: str = "medium"        # low, medium, high, critical
    
    # Evaluation criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    quality_weights: Dict[str, float] = field(default_factory=dict)
    
    def get_context_for_adaptive_system(self) -> Dict[str, Any]:
        """Get context information for adaptive voting systems."""
        complexity_mapping = {
            'trivial': 0.1, 'simple': 0.3, 'moderate': 0.5, 
            'complex': 0.7, 'critical': 0.9
        }
        
        pressure_mapping = {
            'relaxed': 0.1, 'normal': 0.3, 'urgent': 0.7, 'critical': 0.9
        }
        
        stakes_mapping = {
            'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 0.95
        }
        
        return {
            'complexity_level': complexity_mapping.get(self.complexity_level, 0.5),
            'time_pressure': pressure_mapping.get(self.time_pressure, 0.3),
            'stakes_level': stakes_mapping.get(self.stakes_level, 0.5),
            'participant_count': len(self.agent_personas),
            'scenario_type': self.scenario_type.value
        }


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark execution."""
    
    # Model settings
    model_name: str = "gpt-4o-mini"
    model_temperature: float = 0.7
    max_tokens: int = 2000
    
    # Execution settings
    max_messages: int = 20
    timeout_seconds: int = 300
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    
    # Evaluation settings
    save_detailed_logs: bool = True
    save_conversation_history: bool = True
    results_directory: str = "benchmark_results"
    
    # Comparison settings
    compare_with_standard: bool = True
    voting_methods_to_test: List[VotingMethod] = field(default_factory=lambda: [
        VotingMethod.MAJORITY, VotingMethod.QUALIFIED_MAJORITY
    ])
    
    # Adaptive system settings
    enable_adaptive_consensus: bool = True
    enable_semantic_parsing: bool = True
    enable_learning: bool = False  # Disabled for consistent benchmarking


class BenchmarkRunner:
    """
    Main benchmark runner for systematic evaluation of voting systems.
    
    Provides comprehensive testing capabilities with standardized scenarios,
    controlled execution environments, and detailed performance analysis.
    """
    
    def __init__(self, config: Optional[BenchmarkConfiguration] = None):
        self.config = config or BenchmarkConfiguration()
        self.results_dir = Path(self.config.results_directory)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup model client
        self.model_client = OpenAIChatCompletionClient(
            model=self.config.model_name,
            temperature=self.config.model_temperature,
            max_tokens=self.config.max_tokens
        )
        
        logger.info(f"BenchmarkRunner initialized with {self.config.model_name}")
    
    async def run_single_scenario(
        self, 
        scenario: BenchmarkScenario,
        voting_method: VotingMethod,
        system_type: str = "enhanced"
    ) -> BenchmarkMetrics:
        """
        Run a single benchmark scenario with specified configuration.
        
        Args:
            scenario: The benchmark scenario to run
            voting_method: Voting method to use
            system_type: Type of system ("base", "enhanced", "adaptive")
            
        Returns:
            Detailed benchmark metrics for the run
        """
        logger.info(f"Running scenario: {scenario.name} with {voting_method.value}")
        
        # Create metrics collector
        metrics_collector = MetricsCollector()
        metrics_collector.start_collection(
            scenario.name, 
            voting_method.value, 
            len(scenario.agent_personas)
        )
        
        try:
            # Setup phase
            metrics_collector.start_phase('setup')
            agents = await self._create_agents(scenario.agent_personas)
            voting_system = await self._create_voting_system(
                agents, voting_method, system_type, scenario
            )
            metrics_collector.end_phase('setup')
            
            # Execution phase
            metrics_collector.start_phase('execution')
            
            # Run the voting process
            result = await self._execute_voting_process(
                voting_system, scenario.task_prompt, metrics_collector
            )
            
            metrics_collector.end_phase('execution')
            
            # Determine if decision was reached
            decision_reached = result is not None and hasattr(result, 'summary')
            metrics_collector.record_decision_outcome(decision_reached, result)
            
            # Quality assessment (if ground truth available)
            if scenario.ground_truth or scenario.expected_outcome:
                quality_metrics = await self._assess_quality(
                    result, scenario, metrics_collector
                )
                metrics_collector.add_quality_assessment(quality_metrics)
            
            final_metrics = metrics_collector.finalize_collection()
            
            # Save detailed results if configured
            if self.config.save_detailed_logs:
                await self._save_scenario_results(scenario, final_metrics, result)
            
            logger.info(f"Completed scenario {scenario.name}: {'SUCCESS' if decision_reached else 'FAILED'}")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error running scenario {scenario.name}: {e}")
            metrics_collector.record_decision_outcome(False)
            return metrics_collector.finalize_collection()
    
    async def run_comparison(
        self, 
        scenario: BenchmarkScenario,
        voting_method: VotingMethod,
        compare_systems: List[str] = None
    ) -> ComparisonResults:
        """
        Run comparative benchmark between different system types.
        
        Args:
            scenario: Benchmark scenario to run
            voting_method: Voting method to use
            compare_systems: List of systems to compare (default: ["enhanced", "standard"])
            
        Returns:
            Detailed comparison results
        """
        if compare_systems is None:
            compare_systems = ["enhanced", "standard"]
        
        logger.info(f"Running comparison for {scenario.name}: {' vs '.join(compare_systems)}")
        
        results = {}
        
        # Run each system type
        for system_type in compare_systems:
            logger.info(f"Running {system_type} system...")
            
            if system_type == "standard":
                # Run standard group chat for comparison
                metrics = await self._run_standard_group_chat(scenario)
            else:
                # Run voting system
                metrics = await self.run_single_scenario(scenario, voting_method, system_type)
            
            results[system_type] = metrics
            
            # Add delay between runs to avoid rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)
        
        # Create comparison results
        if len(compare_systems) >= 2:
            comparison = ComparisonResults(
                system_a_name=compare_systems[0],
                system_b_name=compare_systems[1], 
                system_a_metrics=results[compare_systems[0]],
                system_b_metrics=results[compare_systems[1]]
            )
            
            # Save comparison results
            if self.config.save_detailed_logs:
                await self._save_comparison_results(scenario, comparison)
            
            return comparison
        else:
            # Single system run - create dummy comparison
            system_name = compare_systems[0]
            return ComparisonResults(
                system_a_name=system_name,
                system_b_name="none",
                system_a_metrics=results[system_name],
                system_b_metrics=BenchmarkMetrics()
            )
    
    async def run_scenario_suite(
        self, 
        scenarios: List[BenchmarkScenario],
        voting_methods: Optional[List[VotingMethod]] = None
    ) -> List[ComparisonResults]:
        """
        Run a complete suite of benchmark scenarios.
        
        Args:
            scenarios: List of scenarios to run
            voting_methods: Voting methods to test (default from config)
            
        Returns:
            List of comparison results for all scenario/method combinations
        """
        if voting_methods is None:
            voting_methods = self.config.voting_methods_to_test
        
        logger.info(f"Running benchmark suite: {len(scenarios)} scenarios × {len(voting_methods)} methods")
        
        all_results = []
        
        for scenario in scenarios:
            logger.info(f"Starting scenario: {scenario.name}")
            
            for voting_method in voting_methods:
                try:
                    result = await self.run_comparison(scenario, voting_method)
                    all_results.append(result)
                    
                    # Progress logging
                    completed = len(all_results)
                    total = len(scenarios) * len(voting_methods)
                    logger.info(f"Progress: {completed}/{total} ({completed/total:.1%})")
                    
                except Exception as e:
                    logger.error(f"Failed comparison {scenario.name} + {voting_method.value}: {e}")
                    continue
                
                # Rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
        
        logger.info(f"Benchmark suite completed: {len(all_results)} results")
        return all_results
    
    # Private helper methods
    
    async def _create_agents(self, agent_personas: List[Dict[str, str]]) -> List[AssistantAgent]:
        """Create agents from persona definitions."""
        agents = []
        
        for persona in agent_personas:
            agent = AssistantAgent(
                name=persona["name"],
                model_client=self.model_client,
                system_message=f"You are a {persona['role']}. {persona.get('description', '')}"
            )
            agents.append(agent)
        
        return agents
    
    async def _create_voting_system(
        self, 
        agents: List[AssistantAgent], 
        voting_method: VotingMethod,
        system_type: str,
        scenario: BenchmarkScenario
    ) -> Union[BaseVotingGroupChat, EnhancedVotingGroupChat]:
        """Create appropriate voting system based on type."""
        
        termination = MaxMessageTermination(self.config.max_messages)
        
        if system_type == "base":
            return BaseVotingGroupChat(
                participants=agents,
                voting_method=voting_method,
                termination_condition=termination
            )
        
        elif system_type == "enhanced":
            return EnhancedVotingGroupChat(
                participants=agents,
                config=VotingSystemConfig(
                    voting_method=voting_method,
                    enable_adaptive_consensus=self.config.enable_adaptive_consensus,
                    enable_semantic_parsing=self.config.enable_semantic_parsing,
                    enable_learning=self.config.enable_learning
                ),
                termination_condition=termination
            )
        
        elif system_type == "adaptive":
            return create_adaptive_voting_system(
                participants=agents,
                complexity_aware=True,
                semantic_parsing=True,
                learning_enabled=self.config.enable_learning
            )
        
        else:
            raise ValueError(f"Unknown system type: {system_type}")
    
    async def _execute_voting_process(
        self,
        voting_system,
        task_prompt: str,
        metrics_collector: MetricsCollector
    ) -> Any:
        """Execute the voting process with metrics collection."""
        
        # Setup metrics collection hook
        if hasattr(voting_system, 'set_metrics_collector'):
            voting_system.set_metrics_collector(metrics_collector)
        
        # Run the voting process
        result = await voting_system.run(task=task_prompt)
        
        return result
    
    async def _run_standard_group_chat(self, scenario: BenchmarkScenario) -> BenchmarkMetrics:
        """Run standard group chat for comparison baseline."""
        
        metrics_collector = MetricsCollector()
        metrics_collector.start_collection(scenario.name, "standard", len(scenario.agent_personas))
        
        try:
            # Create agents
            agents = await self._create_agents(scenario.agent_personas)
            
            # Create standard group chat
            group_chat = RoundRobinGroupChat(
                participants=agents,
                termination_condition=MaxMessageTermination(self.config.max_messages)
            )
            
            # Run standard process
            result = await group_chat.run(task=scenario.task_prompt)
            
            # Record completion
            decision_reached = result is not None
            metrics_collector.record_decision_outcome(decision_reached, result)
            
            return metrics_collector.finalize_collection()
            
        except Exception as e:
            logger.error(f"Standard group chat error: {e}")
            metrics_collector.record_decision_outcome(False)
            return metrics_collector.finalize_collection()
    
    async def _assess_quality(
        self, 
        result: Any, 
        scenario: BenchmarkScenario, 
        metrics_collector: MetricsCollector
    ) -> Any:
        """Assess the quality of the decision result."""
        # This would implement quality assessment logic
        # For now, return a basic assessment
        from .evaluation_metrics import QualityMetrics
        
        return QualityMetrics(
            decision_accuracy=0.8 if result else 0.0,
            consensus_strength=0.7,
            participant_satisfaction=0.75
        )
    
    async def _save_scenario_results(
        self, 
        scenario: BenchmarkScenario, 
        metrics: BenchmarkMetrics, 
        result: Any
    ) -> None:
        """Save detailed scenario results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{scenario.name}_{timestamp}.json"
        
        # Prepare serializable data
        data = {
            'scenario': {
                'name': scenario.name,
                'type': scenario.scenario_type.value,
                'description': scenario.description
            },
            'metrics': {
                'decision_reached': metrics.decision_reached,
                'duration': metrics.performance.total_duration_seconds,
                'messages': metrics.performance.total_messages,
                'tokens': metrics.performance.total_tokens
            },
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved results to {filename}")
    
    async def _save_comparison_results(
        self, 
        scenario: BenchmarkScenario, 
        comparison: ComparisonResults
    ) -> None:
        """Save comparison results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"comparison_{scenario.name}_{timestamp}.json"
        
        # Prepare serializable data
        data = {
            'scenario': scenario.name,
            'comparison': {
                'winner': comparison.overall_winner,
                'system_a': comparison.system_a_name,
                'system_b': comparison.system_b_name,
                'quality_ratio': comparison.quality_comparison.get('overall_quality_ratio', 1.0),
                'time_ratio': comparison.performance_comparison.get('time_ratio', 1.0),
                'message_ratio': comparison.performance_comparison.get('message_ratio', 1.0)
            },
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved comparison to {filename}")


class ResultsAnalyzer:
    """Analyzer for benchmark results with statistical analysis capabilities."""
    
    def __init__(self):
        self.results: List[ComparisonResults] = []
    
    def add_results(self, results: List[ComparisonResults]) -> None:
        """Add results for analysis."""
        self.results.extend(results)
    
    def analyze_overall_performance(self) -> Dict[str, Any]:
        """Analyze overall performance across all results."""
        if not self.results:
            return {}
        
        # Aggregate statistics
        quality_ratios = [r.quality_comparison['overall_quality_ratio'] for r in self.results]
        time_ratios = [r.performance_comparison['time_ratio'] for r in self.results]
        message_ratios = [r.performance_comparison['message_ratio'] for r in self.results]
        
        # Winner counts
        winners = [r.overall_winner for r in self.results]
        winner_counts = {winner: winners.count(winner) for winner in set(winners)}
        
        return {
            'total_comparisons': len(self.results),
            'average_quality_ratio': sum(quality_ratios) / len(quality_ratios),
            'average_time_ratio': sum(time_ratios) / len(time_ratios),
            'average_message_ratio': sum(message_ratios) / len(message_ratios),
            'winner_distribution': winner_counts,
            'voting_advantage_rate': sum(1 for w in winners if 'voting' in w.lower()) / len(winners)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        analysis = self.analyze_overall_performance()
        
        report = f"""
# Benchmark Analysis Report

## Overall Results
- Total Comparisons: {analysis['total_comparisons']}
- Average Quality Ratio: {analysis['average_quality_ratio']:.2f}
- Average Time Ratio: {analysis['average_time_ratio']:.2f}
- Average Message Ratio: {analysis['average_message_ratio']:.2f}

## Winner Distribution
"""
        
        for winner, count in analysis['winner_distribution'].items():
            percentage = count / analysis['total_comparisons'] * 100
            report += f"- {winner}: {count} ({percentage:.1f}%)\n"
        
        report += f"\n## Voting System Advantage Rate: {analysis['voting_advantage_rate']:.1%}"
        
        return report