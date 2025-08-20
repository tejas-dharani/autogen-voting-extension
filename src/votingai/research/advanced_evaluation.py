"""
Advanced Research Evaluation Framework

Research-grade evaluation framework for comprehensive analysis of voting systems,
including sophisticated metrics, experimental design, and statistical analysis.
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np
from collections import defaultdict

from ..consensus import SmartConsensusOrchestrator, DecisionComplexity, ConsensusStrategy
from ..intelligence import SemanticVoteInterpreter
from .evaluation_metrics import QualityMetrics, PerformanceMetrics, BenchmarkMetrics
from .benchmarking_suite import BenchmarkScenario, BenchmarkRunner

logger = logging.getLogger(__name__)


class ExperimentType(str, Enum):
    """Types of research experiments for systematic evaluation."""
    
    EFFICIENCY_VS_QUALITY = "efficiency_vs_quality"
    COMPLEXITY_ADAPTATION = "complexity_adaptation"  
    DELIBERATION_EFFECTIVENESS = "deliberation_effectiveness"
    SEMANTIC_PARSING_ACCURACY = "semantic_parsing_accuracy"
    CONSENSUS_ROBUSTNESS = "consensus_robustness"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    LEARNING_IMPROVEMENT = "learning_improvement"
    BYZANTINE_RESILIENCE = "byzantine_resilience"


@dataclass
class ExperimentalCondition:
    """Definition of experimental conditions for systematic testing."""
    
    condition_name: str
    description: str
    parameters: Dict[str, Any]
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentDesign:
    """Complete experimental design with conditions and analysis plan."""
    
    experiment_name: str
    experiment_type: ExperimentType
    description: str
    
    # Experimental setup
    conditions: List[ExperimentalCondition]
    scenarios: List[BenchmarkScenario] 
    repetitions: int = 3
    
    # Analysis configuration
    primary_metrics: List[str] = field(default_factory=list)
    secondary_metrics: List[str] = field(default_factory=list)
    statistical_tests: List[str] = field(default_factory=list)
    
    # Research hypotheses
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    significance_level: float = 0.05


class ResearchBenchmarkFramework:
    """
    Comprehensive research framework for voting system evaluation.
    
    Provides systematic experimental design, data collection,
    and statistical analysis capabilities for research publications.
    """
    
    def __init__(self, results_directory: str = "research_results"):
        self.results_dir = Path(results_directory)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Research components
        self.consensus_orchestrator = SmartConsensusOrchestrator()
        self.semantic_interpreter = SemanticVoteInterpreter()
        self.quality_evaluator = ConsensusQualityEvaluator()
        self.learning_analyzer = LearningEffectivenessAnalyzer()
        self.scalability_tester = ScalabilityTester()
        
        # Experiment tracking
        self.active_experiments: Dict[str, ExperimentDesign] = {}
        self.experiment_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("ResearchBenchmarkFramework initialized")
    
    def design_experiment(
        self, 
        experiment_type: ExperimentType,
        name: str,
        description: str
    ) -> ExperimentDesign:
        """Design a research experiment with appropriate conditions and metrics."""
        
        if experiment_type == ExperimentType.EFFICIENCY_VS_QUALITY:
            return self._design_efficiency_quality_experiment(name, description)
        elif experiment_type == ExperimentType.COMPLEXITY_ADAPTATION:
            return self._design_complexity_adaptation_experiment(name, description)
        elif experiment_type == ExperimentType.SCALABILITY_ANALYSIS:
            return self._design_scalability_experiment(name, description)
        elif experiment_type == ExperimentType.SEMANTIC_PARSING_ACCURACY:
            return self._design_semantic_accuracy_experiment(name, description)
        else:
            # Generic experiment design
            return self._design_generic_experiment(experiment_type, name, description)
    
    async def run_experiment(self, experiment: ExperimentDesign) -> Dict[str, Any]:
        """Execute a complete research experiment with systematic data collection."""
        
        logger.info(f"Starting experiment: {experiment.experiment_name}")
        experiment_start = time.time()
        
        self.active_experiments[experiment.experiment_name] = experiment
        all_results = []
        
        # Run each condition multiple times
        for condition in experiment.conditions:
            logger.info(f"Running condition: {condition.condition_name}")
            
            condition_results = []
            
            for repetition in range(experiment.repetitions):
                logger.info(f"Repetition {repetition + 1}/{experiment.repetitions}")
                
                # Run scenarios under this condition
                for scenario in experiment.scenarios:
                    try:
                        result = await self._run_experimental_condition(
                            scenario, condition, experiment
                        )
                        result['repetition'] = repetition + 1
                        result['condition'] = condition.condition_name
                        condition_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Failed experimental run: {e}")
                        continue
                
                # Delay between repetitions to avoid rate limiting
                await asyncio.sleep(1.0)
            
            all_results.extend(condition_results)
        
        # Analyze results
        analysis = self._analyze_experiment_results(experiment, all_results)
        
        # Save comprehensive results
        experiment_duration = time.time() - experiment_start
        final_results = {
            'experiment_design': experiment,
            'raw_results': all_results,
            'statistical_analysis': analysis,
            'metadata': {
                'total_runs': len(all_results),
                'duration_minutes': experiment_duration / 60,
                'timestamp': datetime.now().isoformat(),
                'successful_runs': len([r for r in all_results if r.get('success', False)])
            }
        }
        
        # Save to file
        await self._save_experiment_results(experiment.experiment_name, final_results)
        
        logger.info(f"Experiment {experiment.experiment_name} completed: {len(all_results)} runs")
        return final_results
    
    # Private experiment design methods
    
    def _design_efficiency_quality_experiment(
        self, name: str, description: str
    ) -> ExperimentDesign:
        """Design experiment comparing efficiency vs quality trade-offs."""
        
        conditions = [
            ExperimentalCondition(
                "fast_consensus",
                "Optimize for speed with minimal deliberation",
                {"max_discussion_rounds": 0, "voting_method": "majority", "time_limit": 60}
            ),
            ExperimentalCondition(
                "balanced_consensus",
                "Balance speed and quality with moderate deliberation", 
                {"max_discussion_rounds": 2, "voting_method": "qualified_majority", "time_limit": 180}
            ),
            ExperimentalCondition(
                "quality_consensus", 
                "Optimize for quality with extensive deliberation",
                {"max_discussion_rounds": 5, "voting_method": "unanimous", "time_limit": 300}
            )
        ]
        
        return ExperimentDesign(
            experiment_name=name,
            experiment_type=ExperimentType.EFFICIENCY_VS_QUALITY,
            description=description,
            conditions=conditions,
            scenarios=[],  # To be filled by caller
            repetitions=5,
            primary_metrics=["decision_quality", "time_efficiency", "consensus_strength"],
            secondary_metrics=["participant_satisfaction", "resource_utilization"],
            statistical_tests=["anova", "tukey_hsd"],
            null_hypothesis="No difference in efficiency/quality trade-offs between conditions",
            alternative_hypothesis="Significant differences exist in efficiency/quality trade-offs"
        )
    
    def _design_complexity_adaptation_experiment(
        self, name: str, description: str
    ) -> ExperimentDesign:
        """Design experiment testing adaptive complexity handling."""
        
        conditions = [
            ExperimentalCondition(
                "adaptive_strategy",
                "Use adaptive strategy selection based on complexity",
                {"adaptive_consensus": True, "strategy_selection": "automatic"}
            ),
            ExperimentalCondition(
                "fixed_strategy",
                "Use fixed strategy regardless of complexity",
                {"adaptive_consensus": False, "strategy_selection": "majority_fixed"}
            )
        ]
        
        return ExperimentDesign(
            experiment_name=name,
            experiment_type=ExperimentType.COMPLEXITY_ADAPTATION,
            description=description,
            conditions=conditions,
            scenarios=[],
            repetitions=3,
            primary_metrics=["complexity_handling_accuracy", "strategy_appropriateness"],
            secondary_metrics=["decision_quality", "efficiency"],
            statistical_tests=["t_test", "wilcoxon"],
            null_hypothesis="No advantage of adaptive vs fixed strategy selection",
            alternative_hypothesis="Adaptive strategy selection improves outcomes"
        )
    
    def _design_scalability_experiment(
        self, name: str, description: str
    ) -> ExperimentDesign:
        """Design scalability experiment with varying participant counts."""
        
        conditions = [
            ExperimentalCondition(f"agents_{n}", f"Test with {n} agents", {"agent_count": n})
            for n in [3, 5, 7, 10, 15]
        ]
        
        return ExperimentDesign(
            experiment_name=name,
            experiment_type=ExperimentType.SCALABILITY_ANALYSIS,
            description=description,
            conditions=conditions,
            scenarios=[],
            repetitions=3,
            primary_metrics=["scalability_efficiency", "decision_time_scaling"],
            secondary_metrics=["participation_balance", "consensus_quality"],
            statistical_tests=["regression", "correlation"],
            null_hypothesis="System performance scales linearly with participant count",
            alternative_hypothesis="System exhibits non-linear scaling characteristics"
        )
    
    def _design_semantic_accuracy_experiment(
        self, name: str, description: str
    ) -> ExperimentDesign:
        """Design experiment testing semantic parsing accuracy."""
        
        conditions = [
            ExperimentalCondition(
                "semantic_parsing",
                "Use advanced semantic vote interpretation",
                {"semantic_parsing": True, "fallback_threshold": 0.5}
            ),
            ExperimentalCondition(
                "keyword_parsing",
                "Use simple keyword-based parsing",
                {"semantic_parsing": False, "keyword_only": True}
            )
        ]
        
        return ExperimentDesign(
            experiment_name=name,
            experiment_type=ExperimentType.SEMANTIC_PARSING_ACCURACY,
            description=description,
            conditions=conditions,
            scenarios=[],
            repetitions=5,
            primary_metrics=["parsing_accuracy", "confidence_scores"],
            secondary_metrics=["processing_time", "fallback_rate"],
            statistical_tests=["accuracy_test", "confidence_interval"],
            null_hypothesis="No difference in parsing accuracy between methods",
            alternative_hypothesis="Semantic parsing significantly improves accuracy"
        )
    
    def _design_generic_experiment(
        self, experiment_type: ExperimentType, name: str, description: str
    ) -> ExperimentDesign:
        """Design generic experiment template."""
        
        return ExperimentDesign(
            experiment_name=name,
            experiment_type=experiment_type,
            description=description,
            conditions=[],
            scenarios=[],
            repetitions=3,
            primary_metrics=["decision_quality", "efficiency"],
            secondary_metrics=["participant_satisfaction"],
            statistical_tests=["t_test"]
        )
    
    async def _run_experimental_condition(
        self,
        scenario: BenchmarkScenario,
        condition: ExperimentalCondition,
        experiment: ExperimentDesign
    ) -> Dict[str, Any]:
        """Run a single experimental condition."""
        
        # This would integrate with the benchmarking suite
        # For now, return mock results structure
        
        return {
            'scenario': scenario.name,
            'condition': condition.condition_name,
            'success': True,
            'metrics': {
                'decision_quality': 0.8,
                'efficiency': 0.7,
                'duration_seconds': 120.0
            },
            'timestamp': time.time()
        }
    
    def _analyze_experiment_results(
        self, experiment: ExperimentDesign, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform statistical analysis of experimental results."""
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in results:
            condition_groups[result['condition']].append(result)
        
        analysis = {
            'descriptive_statistics': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'conclusions': []
        }
        
        # Calculate descriptive statistics for each condition
        for condition, condition_results in condition_groups.items():
            if condition_results:
                analysis['descriptive_statistics'][condition] = {
                    'n': len(condition_results),
                    'success_rate': sum(1 for r in condition_results if r.get('success', False)) / len(condition_results)
                }
        
        return analysis
    
    async def _save_experiment_results(self, experiment_name: str, results: Dict[str, Any]):
        """Save experiment results to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"experiment_{experiment_name}_{timestamp}.json"
        
        # Convert to serializable format
        import json
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)  # Test serialization
                serializable_results[key] = value
            except (TypeError, ValueError):
                serializable_results[key] = str(value)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Saved experiment results to {filename}")


class ConsensusQualityEvaluator:
    """Evaluates the quality of consensus decisions using multiple criteria."""
    
    def __init__(self):
        self.quality_dimensions = [
            'decision_accuracy', 'consensus_strength', 'participant_satisfaction',
            'process_fairness', 'outcome_stability', 'implementation_feasibility'
        ]
    
    def evaluate_consensus_quality(
        self, 
        decision_result: Any,
        ground_truth: Optional[Any] = None,
        participant_feedback: Optional[List[Dict[str, Any]]] = None
    ) -> QualityMetrics:
        """Evaluate comprehensive quality metrics for a consensus decision."""
        
        quality = QualityMetrics()
        
        # Decision accuracy (if ground truth available)
        if ground_truth:
            quality.decision_accuracy = self._calculate_accuracy(decision_result, ground_truth)
        
        # Consensus strength based on vote distribution
        quality.consensus_strength = self._calculate_consensus_strength(decision_result)
        
        # Participant satisfaction (if feedback available)
        if participant_feedback:
            quality.participant_satisfaction = self._calculate_satisfaction(participant_feedback)
        
        # Process quality metrics
        quality.deliberation_depth = self._assess_deliberation_depth(decision_result)
        quality.argument_quality = self._assess_argument_quality(decision_result)
        
        return quality
    
    def _calculate_accuracy(self, decision: Any, ground_truth: Any) -> float:
        """Calculate decision accuracy against ground truth."""
        # Implementation would depend on decision format
        return 0.8  # Placeholder
    
    def _calculate_consensus_strength(self, decision: Any) -> float:
        """Calculate how strong the consensus was."""
        # Implementation would analyze vote distribution
        return 0.75  # Placeholder
    
    def _calculate_satisfaction(self, feedback: List[Dict[str, Any]]) -> float:
        """Calculate average participant satisfaction."""
        if not feedback:
            return 0.5
        
        satisfaction_scores = [f.get('satisfaction', 0.5) for f in feedback]
        return sum(satisfaction_scores) / len(satisfaction_scores)
    
    def _assess_deliberation_depth(self, decision: Any) -> float:
        """Assess the depth and quality of deliberation."""
        # Implementation would analyze deliberation messages
        return 0.7  # Placeholder
    
    def _assess_argument_quality(self, decision: Any) -> float:
        """Assess the quality of arguments presented."""
        # Implementation would analyze reasoning quality
        return 0.65  # Placeholder


class LearningEffectivenessAnalyzer:
    """Analyzes the effectiveness of learning mechanisms in voting systems."""
    
    def __init__(self):
        self.learning_metrics = [
            'improvement_rate', 'adaptation_speed', 'knowledge_retention',
            'transfer_learning', 'performance_stability'
        ]
    
    def analyze_learning_effectiveness(
        self, 
        learning_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze how effectively the system learns from experience."""
        
        if not learning_history:
            return {metric: 0.0 for metric in self.learning_metrics}
        
        analysis = {}
        
        # Improvement rate over time
        performance_scores = [h.get('performance_score', 0.5) for h in learning_history]
        analysis['improvement_rate'] = self._calculate_improvement_rate(performance_scores)
        
        # Adaptation speed to new conditions
        analysis['adaptation_speed'] = self._calculate_adaptation_speed(learning_history)
        
        # Knowledge retention across sessions
        analysis['knowledge_retention'] = self._calculate_retention(learning_history)
        
        # Transfer learning to new domains
        analysis['transfer_learning'] = self._calculate_transfer_learning(learning_history)
        
        # Performance stability
        analysis['performance_stability'] = self._calculate_stability(performance_scores)
        
        return analysis
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate rate of improvement over time."""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(scores)
        x = list(range(n))
        slope = (n * sum(x[i] * scores[i] for i in range(n)) - sum(x) * sum(scores)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        return max(0.0, slope)  # Only positive improvement
    
    def _calculate_adaptation_speed(self, history: List[Dict[str, Any]]) -> float:
        """Calculate how quickly system adapts to new conditions."""
        # Implementation would analyze adaptation patterns
        return 0.6  # Placeholder
    
    def _calculate_retention(self, history: List[Dict[str, Any]]) -> float:
        """Calculate knowledge retention across sessions."""
        # Implementation would analyze retention patterns
        return 0.7  # Placeholder
    
    def _calculate_transfer_learning(self, history: List[Dict[str, Any]]) -> float:
        """Calculate transfer learning effectiveness."""
        # Implementation would analyze cross-domain performance
        return 0.5  # Placeholder
    
    def _calculate_stability(self, scores: List[float]) -> float:
        """Calculate performance stability (inverse of variance)."""
        if len(scores) < 2:
            return 1.0
        
        variance = statistics.variance(scores)
        return max(0.0, 1.0 - variance)  # Stability as inverse of variance


class ScalabilityTester:
    """Tests system scalability with varying numbers of participants."""
    
    def __init__(self):
        self.scalability_metrics = [
            'time_complexity', 'space_complexity', 'communication_efficiency',
            'decision_quality_scaling', 'participation_balance_scaling'
        ]
    
    async def test_scalability(
        self,
        base_scenario: BenchmarkScenario,
        participant_counts: List[int]
    ) -> Dict[str, Any]:
        """Test system scalability across different participant counts."""
        
        results = {}
        
        for count in participant_counts:
            logger.info(f"Testing scalability with {count} participants")
            
            # Create scaled scenario
            scaled_scenario = self._scale_scenario(base_scenario, count)
            
            # Run benchmark
            try:
                # This would integrate with actual benchmarking
                result = await self._run_scaled_benchmark(scaled_scenario, count)
                results[count] = result
                
            except Exception as e:
                logger.error(f"Scalability test failed for {count} participants: {e}")
                results[count] = {'error': str(e)}
        
        # Analyze scalability patterns
        analysis = self._analyze_scalability_patterns(results)
        
        return {
            'raw_results': results,
            'scalability_analysis': analysis,
            'scaling_coefficients': self._calculate_scaling_coefficients(results)
        }
    
    def _scale_scenario(self, base_scenario: BenchmarkScenario, participant_count: int) -> BenchmarkScenario:
        """Create scaled version of scenario with specified participant count."""
        
        # Generate additional personas if needed
        base_personas = base_scenario.agent_personas
        scaled_personas = base_personas.copy()
        
        while len(scaled_personas) < participant_count:
            persona_num = len(scaled_personas) + 1
            scaled_personas.append({
                "name": f"Agent_{persona_num}",
                "role": f"Participant {persona_num}",
                "description": f"Additional participant for scalability testing"
            })
        
        # Create scaled scenario
        scaled_scenario = BenchmarkScenario(
            name=f"{base_scenario.name}_scaled_{participant_count}",
            scenario_type=base_scenario.scenario_type,
            description=f"Scaled version with {participant_count} participants",
            task_prompt=base_scenario.task_prompt,
            agent_personas=scaled_personas[:participant_count],
            complexity_level=base_scenario.complexity_level,
            stakes_level=base_scenario.stakes_level,
            time_pressure=base_scenario.time_pressure
        )
        
        return scaled_scenario
    
    async def _run_scaled_benchmark(
        self, scenario: BenchmarkScenario, participant_count: int
    ) -> Dict[str, Any]:
        """Run benchmark with scaled scenario."""
        
        # Mock implementation - would integrate with actual benchmarking
        base_time = 30.0  # Base time for 3 participants
        scaling_factor = 1.2  # Time increases with more participants
        
        estimated_time = base_time * (participant_count ** scaling_factor)
        
        return {
            'participant_count': participant_count,
            'duration_seconds': estimated_time,
            'messages': participant_count * 3,  # Approximation
            'decision_reached': True,
            'quality_score': max(0.5, 0.9 - (participant_count - 3) * 0.05)  # Quality decreases slightly
        }
    
    def _analyze_scalability_patterns(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scalability patterns from results."""
        
        participant_counts = sorted(results.keys())
        durations = [results[n].get('duration_seconds', 0) for n in participant_counts]
        quality_scores = [results[n].get('quality_score', 0) for n in participant_counts]
        
        return {
            'time_scaling_pattern': 'polynomial' if len(durations) > 1 else 'unknown',
            'quality_degradation_rate': self._calculate_degradation_rate(quality_scores),
            'optimal_participant_range': self._find_optimal_range(results),
            'scalability_score': self._calculate_scalability_score(results)
        }
    
    def _calculate_scaling_coefficients(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate scaling coefficients for different metrics."""
        
        # Simple implementation - would be more sophisticated in practice
        return {
            'time_scaling_coefficient': 1.2,
            'quality_scaling_coefficient': -0.05,
            'communication_scaling_coefficient': 1.0
        }
    
    def _calculate_degradation_rate(self, quality_scores: List[float]) -> float:
        """Calculate rate of quality degradation with scale."""
        if len(quality_scores) < 2:
            return 0.0
        
        # Simple linear regression to find degradation rate
        return max(0.0, quality_scores[0] - quality_scores[-1]) / len(quality_scores)
    
    def _find_optimal_range(self, results: Dict[int, Dict[str, Any]]) -> Tuple[int, int]:
        """Find optimal participant count range."""
        # Simple implementation - would be more sophisticated
        return (3, 7)  # Typical optimal range
    
    def _calculate_scalability_score(self, results: Dict[int, Dict[str, Any]]) -> float:
        """Calculate overall scalability score."""
        # Implementation would consider multiple factors
        return 0.75  # Placeholder