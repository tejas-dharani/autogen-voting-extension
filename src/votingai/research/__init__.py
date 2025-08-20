"""
Research and Evaluation Module

Microsoft Research-grade benchmarking, evaluation, and experimental analysis
components for systematic evaluation of voting systems.
"""

# Core evaluation metrics
from .evaluation_metrics import (
    QualityMetrics,
    PerformanceMetrics,
    ConsensusMetrics,
    BenchmarkMetrics,
    MetricsCollector,
    ComparisonResults
)

# Advanced evaluation framework
from .advanced_evaluation import (
    ExperimentType,
    ExperimentDesign,
    ExperimentalCondition,
    ResearchBenchmarkFramework as ResearchFramework
)

# Benchmarking suite
from .benchmarking_suite import (
    BenchmarkRunner,
    BenchmarkScenario,
    ScenarioType,
    BenchmarkConfiguration,
    ResultsAnalyzer
)

# Advanced evaluation
from .advanced_evaluation import (
    ResearchBenchmarkFramework,
    ConsensusQualityEvaluator,
    LearningEffectivenessAnalyzer,
    ScalabilityTester
)

__all__ = [
    # Evaluation metrics
    "QualityMetrics",
    "PerformanceMetrics",
    "ConsensusMetrics",
    "BenchmarkMetrics",
    "MetricsCollector",
    "ComparisonResults",
    
    # Advanced evaluation framework  
    "ExperimentType",
    "ExperimentDesign",
    "ExperimentalCondition",
    "ResearchFramework",
    
    # Benchmarking suite
    "BenchmarkRunner",
    "BenchmarkScenario",
    "ScenarioType",
    "BenchmarkConfiguration",
    "ResultsAnalyzer",
    
    # Advanced evaluation
    "ResearchBenchmarkFramework",
    "ConsensusQualityEvaluator",
    "LearningEffectivenessAnalyzer",
    "ScalabilityTester"
]