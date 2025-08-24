"""
Research and Evaluation Module

Advanced benchmarking, evaluation, and experimental analysis
components for systematic evaluation of voting systems.
"""

# Core evaluation metrics
# Advanced evaluation framework
# Advanced evaluation
from .advanced_evaluation import (
    ConsensusQualityEvaluator,
    ExperimentalCondition,
    ExperimentDesign,
    ExperimentType,
    LearningEffectivenessAnalyzer,
    ResearchBenchmarkFramework,
    ScalabilityTester,
)
from .advanced_evaluation import (
    ResearchBenchmarkFramework as ResearchFramework,
)

# Benchmarking suite
from .benchmarking_suite import (
    BenchmarkConfiguration,
    BenchmarkRunner,
    BenchmarkScenario,
    ResultsAnalyzer,
    ScenarioType,
)
from .evaluation_metrics import (
    BenchmarkMetrics,
    ComparisonResults,
    ConsensusMetrics,
    MetricsCollector,
    PerformanceMetrics,
    QualityMetrics,
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
    "ScalabilityTester",
]
