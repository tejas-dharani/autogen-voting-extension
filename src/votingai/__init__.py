"""
VotingAI - Democratic Consensus System for AI Agent Teams

A comprehensive, production-ready voting system for AI agent teams with
advanced enhancements, adaptive consensus mechanisms, and enterprise security.

Refactored Architecture:
- core: Fundamental voting protocols and base implementations
- consensus: Advanced adaptive consensus and deliberation algorithms
- intelligence: Semantic interpretation and natural language processing
- security: Cryptographic integrity, audit, and Byzantine fault tolerance
- research: Benchmarking, evaluation, and advanced testing frameworks
- utilities: Configuration management and common utilities
"""

from typing import Any, Dict

# Core voting system (foundational components)
# Consensus algorithms and strategies
from .consensus import (
    AdaptiveStrategySelector,
    ComplexityClassifier,
    ConsensusRecommendation,
    ConsensusStrategy,
    ContextualMetrics,
    ConvergenceAnalyzer,
    # Adaptive consensus strategies
    DecisionComplexity,
    DeliberationRound,
    DeliberationSummary,
    LearningFramework,
    PerformanceInsights,
    # Smart orchestration
    SmartConsensusOrchestrator,
    # Deliberation engine
    StructuredDeliberationEngine,
)
from .core import (
    # Base voting system
    BaseVotingGroupChat,
    # Core manager (Byzantine detector moved to security)
    CoreVotingManager,
    ProposalContent,
    ProposalMessage,
    VoteContent,
    VoteMessage,
    # Voting protocols
    VoteType,
    VotingGroupChatConfiguration,
    VotingManagerState,
    VotingMethod,
    VotingPhase,
    VotingResult,
    VotingResultMessage,
)

# Intelligence and semantic understanding
from .intelligence import (
    ConfidenceLevel,
    ContentAnalysisResult,
    ContextualAnalyzer,
    IntentionClassifier,
    MessageInsightExtractor,
    NaturalLanguageProcessor,
    ParsingStatistics,
    # Natural language processing
    PatternLibrary,
    SemanticVoteInterpreter,
    SemanticVoteResult,
    # Semantic interpretation
    VoteIntention,
    # Vote understanding
    VoteUnderstandingEngine,
)

# Research and evaluation framework
# Note: Research components available but not exported by default
# Security and integrity
from .security import (
    # Audit framework
    AuditLogger,
    # Byzantine fault tolerance (now properly in security module)
    ByzantineFaultDetector,
    CryptographicIntegrity,
    IByzantineDetectionStrategy,
    ReputationBasedDetectionStrategy,
    # Cryptographic services
    SecurityValidator,
)

# Configuration and utilities
from .utilities import (
    DEFAULT_MODEL,
    ConfigurationError,
    ErrorCodes,
    LoggingConfiguration,
    ModelConfiguration,
    ProcessingError,
    SecurityError,
    # Configuration management
    VotingSystemConfig,
    # Common types and errors
    VotingSystemError,
)

# Core functionality exports

# Version and metadata
__version__ = "2.0.0"  # Major version bump for architectural refactoring
__author__ = "VotingAI Development Team"
__description__ = "Enterprise-grade democratic consensus system for AI agent teams"

# Core exports for most common use cases
__all__ = [
    # === CORE VOTING SYSTEM ===
    # Protocols and data structures
    "VoteType",
    "VotingMethod",
    "VotingPhase",
    "VoteContent",
    "ProposalContent",
    "VotingResult",
    # Base voting implementation
    "BaseVotingGroupChat",
    "VotingGroupChatConfiguration",
    "VoteMessage",
    "ProposalMessage",
    "VotingResultMessage",
    # Core management
    "CoreVotingManager",
    "VotingManagerState",
    # === ENHANCED VOTING SYSTEM ===
    # Enhanced components moved to core - use BaseVotingGroupChat
    # === CONSENSUS ALGORITHMS ===
    # Strategy selection
    "DecisionComplexity",
    "ConsensusStrategy",
    "AdaptiveStrategySelector",
    "ContextualMetrics",
    "ComplexityClassifier",
    # Deliberation
    "StructuredDeliberationEngine",
    "DeliberationRound",
    "DeliberationSummary",
    "ConvergenceAnalyzer",
    # Orchestration
    "SmartConsensusOrchestrator",
    "ConsensusRecommendation",
    "LearningFramework",
    "PerformanceInsights",
    # === INTELLIGENCE & NLP ===
    # Semantic interpretation
    "VoteIntention",
    "ConfidenceLevel",
    "SemanticVoteResult",
    "SemanticVoteInterpreter",
    # NLP components
    "PatternLibrary",
    "ContextualAnalyzer",
    "ContentAnalysisResult",
    "NaturalLanguageProcessor",
    # Vote understanding
    "VoteUnderstandingEngine",
    "IntentionClassifier",
    "MessageInsightExtractor",
    "ParsingStatistics",
    # === SECURITY & INTEGRITY ===
    "SecurityValidator",
    "CryptographicIntegrity",
    "AuditLogger",
    "ByzantineFaultDetector",
    "IByzantineDetectionStrategy",
    "ReputationBasedDetectionStrategy",
    # === CONFIGURATION & UTILITIES ===
    "VotingSystemConfig",
    "ModelConfiguration",
    "LoggingConfiguration",
    "DEFAULT_MODEL",
    "VotingSystemError",
    "ConfigurationError",
    "SecurityError",
    "ProcessingError",
    "ErrorCodes",
    # === END OF EXPORTS ==="
]


def get_version_info() -> Dict[str, Any]:
    """Get detailed version and component information."""
    return {
        "version": __version__,
        "description": __description__,
        "architecture": "Modular Enterprise Architecture",
        "components": {
            "core": "Fundamental voting protocols and base implementations",
            "consensus": "Adaptive consensus algorithms and deliberation strategies",
            "intelligence": "Semantic interpretation and natural language processing",
            "security": "Cryptographic integrity and Byzantine fault tolerance",
            "research": "Benchmarking, evaluation, and advanced testing frameworks",
            "utilities": "Configuration management and common utilities",
        },
        "development_status": "Active Development",
    }


# Module-level documentation for discoverability
def list_voting_systems() -> Dict[str, str]:
    """List available voting system configurations."""
    return {"BaseVotingGroupChat": "Core voting system with essential features"}
