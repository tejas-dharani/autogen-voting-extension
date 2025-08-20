"""
VotingAI - Microsoft Research Grade Democratic Consensus System

A comprehensive, production-ready voting system for AI agent teams with
research-grade enhancements, adaptive consensus mechanisms, and enterprise security.

Refactored Architecture:
- core: Fundamental voting protocols and base implementations
- consensus: Advanced adaptive consensus and deliberation algorithms  
- intelligence: Semantic interpretation and natural language processing
- system: Enhanced voting frameworks with integrated features
- security: Cryptographic integrity, audit, and Byzantine fault tolerance
- utilities: Configuration management and common utilities
"""

# Core voting system (foundational components)
from .core import (
    # Voting protocols
    VoteType,
    VotingMethod, 
    VotingPhase,
    VoteContent,
    ProposalContent,
    VotingResult,
    
    # Base voting system
    BaseVotingGroupChat,
    VotingGroupChatConfiguration,
    VoteMessage,
    ProposalMessage,
    VotingResultMessage,
    
    # Core manager
    CoreVotingManager,
    VotingManagerState,
    ByzantineFaultDetector
)

# Enhanced voting system (production-ready with all features)
from .system import (
    # Enhanced framework with adaptive features
    EnhancedVotingGroupChat,
    VotingSystemConfiguration,
    
    # Factory functions for different use cases
    create_adaptive_voting_system,
    create_research_voting_system,
    create_lightweight_voting_system,
    create_standard_voting_system,
    
    # Advanced management
    AdvancedVotingManager,
    EnhancedVotingPhase,
    SystemMetrics
)

# Consensus algorithms and strategies  
from .consensus import (
    # Adaptive consensus strategies
    DecisionComplexity,
    ConsensusStrategy,
    AdaptiveStrategySelector,
    ContextualMetrics,
    ComplexityClassifier,
    
    # Deliberation engine
    StructuredDeliberationEngine,
    DeliberationRound,
    DeliberationSummary,
    ConvergenceAnalyzer,
    
    # Smart orchestration
    SmartConsensusOrchestrator,
    ConsensusRecommendation,
    LearningFramework,
    PerformanceInsights
)

# Intelligence and semantic understanding
from .intelligence import (
    # Semantic interpretation
    VoteIntention,
    ConfidenceLevel,
    SemanticVoteResult,
    SemanticVoteInterpreter,
    
    # Natural language processing
    PatternLibrary,
    ContextualAnalyzer,
    ContentAnalysisResult,
    NaturalLanguageProcessor,
    
    # Vote understanding
    VoteUnderstandingEngine,
    IntentionClassifier,
    MessageInsightExtractor,
    ParsingStatistics
)

# Security and integrity
from .security import (
    # Cryptographic services
    SecurityValidator,
    CryptographicIntegrity,
    
    # Audit framework
    AuditLogger,
    
    # Byzantine protection (reexport from core for convenience)
    # ByzantineFaultDetector - already exported from core
)

# Configuration and utilities
from .utilities import (
    # Configuration management
    VotingSystemConfig,
    ModelConfiguration,
    LoggingConfiguration,
    DEFAULT_MODEL,
    
    # Common types and errors
    VotingSystemError,
    ConfigurationError,
    SecurityError,
    ProcessingError,
    ErrorCodes
)

# Backward compatibility exports (legacy names)
VotingGroupChat = BaseVotingGroupChat
VotingGroupChatManager = CoreVotingManager

# Legacy adaptive consensus exports
SmartConsensusEngine = SmartConsensusOrchestrator  # Renamed for clarity
StructuredDeliberation = StructuredDeliberationEngine  # Renamed for clarity

# Legacy semantic parser export
SemanticVoteParser = SemanticVoteInterpreter  # Renamed for clarity

# Version and metadata
__version__ = "2.0.0"  # Major version bump for architectural refactoring
__author__ = "Microsoft Research Standards Implementation"
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
    "ByzantineFaultDetector",
    
    # === ENHANCED VOTING SYSTEM ===
    # Enhanced framework
    "EnhancedVotingGroupChat",
    "VotingSystemConfiguration",
    
    # Factory functions
    "create_adaptive_voting_system",
    "create_research_voting_system",
    "create_lightweight_voting_system", 
    "create_standard_voting_system",
    
    # Advanced features
    "AdvancedVotingManager",
    "EnhancedVotingPhase",
    "SystemMetrics",
    
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
    
    # === BACKWARD COMPATIBILITY ===
    "VotingGroupChat",          # -> BaseVotingGroupChat
    "VotingGroupChatManager",   # -> CoreVotingManager
    "SmartConsensusEngine",     # -> SmartConsensusOrchestrator
    "StructuredDeliberation",   # -> StructuredDeliberationEngine
    "SemanticVoteParser",       # -> SemanticVoteInterpreter
]


def get_version_info():
    """Get detailed version and component information."""
    return {
        "version": __version__,
        "description": __description__,
        "architecture": "Microsoft Research Standards",
        "components": {
            "core": "Fundamental voting protocols and base implementations",
            "consensus": "Adaptive consensus algorithms and deliberation strategies", 
            "intelligence": "Semantic interpretation and natural language processing",
            "system": "Enhanced voting frameworks with integrated features",
            "security": "Cryptographic integrity and Byzantine fault tolerance",
            "utilities": "Configuration management and common utilities"
        },
        "compatibility": {
            "backward_compatible": True,
            "legacy_exports": [
                "VotingGroupChat", "VotingGroupChatManager", 
                "SmartConsensusEngine", "StructuredDeliberation", 
                "SemanticVoteParser"
            ]
        }
    }


# Module-level documentation for discoverability
def list_voting_systems():
    """List available voting system configurations."""
    return {
        "BaseVotingGroupChat": "Core voting system with essential features",
        "EnhancedVotingGroupChat": "Full-featured system with adaptive consensus",
        "create_adaptive_voting_system": "Factory for complexity-aware voting",
        "create_research_voting_system": "Factory for research and evaluation",
        "create_lightweight_voting_system": "Factory for high-performance scenarios",
        "create_standard_voting_system": "Factory for backward compatibility"
    }