"""
Core Voting System Components

This module contains the fundamental voting system infrastructure following

"""

# Core voting protocols and enums
# Refactored components (security moved to security module)
from ..security.byzantine_fault_detector import ByzantineFaultDetector

# Base voting system classes
from .base_voting_system import (
    BaseVotingGroupChat,
    ProposalMessage,
    VoteMessage,
    VotingGroupChatConfiguration,
    VotingResultMessage,
)
from .speaker_selection_service import SpeakerSelectionService

# Core voting manager (refactored)
from .voting_manager import (
    RefactoredVotingManager as CoreVotingManager,
)
from .voting_manager import (
    VotingManagerState,
    create_enterprise_voting_manager,
)
from .voting_protocols import ProposalContent, VoteContent, VoteType, VotingMethod, VotingPhase, VotingResult
from .voting_strategies import IVotingStrategy, VotingStrategyFactory

__all__ = [
    # Protocols and enums
    "VoteType",
    "VotingMethod",
    "VotingPhase",
    "VoteContent",
    "ProposalContent",
    "VotingResult",
    # Base system classes
    "BaseVotingGroupChat",
    "VotingGroupChatConfiguration",
    "VoteMessage",
    "ProposalMessage",
    "VotingResultMessage",
    # Core manager (refactored)
    "CoreVotingManager",
    "VotingManagerState",
    "create_enterprise_voting_manager",
    # Refactored components
    "ByzantineFaultDetector",
    "VotingStrategyFactory",
    "IVotingStrategy",
    "SpeakerSelectionService",
]
