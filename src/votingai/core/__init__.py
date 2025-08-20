"""
Core Voting System Components

This module contains the fundamental voting system infrastructure following
Microsoft Research architectural patterns and naming conventions.
"""

# Core voting protocols and enums
from .voting_protocols import (
    VoteType,
    VotingMethod,
    VotingPhase,
    VoteContent,
    ProposalContent,
    VotingResult
)

# Base voting system classes
from .base_voting_system import (
    BaseVotingGroupChat,
    VotingGroupChatConfiguration,
    VoteMessage,
    ProposalMessage,
    VotingResultMessage
)

# Core voting manager
from .voting_manager import (
    CoreVotingManager,
    VotingManagerState,
    ByzantineFaultDetector
)

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
    
    # Core manager
    "CoreVotingManager",
    "VotingManagerState",
    "ByzantineFaultDetector"
]