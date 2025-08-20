"""
Voting Protocols and Data Structures

Defines the fundamental protocols, enums, and data structures used throughout
the voting system. Following Microsoft Research naming conventions.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from ..security.cryptographic_services import SecurityValidator

# Security constants
MAX_PROPOSAL_LENGTH = 10000
MAX_REASONING_LENGTH = 5000
MAX_OPTION_LENGTH = 500
MAX_OPTIONS_COUNT = 20


class VotingMethod(str, Enum):
    """Enumeration of supported voting methods for consensus building."""
    
    MAJORITY = "majority"                # >50% of votes required
    PLURALITY = "plurality"              # Most votes wins (simple)
    UNANIMOUS = "unanimous"              # All voters must agree
    QUALIFIED_MAJORITY = "qualified_majority"  # Configurable threshold (e.g., 2/3)
    RANKED_CHOICE = "ranked_choice"      # Ranked choice voting with elimination


class VoteType(str, Enum):
    """Enumeration of vote types that can be cast."""
    
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class VotingPhase(str, Enum):
    """Enumeration of voting process phases."""
    
    PROPOSAL = "proposal"        # Initial proposal or discussion
    VOTING = "voting"           # Collecting votes
    CONSENSUS = "consensus"     # Consensus reached
    DISCUSSION = "discussion"   # Additional discussion needed


class VoteContent(BaseModel):
    """Content structure for vote messages."""
    
    vote: VoteType
    proposal_id: str
    reasoning: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ranked_choices: Optional[List[str]] = None  # For ranked choice voting
    signature: Optional[str] = None             # Cryptographic signature for integrity
    timestamp: Optional[str] = None             # Vote timestamp for audit trail

    def model_post_init(self, __context: Any) -> None:
        """Validate and sanitize vote content after initialization."""
        if self.reasoning:
            self.reasoning = SecurityValidator.sanitize_text(self.reasoning, MAX_REASONING_LENGTH)

        if self.ranked_choices:
            if len(self.ranked_choices) > MAX_OPTIONS_COUNT:
                raise ValueError(f"Too many ranked choices (max {MAX_OPTIONS_COUNT})")
            self.ranked_choices = [
                SecurityValidator.sanitize_text(choice, MAX_OPTION_LENGTH) 
                for choice in self.ranked_choices
            ]


class ProposalContent(BaseModel):
    """Content structure for proposal messages."""
    
    proposal_id: str
    title: str
    description: str
    options: List[str] = Field(default_factory=list)  # For multiple choice proposals
    requires_discussion: bool = False
    deadline: Optional[str] = None
    created_timestamp: Optional[str] = None           # Proposal creation timestamp
    proposer_signature: Optional[str] = None          # Cryptographic signature from proposer

    def model_post_init(self, __context: Any) -> None:
        """Validate and sanitize proposal content after initialization."""
        # Auto-generate secure ID if not provided
        if not self.proposal_id:
            self.proposal_id = SecurityValidator.generate_proposal_id()

        # Validate and sanitize text fields
        self.title = SecurityValidator.sanitize_text(self.title, MAX_OPTION_LENGTH)
        self.description = SecurityValidator.sanitize_text(self.description, MAX_PROPOSAL_LENGTH)

        # Validate options
        if len(self.options) > MAX_OPTIONS_COUNT:
            raise ValueError(f"Too many options (max {MAX_OPTIONS_COUNT})")
        self.options = [
            SecurityValidator.sanitize_text(option, MAX_OPTION_LENGTH) 
            for option in self.options
        ]


class VotingResult(BaseModel):
    """Structure for voting results."""
    
    proposal_id: str
    result: Literal["approved", "rejected", "no_consensus"]
    votes_summary: Dict[str, int]           # vote_type -> count
    winning_option: Optional[str] = None
    total_voters: int
    participation_rate: float
    confidence_average: float
    detailed_votes: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Enhanced result metadata
    byzantine_resilient: bool = True
    suspicious_agents: List[str] = Field(default_factory=list)
    reputation_adjusted: bool = False
    weighted_votes_summary: Optional[Dict[str, float]] = None