"""
Common Types and Error Definitions

Shared type definitions, error classes, and constants used throughout
the voting system for consistency and maintainability.
"""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCodes(str, Enum):
    """Standard error codes for the voting system."""

    # Configuration errors
    INVALID_CONFIG = "INVALID_CONFIG"
    MISSING_REQUIRED_PARAM = "MISSING_REQUIRED_PARAM"
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"

    # Security errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    SIGNATURE_VERIFICATION_FAILED = "SIGNATURE_VERIFICATION_FAILED"
    REPLAY_ATTACK_DETECTED = "REPLAY_ATTACK_DETECTED"
    BYZANTINE_BEHAVIOR_DETECTED = "BYZANTINE_BEHAVIOR_DETECTED"

    # Processing errors
    SEMANTIC_PARSING_FAILED = "SEMANTIC_PARSING_FAILED"
    VOTE_VALIDATION_FAILED = "VOTE_VALIDATION_FAILED"
    CONSENSUS_CALCULATION_FAILED = "CONSENSUS_CALCULATION_FAILED"
    PROPOSAL_VALIDATION_FAILED = "PROPOSAL_VALIDATION_FAILED"

    # System errors
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED"
    INTERNAL_SYSTEM_ERROR = "INTERNAL_SYSTEM_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

    # Data errors
    INVALID_INPUT_FORMAT = "INVALID_INPUT_FORMAT"
    DATA_CORRUPTION_DETECTED = "DATA_CORRUPTION_DETECTED"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"


class VotingSystemError(Exception):
    """Base exception class for all voting system errors."""

    def __init__(self, message: str, error_code: ErrorCodes, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
            "error_type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        details_str = f" (Details: {self.details})" if self.details else ""
        return f"[{self.error_code.value}] {self.message}{details_str}"


class ConfigurationError(VotingSystemError):
    """Error in system configuration."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCodes.INVALID_CONFIG, details)


class SecurityError(VotingSystemError):
    """Security-related error."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCodes = ErrorCodes.AUTHENTICATION_FAILED,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)


class ProcessingError(VotingSystemError):
    """Error during vote or proposal processing."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCodes = ErrorCodes.INTERNAL_SYSTEM_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)


class TimeoutError(VotingSystemError):
    """Operation timeout error."""

    def __init__(self, message: str, timeout_seconds: float):
        super().__init__(message, ErrorCodes.TIMEOUT_EXCEEDED, {"timeout_seconds": timeout_seconds})


class ValidationError(VotingSystemError):
    """Input validation error."""

    def __init__(self, message: str, field_name: Optional[str] = None, invalid_value: Optional[Any] = None):
        details: Dict[str, Any] = {}
        if field_name:
            details["field_name"] = field_name
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)

        super().__init__(message, ErrorCodes.INVALID_INPUT_FORMAT, details)


# Type aliases for common patterns
VoteData = Dict[str, Any]
ProposalData = Dict[str, Any]
ConfigurationData = Dict[str, Any]
MetricsData = Dict[str, Any]
AuditData = Dict[str, Any]
