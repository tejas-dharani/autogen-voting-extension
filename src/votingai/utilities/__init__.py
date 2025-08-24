"""
Utilities Module

Common utilities, configuration management, and shared type definitions
used throughout the voting system.
"""

# Configuration management
# Common types and constants
from .common_types import ConfigurationError, ErrorCodes, ProcessingError, SecurityError, VotingSystemError
from .configuration_management import DEFAULT_MODEL, LoggingConfiguration, ModelConfiguration, VotingSystemConfig

__all__ = [
    # Configuration
    "VotingSystemConfig",
    "ModelConfiguration",
    "LoggingConfiguration",
    "DEFAULT_MODEL",
    # Common types
    "VotingSystemError",
    "ConfigurationError",
    "SecurityError",
    "ProcessingError",
    "ErrorCodes",
]
