"""
Configuration Management

Enhanced configuration system with model settings, logging, and environment management.
Refactored from config.py with Microsoft Research standards.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels for the voting system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfiguration:
    """Configuration for AI model settings."""
    
    default_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    @classmethod
    def from_environment(cls) -> 'ModelConfiguration':
        """Create configuration from environment variables."""
        return cls(
            default_model=os.getenv('VOTINGAI_DEFAULT_MODEL', cls.default_model),
            fallback_model=os.getenv('VOTINGAI_FALLBACK_MODEL', cls.fallback_model),
            max_tokens=int(os.getenv('VOTINGAI_MAX_TOKENS', cls.max_tokens)),
            temperature=float(os.getenv('VOTINGAI_TEMPERATURE', cls.temperature)),
            timeout_seconds=int(os.getenv('VOTINGAI_TIMEOUT', cls.timeout_seconds)),
            retry_attempts=int(os.getenv('VOTINGAI_RETRIES', cls.retry_attempts))
        )


@dataclass
class LoggingConfiguration:
    """Configuration for logging system."""
    
    level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = False
    log_directory: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_audit_logging: bool = True
    
    @classmethod
    def from_environment(cls) -> 'LoggingConfiguration':
        """Create configuration from environment variables."""
        return cls(
            level=LogLevel(os.getenv('VOTINGAI_LOG_LEVEL', cls.level.value)),
            enable_file_logging=os.getenv('VOTINGAI_FILE_LOGGING', 'false').lower() == 'true',
            log_directory=os.getenv('VOTINGAI_LOG_DIR', cls.log_directory),
            max_file_size_mb=int(os.getenv('VOTINGAI_LOG_MAX_SIZE', cls.max_file_size_mb)),
            backup_count=int(os.getenv('VOTINGAI_LOG_BACKUPS', cls.backup_count)),
            enable_audit_logging=os.getenv('VOTINGAI_AUDIT_LOGGING', 'true').lower() == 'true'
        )


@dataclass
class VotingSystemConfig:
    """Complete configuration for the voting system."""
    
    model: ModelConfiguration
    logging: LoggingConfiguration
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_concurrent_votes: int = 50
    
    # Feature flags
    enable_semantic_parsing: bool = True
    enable_adaptive_consensus: bool = True
    enable_byzantine_protection: bool = True
    enable_metrics_collection: bool = True
    
    # Security settings
    enable_cryptographic_signatures: bool = True
    require_agent_authentication: bool = True
    max_proposal_length: int = 10000
    max_reasoning_length: int = 5000
    
    @classmethod
    def from_environment(cls) -> 'VotingSystemConfig':
        """Create complete configuration from environment variables."""
        return cls(
            model=ModelConfiguration.from_environment(),
            logging=LoggingConfiguration.from_environment(),
            enable_caching=os.getenv('VOTINGAI_ENABLE_CACHING', 'true').lower() == 'true',
            cache_ttl_seconds=int(os.getenv('VOTINGAI_CACHE_TTL', '300')),
            max_concurrent_votes=int(os.getenv('VOTINGAI_MAX_CONCURRENT', '50')),
            enable_semantic_parsing=os.getenv('VOTINGAI_SEMANTIC_PARSING', 'true').lower() == 'true',
            enable_adaptive_consensus=os.getenv('VOTINGAI_ADAPTIVE_CONSENSUS', 'true').lower() == 'true',
            enable_byzantine_protection=os.getenv('VOTINGAI_BYZANTINE_PROTECTION', 'true').lower() == 'true',
            enable_metrics_collection=os.getenv('VOTINGAI_METRICS', 'true').lower() == 'true',
            enable_cryptographic_signatures=os.getenv('VOTINGAI_CRYPTO_SIGNATURES', 'true').lower() == 'true',
            require_agent_authentication=os.getenv('VOTINGAI_REQUIRE_AUTH', 'true').lower() == 'true',
            max_proposal_length=int(os.getenv('VOTINGAI_MAX_PROPOSAL_LENGTH', '10000')),
            max_reasoning_length=int(os.getenv('VOTINGAI_MAX_REASONING_LENGTH', '5000'))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'default_model': self.model.default_model,
                'fallback_model': self.model.fallback_model,
                'max_tokens': self.model.max_tokens,
                'temperature': self.model.temperature,
                'timeout_seconds': self.model.timeout_seconds,
                'retry_attempts': self.model.retry_attempts
            },
            'logging': {
                'level': self.logging.level.value,
                'enable_file_logging': self.logging.enable_file_logging,
                'log_directory': self.logging.log_directory,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count,
                'enable_audit_logging': self.logging.enable_audit_logging
            },
            'performance': {
                'enable_caching': self.enable_caching,
                'cache_ttl_seconds': self.cache_ttl_seconds,
                'max_concurrent_votes': self.max_concurrent_votes
            },
            'features': {
                'enable_semantic_parsing': self.enable_semantic_parsing,
                'enable_adaptive_consensus': self.enable_adaptive_consensus,
                'enable_byzantine_protection': self.enable_byzantine_protection,
                'enable_metrics_collection': self.enable_metrics_collection
            },
            'security': {
                'enable_cryptographic_signatures': self.enable_cryptographic_signatures,
                'require_agent_authentication': self.require_agent_authentication,
                'max_proposal_length': self.max_proposal_length,
                'max_reasoning_length': self.max_reasoning_length
            }
        }


# Default model configuration
DEFAULT_MODEL = "gpt-4o-mini"

# Global configuration instance
_global_config: Optional[VotingSystemConfig] = None


def get_global_config() -> VotingSystemConfig:
    """Get or create global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = VotingSystemConfig.from_environment()
    return _global_config


def set_global_config(config: VotingSystemConfig) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config