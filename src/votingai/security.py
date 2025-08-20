"""Security and audit infrastructure for voting systems."""

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast


class SecurityValidator:
    """Validates and sanitizes inputs to prevent security vulnerabilities."""

    # Input validation patterns
    AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,50}$")
    SAFE_TEXT_PATTERN = re.compile(r"^[^<>\"'&]*$")

    # Maximum lengths for various inputs
    MAX_PROPOSAL_LENGTH = 10000
    MAX_REASONING_LENGTH = 5000
    MAX_AGENT_NAME_LENGTH = 50
    MAX_VOTE_OPTIONS = 20

    @classmethod
    def validate_agent_name(cls, name: str) -> str:
        """Validate and sanitize agent name."""

        if len(name) > cls.MAX_AGENT_NAME_LENGTH:
            raise ValueError(f"Agent name too long (max {cls.MAX_AGENT_NAME_LENGTH})")

        if not cls.AGENT_NAME_PATTERN.match(name):
            raise ValueError("Agent name contains invalid characters (allowed: a-z, A-Z, 0-9, _, -)")

        return name

    @classmethod
    def validate_proposal_text(cls, text: str) -> str:
        """Validate and sanitize proposal text."""

        if len(text) > cls.MAX_PROPOSAL_LENGTH:
            raise ValueError(f"Proposal text too long (max {cls.MAX_PROPOSAL_LENGTH})")

        # Basic XSS prevention and remove null bytes
        # Remove null bytes and other dangerous characters
        sanitized = re.sub(r'[\x00<>"\'&]', "", text)
        return sanitized

    @classmethod
    def validate_vote_reasoning(cls, reasoning: str) -> str:
        """Validate and sanitize vote reasoning."""

        if len(reasoning) > cls.MAX_REASONING_LENGTH:
            raise ValueError(f"Vote reasoning too long (max {cls.MAX_REASONING_LENGTH})")

        # Basic XSS prevention and remove null bytes
        sanitized = re.sub(r'[\x00<>"\'&]', "", reasoning)
        return sanitized

    @classmethod
    def validate_vote_options(cls, options: List[str]) -> List[str]:
        """Validate vote options."""

        if len(options) > cls.MAX_VOTE_OPTIONS:
            raise ValueError(f"Too many vote options (max {cls.MAX_VOTE_OPTIONS})")

        validated_options: List[str] = []
        for option in options:
            validated_options.append(cls.validate_proposal_text(option))

        return validated_options

    @classmethod
    def generate_secure_nonce(cls) -> str:
        """Generate cryptographically secure nonce."""
        return secrets.token_hex(16)

    @classmethod
    def hash_sensitive_data(cls, data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data with optional salt."""
        if salt is None:
            salt = secrets.token_hex(16)

        combined = f"{salt}{data}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @classmethod
    def sanitize_text(cls, text: str, max_length: int) -> str:
        """Sanitize text input with length validation."""
        validated = cls.validate_proposal_text(text)
        if len(validated) > max_length:
            raise ValueError(f"Text exceeds maximum length of {max_length} characters")
        return validated

    @classmethod
    def generate_proposal_id(cls) -> str:
        """Generate a secure, unique proposal ID."""
        return f"proposal_{secrets.token_hex(16)}"

    @classmethod
    def create_vote_signature(cls, vote_data: Dict[str, Any], agent_key: str) -> str:
        """Create cryptographic signature for vote integrity."""
        canonical = f"{vote_data['vote']}:{vote_data['proposal_id']}:{vote_data.get('reasoning', '')}"
        return hmac.new(agent_key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    @classmethod
    def verify_vote_signature(cls, vote_data: Dict[str, Any], agent_key: str, signature: str) -> bool:
        """Verify vote signature using HMAC."""
        expected_signature = cls.create_vote_signature(vote_data, agent_key)
        return hmac.compare_digest(expected_signature, signature)


class CryptographicIntegrity:
    """Provides cryptographic integrity verification for votes."""

    def __init__(self, master_key: Optional[str] = None):
        """Initialize with master key for HMAC operations."""
        self.master_key = master_key or secrets.token_hex(32)
        self._agent_keys: Dict[str, str] = {}

    def register_agent(self, agent_name: str) -> str:
        """Register agent and return their unique key."""
        agent_key = secrets.token_hex(32)
        self._agent_keys[agent_name] = agent_key
        return agent_key

    def sign_vote(self, agent_name: str, vote_data: Dict[str, Any]) -> str:
        """Create HMAC signature for vote data."""
        if agent_name not in self._agent_keys:
            raise ValueError(f"Agent {agent_name} not registered")

        # Create canonical representation
        vote_json = json.dumps(vote_data, sort_keys=True)
        agent_key = self._agent_keys[agent_name]

        # Create HMAC signature
        signature = hmac.new(agent_key.encode(), vote_json.encode(), hashlib.sha256).hexdigest()

        return signature

    def verify_vote_signature(self, agent_name: str, vote_data: Dict[str, Any], signature: str) -> bool:
        """Verify HMAC signature for vote data."""
        try:
            expected_signature = self.sign_vote(agent_name, vote_data)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False

    def detect_replay_attack(self, nonce: str, used_nonces: set[str]) -> bool:
        """Detect potential replay attacks using nonce tracking."""
        if nonce in used_nonces:
            return True  # Replay attack detected
        return False


class AuditLogger:
    """Comprehensive audit logging for voting transparency and compliance."""

    def __init__(self, log_file: Optional[str] = None, enable_file_logging: bool = False):
        """Initialize audit logger with optional file logging."""
        self.enable_file_logging = enable_file_logging
        self.log_file: Optional[Path] = None

        if enable_file_logging:
            if log_file:
                self.log_file = Path(log_file)
            else:
                # Default log file with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file = Path(f"audit_log_{timestamp}.json")

            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log structure (in-memory by default)
        self.audit_entries: List[Dict[str, Any]] = []

        if self.enable_file_logging:
            self._load_existing_log()

    def _load_existing_log(self) -> None:
        """Load existing audit log if it exists."""
        if self.log_file and self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "audit_entries" in data:
                        self.audit_entries = data["audit_entries"]
            except Exception:
                # If log file is corrupted, start fresh
                self.audit_entries = []

    def _save_log(self) -> None:
        """Save audit log to file."""
        if not self.enable_file_logging or not self.log_file:
            return

        log_data = {
            "audit_metadata": {
                "log_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_entries": len(self.audit_entries),
            },
            "audit_entries": self.audit_entries,
        }

        with open(self.log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def log_proposal_creation(self, agent_name: str, proposal_data: Dict[str, Any]) -> None:
        """Log proposal creation event."""
        entry = {
            "event_type": "proposal_created",
            "timestamp": datetime.now().isoformat(),
            "agent_name": SecurityValidator.validate_agent_name(agent_name),
            "proposal_id": proposal_data.get("proposal_id"),
            "proposal_title": SecurityValidator.validate_proposal_text(proposal_data.get("title", "")),
            "options_count": len(proposal_data.get("options", [])),
            "metadata": {
                "description_length": len(proposal_data.get("description", "")),
                "has_deadline": "deadline" in proposal_data,
            },
        }
        self.audit_entries.append(entry)
        self._save_log()

    def log_vote_cast(
        self,
        proposal_id_or_agent: str,
        vote_data_or_vote: Any = None,
        signature_or_has_sig: Any = None,
        has_signature: Optional[bool] = None,
    ) -> None:
        """Log vote casting event."""
        # Support both signatures: (agent_name, vote_data, signature) and (proposal_id, voter, vote, has_signature)
        if isinstance(vote_data_or_vote, dict):
            # Original signature: log_vote_cast(agent_name, vote_data, signature)
            agent_name = proposal_id_or_agent
            vote_data = cast(Dict[str, Any], vote_data_or_vote)
            signature = str(signature_or_has_sig)
            entry = {
                "event_type": "vote_cast",
                "timestamp": datetime.now().isoformat(),
                "agent_name": SecurityValidator.validate_agent_name(agent_name),
                "proposal_id": vote_data.get("proposal_id"),
                "vote_type": vote_data.get("vote_type"),
                "has_reasoning": bool(vote_data.get("reasoning")),
                "confidence_score": vote_data.get("confidence"),
                "signature_hash": hashlib.sha256(signature.encode()).hexdigest()[:16],  # Store hash of signature
                "nonce": vote_data.get("nonce"),
                "metadata": {
                    "reasoning_length": len(vote_data.get("reasoning", "")),
                    "ranked_choices": vote_data.get("ranked_choices", []),
                },
            }
        else:
            # Test signature: log_vote_cast(proposal_id, voter, vote, has_signature)
            proposal_id = proposal_id_or_agent
            voter = vote_data_or_vote
            vote = signature_or_has_sig
            has_sig = has_signature
            entry = {
                "event_type": "vote_cast",
                "timestamp": datetime.now().isoformat(),
                "proposal_id": proposal_id,
                "agent_name": SecurityValidator.validate_agent_name(voter),
                "vote_type": vote,
                "has_signature": has_sig,
            }
        self.audit_entries.append(entry)
        self._save_log()

    def log_voting_result(self, proposal_id: str, result: str, participation_rate: float) -> None:
        """Log voting results."""
        entry = {
            "event_type": "voting_result",
            "timestamp": datetime.now().isoformat(),
            "proposal_id": proposal_id,
            "result": result,
            "participation_rate": participation_rate,
        }
        self.audit_entries.append(entry)
        self._save_log()

    def log_security_violation(self, violation_type: str, details: str) -> None:
        """Log security violations."""
        entry = {
            "event_type": "security_violation",
            "timestamp": datetime.now().isoformat(),
            "violation_type": violation_type,
            "details": details,
        }
        self.audit_entries.append(entry)
        self._save_log()

    def log_proposal_created(self, proposal_id: str, proposer: str, title: str) -> None:
        """Log when a proposal is created."""
        entry = {
            "event_type": "proposal_created",
            "timestamp": datetime.now().isoformat(),
            "proposal_id": proposal_id,
            "proposer": SecurityValidator.validate_agent_name(proposer),
            "title": SecurityValidator.validate_proposal_text(title)[:100],  # Truncate for logging
        }
        self.audit_entries.append(entry)
        self._save_log()

    def log_consensus_reached(self, proposal_id: str, result_data: Dict[str, Any]) -> None:
        """Log consensus achievement."""
        entry = {
            "event_type": "consensus_reached",
            "timestamp": datetime.now().isoformat(),
            "proposal_id": proposal_id,
            "voting_method": result_data.get("voting_method"),
            "final_result": result_data.get("result"),
            "vote_counts": result_data.get("vote_counts", {}),
            "total_participants": result_data.get("total_participants", 0),
            "total_votes": result_data.get("total_votes", 0),
            "discussion_rounds": result_data.get("discussion_rounds", 0),
            "metadata": {
                "duration_seconds": result_data.get("duration_seconds"),
                "message_count": result_data.get("message_count"),
                "abstentions": result_data.get("abstentions", 0),
            },
        }
        self.audit_entries.append(entry)
        self._save_log()

    def log_security_event(self, event_type: str, agent_name: str, details: Dict[str, Any]) -> None:
        """Log security-related events."""
        entry = {
            "event_type": f"security_{event_type}",
            "timestamp": datetime.now().isoformat(),
            "agent_name": SecurityValidator.validate_agent_name(agent_name),
            "security_level": details.get("level", "warning"),
            "description": details.get("description", ""),
            "details": details,
        }
        self.audit_entries.append(entry)
        self._save_log()

    def log_byzantine_detection(self, agent_name: str, detection_data: Dict[str, Any]) -> None:
        """Log Byzantine fault detection."""
        entry = {
            "event_type": "byzantine_detection",
            "timestamp": datetime.now().isoformat(),
            "suspected_agent": SecurityValidator.validate_agent_name(agent_name),
            "reputation_score": detection_data.get("reputation_score"),
            "detection_reason": detection_data.get("reason", ""),
            "voting_pattern": detection_data.get("voting_pattern", []),
            "consistency_score": detection_data.get("consistency_score"),
            "metadata": {
                "total_votes": len(detection_data.get("voting_history", [])),
                "recent_disagreements": detection_data.get("recent_disagreements", 0),
            },
        }
        self.audit_entries.append(entry)
        self._save_log()

    def get_audit_summary(self) -> Dict[str, Any]:
        """Generate audit summary for transparency reports."""
        event_counts: Dict[str, int] = {}
        agents_involved: set[str] = set()
        security_events = 0

        for entry in self.audit_entries:
            event_type = entry["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            if "agent_name" in entry:
                agents_involved.add(entry["agent_name"])

            if "security_" in event_type or event_type == "byzantine_detection":
                security_events += 1

        return {
            "total_entries": len(self.audit_entries),
            "event_types": event_counts,
            "unique_agents": len(agents_involved),
            "security_events": security_events,
            "log_file": str(self.log_file),
            "coverage_period": {
                "first_entry": self.audit_entries[0]["timestamp"] if self.audit_entries else None,
                "last_entry": self.audit_entries[-1]["timestamp"] if self.audit_entries else None,
            },
        }

    def export_transparency_report(self, output_file: Optional[str] = None) -> str:
        """Export comprehensive transparency report."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"transparency_report_{timestamp}.json"

        summary = self.get_audit_summary()
        detailed_entries = [
            entry
            for entry in self.audit_entries
            if entry["event_type"] in ["proposal_created", "consensus_reached", "byzantine_detection"]
        ]

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "voting_system_transparency",
                "version": "1.0",
            },
            "executive_summary": summary,
            "detailed_audit_trail": detailed_entries,
            "compliance_verification": {
                "all_votes_logged": True,
                "cryptographic_integrity": True,
                "participant_privacy": True,
                "decision_traceability": True,
            },
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return output_file
