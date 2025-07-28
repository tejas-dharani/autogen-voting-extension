# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial development setup
- Core project structure and configuration

## [0.1.0] - 2025-07-28

### Added
- **VotingGroupChat** - Democratic consensus system for AutoGen multi-agent teams
- **Multiple Voting Methods**:
  - Majority voting (>50% approval)
  - Plurality voting (most votes wins)
  - Unanimous consensus (all voters must agree)
  - Qualified majority (configurable threshold)
  - Ranked choice voting (with preference elimination)
- **Advanced Configuration Options**:
  - Configurable voting thresholds
  - Discussion rounds before final decisions
  - Abstention support with flexible participation rules
  - Reasoning requirements for transparent decision-making
  - Confidence scoring for vote quality assessment
  - Auto-proposer selection for structured workflows
- **Rich Message Types**:
  - `ProposalMessage` - Structured proposals with options and metadata
  - `VoteMessage` - Votes with reasoning, confidence scores, and ranked choices
  - `VotingResultMessage` - Comprehensive results with participation analytics
- **State Management**:
  - Persistent voting state across conversations
  - Phase tracking (Proposal → Voting → Discussion → Consensus)
  - Vote audit trails with detailed logging
  - Automatic result calculation and consensus detection
- **Examples and Documentation**:
  - Code review voting scenarios
  - Architecture decision workflows
  - Content moderation processes
  - Feature prioritization examples
- **Professional Development Setup**:
  - Modern Python packaging with `pyproject.toml`
  - Comprehensive CI/CD pipeline with GitHub Actions
  - Code quality tools (Black, isort, flake8, mypy)
  - Pre-commit hooks for development workflow
  - Security scanning with bandit
  - Test coverage reporting
  - Type checking support with `py.typed` marker

### Technical Details
- **Python Support**: 3.8+
- **Dependencies**: AutoGen 0.7.1+, Pydantic 2.0+
- **Type Safety**: Full type hints and mypy support
- **Testing**: pytest with async support and coverage
- **Documentation**: Sphinx-ready with examples
- **License**: MIT

### Breaking Changes
- None (initial release)

### Security
- Added bandit security scanning
- Implemented dependency review for pull requests
- Added safety checks for known vulnerabilities

### Performance
- Optimized for AutoGen 0.7.1 architecture
- Efficient state management with minimal overhead
- Async-first design for scalable multi-agent workflows

---

## Release Notes

### v0.1.0 - Initial Release

This is the initial release of the AutoGen Voting Extension, providing democratic consensus capabilities for Microsoft AutoGen multi-agent systems. The extension has been designed as a standalone package following Microsoft's recommendation for community extensions.

**Key Features:**
- 🗳️ Five different voting methods for various decision-making scenarios
- ⚙️ Highly configurable with enterprise-grade flexibility
- 📨 Rich message types for transparent voting processes
- 🔄 Robust state management with audit trails
- 🎯 Real-world examples for immediate implementation

**Use Cases:**
- Code review approval workflows
- Architecture decision processes
- Content moderation systems
- Feature prioritization meetings
- Any scenario requiring group consensus

For detailed usage instructions, see the [README.md](README.md) and check out the examples in the `/examples` directory.

**Migration from AutoGen PR:**
This extension was originally designed as a contribution to the Microsoft AutoGen repository but has been converted to a standalone extension package per maintainer guidance. All functionality remains intact with improved packaging and distribution.

---

[Unreleased]: https://github.com/your-username/autogen-voting-extension/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/autogen-voting-extension/releases/tag/v0.1.0