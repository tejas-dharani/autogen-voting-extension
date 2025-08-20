"""
Base Voting System Implementation

Core implementation of the democratic voting system for AI agent teams.
Refactored from voting_group_chat.py following Microsoft Research standards.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, List, Optional, cast

from autogen_agentchat.base import ChatAgent, Team, TerminationCondition
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MessageFactory,
    StructuredMessage,
    TextMessage,
)
from autogen_agentchat.state import BaseGroupChatManagerState
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.teams._group_chat._base_group_chat_manager import BaseGroupChatManager
from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from autogen_core import AgentRuntime, Component, ComponentModel
from pydantic import BaseModel, Field
from typing_extensions import Self

from .voting_protocols import (
    VotingMethod, VoteType, VotingPhase, VoteContent, 
    ProposalContent, VotingResult
)
from .voting_manager import CoreVotingManager

logger = logging.getLogger(__name__)


class VoteMessage(StructuredMessage[VoteContent]):
    """Message containing a vote from an agent."""

    content: VoteContent

    def to_model_text(self) -> str:
        text = f"Vote: {self.content.vote.value}"
        if self.content.reasoning:
            text += f" - Reasoning: {self.content.reasoning}"
        if self.content.confidence < 1.0:
            text += f" (Confidence: {self.content.confidence:.2f})"
        return text


class ProposalMessage(StructuredMessage[ProposalContent]):
    """Message containing a proposal for voting."""

    content: ProposalContent

    def to_model_text(self) -> str:
        text = f"Proposal: {self.content.title}\n{self.content.description}"
        if self.content.options:
            text += f"\nOptions: {', '.join(self.content.options)}"
        return text


class VotingResultMessage(StructuredMessage[VotingResult]):
    """Message containing voting results."""

    content: VotingResult

    def to_model_text(self) -> str:
        result = self.content
        text = f"Voting Result: {result.result.upper()}\n"
        text += f"Participation: {result.participation_rate:.1%} ({result.total_voters} voters)\n"
        text += f"Average Confidence: {result.confidence_average:.2f}\n"

        for vote_type, count in result.votes_summary.items():
            text += f"{vote_type}: {count} votes\n"

        if result.winning_option:
            text += f"Winning Option: {result.winning_option}"

        return text


class VotingGroupChatConfiguration(BaseModel):
    """Configuration model for the base voting group chat."""

    participants: List[ComponentModel]
    termination_condition: Optional[ComponentModel] = None
    max_turns: Optional[int] = None
    voting_method: VotingMethod = VotingMethod.MAJORITY
    qualified_majority_threshold: float = Field(default=0.67, ge=0.5, le=1.0)
    allow_abstentions: bool = True
    require_reasoning: bool = False
    max_discussion_rounds: int = 3
    auto_propose_speaker: Optional[str] = None
    emit_team_events: bool = False
    enable_audit_logging: bool = True
    enable_file_logging: bool = False


class BaseVotingGroupChat(BaseGroupChat, Component[VotingGroupChatConfiguration]):
    """
    Base voting group chat implementation for democratic consensus building.
    
    This is a foundational class that provides core voting functionality.
    Enhanced versions with adaptive consensus and semantic parsing are available
    in the system module.
    
    Examples:
        Basic voting for simple decisions:
        
        ```python
        import asyncio
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_agentchat.agents import AssistantAgent
        from votingai.core import BaseVotingGroupChat, VotingMethod
        from autogen_agentchat.conditions import MaxMessageTermination

        async def main():
            model_client = OpenAIChatCompletionClient(model="gpt-4o")

            # Create participants
            reviewer1 = AssistantAgent("Reviewer1", model_client)
            reviewer2 = AssistantAgent("Reviewer2", model_client)
            reviewer3 = AssistantAgent("Reviewer3", model_client)

            # Create voting system
            voting_chat = BaseVotingGroupChat(
                participants=[reviewer1, reviewer2, reviewer3],
                voting_method=VotingMethod.MAJORITY,
                termination_condition=MaxMessageTermination(15),
            )

            result = await voting_chat.run(task="Approve the proposed changes")
            print(result)

        asyncio.run(main())
        ```
    """

    component_config_schema = VotingGroupChatConfiguration
    component_provider_override = "autogen_agentchat.teams.BaseVotingGroupChat"

    def __init__(
        self,
        participants: List[ChatAgent],
        voting_method: VotingMethod = VotingMethod.MAJORITY,
        qualified_majority_threshold: float = 0.67,
        allow_abstentions: bool = True,
        require_reasoning: bool = False,
        max_discussion_rounds: int = 3,
        auto_propose_speaker: Optional[str] = None,
        termination_condition: Optional[TerminationCondition] = None,
        max_turns: Optional[int] = None,
        runtime: Optional[AgentRuntime] = None,
        custom_message_types: Optional[List[type[BaseAgentEvent | BaseChatMessage]]] = None,
        emit_team_events: bool = False,
        enable_audit_logging: bool = True,
        enable_file_logging: bool = False,
    ) -> None:
        # Validate participants
        if len(participants) < 2:
            raise ValueError("Voting requires at least 2 participants.")

        if auto_propose_speaker and auto_propose_speaker not in [p.name for p in participants]:
            raise ValueError(f"auto_propose_speaker '{auto_propose_speaker}' not found in participants.")

        if not (0.5 <= qualified_majority_threshold <= 1.0):
            raise ValueError("qualified_majority_threshold must be between 0.5 and 1.0")

        # Add voting message types to custom types
        voting_message_types: List[type[BaseAgentEvent | BaseChatMessage]] = [
            VoteMessage,
            ProposalMessage,
            VotingResultMessage,
        ]
        if custom_message_types:
            custom_message_types.extend(voting_message_types)
        else:
            custom_message_types = voting_message_types

        super().__init__(
            name="BaseVotingGroupChat",
            description="Base group chat team for democratic consensus through voting",
            participants=cast(List[ChatAgent | Team], participants),
            group_chat_manager_name="CoreVotingManager",
            group_chat_manager_class=CoreVotingManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types,
            emit_team_events=emit_team_events,
        )

        # Store voting configuration
        self._voting_method = voting_method
        self._qualified_majority_threshold = qualified_majority_threshold
        self._allow_abstentions = allow_abstentions
        self._require_reasoning = require_reasoning
        self._max_discussion_rounds = max_discussion_rounds
        self._auto_propose_speaker = auto_propose_speaker
        self._enable_audit_logging = enable_audit_logging
        self._enable_file_logging = enable_file_logging

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: Optional[TerminationCondition],
        max_turns: Optional[int],
        message_factory: MessageFactory,
    ) -> Callable[[], CoreVotingManager]:
        def _factory() -> CoreVotingManager:
            return CoreVotingManager(
                name=name,
                group_topic_type=group_topic_type,
                output_topic_type=output_topic_type,
                participant_topic_types=participant_topic_types,
                participant_names=participant_names,
                participant_descriptions=participant_descriptions,
                output_message_queue=output_message_queue,
                termination_condition=termination_condition,
                max_turns=max_turns,
                message_factory=message_factory,
                voting_method=self._voting_method,
                qualified_majority_threshold=self._qualified_majority_threshold,
                allow_abstentions=self._allow_abstentions,
                require_reasoning=self._require_reasoning,
                max_discussion_rounds=self._max_discussion_rounds,
                auto_propose_speaker=self._auto_propose_speaker,
                emit_team_events=self._emit_team_events,
                enable_audit_logging=self._enable_audit_logging,
                enable_file_logging=self._enable_file_logging,
            )

        return _factory

    def _to_config(self) -> VotingGroupChatConfiguration:
        """Convert to configuration object."""
        return VotingGroupChatConfiguration(
            participants=[participant.dump_component() for participant in self._participants],
            termination_condition=self._termination_condition.dump_component() if self._termination_condition else None,
            max_turns=self._max_turns,
            voting_method=self._voting_method,
            qualified_majority_threshold=self._qualified_majority_threshold,
            allow_abstentions=self._allow_abstentions,
            require_reasoning=self._require_reasoning,
            max_discussion_rounds=self._max_discussion_rounds,
            auto_propose_speaker=self._auto_propose_speaker,
            emit_team_events=self._emit_team_events,
            enable_audit_logging=self._enable_audit_logging,
            enable_file_logging=self._enable_file_logging,
        )

    @classmethod
    def _from_config(cls, config: VotingGroupChatConfiguration) -> Self:
        """Create from configuration object."""
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )

        return cls(
            participants=participants,
            voting_method=config.voting_method,
            qualified_majority_threshold=config.qualified_majority_threshold,
            allow_abstentions=config.allow_abstentions,
            require_reasoning=config.require_reasoning,
            max_discussion_rounds=config.max_discussion_rounds,
            auto_propose_speaker=config.auto_propose_speaker,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
            emit_team_events=config.emit_team_events,
            enable_audit_logging=config.enable_audit_logging,
            enable_file_logging=config.enable_file_logging,
        )