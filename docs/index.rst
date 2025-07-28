AutoGen Voting Extension Documentation
======================================

.. image:: https://badge.fury.io/py/autogen-voting-extension.svg
   :target: https://badge.fury.io/py/autogen-voting-extension
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A powerful extension for Microsoft AutoGen that enables **democratic consensus** in multi-agent systems through configurable voting mechanisms. Perfect for code reviews, architecture decisions, content moderation, and any scenario requiring transparent group decision-making.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install AutoGen dependencies
   pip install autogen-agentchat autogen-ext[openai]

   # Install the voting extension
   pip install autogen-voting-extension

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from autogen_ext.models.openai import OpenAIChatCompletionClient
   from autogen_agentchat.agents import AssistantAgent
   from autogen_agentchat.conditions import MaxMessageTermination

   from autogen_voting import VotingGroupChat, VotingMethod

   async def main():
       model_client = OpenAIChatCompletionClient(model="gpt-4o")
       
       # Create voting agents
       agents = [
           AssistantAgent("Alice", model_client, system_message="Expert in backend systems"),
           AssistantAgent("Bob", model_client, system_message="Frontend specialist"), 
           AssistantAgent("Carol", model_client, system_message="Security expert")
       ]
       
       # Create voting team
       voting_team = VotingGroupChat(
           participants=agents,
           voting_method=VotingMethod.MAJORITY,
           require_reasoning=True,
           max_discussion_rounds=2,
           termination_condition=MaxMessageTermination(20)
       )
       
       # Run voting process
       result = await voting_team.run(task="""
           Proposal: Should we migrate our API from REST to GraphQL?
           
           Please vote APPROVE or REJECT with detailed reasoning.
       """)
       
       print(f"Decision: {result}")

   asyncio.run(main())

Features
--------

🗳️ **Multiple Voting Methods**
   - Majority voting (>50% approval)
   - Plurality voting (most votes wins)
   - Unanimous consensus (all voters must agree)
   - Qualified majority (configurable threshold)
   - Ranked choice voting (with preference elimination)

⚙️ **Advanced Configuration**
   - Configurable voting thresholds
   - Discussion rounds before final decisions
   - Abstention support with flexible participation rules
   - Reasoning requirements for transparent decision-making
   - Confidence scoring for vote quality assessment
   - Auto-proposer selection for structured workflows

📨 **Rich Message Types**
   - ``ProposalMessage`` - Structured proposals with options and metadata
   - ``VoteMessage`` - Votes with reasoning, confidence scores, and ranked choices
   - ``VotingResultMessage`` - Comprehensive results with participation analytics

🔄 **State Management**
   - Persistent voting state across conversations
   - Phase tracking (Proposal → Voting → Discussion → Consensus)
   - Vote audit trails with detailed logging
   - Automatic result calculation and consensus detection

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   voting_methods
   configuration
   examples
   best_practices

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/voting_group_chat
   api/message_types
   api/enums

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/code_review
   examples/architecture_decisions
   examples/content_moderation
   examples/feature_prioritization

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Use Cases
---------

**Code Review Voting** 👨‍💻
   Perfect for collaborative code reviews with multiple reviewers using qualified majority voting.

**Architecture Decisions** 🏗️
   Use ranked choice voting for complex architectural decisions with multiple options.

**Content Moderation** 🛡️
   Majority voting for content approval/rejection with abstention support.

**Feature Prioritization** 📈
   Unanimous consensus for high-stakes feature prioritization decisions.

Links
-----

* **GitHub Repository**: https://github.com/your-username/autogen-voting-extension
* **PyPI Package**: https://pypi.org/project/autogen-voting-extension/
* **AutoGen Documentation**: https://microsoft.github.io/autogen/
* **Issues & Support**: https://github.com/your-username/autogen-voting-extension/issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`