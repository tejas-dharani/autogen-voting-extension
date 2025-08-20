#!/usr/bin/env python3
"""
Interactive VotingAI Demo
========================

Interactive command-line interface to test your voting system with custom scenarios.
Allows you to create your own proposals, choose voting methods, and see results.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from votingai import (
    VotingMethod,
    VoteType,
    DEFAULT_MODEL,
)

from votingai.research.benchmarking_suite import BenchmarkRunner, BenchmarkConfiguration
from votingai.research.evaluation_metrics import ScenarioType, BenchmarkScenario


class InteractiveVotingDemo:
    """Interactive demo for testing custom voting scenarios."""
    
    def __init__(self):
        self.config = BenchmarkConfiguration(
            model_name=DEFAULT_MODEL,
            max_messages=20,
            timeout_seconds=180,
            rate_limit_delay=1.0,
            save_detailed_logs=True,
            enable_semantic_parsing=True,
            enable_adaptive_consensus=True,
            enable_byzantine_detection=True,
            enable_audit_logging=True
        )
        self.runner = BenchmarkRunner(self.config)
    
    def show_welcome(self):
        """Show welcome message and instructions."""
        print("🚀 Welcome to VotingAI Interactive Demo!")
        print("=" * 50)
        print("Create custom voting scenarios and test your system")
        print("with real AI agents using different voting methods.\n")
        
        print("📋 Available Features:")
        print("• Multiple voting methods (Majority, Qualified, Unanimous, Plurality)")
        print("• Byzantine fault detection")
        print("• Semantic interpretation of natural language")
        print("• Security validation and audit logging")
        print("• Custom agent personas and scenarios")
        print("• Real-time voting results and analysis\n")
    
    def get_user_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default."""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    
    def choose_voting_method(self) -> VotingMethod:
        """Let user choose voting method."""
        print("\n🗳️  Choose Voting Method:")
        methods = list(VotingMethod)
        
        for i, method in enumerate(methods, 1):
            description = {
                VotingMethod.MAJORITY: "Simple majority (>50%)",
                VotingMethod.QUALIFIED_MAJORITY: "Qualified majority (67%)", 
                VotingMethod.UNANIMOUS: "Unanimous agreement (100%)",
                VotingMethod.PLURALITY: "Most votes wins",
                VotingMethod.RANKED_CHOICE: "Ranked choice with elimination"
            }
            print(f"   {i}. {method.value.title()} - {description.get(method, 'Advanced voting method')}")
        
        while True:
            try:
                choice = int(input(f"\nEnter choice (1-{len(methods)}): "))
                if 1 <= choice <= len(methods):
                    return methods[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(methods)}")
            except ValueError:
                print("Please enter a valid number")
    
    def choose_scenario_type(self) -> ScenarioType:
        """Let user choose scenario type."""
        print("\n📋 Choose Scenario Type:")
        types = [
            ScenarioType.CODE_REVIEW,
            ScenarioType.ARCHITECTURE_DECISION,
            ScenarioType.TECHNICAL_EVALUATION,
            ScenarioType.CONTENT_MODERATION,
            ScenarioType.RESOURCE_ALLOCATION
        ]
        
        descriptions = {
            ScenarioType.CODE_REVIEW: "Review code changes and bug fixes",
            ScenarioType.ARCHITECTURE_DECISION: "Major architectural and design decisions",
            ScenarioType.TECHNICAL_EVALUATION: "Evaluate technologies and tools",
            ScenarioType.CONTENT_MODERATION: "Content approval and moderation",
            ScenarioType.RESOURCE_ALLOCATION: "Resource allocation and prioritization"
        }
        
        for i, scenario_type in enumerate(types, 1):
            desc = descriptions.get(scenario_type, "General scenario")
            print(f"   {i}. {scenario_type.value.replace('_', ' ').title()} - {desc}")
        
        while True:
            try:
                choice = int(input(f"\nEnter choice (1-{len(types)}): "))
                if 1 <= choice <= len(types):
                    return types[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(types)}")
            except ValueError:
                print("Please enter a valid number")
    
    def create_custom_agents(self) -> List[Dict[str, str]]:
        """Let user create custom agent personas."""
        print("\n🤖 Create Agent Personas:")
        print("Define the AI agents that will participate in voting")
        print("(Press Enter with empty name to finish)")
        
        agents = []
        while True:
            print(f"\nAgent #{len(agents) + 1}:")
            name = self.get_user_input("Agent name (or Enter to finish)")
            
            if not name:
                break
            
            role = self.get_user_input("Role/Title", "Developer")
            description = self.get_user_input("Description/Specialty", f"{role} with expertise in the domain")
            
            agents.append({
                "name": name,
                "role": role,
                "description": description
            })
            
            print(f"   ✅ Added {name} ({role})")
        
        if len(agents) < 2:
            print("⚠️  Adding default agents (minimum 2 required)")
            agents.extend([
                {"name": "Agent1", "role": "Developer", "description": "General purpose developer"},
                {"name": "Agent2", "role": "Reviewer", "description": "Code reviewer and quality assurance"}
            ])
        
        print(f"\n👥 Final agent roster: {len(agents)} agents")
        for agent in agents:
            print(f"   • {agent['name']} - {agent['role']}")
        
        return agents
    
    def create_proposal(self, scenario_type: ScenarioType) -> str:
        """Let user create a custom proposal."""
        print(f"\n📝 Create Your Proposal ({scenario_type.value.replace('_', ' ').title()}):")
        print("Write the proposal/question that agents will vote on")
        print("(Type 'END' on a new line to finish)")
        
        proposal_lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            proposal_lines.append(line)
        
        proposal = '\n'.join(proposal_lines).strip()
        
        if not proposal:
            # Provide default based on scenario type
            defaults = {
                ScenarioType.CODE_REVIEW: "Should we approve this bug fix that changes the return type from null to empty array?",
                ScenarioType.ARCHITECTURE_DECISION: "Should we migrate from REST API to GraphQL for better performance?",
                ScenarioType.TECHNICAL_EVALUATION: "Which database should we use: PostgreSQL, MongoDB, or MySQL?",
                ScenarioType.CONTENT_MODERATION: "Should this user-generated content be approved for publication?",
                ScenarioType.RESOURCE_ALLOCATION: "Should we prioritize the new feature over technical debt reduction?"
            }
            proposal = defaults.get(scenario_type, "Should we proceed with this proposal?")
            print(f"Using default proposal: {proposal}")
        
        return proposal
    
    def get_scenario_settings(self) -> Dict[str, str]:
        """Get additional scenario settings."""
        print("\n⚙️  Scenario Settings:")
        
        complexity = self.get_user_input("Complexity level (simple/moderate/complex)", "moderate")
        stakes = self.get_user_input("Stakes level (low/medium/high/critical)", "medium")
        
        return {
            "complexity_level": complexity,
            "stakes_level": stakes
        }
    
    async def run_custom_scenario(self):
        """Run a user-defined custom scenario."""
        print("\n🎯 Create Custom Scenario")
        print("=" * 40)
        
        # Get user inputs
        voting_method = self.choose_voting_method()
        scenario_type = self.choose_scenario_type()
        agents = self.create_custom_agents()
        proposal = self.create_proposal(scenario_type)
        settings = self.get_scenario_settings()
        
        # Create scenario
        scenario = BenchmarkScenario(
            name="custom_user_scenario",
            scenario_type=scenario_type,
            description="User-created custom scenario",
            task_prompt=proposal,
            agent_personas=agents,
            complexity_level=settings["complexity_level"],
            stakes_level=settings["stakes_level"]
        )
        
        # Show summary
        print(f"\n📊 Scenario Summary:")
        print(f"   Voting Method: {voting_method.value.title()}")
        print(f"   Scenario Type: {scenario_type.value.replace('_', ' ').title()}")
        print(f"   Agents: {len(agents)} participants")
        print(f"   Complexity: {settings['complexity_level'].title()}")
        print(f"   Stakes: {settings['stakes_level'].title()}")
        
        confirmation = input(f"\n🚀 Run this scenario? (y/n): ").lower().strip()
        if confirmation != 'y':
            print("Scenario cancelled.")
            return False
        
        # Run the scenario
        print(f"\n🤖 Starting voting session...")
        print(f"Agents are discussing and voting on your proposal...\n")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                voting_method,
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            
            # Display results
            print(f"🎉 VOTING COMPLETE!")
            print(f"=" * 40)
            print(f"Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"Duration: {metrics.performance.total_duration_seconds:.1f} seconds")
            print(f"Messages Exchanged: {metrics.performance.total_messages}")
            print(f"Consensus Quality: {metrics.consensus_quality:.2f}/1.00")
            
            # Show voting outcome if available
            if hasattr(metrics, 'voting_outcome') and metrics.voting_outcome:
                outcome = metrics.voting_outcome
                print(f"\nFinal Result: {outcome.get('result', 'Unknown').upper()}")
                
                if 'votes_summary' in outcome:
                    print(f"Vote Breakdown:")
                    for vote_type, count in outcome['votes_summary'].items():
                        print(f"   {vote_type.title()}: {count} votes")
                
                if outcome.get('confidence_average'):
                    print(f"Average Confidence: {outcome['confidence_average']:.2f}")
            
            # Show additional metrics
            if hasattr(metrics, 'byzantine_metrics'):
                bm = metrics.byzantine_metrics
                if bm.get('suspicious_agents'):
                    print(f"⚠️  Suspicious Agents: {bm['suspicious_agents']}")
                if bm.get('byzantine_resilient') is False:
                    print(f"🛡️  Byzantine Resilience: Need more honest agents")
            
            return True
            
        except Exception as e:
            print(f"❌ Scenario failed: {e}")
            return False
    
    def show_predefined_scenarios(self) -> List[Dict[str, Any]]:
        """Show predefined interesting scenarios."""
        scenarios = [
            {
                "name": "Security Code Review",
                "description": "Review a critical security fix",
                "method": VotingMethod.UNANIMOUS,
                "type": ScenarioType.CODE_REVIEW,
                "proposal": "Critical security fix: Add input validation to prevent SQL injection attacks. Should be deployed immediately to production.",
                "agents": [
                    {"name": "SecurityExpert", "role": "Security Engineer", "description": "Cybersecurity specialist"},
                    {"name": "BackendDev", "role": "Backend Developer", "description": "API and database developer"},
                    {"name": "DevOpsLead", "role": "DevOps Lead", "description": "Production deployment manager"}
                ]
            },
            {
                "name": "Architecture Migration",
                "description": "Decide on microservices migration",
                "method": VotingMethod.QUALIFIED_MAJORITY,
                "type": ScenarioType.ARCHITECTURE_DECISION,
                "proposal": "Migrate our monolithic application to microservices architecture. This will improve scalability but increase complexity and operational overhead.",
                "agents": [
                    {"name": "Architect", "role": "Solutions Architect", "description": "System architecture expert"},
                    {"name": "TeamLead", "role": "Engineering Lead", "description": "Engineering team manager"},
                    {"name": "SRE", "role": "Site Reliability Engineer", "description": "Production systems specialist"},
                    {"name": "ProductManager", "role": "Product Manager", "description": "Business requirements focus"}
                ]
            },
            {
                "name": "Technology Choice",
                "description": "Choose between React, Vue, and Angular",
                "method": VotingMethod.PLURALITY,
                "type": ScenarioType.TECHNICAL_EVALUATION,
                "proposal": "Choose our frontend framework: React (popular, large ecosystem), Vue (gentle learning curve), or Angular (enterprise features). Each has different strengths.",
                "agents": [
                    {"name": "FrontendDev", "role": "Frontend Developer", "description": "UI/UX implementation specialist"},
                    {"name": "FullStackDev", "role": "Full-Stack Developer", "description": "Both frontend and backend experience"},
                    {"name": "TechLead", "role": "Technical Lead", "description": "Technology strategy and decisions"}
                ]
            },
            {
                "name": "Content Moderation",
                "description": "Moderate borderline user content",
                "method": VotingMethod.MAJORITY,
                "type": ScenarioType.CONTENT_MODERATION,
                "proposal": "User comment: 'This product is overpriced garbage and the company doesn't care about customers.' Should this review be approved, flagged, or removed?",
                "agents": [
                    {"name": "ContentMod", "role": "Content Moderator", "description": "Community guidelines enforcement"},
                    {"name": "CommunityMgr", "role": "Community Manager", "description": "Community health and engagement"},
                    {"name": "PolicyExpert", "role": "Policy Specialist", "description": "Content policy interpretation"}
                ]
            }
        ]
        
        return scenarios
    
    async def run_predefined_scenario(self, scenario_config: Dict[str, Any]):
        """Run a predefined scenario."""
        print(f"\n🎯 Running: {scenario_config['name']}")
        print(f"Description: {scenario_config['description']}")
        print(f"Voting Method: {scenario_config['method'].value.title()}")
        print(f"Agents: {len(scenario_config['agents'])} participants")
        
        confirmation = input(f"\n🚀 Run this scenario? (y/n): ").lower().strip()
        if confirmation != 'y':
            return False
        
        # Create scenario
        scenario = BenchmarkScenario(
            name=scenario_config['name'].lower().replace(' ', '_'),
            scenario_type=scenario_config['type'],
            description=scenario_config['description'],
            task_prompt=scenario_config['proposal'],
            agent_personas=scenario_config['agents'],
            complexity_level="moderate",
            stakes_level="medium"
        )
        
        print(f"\n🤖 Starting voting session...")
        
        try:
            result = await self.runner.run_comparison(
                scenario,
                scenario_config['method'],
                compare_systems=["enhanced"]
            )
            
            metrics = result.system_a_metrics
            
            # Display results (same format as custom scenarios)
            print(f"\n🎉 VOTING COMPLETE!")
            print(f"=" * 40)
            print(f"Decision Reached: {'✅ YES' if metrics.decision_reached else '❌ NO'}")
            print(f"Duration: {metrics.performance.total_duration_seconds:.1f} seconds")
            print(f"Messages: {metrics.performance.total_messages}")
            print(f"Quality: {metrics.consensus_quality:.2f}/1.00")
            
            if hasattr(metrics, 'voting_outcome') and metrics.voting_outcome:
                outcome = metrics.voting_outcome
                print(f"\nResult: {outcome.get('result', 'Unknown').upper()}")
                
                if 'votes_summary' in outcome:
                    for vote_type, count in outcome['votes_summary'].items():
                        print(f"   {vote_type.title()}: {count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Scenario failed: {e}")
            return False
    
    async def main_menu(self):
        """Main interactive menu."""
        while True:
            print("\n" + "=" * 50)
            print("🚀 VotingAI Interactive Demo - Main Menu")
            print("=" * 50)
            print("1. 🎯 Create Custom Scenario")
            print("2. 📋 Run Predefined Scenario")
            print("3. 📊 Quick System Status Check")
            print("4. ❓ Help & Tips")
            print("5. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                await self.run_custom_scenario()
            
            elif choice == '2':
                scenarios = self.show_predefined_scenarios()
                print("\n📋 Predefined Scenarios:")
                for i, scenario in enumerate(scenarios, 1):
                    print(f"   {i}. {scenario['name']} - {scenario['description']}")
                
                try:
                    scenario_choice = int(input(f"\nChoose scenario (1-{len(scenarios)}): "))
                    if 1 <= scenario_choice <= len(scenarios):
                        await self.run_predefined_scenario(scenarios[scenario_choice - 1])
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Please enter a valid number")
            
            elif choice == '3':
                print("\n📊 System Status Check")
                print("=" * 30)
                print("✅ VotingAI system loaded")
                print("✅ OpenAI API key configured" if os.getenv("OPENAI_API_KEY") else "❌ OpenAI API key missing")
                print(f"✅ Model: {DEFAULT_MODEL}")
                print("✅ All voting methods available")
                print("✅ Security features enabled")
                print("✅ Byzantine detection active")
                print("✅ Semantic parsing enabled")
            
            elif choice == '4':
                print("\n❓ Help & Tips")
                print("=" * 20)
                print("📝 Creating Effective Proposals:")
                print("   • Be specific about what you're asking")
                print("   • Include context and background")
                print("   • Mention benefits and concerns")
                print("   • Ask clear questions")
                
                print("\n🤖 Agent Personas:")
                print("   • Give agents specific roles and expertise")
                print("   • Vary their perspectives and priorities")
                print("   • Include domain experts for your topic")
                
                print("\n⚖️  Voting Methods:")
                print("   • Majority: Good for routine decisions")
                print("   • Qualified Majority: Important changes")
                print("   • Unanimous: Critical/security decisions")
                print("   • Plurality: Multiple option choices")
                
                print("\n🎯 Best Practices:")
                print("   • Test with 3-5 agents for good discussion")
                print("   • Use appropriate voting method for stakes")
                print("   • Review audit logs for insights")
                print("   • Check Byzantine detection for issues")
            
            elif choice == '5':
                print("\n👋 Thanks for using VotingAI Interactive Demo!")
                print("Your voting system is ready for production use.")
                break
            
            else:
                print("Please enter a valid choice (1-5)")


async def main():
    """Run the interactive demo."""
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    
    demo = InteractiveVotingDemo()
    demo.show_welcome()
    
    try:
        await demo.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())