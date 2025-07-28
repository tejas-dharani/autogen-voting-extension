"""Analysis and visualization tools for benchmark results."""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """Load all result files matching the pattern."""
        results = []
        
        for file_path in self.results_dir.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Add filename for reference
                    data['source_file'] = file_path.name
                    results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def create_efficiency_comparison_chart(self, results: List[Dict[str, Any]], save_path: str = None):
        """Create efficiency comparison charts."""
        
        # Extract efficiency data
        scenarios = []
        time_ratios = []
        message_ratios = []
        token_ratios = []
        voting_methods = []
        
        for result in results:
            if 'efficiency_comparison' in result:
                scenarios.append(result.get('scenario_name', 'Unknown'))
                time_ratios.append(result['efficiency_comparison']['time_ratio'])
                message_ratios.append(result['efficiency_comparison']['message_ratio'])
                token_ratios.append(result['efficiency_comparison']['token_ratio'])
                
                # Extract voting method from filename or data
                method = 'majority'  # default
                if 'qualified_majority' in result.get('source_file', ''):
                    method = 'qualified_majority'
                elif 'unanimous' in result.get('source_file', ''):
                    method = 'unanimous'
                voting_methods.append(method)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Scenario': scenarios,
            'Time Ratio': time_ratios,
            'Message Ratio': message_ratios,
            'Token Ratio': token_ratios,
            'Voting Method': voting_methods
        })
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Voting vs Standard GroupChat Efficiency Comparison', fontsize=16)
        
        # Time ratio comparison
        sns.barplot(data=df, x='Scenario', y='Time Ratio', hue='Voting Method', ax=axes[0,0])
        axes[0,0].set_title('Time Efficiency (Voting/Standard)')
        axes[0,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend()
        
        # Message ratio comparison
        sns.barplot(data=df, x='Scenario', y='Message Ratio', hue='Voting Method', ax=axes[0,1])
        axes[0,1].set_title('Message Efficiency (Voting/Standard)')
        axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        
        # Token ratio comparison
        sns.barplot(data=df, x='Scenario', y='Token Ratio', hue='Voting Method', ax=axes[1,0])
        axes[1,0].set_title('Token Efficiency (Voting/Standard)')
        axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        
        # Combined efficiency scatter
        axes[1,1].scatter(df['Time Ratio'], df['Token Ratio'], 
                         c=[{'majority': 0, 'qualified_majority': 1, 'unanimous': 2}[m] for m in df['Voting Method']],
                         alpha=0.7, s=100)
        axes[1,1].set_xlabel('Time Ratio')
        axes[1,1].set_ylabel('Token Ratio')
        axes[1,1].set_title('Time vs Token Efficiency')
        axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[1,1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Efficiency comparison chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def create_voting_method_comparison(self, results: List[Dict[str, Any]], save_path: str = None):
        """Compare performance across different voting methods."""
        
        # Group results by voting method
        method_data = {'majority': [], 'qualified_majority': [], 'unanimous': []}
        
        for result in results:
            method = 'majority'  # default
            if 'qualified_majority' in result.get('source_file', ''):
                method = 'qualified_majority'
            elif 'unanimous' in result.get('source_file', ''):
                method = 'unanimous'
            
            if 'voting_metrics' in result:
                method_data[method].append({
                    'duration': result['voting_metrics']['duration_seconds'],
                    'messages': result['voting_metrics']['total_messages'],
                    'decision_reached': result['voting_metrics']['decision_reached'],
                    'scenario': result.get('scenario_name', 'Unknown')
                })
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Voting Method Performance Comparison', fontsize=16)
        
        # Duration comparison
        durations = []
        methods = []
        for method, data in method_data.items():
            for item in data:
                durations.append(item['duration'])
                methods.append(method.replace('_', ' ').title())
        
        sns.boxplot(x=methods, y=durations, ax=axes[0,0])
        axes[0,0].set_title('Duration Distribution by Voting Method')
        axes[0,0].set_ylabel('Duration (seconds)')
        
        # Message count comparison
        messages = []
        methods_msg = []
        for method, data in method_data.items():
            for item in data:
                messages.append(item['messages'])
                methods_msg.append(method.replace('_', ' ').title())
        
        sns.boxplot(x=methods_msg, y=messages, ax=axes[0,1])
        axes[0,1].set_title('Message Count Distribution by Voting Method')
        axes[0,1].set_ylabel('Message Count')
        
        # Success rate comparison
        success_rates = {}
        for method, data in method_data.items():
            if data:
                success_rate = sum(1 for item in data if item['decision_reached']) / len(data)
                success_rates[method.replace('_', ' ').title()] = success_rate
        
        axes[1,0].bar(success_rates.keys(), success_rates.values())
        axes[1,0].set_title('Decision Success Rate by Voting Method')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].set_ylim(0, 1.1)
        
        # Average performance radar
        methods_clean = list(success_rates.keys())
        avg_durations = []
        avg_messages = []
        
        for method in ['majority', 'qualified_majority', 'unanimous']:
            if method_data[method]:
                avg_durations.append(np.mean([item['duration'] for item in method_data[method]]))
                avg_messages.append(np.mean([item['messages'] for item in method_data[method]]))
            else:
                avg_durations.append(0)
                avg_messages.append(0)
        
        x = np.arange(len(methods_clean))
        width = 0.35
        
        axes[1,1].bar(x - width/2, avg_durations, width, label='Avg Duration', alpha=0.7)
        axes[1,1].bar(x + width/2, avg_messages, width, label='Avg Messages', alpha=0.7)
        axes[1,1].set_title('Average Performance Metrics')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(methods_clean)
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Voting method comparison chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def create_scenario_analysis(self, results: List[Dict[str, Any]], save_path: str = None):
        """Analyze performance across different scenario types."""
        
        # Group by scenario type
        scenario_types = {}
        
        for result in results:
            scenario_name = result.get('scenario_name', 'Unknown')
            scenario_type = scenario_name.split('_')[0]  # Simple categorization
            
            if scenario_type not in scenario_types:
                scenario_types[scenario_type] = []
            
            if 'efficiency_comparison' in result:
                scenario_types[scenario_type].append({
                    'time_ratio': result['efficiency_comparison']['time_ratio'],
                    'message_ratio': result['efficiency_comparison']['message_ratio'],
                    'token_ratio': result['efficiency_comparison']['token_ratio'],
                    'voting_success': result['voting_metrics']['decision_reached'],
                    'standard_success': result['standard_metrics']['decision_reached']
                })
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Analysis by Scenario Type', fontsize=16)
        
        # Time ratio by scenario type
        time_data = []
        scenario_labels = []
        for scenario_type, data in scenario_types.items():
            for item in data:
                time_data.append(item['time_ratio'])
                scenario_labels.append(scenario_type.title())
        
        sns.boxplot(x=scenario_labels, y=time_data, ax=axes[0,0])
        axes[0,0].set_title('Time Ratio Distribution by Scenario Type')
        axes[0,0].set_ylabel('Time Ratio (Voting/Standard)')
        axes[0,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Success rate comparison
        success_comparison = {}
        for scenario_type, data in scenario_types.items():
            if data:
                voting_success = sum(1 for item in data if item['voting_success']) / len(data)
                standard_success = sum(1 for item in data if item['standard_success']) / len(data)
                success_comparison[scenario_type.title()] = {
                    'Voting': voting_success,
                    'Standard': standard_success
                }
        
        # Convert to DataFrame for plotting
        success_df = pd.DataFrame(success_comparison).T
        success_df.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Success Rate by Scenario Type')
        axes[0,1].set_ylabel('Success Rate')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Efficiency heatmap
        efficiency_matrix = []
        scenario_names = []
        
        for scenario_type, data in scenario_types.items():
            if data:
                avg_time = np.mean([item['time_ratio'] for item in data])
                avg_message = np.mean([item['message_ratio'] for item in data])
                avg_token = np.mean([item['token_ratio'] for item in data])
                
                efficiency_matrix.append([avg_time, avg_message, avg_token])
                scenario_names.append(scenario_type.title())
        
        if efficiency_matrix:
            efficiency_df = pd.DataFrame(
                efficiency_matrix,
                index=scenario_names,
                columns=['Time Ratio', 'Message Ratio', 'Token Ratio']
            )
            
            sns.heatmap(efficiency_df, annot=True, fmt='.2f', ax=axes[1,0], cmap='RdYlBu_r')
            axes[1,0].set_title('Average Efficiency Ratios by Scenario Type')
        
        # Performance scatter
        for scenario_type, data in scenario_types.items():
            time_ratios = [item['time_ratio'] for item in data]
            token_ratios = [item['token_ratio'] for item in data]
            axes[1,1].scatter(time_ratios, token_ratios, label=scenario_type.title(), alpha=0.7, s=60)
        
        axes[1,1].set_xlabel('Time Ratio')
        axes[1,1].set_ylabel('Token Ratio')
        axes[1,1].set_title('Time vs Token Efficiency by Scenario Type')
        axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[1,1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scenario analysis chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def generate_summary_report(self, results: List[Dict[str, Any]], output_file: str = None):
        """Generate a comprehensive summary report."""
        
        if not results:
            print("No results to analyze.")
            return
        
        report = []
        report.append("# AutoGen Voting Extension Benchmark Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Comparisons: {len(results)}")
        report.append("")
        
        # Overall efficiency summary
        time_ratios = [r['efficiency_comparison']['time_ratio'] for r in results if 'efficiency_comparison' in r]
        message_ratios = [r['efficiency_comparison']['message_ratio'] for r in results if 'efficiency_comparison' in r]
        token_ratios = [r['efficiency_comparison']['token_ratio'] for r in results if 'efficiency_comparison' in r]
        
        if time_ratios:
            report.append("## Overall Efficiency Summary")
            report.append(f"- Average Time Ratio (Voting/Standard): {np.mean(time_ratios):.2f}")
            report.append(f"- Average Message Ratio: {np.mean(message_ratios):.2f}")
            report.append(f"- Average Token Ratio: {np.mean(token_ratios):.2f}")
            report.append(f"- Time ratios range: {min(time_ratios):.2f} - {max(time_ratios):.2f}")
            report.append("")
        
        # Decision quality summary
        voting_successes = sum(1 for r in results if r.get('voting_metrics', {}).get('decision_reached', False))
        standard_successes = sum(1 for r in results if r.get('standard_metrics', {}).get('decision_reached', False))
        
        report.append("## Decision Quality Summary")
        report.append(f"- Voting Success Rate: {voting_successes/len(results):.1%}")
        report.append(f"- Standard Success Rate: {standard_successes/len(results):.1%}")
        report.append("")
        
        # Key findings
        report.append("## Key Findings")
        
        avg_time_ratio = np.mean(time_ratios) if time_ratios else 1.0
        avg_message_ratio = np.mean(message_ratios) if message_ratios else 1.0
        
        if avg_time_ratio < 1.0:
            report.append("- ✅ Voting approach is generally faster than standard GroupChat")
        elif avg_time_ratio > 1.2:
            report.append("- ⚠️ Voting approach takes significantly longer than standard GroupChat")
        else:
            report.append("- ➡️ Voting and standard approaches have similar performance")
        
        if avg_message_ratio < 1.0:
            report.append("- ✅ Voting approach uses fewer messages")
        else:
            report.append("- ⚠️ Voting approach requires more messages")
        
        if voting_successes > standard_successes:
            report.append("- ✅ Voting approach has higher success rate")
        elif voting_successes == standard_successes:
            report.append("- ➡️ Both approaches have similar success rates")
        else:
            report.append("- ⚠️ Standard approach has higher success rate")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("Based on the benchmark results:")
        report.append("")
        
        if avg_time_ratio < 1.1 and voting_successes >= standard_successes:
            report.append("- **Recommended**: Use voting for decision-making scenarios")
            report.append("- Voting provides comparable or better performance with democratic consensus")
        elif avg_time_ratio > 1.5:
            report.append("- **Consider carefully**: Voting adds significant overhead")
            report.append("- Best suited for high-stakes decisions where consensus quality matters more than speed")
        else:
            report.append("- **Context-dependent**: Choose based on specific use case requirements")
            report.append("- Use voting when transparent decision-making is important")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to: {output_file}")
        
        print(report_text)
        return report_text


def main():
    """Main function to run analysis on existing results."""
    analyzer = BenchmarkAnalyzer()
    
    # Load results
    results = analyzer.load_results()
    
    if not results:
        print("No benchmark results found. Run benchmarks first with:")
        print("python run_benchmarks.py --full")
        return
    
    print(f"Loaded {len(results)} benchmark results")
    
    # Generate visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    analyzer.create_efficiency_comparison_chart(
        results, 
        save_path=f"benchmark_results/efficiency_comparison_{timestamp}.png"
    )
    
    analyzer.create_voting_method_comparison(
        results,
        save_path=f"benchmark_results/voting_method_comparison_{timestamp}.png"
    )
    
    analyzer.create_scenario_analysis(
        results,
        save_path=f"benchmark_results/scenario_analysis_{timestamp}.png"
    )
    
    # Generate summary report
    analyzer.generate_summary_report(
        results,
        output_file=f"benchmark_results/summary_report_{timestamp}.md"
    )


if __name__ == "__main__":
    main()