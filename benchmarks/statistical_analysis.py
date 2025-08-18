"""Advanced statistical analysis for voting system evaluation."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis following academic research standards."""
    
    def __init__(self, significance_level: float = 0.01):
        """Initialize with significance level (default: 0.01 for Bonferroni correction)."""
        self.significance_level = significance_level
        self.comparison_count = 0  # Track number of comparisons for Bonferroni correction
    
    def calculate_effect_sizes(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Calculate Cohen's d and other effect size measures."""
        group1_array = np.array(group1)
        group2_array = np.array(group2)
        
        # Cohen's d
        pooled_std = np.sqrt(((len(group1_array) - 1) * np.var(group1_array, ddof=1) + 
                             (len(group2_array) - 1) * np.var(group2_array, ddof=1)) / 
                            (len(group1_array) + len(group2_array) - 2))
        
        cohens_d = (np.mean(group1_array) - np.mean(group2_array)) / pooled_std if pooled_std > 0 else 0.0
        
        # Glass's delta (using group2 as control)
        glass_delta = (np.mean(group1_array) - np.mean(group2_array)) / np.std(group2_array, ddof=1) if np.std(group2_array, ddof=1) > 0 else 0.0
        
        # Hedge's g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(group1_array) + len(group2_array)) - 9))
        hedges_g = cohens_d * correction_factor
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'hedges_g': hedges_g,
            'interpretation': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def bootstrap_confidence_intervals(
        self, 
        data: List[float], 
        statistic_func: callable = np.mean, 
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence intervals."""
        data_array = np.array(data)
        
        # Use scipy's bootstrap function
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        def statistic_wrapper(x, axis):
            return statistic_func(x, axis=axis)
        
        result = bootstrap(
            (data_array,), 
            statistic_wrapper, 
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            random_state=rng
        )
        
        statistic_value = statistic_func(data_array)
        ci_lower = result.confidence_interval.low
        ci_upper = result.confidence_interval.high
        
        return statistic_value, ci_lower, ci_upper
    
    def bonferroni_correction(self, p_values: List[float]) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple comparisons."""
        p_values_array = np.array(p_values)
        n_comparisons = len(p_values)
        
        # Bonferroni corrected p-values
        corrected_p_values = p_values_array * n_comparisons
        corrected_p_values = np.minimum(corrected_p_values, 1.0)  # Cap at 1.0
        
        # Adjusted significance level
        adjusted_alpha = self.significance_level / n_comparisons
        
        # Significant results after correction
        significant_results = corrected_p_values < self.significance_level
        
        return {
            'original_p_values': p_values,
            'corrected_p_values': corrected_p_values.tolist(),
            'adjusted_alpha': adjusted_alpha,
            'significant_after_correction': significant_results.tolist(),
            'n_comparisons': n_comparisons,
            'family_wise_error_rate': self.significance_level
        }
    
    def power_analysis(
        self, 
        effect_size: float, 
        sample_size: int, 
        significance_level: float = None
    ) -> Dict[str, float]:
        """Calculate statistical power for given parameters."""
        if significance_level is None:
            significance_level = self.significance_level
        
        # Calculate power using t-test approximation
        # This is a simplified power calculation for two-sample t-test
        degrees_of_freedom = 2 * sample_size - 2
        critical_t = stats.t.ppf(1 - significance_level/2, degrees_of_freedom)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size / 2)
        
        # Power calculation
        power = 1 - stats.nct.cdf(critical_t, degrees_of_freedom, ncp) + stats.nct.cdf(-critical_t, degrees_of_freedom, ncp)
        
        return {
            'statistical_power': power,
            'effect_size': effect_size,
            'sample_size': sample_size,
            'significance_level': significance_level,
            'interpretation': 'adequate' if power >= 0.8 else 'inadequate'
        }
    
    def comprehensive_comparison(
        self, 
        voting_data: List[float], 
        standard_data: List[float],
        metric_name: str = "performance"
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical comparison between voting and standard approaches."""
        
        # Basic descriptive statistics
        voting_stats = {
            'mean': np.mean(voting_data),
            'std': np.std(voting_data, ddof=1),
            'median': np.median(voting_data),
            'n': len(voting_data)
        }
        
        standard_stats = {
            'mean': np.mean(standard_data),
            'std': np.std(standard_data, ddof=1),
            'median': np.median(standard_data),
            'n': len(standard_data)
        }
        
        # Statistical tests
        # 1. Shapiro-Wilk test for normality
        voting_normality = stats.shapiro(voting_data)
        standard_normality = stats.shapiro(standard_data)
        
        # 2. Levene's test for equal variances
        levene_test = stats.levene(voting_data, standard_data)
        
        # 3. Choose appropriate test based on assumptions
        if voting_normality.pvalue > 0.05 and standard_normality.pvalue > 0.05:
            # Data is normal, use t-test
            if levene_test.pvalue > 0.05:
                # Equal variances
                ttest_result = stats.ttest_ind(voting_data, standard_data, equal_var=True)
                test_used = "independent_t_test_equal_var"
            else:
                # Unequal variances (Welch's t-test)
                ttest_result = stats.ttest_ind(voting_data, standard_data, equal_var=False)
                test_used = "welch_t_test"
        else:
            # Data is not normal, use Mann-Whitney U test
            ttest_result = stats.mannwhitneyu(voting_data, standard_data, alternative='two-sided')
            test_used = "mann_whitney_u_test"
        
        # Effect size
        effect_sizes = self.calculate_effect_sizes(voting_data, standard_data)
        
        # Bootstrap confidence intervals
        voting_mean_ci = self.bootstrap_confidence_intervals(voting_data, np.mean)
        standard_mean_ci = self.bootstrap_confidence_intervals(standard_data, np.mean)
        
        # Difference in means with CI
        differences = [v - s for v in voting_data for s in standard_data]
        difference_ci = self.bootstrap_confidence_intervals(differences, np.mean)
        
        # Power analysis
        power_analysis = self.power_analysis(
            effect_sizes['cohens_d'], 
            min(len(voting_data), len(standard_data))
        )
        
        # Track comparison for Bonferroni correction
        self.comparison_count += 1
        
        return {
            'metric_name': metric_name,
            'voting_statistics': voting_stats,
            'standard_statistics': standard_stats,
            'statistical_test': {
                'test_used': test_used,
                'statistic': float(ttest_result.statistic),
                'p_value': float(ttest_result.pvalue),
                'significant': ttest_result.pvalue < self.significance_level
            },
            'assumption_tests': {
                'voting_normality': {
                    'statistic': float(voting_normality.statistic),
                    'p_value': float(voting_normality.pvalue),
                    'normal': voting_normality.pvalue > 0.05
                },
                'standard_normality': {
                    'statistic': float(standard_normality.statistic),
                    'p_value': float(standard_normality.pvalue),
                    'normal': standard_normality.pvalue > 0.05
                },
                'equal_variances': {
                    'statistic': float(levene_test.statistic),
                    'p_value': float(levene_test.pvalue),
                    'equal_var': levene_test.pvalue > 0.05
                }
            },
            'effect_sizes': effect_sizes,
            'confidence_intervals': {
                'voting_mean': {
                    'estimate': voting_mean_ci[0],
                    'lower': voting_mean_ci[1],
                    'upper': voting_mean_ci[2]
                },
                'standard_mean': {
                    'estimate': standard_mean_ci[0],
                    'lower': standard_mean_ci[1],
                    'upper': standard_mean_ci[2]
                },
                'difference': {
                    'estimate': difference_ci[0],
                    'lower': difference_ci[1],
                    'upper': difference_ci[2]
                }
            },
            'power_analysis': power_analysis,
            'improvement_percentage': ((voting_stats['mean'] - standard_stats['mean']) / standard_stats['mean']) * 100 if standard_stats['mean'] != 0 else 0
        }
    
    def analyze_benchmark_results(self, results_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comprehensive benchmark results with advanced statistics."""
        
        # Extract data for different metrics
        metrics_data = {
            'duration_seconds': {'voting': [], 'standard': []},
            'total_messages': {'voting': [], 'standard': []},
            'token_usage': {'voting': [], 'standard': []}
        }
        
        # Collect p-values for multiple comparison correction
        all_p_values = []
        comparisons = []
        
        # Extract data
        for result in results_data:
            if 'voting_metrics' in result and 'standard_metrics' in result:
                voting = result['voting_metrics']
                standard = result['standard_metrics']
                
                for metric in metrics_data:
                    if metric in voting and metric in standard:
                        metrics_data[metric]['voting'].append(voting[metric])
                        metrics_data[metric]['standard'].append(standard[metric])
        
        # Perform comprehensive analysis for each metric
        analysis_results = {}
        
        for metric_name, data in metrics_data.items():
            if len(data['voting']) > 0 and len(data['standard']) > 0:
                comparison_result = self.comprehensive_comparison(
                    data['voting'], 
                    data['standard'], 
                    metric_name
                )
                analysis_results[metric_name] = comparison_result
                all_p_values.append(comparison_result['statistical_test']['p_value'])
                comparisons.append(metric_name)
        
        # Apply Bonferroni correction
        if all_p_values:
            bonferroni_results = self.bonferroni_correction(all_p_values)
            
            # Update significance after correction
            for i, metric_name in enumerate(comparisons):
                analysis_results[metric_name]['bonferroni_correction'] = {
                    'corrected_p_value': bonferroni_results['corrected_p_values'][i],
                    'significant_after_correction': bonferroni_results['significant_after_correction'][i],
                    'adjusted_alpha': bonferroni_results['adjusted_alpha']
                }
        
        # Generate summary
        summary = self._generate_summary(analysis_results)
        
        return {
            'individual_analyses': analysis_results,
            'multiple_comparison_correction': bonferroni_results if all_p_values else None,
            'summary': summary,
            'methodology': {
                'significance_level': self.significance_level,
                'correction_method': 'bonferroni',
                'bootstrap_samples': 1000,
                'effect_size_measures': ['cohens_d', 'hedges_g'],
                'confidence_level': 0.95
            }
        }
    
    def _generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of statistical analysis."""
        significant_improvements = []
        large_effects = []
        
        for metric_name, result in analysis_results.items():
            # Check for significance after Bonferroni correction
            is_significant = result.get('bonferroni_correction', {}).get('significant_after_correction', False)
            
            if is_significant:
                improvement = result['improvement_percentage']
                effect_size = result['effect_sizes']['cohens_d']
                
                if improvement < 0:  # Voting is better (lower is better for time/messages)
                    significant_improvements.append({
                        'metric': metric_name,
                        'improvement_percentage': abs(improvement),
                        'effect_size': abs(effect_size),
                        'interpretation': result['effect_sizes']['interpretation']
                    })
                
                if abs(effect_size) > 0.8:  # Large effect
                    large_effects.append({
                        'metric': metric_name,
                        'effect_size': abs(effect_size),
                        'interpretation': result['effect_sizes']['interpretation']
                    })
        
        return {
            'total_comparisons': len(analysis_results),
            'significant_improvements': significant_improvements,
            'large_effects': large_effects,
            'meets_statistical_rigor': len(significant_improvements) > 0 and len(large_effects) > 0,
            'bonferroni_alpha': self.significance_level / max(len(analysis_results), 1),
            'statistical_power_adequate': all(
                result['power_analysis']['statistical_power'] >= 0.8 
                for result in analysis_results.values()
            )
        }
    
    def generate_statistical_report(self, analysis_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate comprehensive statistical report."""
        if not output_file:
            output_file = f"statistical_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'report_metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'analysis_type': 'research_grade_statistical_analysis',
                'significance_level': self.significance_level,
                'multiple_comparison_correction': 'bonferroni'
            },
            'executive_summary': analysis_results['summary'],
            'detailed_results': analysis_results,
            'statistical_validity': {
                'assumptions_tested': True,
                'effect_sizes_calculated': True,
                'confidence_intervals_provided': True,
                'multiple_comparisons_corrected': True,
                'power_analysis_performed': True,
                'bootstrap_resampling_used': True
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file