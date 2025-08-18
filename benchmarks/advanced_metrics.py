"""Advanced metrics for fairness, safety, and ethical AI evaluation in voting systems."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FairnessMetrics:
    """Comprehensive fairness metrics for voting systems."""
    
    # Demographic parity metrics
    demographic_parity: float = 0.0
    demographic_parity_difference: float = 0.0
    
    # Equalized odds metrics  
    equalized_odds_difference: float = 0.0
    true_positive_rate_difference: float = 0.0
    false_positive_rate_difference: float = 0.0
    
    # Individual fairness
    individual_fairness_score: float = 0.0
    consistency_score: float = 0.0
    
    # Representation metrics
    participation_parity: float = 0.0
    voice_equality_score: float = 0.0
    
    # Metadata
    protected_attributes: List[str] = field(default_factory=list)
    sample_size_by_group: Dict[str, int] = field(default_factory=dict)


@dataclass  
class SafetyMetrics:
    """Safety and harm detection metrics."""
    
    # Toxicity detection
    toxicity_score: float = 0.0
    severe_toxicity_score: float = 0.0
    identity_attack_score: float = 0.0
    insult_score: float = 0.0
    profanity_score: float = 0.0
    threat_score: float = 0.0
    
    # Bias amplification  
    bias_amplification_score: float = 0.0
    stereotype_reinforcement: float = 0.0
    
    # Harmful content categories
    harmful_categories: Dict[str, float] = field(default_factory=dict)
    
    # Content quality
    reasoning_quality_score: float = 0.0
    factual_accuracy_score: float = 0.0
    
    # Safety metadata
    flagged_content_count: int = 0
    total_content_evaluated: int = 0


@dataclass
class QualityMetrics:
    """Advanced quality metrics beyond basic accuracy."""
    
    # Classification metrics
    precision: float = 0.0
    recall: float = 0.0  
    f1_score: float = 0.0
    specificity: float = 0.0
    
    # Calibration metrics
    expected_calibration_error: float = 0.0  # ECE
    maximum_calibration_error: float = 0.0  # MCE
    brier_score: float = 0.0
    
    # Confidence and uncertainty
    confidence_accuracy_correlation: float = 0.0
    overconfidence_score: float = 0.0
    uncertainty_score: float = 0.0
    
    # Decision quality
    consensus_quality_score: float = 0.0
    reasoning_coherence: float = 0.0
    evidence_quality: float = 0.0


class AdvancedMetricsCalculator:
    """Calculate advanced fairness, safety, and quality metrics."""
    
    def __init__(self):
        self.protected_attributes = ["agent_type", "expertise_level", "model_version"]
        
    def calculate_fairness_metrics(
        self, 
        decisions: List[Dict[str, Any]], 
        agent_attributes: Dict[str, Dict[str, Any]]
    ) -> FairnessMetrics:
        """Calculate comprehensive fairness metrics."""
        
        # Group agents by protected attributes
        groups = self._group_by_attributes(decisions, agent_attributes)
        
        # Calculate demographic parity
        dp, dp_diff = self._calculate_demographic_parity(groups)
        
        # Calculate equalized odds
        eo_diff, tpr_diff, fpr_diff = self._calculate_equalized_odds(groups)
        
        # Calculate participation parity
        participation_parity = self._calculate_participation_parity(groups)
        
        # Calculate voice equality
        voice_equality = self._calculate_voice_equality(decisions, agent_attributes)
        
        # Calculate individual fairness
        individual_fairness = self._calculate_individual_fairness(decisions)
        
        return FairnessMetrics(
            demographic_parity=dp,
            demographic_parity_difference=dp_diff,
            equalized_odds_difference=eo_diff,
            true_positive_rate_difference=tpr_diff,
            false_positive_rate_difference=fpr_diff,
            individual_fairness_score=individual_fairness,
            participation_parity=participation_parity,
            voice_equality_score=voice_equality,
            protected_attributes=self.protected_attributes,
            sample_size_by_group={k: len(v) for k, v in groups.items()}
        )
    
    def calculate_safety_metrics(
        self, 
        content_data: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]]
    ) -> SafetyMetrics:
        """Calculate safety and harm detection metrics."""
        
        # Simulate toxicity detection (in production, use actual toxicity API)
        toxicity_scores = self._simulate_toxicity_detection(content_data)
        
        # Calculate bias amplification
        bias_amplification = self._calculate_bias_amplification(decisions)
        
        # Evaluate reasoning quality
        reasoning_quality = self._evaluate_reasoning_quality(content_data)
        
        # Count harmful content
        harmful_count = sum(1 for score in toxicity_scores if score > 0.5)
        
        return SafetyMetrics(
            toxicity_score=np.mean(toxicity_scores),
            bias_amplification_score=bias_amplification,
            reasoning_quality_score=reasoning_quality,
            flagged_content_count=harmful_count,
            total_content_evaluated=len(content_data),
            harmful_categories={
                "toxicity": np.mean([s for s in toxicity_scores if s > 0.1]),
                "bias": bias_amplification
            }
        )
    
    def calculate_quality_metrics(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        confidence_scores: List[float]
    ) -> QualityMetrics:
        """Calculate advanced quality metrics."""
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(ground_truth)
        confidences = np.array(confidence_scores)
        
        # Calculate basic classification metrics
        precision, recall, f1 = self._calculate_classification_metrics(y_pred, y_true)
        
        # Calculate calibration metrics
        ece = self._calculate_expected_calibration_error(y_pred, y_true, confidences)
        mce = self._calculate_maximum_calibration_error(y_pred, y_true, confidences)
        brier = self._calculate_brier_score(y_pred, y_true, confidences)
        
        # Calculate confidence metrics
        conf_acc_corr = np.corrcoef(confidences, y_pred == y_true)[0, 1]
        overconfidence = self._calculate_overconfidence(y_pred, y_true, confidences)
        
        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            confidence_accuracy_correlation=conf_acc_corr if not np.isnan(conf_acc_corr) else 0.0,
            overconfidence_score=overconfidence
        )
    
    def _group_by_attributes(
        self, 
        decisions: List[Dict[str, Any]], 
        agent_attributes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group decisions by protected attributes."""
        groups = {}
        
        for decision in decisions:
            agent_name = decision.get("agent_name", "")
            if agent_name in agent_attributes:
                attrs = agent_attributes[agent_name]
                # Create group key from protected attributes
                group_key = "_".join([
                    str(attrs.get(attr, "unknown")) 
                    for attr in self.protected_attributes
                ])
                
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(decision)
        
        return groups
    
    def _calculate_demographic_parity(self, groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[float, float]:
        """Calculate demographic parity metrics."""
        if len(groups) < 2:
            return 1.0, 0.0
        
        positive_rates = {}
        for group, decisions in groups.items():
            positive_count = sum(1 for d in decisions if d.get("decision") == "approve")
            positive_rates[group] = positive_count / len(decisions) if decisions else 0
        
        rates = list(positive_rates.values())
        demographic_parity = min(rates) / max(rates) if max(rates) > 0 else 1.0
        demographic_parity_diff = max(rates) - min(rates)
        
        return demographic_parity, demographic_parity_diff
    
    def _calculate_equalized_odds(self, groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[float, float, float]:
        """Calculate equalized odds metrics."""
        if len(groups) < 2:
            return 0.0, 0.0, 0.0
        
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group, decisions in groups.items():
            tp = sum(1 for d in decisions if d.get("prediction") == "positive" and d.get("ground_truth") == "positive")
            fp = sum(1 for d in decisions if d.get("prediction") == "positive" and d.get("ground_truth") == "negative")
            tn = sum(1 for d in decisions if d.get("prediction") == "negative" and d.get("ground_truth") == "negative")
            fn = sum(1 for d in decisions if d.get("prediction") == "negative" and d.get("ground_truth") == "positive")
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_by_group[group] = tpr
            fpr_by_group[group] = fpr
        
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0
        eo_diff = max(tpr_diff, fpr_diff)
        
        return eo_diff, tpr_diff, fpr_diff
    
    def _calculate_participation_parity(self, groups: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate participation parity across groups."""
        if len(groups) < 2:
            return 1.0
        
        participation_rates = {
            group: len(decisions) for group, decisions in groups.items()
        }
        
        rates = list(participation_rates.values())
        return min(rates) / max(rates) if max(rates) > 0 else 1.0
    
    def _calculate_voice_equality(
        self, 
        decisions: List[Dict[str, Any]], 
        agent_attributes: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate voice equality score."""
        # Simulate voice equality based on message length and reasoning quality
        voice_scores = []
        
        for decision in decisions:
            reasoning_length = len(decision.get("reasoning", ""))
            confidence = decision.get("confidence", 0.5)
            voice_score = min(1.0, (reasoning_length / 100) * confidence)
            voice_scores.append(voice_score)
        
        return np.mean(voice_scores) if voice_scores else 0.0
    
    def _calculate_individual_fairness(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate individual fairness score."""
        # Simulate individual fairness based on consistency of similar decisions
        consistency_scores = []
        
        for i, decision1 in enumerate(decisions):
            similar_decisions = []
            for j, decision2 in enumerate(decisions):
                if i != j:
                    # Simulate similarity based on decision context
                    similarity = self._calculate_decision_similarity(decision1, decision2)
                    if similarity > 0.8:  # High similarity threshold
                        similar_decisions.append(decision2)
            
            if similar_decisions:
                # Check consistency of outcomes for similar decisions
                same_outcome = sum(
                    1 for d in similar_decisions 
                    if d.get("decision") == decision1.get("decision")
                )
                consistency = same_outcome / len(similar_decisions)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_decision_similarity(self, decision1: Dict[str, Any], decision2: Dict[str, Any]) -> float:
        """Calculate similarity between two decisions."""
        # Simple similarity based on scenario type
        if decision1.get("scenario_type") == decision2.get("scenario_type"):
            return 0.9
        return 0.1
    
    def _simulate_toxicity_detection(self, content_data: List[Dict[str, Any]]) -> List[float]:
        """Simulate toxicity detection (replace with actual API in production)."""
        toxicity_scores = []
        
        # Define harmful patterns
        harmful_patterns = [
            r'\b(hate|stupid|idiot|moron)\b',
            r'\b(kill|destroy|eliminate)\b',
            r'\b(never|always|completely wrong)\b'
        ]
        
        for content in content_data:
            text = content.get("reasoning", "") + " " + content.get("content", "")
            text = text.lower()
            
            # Count harmful patterns
            harmful_count = sum(
                len(re.findall(pattern, text)) for pattern in harmful_patterns
            )
            
            # Convert to toxicity score (0-1)
            toxicity_score = min(1.0, harmful_count * 0.3)
            toxicity_scores.append(toxicity_score)
        
        return toxicity_scores
    
    def _calculate_bias_amplification(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate bias amplification score."""
        # Simulate bias detection based on decision patterns
        bias_indicators = 0
        total_comparisons = 0
        
        for decision in decisions:
            # Check for potential bias indicators in reasoning
            reasoning = decision.get("reasoning", "").lower()
            
            # Look for biased language patterns
            bias_patterns = [
                "obviously",
                "clearly wrong",
                "anyone can see",
                "it's obvious that"
            ]
            
            for pattern in bias_patterns:
                if pattern in reasoning:
                    bias_indicators += 1
            
            total_comparisons += 1
        
        return bias_indicators / total_comparisons if total_comparisons > 0 else 0.0
    
    def _evaluate_reasoning_quality(self, content_data: List[Dict[str, Any]]) -> float:
        """Evaluate quality of reasoning provided."""
        quality_scores = []
        
        for content in content_data:
            reasoning = content.get("reasoning", "")
            
            # Simple quality metrics
            length_score = min(1.0, len(reasoning) / 200)  # Prefer detailed reasoning
            
            # Check for evidence keywords
            evidence_keywords = ["because", "due to", "evidence", "data", "analysis", "research"]
            evidence_score = sum(1 for keyword in evidence_keywords if keyword in reasoning.lower())
            evidence_score = min(1.0, evidence_score / 3)
            
            # Check for logical structure
            structure_keywords = ["first", "second", "therefore", "however", "additionally"]
            structure_score = sum(1 for keyword in structure_keywords if keyword in reasoning.lower())
            structure_score = min(1.0, structure_score / 2)
            
            overall_quality = (length_score + evidence_score + structure_score) / 3
            quality_scores.append(overall_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_classification_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if len(y_pred) == 0:
            return 0.0, 0.0, 0.0
        
        # Convert to binary if needed
        if len(set(y_pred)) > 2:
            # Multi-class to binary (approved vs not approved)
            y_pred_binary = (y_pred == "approve").astype(int)
            y_true_binary = (y_true == "approve").astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
            y_true_binary = y_true.astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _calculate_expected_calibration_error(
        self, y_pred: np.ndarray, y_true: np.ndarray, confidences: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if len(confidences) == 0:
            return 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_maximum_calibration_error(
        self, y_pred: np.ndarray, y_true: np.ndarray, confidences: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error (MCE)."""
        if len(confidences) == 0:
            return 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def _calculate_brier_score(self, y_pred: np.ndarray, y_true: np.ndarray, confidences: np.ndarray) -> float:
        """Calculate Brier Score."""
        if len(confidences) == 0:
            return 0.0
        
        # Convert predictions to probabilities
        y_true_binary = (y_true == y_pred).astype(float)
        return np.mean((confidences - y_true_binary) ** 2)
    
    def _calculate_overconfidence(self, y_pred: np.ndarray, y_true: np.ndarray, confidences: np.ndarray) -> float:
        """Calculate overconfidence score."""
        if len(confidences) == 0:
            return 0.0
        
        correct = (y_pred == y_true).astype(float)
        overconfidence = np.mean(np.maximum(0, confidences - correct))
        return overconfidence