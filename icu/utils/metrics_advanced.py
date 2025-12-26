"""
icu/utils/metrics_advanced.py
--------------------------------------------------------------------------------
SOTA Clinical & RL Metrics focused on Safety and Reliability.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Measures the gap between predicted probabilities and actual accuracy.
    """
    if probs.dim() > 1 and probs.shape[1] > 1:
        # Multiclass: take max prob (confidence)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
    else:
        # Binary
        confidences = probs
        accuracies = labels.float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select items in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def compute_overconfidence_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes Overconfidence Error (OE) - SOTA for medical trust.
    Ref: 'On Calibration of Modern Neural Networks' & KDD 2024 healthcare standards.
    Formula: OE = sum(|Bb|/n * conf(Bb) * max(0, conf(Bb) - acc(Bb)))
    """
    if probs.dim() > 1 and probs.shape[1] > 1:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()
    else:
        confidences = probs
        accuracies = labels.float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    oe = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            acc_b = accuracies[in_bin].mean().item()
            conf_b = confidences[in_bin].mean().item()
            # Safety critical: Only penalize if confidence > accuracy
            oe += prop_in_bin * conf_b * max(0, conf_b - acc_b)
            
    return oe

def compute_bootstrap_ci(
    data: torch.Tensor, 
    metric_fn: Any, 
    n_resamples: int = 1000, 
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Computes Bootstrap Confidence Intervals for a given metric.
    Standard for SOTA clinical papers (Nature Medicine / Lancet).
    """
    results = []
    n = len(data)
    # Convert to numpy for faster resampling if it's a tensor
    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.array(data)

    for _ in range(n_resamples):
        resample_idx = np.random.choice(n, size=n, replace=True)
        resampled_data = data_np[resample_idx]
        results.append(metric_fn(resampled_data))
    
    results = np.sort(results)
    alpha = 1.0 - confidence_level
    lower_idx = int(alpha / 2 * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples)
    
    return {
        "mean": float(np.mean(results)),
        "lower": float(results[lower_idx]),
        "upper": float(results[upper_idx]),
        "std": float(np.std(results))
    }

def compute_action_continuity(actions: torch.Tensor) -> float:
    """
    Computes Action Continuity (Smoothness).
    Formula: 1 - mean(abs(a_t - a_{t-1}))
    Higher is smoother. Essential for clinical trust.
    """
    if actions.shape[1] < 2:
        return 1.0
    # actions: [B, T, D]
    diffs = torch.abs(actions[:, 1:] - actions[:, :-1])
    # Normalize by max delta to keep in [0, 1]
    # Assuming actions are normalized or have a known range
    smoothness = 1.0 - diffs.mean().item()
    return max(0.0, smoothness)

def compute_demographic_accuracy_gaps(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    demographics: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Computes accuracy gaps across demographics (Gender, Age).
    Requirement for 2025 SOTA (Fairness Audit).
    """
    gaps = {}
    # Convert to binary choices for simple gap analysis
    pred_choices = (preds > 0.5).long()
    
    for group_name, group_mask in demographics.items():
        if group_mask.any():
            acc = (pred_choices[group_mask] == labels[group_mask]).float().mean().item()
            gaps[f"acc_{group_name}"] = acc
            
    return gaps

def compute_policy_entropy(probs: torch.Tensor) -> float:
    """
    Computes Shannon Entropy of the decision distribution.
    Lower entropy indicates more decisive (but potentially collapsed) behavior.
    """
    # -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
    return entropy.item()

def compute_discordance_rate(pred_actions: torch.Tensor, clinician_actions: torch.Tensor) -> float:
    """
    Computes Discordance Rate between AI and Human Clinicians.
    """
    mismatch = (pred_actions != clinician_actions).float().mean()
    return mismatch.item()

def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Explained Variance: 1 - Var(y_true - y_pred) / Var(y_true)
    """
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return 0.0
    var_diff = torch.var(y_true - y_pred)
    return (1.0 - var_diff / var_y).item()
