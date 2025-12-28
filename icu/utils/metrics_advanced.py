"""
icu/utils/metrics_advanced.py
--------------------------------------------------------------------------------
APEX-MoE SOTA Clinical & RL Metrics (v3.5 - Safety Critical)

"Do you know the meaning of saving lives?"
This module implements the Gold Standard for evaluating AI in high-stakes
healthcare settings. It prioritizes:
1. Calibration (ACE/ECE): Can we trust the confidence?
2. Clinical Utility (Net Benefit): Does using this model actually help patients?
3. Safety (Overconfidence/Smoothness): Does the model fail gracefully?
4. Fairness (Equalized Odds): Is the model unbiased across demographics?

References:
- Ovadia et al. (2019) "Can You Trust Your Model's Uncertainty?"
- Vickers et al. (2006) "Decision Curve Analysis"
- Hardt et al. (2016) "Equality of Opportunity in Supervised Learning"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# ==============================================================================
# 1. Calibration & Reliability (Trust)
# ==============================================================================

def compute_ece(
    probs: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 10,
    adaptive: bool = True
) -> float:
    """
    Computes Expected Calibration Error (ECE) or Adaptive Calibration Error (ACE).
    
    Why this matters: In Sepsis (2.9% prevalence), fixed bins are mostly empty.
    Adaptive binning (ACE) ensures every bin has data, giving a true measure of
    reliability for rare events.
    
    Args:
        probs: [N, C] or [N] probabilities.
        labels: [N] ground truth.
        n_bins: Number of confidence bins.
        adaptive: If True, uses Quantile binning (ACE). If False, fixed width (ECE).
    """
    if probs.dim() > 1 and probs.shape[1] > 1:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()
    else:
        confidences = probs
        accuracies = labels.float()

    ece = 0.0
    total_samples = confidences.size(0)
    
    if adaptive:
        # ACE: Quantile-based binning (equal number of samples per bin)
        # Sort confidences to find quantiles
        sorted_conf, sorted_idx = torch.sort(confidences)
        sorted_acc = accuracies[sorted_idx]
        
        bin_size = total_samples // n_bins
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else total_samples
            
            if start >= end: break
            
            bin_conf = sorted_conf[start:end]
            bin_acc = sorted_acc[start:end]
            
            avg_conf = bin_conf.mean().item()
            avg_acc = bin_acc.mean().item()
            
            # Weight by bin size (though roughly equal in ACE)
            weight = (end - start) / total_samples
            ece += weight * np.abs(avg_conf - avg_acc)
    else:
        # Standard ECE: Fixed width bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes the Brier Score (Mean Squared Error of Probabilities).
    A 'Strictly Proper Scoring Rule'. Unlike AUC, this penalizes being 
    uncertain when you should be sure, and sure when you are wrong.
    """
    if probs.dim() > 1 and probs.shape[1] > 1:
        # One-hot encode labels for multiclass Brier
        target_one_hot = F.one_hot(labels.long(), num_classes=probs.shape[1]).float()
        mse = F.mse_loss(probs, target_one_hot, reduction='mean')
    else:
        # Binary
        mse = F.mse_loss(probs.float(), labels.float(), reduction='mean')
        
    return mse.item()

def compute_overconfidence_error(
    probs: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 10
) -> float:
    """
    Computes Overconfidence Error (OE).
    Penalizes the model specifically for high-confidence errors.
    Critical for "Do No Harm": It is worse to be confidently wrong than unsure.
    """
    if probs.dim() > 1 and probs.shape[1] > 1:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()
    else:
        confidences = probs
        accuracies = labels.float()

    # We want to punish: Confidence > Accuracy
    # OE = E[ confidence * max(0, confidence - accuracy) ]
    
    # Vectorized computation without explicit bins for finer granularity 
    # or keep binning for stability. Sticking to binning for stability.
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    oe = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            acc_b = accuracies[in_bin].mean().item()
            conf_b = confidences[in_bin].mean().item()
            # The penalty term
            oe += prop_in_bin * conf_b * max(0.0, conf_b - acc_b)
            
    return oe

# ==============================================================================
# 2. Clinical Utility (Net Benefit)
# ==============================================================================

def compute_net_benefit(
    probs: torch.Tensor, 
    labels: torch.Tensor,
    thresholds: Optional[List[float]] = None
) -> Dict[float, float]:
    """
    Computes Net Benefit for Decision Curve Analysis (DCA).
    Formula: NB = (TP / N) - (FP / N) * (pt / (1 - pt))
    
    This is the ONLY metric that tells you if the model is clinically useful
    at a specific risk tolerance (threshold).
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
    probs_flat = probs.view(-1)
    labels_flat = labels.view(-1)
    n = len(labels_flat)
    
    results = {}
    
    for pt in thresholds:
        if pt >= 1.0 or pt <= 0.0: continue
        
        # Hard predictions at this threshold
        preds = (probs_flat >= pt).float()
        
        tp = ((preds == 1) & (labels_flat == 1)).float().sum().item()
        fp = ((preds == 1) & (labels_flat == 0)).float().sum().item()
        
        # Weight of False Positive relative to True Positive
        w = pt / (1.0 - pt)
        
        nb = (tp / n) - (fp / n) * w
        results[pt] = nb
        
    return results

# ==============================================================================
# 3. RL Safety & Stability
# ==============================================================================

def compute_action_continuity(actions: torch.Tensor) -> float:
    """
    Computes Action Smoothness/Continuity.
    In ICU, we don't want 'flickering' treatments (on/off/on/off).
    Returns a score in [0, 1] where 1 is perfectly smooth.
    """
    if actions.shape[1] < 2:
        return 1.0
        
    # Check if discrete (integers) or continuous (logits/probs)
    is_discrete = actions.dtype in [torch.long, torch.int, torch.bool]
    
    if is_discrete:
        # Hamming distance / Switching rate logic
        # Count how many times action changes from t to t+1
        changes = (actions[:, 1:] != actions[:, :-1]).float()
        mean_changes = changes.mean().item()
        # Invert so higher is better (smoothness)
        return 1.0 - mean_changes
    else:
        # Euclidean smoothness for continuous doses
        diffs = torch.abs(actions[:, 1:] - actions[:, :-1])
        # Normalize assuming action range is roughly [0, 1] or similar
        # If unknown, we just return 1 - mean_diff clamped
        smoothness = 1.0 - diffs.mean().item()
        return max(0.0, smoothness)

def compute_policy_entropy(probs: torch.Tensor) -> float:
    """
    Computes Shannon Entropy of the decision distribution.
    H(p) = - sum(p * log(p))
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    return entropy.item()

def compute_discordance_rate(
    pred_actions: torch.Tensor, 
    clinician_actions: torch.Tensor
) -> float:
    """
    Computes Discordance Rate between AI and Human Clinicians.
    Lower is often 'safer' for imitation learning, but high discordance
    might mean the AI is smarter (or dumber). Needs context.
    """
    mismatch = (pred_actions != clinician_actions).float().mean()
    return mismatch.item()

# ==============================================================================
# 4. Fairness (Equalized Odds)
# ==============================================================================

def compute_demographic_accuracy_gaps(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    demographics: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Computes Fairness Audit using Equalized Odds principles.
    Instead of just accuracy, we check TPR (Recall) and FPR gaps.
    
    Returns:
        Dict with 'gap_acc', 'gap_tpr', 'gap_fpr' for each group.
    """
    results = {}
    pred_binary = (preds > 0.5).float()
    
    for group_name, group_mask in demographics.items():
        if not group_mask.any():
            continue
            
        # 1. Filter data for this group
        g_preds = pred_binary[group_mask]
        g_labels = labels[group_mask]
        
        # 2. Compute Metrics
        # Accuracy
        acc = (g_preds == g_labels).float().mean().item()
        
        # TPR (Recall) = TP / (TP + FN)
        tp = ((g_preds == 1) & (g_labels == 1)).float().sum()
        p_count = (g_labels == 1).float().sum()
        tpr = (tp / p_count).item() if p_count > 0 else 0.0
        
        # FPR = FP / (FP + TN)
        fp = ((g_preds == 1) & (g_labels == 0)).float().sum()
        n_count = (g_labels == 0).float().sum()
        fpr = (fp / n_count).item() if n_count > 0 else 0.0
        
        results[f"{group_name}_acc"] = acc
        results[f"{group_name}_tpr"] = tpr
        results[f"{group_name}_fpr"] = fpr
        
    return results

# ==============================================================================
# 5. Robust Statistics (Bootstrap)
# ==============================================================================

def compute_bootstrap_ci(
    data: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
    metric_fn: Callable, 
    n_resamples: int = 1000, 
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Computes robust Bootstrap Confidence Intervals.
    Handles single tensors, tuples (pred, label), and dictionary outputs.
    """
    # 1. Prepare data for indexing
    if isinstance(data, (tuple, list)):
        n = len(data[0])
        data_is_tuple = True
        # Ensure all tuple elements are numpy/cpu
        data_np = tuple(
            d.detach().cpu().numpy() if torch.is_tensor(d) else np.array(d) 
            for d in data
        )
    else:
        n = len(data)
        data_is_tuple = False
        data_np = data.detach().cpu().numpy() if torch.is_tensor(data) else np.array(data)

    collected_results = []
    
    # 2. Resampling Loop
    device = data[0].device if data_is_tuple and torch.is_tensor(data[0]) else (
        data.device if torch.is_tensor(data) else "cpu"
    )

    for _ in range(n_resamples):
        indices = np.random.choice(n, size=n, replace=True)
        
        if data_is_tuple:
            # Resample all inputs synchronously
            resampled_batch = tuple(
                torch.tensor(d[indices], device=device) for d in data_np
            )
            # Unpack for the function
            val = metric_fn(*resampled_batch)
        else:
            resampled_batch = torch.tensor(data_np[indices], device=device)
            val = metric_fn(resampled_batch)
            
        collected_results.append(val)
        
    # 3. Aggregation & Statistics
    alpha = 1.0 - confidence_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100
    
    # Handle Dictionary Output (e.g., from Net Benefit)
    if isinstance(collected_results[0], dict):
        final_stats = {}
        keys = collected_results[0].keys()
        for k in keys:
            vals = [r[k] for r in collected_results]
            final_stats[k] = {
                "mean": float(np.mean(vals)),
                "lower": float(np.percentile(vals, lower_p)),
                "upper": float(np.percentile(vals, upper_p)),
                "std": float(np.std(vals))
            }
        return final_stats
        
    # Handle Scalar Output
    else:
        vals = np.array(collected_results)
        return {
            "mean": float(np.mean(vals)),
            "lower": float(np.percentile(vals, lower_p)),
            "upper": float(np.percentile(vals, upper_p)),
            "std": float(np.std(vals))
        }

def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Explained Variance: 1 - Var(error) / Var(true)
    """
    var_y = torch.var(y_true)
    if var_y < 1e-9:
        return 0.0
    var_diff = torch.var(y_true - y_pred)
    return (1.0 - var_diff / var_y).item()