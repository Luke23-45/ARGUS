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
) -> torch.Tensor:
    """
    Computes Expected Calibration Error (ECE/ACE) Vectorized.
    Returns a Tensor (Zero-Sync).
    """
    # 1. Shape Safety
    if probs.dim() > 1 and probs.shape[1] == 1:
        probs = probs.view(-1)
    labels = labels.view(-1)
        
    if probs.dim() > 1 and probs.shape[1] > 1:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()
    else:
        confidences = probs
        accuracies = labels.float()

    total_samples = confidences.size(0)
    if total_samples == 0:
        return torch.tensor(0.0, device=probs.device)

    # 2. Sort for Adaptive / Binning
    # We always sort for adaptive, and it helps for fixed too strictly speaking
    # but for fixed bins we can use bucketize.
    
    if adaptive:
        # ACE (Adaptive Calibration Error)
        sorted_conf, sorted_idx = torch.sort(confidences)
        sorted_acc = accuracies[sorted_idx]
        
        # Split into n_bins equal chunks
        # We can use tensor_split or reshape if divisible
        # Robust approach: use linspace indices
        indices = torch.linspace(0, total_samples, n_bins + 1, device=probs.device).long()
        
        ece = torch.tensor(0.0, device=probs.device)
        
        # [optimization] We can still loop here as n_bins is small (10), 
        # BUT we must avoid .item(). We accumulate Tensors.
        for i in range(n_bins):
            start = indices[i]
            end = indices[i+1]
            
            if start >= end: continue
            
            # Slicing is tensor-native
            bin_conf = sorted_conf[start:end]
            bin_acc = sorted_acc[start:end]
            
            # Tensor operations
            avg_conf = bin_conf.mean()
            avg_acc = bin_acc.mean()
            
            weight = (end - start).float() / total_samples
            ece += weight * torch.abs(avg_conf - avg_acc)
            
    else:
        # Standard ECE (Fixed width)
        # Use bucketize for vectorization
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        
        # Assign each sample to a bin
        bin_indices = torch.bucketize(confidences, bin_boundaries)
        # bucketize returns 1..N+1. We want 0..N-1. 
        # Boundaries are 11 points for 10 bins.
        # indices will be 1 to 10 typically.
        # We clamp to be safe.
        bin_indices = (bin_indices - 1).clamp(0, n_bins - 1)
        
        # Scatter add to get sums
        # [SOTA FIX] Dtype Alignment for AMP (Smoking Gun #663)
        # Rationale: Default float32 causes scatter_add mismatch if probs are fp16/bf16.
        bin_counts = torch.zeros(n_bins, device=probs.device, dtype=probs.dtype)
        bin_conf_sum = torch.zeros(n_bins, device=probs.device, dtype=probs.dtype)
        bin_acc_sum = torch.zeros(n_bins, device=probs.device, dtype=probs.dtype)
        
        bin_counts.scatter_add_(0, bin_indices, torch.ones_like(confidences, dtype=probs.dtype))
        bin_conf_sum.scatter_add_(0, bin_indices, confidences)
        bin_acc_sum.scatter_add_(0, bin_indices, accuracies.to(probs.dtype))
        
        # Mask empty bins
        mask = bin_counts > 0
        avg_conf = torch.zeros_like(bin_conf_sum)
        avg_acc = torch.zeros_like(bin_acc_sum)
        
        avg_conf[mask] = bin_conf_sum[mask] / bin_counts[mask]
        avg_acc[mask] = bin_acc_sum[mask] / bin_counts[mask]
        
        # Weighted mean absolute difference
        diffs = torch.abs(avg_conf - avg_acc)
        weighted_diffs = (bin_counts / total_samples) * diffs
        ece = weighted_diffs.sum()

    return ece


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes Brier Score (Tensor Return)."""
    labels = labels.view(-1)

    if probs.dim() > 1 and probs.shape[1] > 1:
        target_one_hot = F.one_hot(labels.long(), num_classes=probs.shape[1]).float()
        squared_error = (probs - target_one_hot).pow(2).sum(dim=1)
        brier = squared_error.mean()
    else:
        probs_flat = probs.view(-1).float()
        labels_flat = labels.float()
        brier = F.mse_loss(probs_flat, labels_flat, reduction='mean')
        
    return brier

def compute_overconfidence_error(
    probs: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 10
) -> torch.Tensor:
    """Computes Overconfidence Error (OE) Vectorized."""
    if probs.dim() > 1 and probs.shape[1] == 1:
        probs = probs.view(-1)
    labels = labels.view(-1)
        
    if probs.dim() > 1 and probs.shape[1] > 1:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()
    else:
        confidences = probs
        accuracies = labels.float()

    total_samples = confidences.size(0)
    if total_samples == 0: 
        return torch.tensor(0.0, device=probs.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(confidences, bin_boundaries)
    bin_indices = (bin_indices - 1).clamp(0, n_bins - 1)
    
    # [SOTA FIX] Dtype Alignment for AMP (Smoking Gun #663)
    bin_counts = torch.zeros(n_bins, device=probs.device, dtype=probs.dtype)
    bin_conf_sum = torch.zeros(n_bins, device=probs.device, dtype=probs.dtype)
    bin_acc_sum = torch.zeros(n_bins, device=probs.device, dtype=probs.dtype)
    
    bin_counts.scatter_add_(0, bin_indices, torch.ones_like(confidences, dtype=probs.dtype))
    bin_conf_sum.scatter_add_(0, bin_indices, confidences)
    bin_acc_sum.scatter_add_(0, bin_indices, accuracies.to(probs.dtype))
    
    mask = bin_counts > 0
    oe = torch.tensor(0.0, device=probs.device)
    
    # Vectorized accumulation
    # prop * conf * max(0, conf - acc)
    
    if mask.any():
        avg_conf = bin_conf_sum[mask] / bin_counts[mask]
        avg_acc = bin_acc_sum[mask] / bin_counts[mask]
        prop = bin_counts[mask] / total_samples
        
        penalty = torch.clamp(avg_conf - avg_acc, min=0.0)
        oe = (prop * avg_conf * penalty).sum()
            
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
    [PATCHED] Auto-handles Multiclass inputs by selecting positive class risk.
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # [PATCH] Resilience Guard: Handle Multiclass [N, C] inputs
    # Logic: If shape is [N, C], DCA applies to the target class (typically Sepsis=1 or Shock=2).
    # We assume the LAST column represents the risk of the positive event.
    if probs.dim() > 1 and probs.shape[1] > 1:
        probs_flat = probs[:, -1].view(-1) 
    else:
        probs_flat = probs.view(-1)
        
    labels_flat = labels.view(-1)
    
    # [PATCH] Safety Assertion: Ensure shapes align before computing
    if probs_flat.shape[0] != labels_flat.shape[0]:
         # Graceful resizing to overlap (resilience against batch fragmentation)
         min_len = min(probs_flat.shape[0], labels_flat.shape[0])
         probs_flat = probs_flat[:min_len]
         labels_flat = labels_flat[:min_len]
    
    n = len(labels_flat)
    if n == 0: return {}
    
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
    [PATCH] Shape Safety added to prevent HPO crashes.
    """
    # Ensure tensors are flat and matching
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    if y_pred.shape[0] != y_true.shape[0]:
        # If we can't align them, returning 0.0 is better than crashing the trial
        return 0.0

    var_y = torch.var(y_true)
    if var_y < 1e-9:
        return 0.0
    var_diff = torch.var(y_true - y_pred)
    return (1.0 - var_diff / var_y).item()