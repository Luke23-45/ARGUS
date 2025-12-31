"""
working/robust_losses.py
------------------------
SOTA Robust Losses for Clinical Diffusion RL.
"""

import torch
import torch.nn.functional as F

def smooth_l1_critic_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0):
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='mean')

def compute_explained_variance(pred: torch.Tensor, target: torch.Tensor):
    var_y = torch.var(target) + 1e-8
    return 1.0 - torch.var(target - pred) / var_y

def physiological_violation_loss(x_pred: torch.Tensor, bounds: float = 2.5, weight: float = 1.0):
    """
    [CRITICAL ALIGNMENT] Bounds changed from 1.5 to 2.5.
    Matches icu/models/diffusion_icu.py for system-wide consistency.
    """
    violation = torch.relu(torch.abs(x_pred) - bounds)
    return weight * (violation ** 2).mean()