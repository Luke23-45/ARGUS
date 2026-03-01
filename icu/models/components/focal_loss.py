"""
icu/models/components/focal_loss.py
----------------------------------
[v2026 SOTA] Focal Loss for Imbalanced Clinical Classification.

RATIONALE:
Standard BCE gradients are dominated by the majority (non-sepsis) class.
Focal Loss down-weights well-classified easy examples to focus on 'Hard' sepsis transitions.
Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    SOTA Focal Loss for imbalanced clinical classification.
    
    Calibration (from ry.md Forensic Audit):
    - alpha=0.85: Boosts the minority (Sepsis) class. 
      (Note: 0.25 is for object detection where foreground is common).
    - gamma=2.0: Standard focusing parameter.
    """
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [B, C]
            targets: Binary targets [B, C]
        """
        # Ensure targets are float for BCE
        targets = targets.float()
        
        # 1. Compute Base Cross Entropy
        # We use BCEWithLogits for maximum numerical stability
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. Compute Probability of the Target Class (pt)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # 3. Apply Focal Weighting
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        
        # 4. Apply Class Weighting (Alpha)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
            
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"
