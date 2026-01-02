import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RiskAwareAsymmetricLoss(nn.Module):
    """
    [Phase 1] SOTA Risk-Aware Asymmetric Loss.
    
    Combines:
    1. ASL: Asymmetric probability shifting to handle imbalanced Sepsis cases.
    2. Critical Penalty: Multiplies the loss for patients in 'Red Zones'.
    3. Class Weights: Optional per-class weighting for imbalanced data.
    
    Logic:
    - Base: Standard Asymmetric Loss (gamma_neg > gamma_pos).
    - Dynamic: loss_total = ASL_loss * (1.0 + risk_coef * penalty_multiplier)
    
    This ensures that for a patient in shock, ANY mistake (False Negative or 
    False Positive) results in a significantly higher gradient.
    """
    def __init__(self, 
                 gamma_neg: float = 4, 
                 gamma_pos: float = 1, 
                 clip: float = 0.05, 
                 eps: float = 1e-8,
                 critical_multiplier: float = 2.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.critical_multiplier = critical_multiplier

    def forward(self, x, y, risk_coef: torch.Tensor, class_weights: Optional[torch.Tensor] = None):
        """
        x: logits [B, C]
        y: targets [B, C]
        risk_coef: [B] risk normalized to [0, 1]
        class_weights: Optional [C] weights for each class
        """
        # 1. Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        # 2. Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 3. Basic Cross Entropy
        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # 4. Asymmetric Focusing (Standard ASL)
        pt = xs_pos * y + xs_neg * (1 - y)
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        
        # 5. Base ASL Loss (Unreduced) [B, C]
        asl_loss = -one_sided_w * (loss_pos + loss_neg)
        
        # 6. Apply Class Weights (if provided)
        if class_weights is not None:
            # class_weights: [C] -> broadcast to [B, C]
            asl_loss = asl_loss * class_weights.unsqueeze(0)
        
        # 7. Apply Critical Penalty (Dynamic)
        # We want weight = 1.0 when risk=0, and weight = critical_multiplier when risk=1.
        risk_weight = 1.0 + (risk_coef.unsqueeze(-1) * (self.critical_multiplier - 1.0))
        
        final_loss = asl_loss * risk_weight
        
        # [FIX 2] Return .mean() instead of .sum() for proper multi-task scaling
        return final_loss.mean()

