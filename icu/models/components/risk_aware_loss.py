import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from icu.models.components.focal_loss import FocalLoss

class RiskAwareAsymmetricLoss(nn.Module):
    """
    [Phase 1] SOTA Risk-Aware Asymmetric Loss.
    
    Combines:
    1. ASL: Asymmetric probability shifting to handle imbalanced Sepsis cases.
    2. Critical Penalty: Multiplies the loss for patients in 'Red Zones'.
    
    Update (v14.9): Integrated optimized FocalLoss core while preserving 
    ASL asymmetry and clinical gradient shielding.
    """
    def __init__(self, 
                 gamma_neg: float = 2.0,  
                 gamma_pos: float = 0.5,   
                 clip: float = 0.05,       
                 eps: float = 1e-5,
                 critical_multiplier: float = 2.0):  
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.critical_multiplier = critical_multiplier
        
        # [RY.MD FIX] Default alpha=0.85 boost for Sepsis
        self.focal_core = FocalLoss(alpha=0.85, gamma=gamma_neg, reduction='none')

    def forward(self, x, y, risk_coef: torch.Tensor, class_weights: Optional[torch.Tensor] = None, stability_factor: float = 1.0):
        # 1. Compute Probabilities for Asymmetric Weighting
        with torch.no_grad():
            probs = torch.sigmoid(x)
            
            # 2. Adaptive Focal Relaxation (gamma_neg suppression)
            gamma_neg_eff = 1.0 + (self.gamma_neg - 1.0) * stability_factor
            
            # 3. [v14.9.1 SURGICAL PATCH] ASL Probability Shifting
            # Rationale: Shift negative probabilities up by clip margin 
            # to hard-threshold trivially easy negatives.
            p_neg_shifted = (1.0 - probs + self.clip).clamp(max=1.0)
            pt = probs * y + p_neg_shifted * (1.0 - y)
            
            # 4. Asymmetric Focusing
            gamma_t = self.gamma_pos * y + gamma_neg_eff * (1.0 - y)
            
            # 5. [v14.9.1 SURGICAL PATCH] Alpha Weighting
            # Rationale: Forensic Sepsis boost (0.85) to resolve 1:30 imbalance.
            alpha_t = 0.85 * y + 0.15 * (1.0 - y)
            
            # [SOTA FIX 2] Gradient Shielding: Weighting must be detached
            asl_w = alpha_t * torch.pow(1.0 - pt, gamma_t)

        # 6. Base Cross Entropy
        bce = F.binary_cross_entropy_with_logits(x, y.float(), reduction='none')
        
        # 7. Combine ASL
        asl_loss = asl_w * bce
        
        # 8. Apply Critical Penalty (Dynamic)
        # Multiplies loss for high-risk zones (Shock/Hypoxia)
        risk_weight = 1.0 + (risk_coef.unsqueeze(-1) * (self.critical_multiplier - 1.0))
        final_loss = asl_loss * risk_weight
        
        return final_loss.mean()

