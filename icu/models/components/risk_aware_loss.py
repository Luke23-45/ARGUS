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
    """
    def __init__(self, 
                 gamma_neg: float = 4.0, 
                 gamma_pos: float = 1.0, 
                 clip: float = 0.0,
                 eps: float = 1e-5, # [SOTA FIX 1] FP16 Hardware Safety (Prevents Tensor-Core 0.0 flush)
                 critical_multiplier: float = 5.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.critical_multiplier = critical_multiplier

    def forward(self, x, y, risk_coef: torch.Tensor, class_weights: Optional[torch.Tensor] = None, stability_factor: float = 1.0):
        # 1. Independent Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # 2. Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 3. Basic Cross Entropy (Protected by 1e-5 floor)
        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # 4. Adaptive Focal Relaxation
        gamma_neg_eff = 1.0 + (self.gamma_neg - 1.0) * stability_factor
        
        # 5. [SOTA FIX 2] Gradient Shielding
        # Modulating weights MUST be detached to prevent the derivative of (1-pt)^gamma 
        # from overpowering and corrupting the directional BCE gradient.
        with torch.no_grad():
            pt = xs_pos * y + xs_neg * (1.0 - y)
            one_sided_gamma = self.gamma_pos * y + gamma_neg_eff * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
        
        # 6. Base ASL Loss (Unreduced) [B, C]
        asl_loss = -one_sided_w * (loss_pos + loss_neg)
        
        # [SOTA FIX 3] Alarm Fatigue Prevention: Class weights intentionally ignored 
        # here because ASL gamma parameters naturally handle 1:30 class imbalances.
        
        # 7. Apply Critical Penalty (Dynamic)
        risk_weight = 1.0 + (risk_coef.unsqueeze(-1) * (self.critical_multiplier - 1.0))
        final_loss = asl_loss * risk_weight
        
        return final_loss.mean()

