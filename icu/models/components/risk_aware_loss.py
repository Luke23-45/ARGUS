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
                 gamma_neg: float = 2.0,  # [ry.md FIX 2a] Reduced from 4.0 to prevent healthy gradient starvation
                                           # (4.0 caused 16x-123x suppression; 2.0 gives 4x-11x — clinical balance)
                 gamma_pos: float = 0.5,   # [ry.md FIX 2b] Reduced from 1.0 (0.5 = mild focal, preserves precision)
                                           # Note: ry.md suggested 0.0 but simulation shows that removes ALL positive
                                           # modulation, which can hurt precision on well-classified positives.
                 clip: float = 0.05,       # [ry.md FIX 2c] Asymmetric clipping for noisy negative label robustness
                 eps: float = 1e-5,
                 critical_multiplier: float = 2.0):  # [ry.md FIX 2d] Reduced from 5.0 to prevent alarm fatigue
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

