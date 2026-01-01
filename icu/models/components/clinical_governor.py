import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceAwareGovernor(nn.Module):
    """
    [Phase 5] Teacher-Guided Safety Steering.
    
    Uses the EMA Teacher as a 'Moral Compass' for the Student.
    
    Logic:
    1. Epistemic Uncertainty (U): Euclidean distance between Student and Teacher latents.
    2. Distrust (D): Product of Uncertainty and Clinical Risk (Phase 1).
    3. Threshold Scaling: The more we distrust the student, the tighter the 
       dynamic thresholding percentile (p) becomes.
    """
    def __init__(self, base_p: float = 0.99, min_p: float = 0.90):
        super().__init__()
        self.base_p = base_p
        self.min_p = min_p
        
    def calculate_distrust(self, student_latents: torch.Tensor, teacher_latents: torch.Tensor, risk_coef: torch.Tensor) -> torch.Tensor:
        """
        Calculates a Distrust score [B, T].
        """
        # 1. Epistemic Uncertainty (Normalized delta)
        # Reduction none -> [B, T, C] -> mean(-1) -> [B, T]
        delta = F.mse_loss(student_latents, teacher_latents, reduction='none').mean(dim=-1)
        # Normalize uncertainty to [0, 1]
        uncertainty = torch.clamp(delta * 2.0, 0, 1)
        
        # 2. Final Distrust (Broadcasting risk_coef [B] over T)
        distrust = torch.max(uncertainty, risk_coef.unsqueeze(-1) * uncertainty)
        
        return distrust

    def get_dynamic_percentile(self, distrust: torch.Tensor) -> torch.Tensor:
        """
        Scales the percentile from base_p down to min_p.
        distrust: [B, T]
        Returns: p_eff [B] (averaged over time for stable thresholding)
        """
        distrust_b = distrust.mean(dim=1) # [B]
        p_eff = self.base_p - (self.base_p - self.min_p) * distrust_b
        return p_eff

    def apply_governance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        [SOTA] Confidence-Aware Dynamic Thresholding.
        x: Latent tensor [B, T, C]
        p: Effective percentile per sample [B]
        """
        B = x.shape[0]
        x_out = x.clone()
        
        for i in range(B):
            # Flatten T, C for quantile calculation
            s = x[i].abs().view(-1)
            # [FIX] torch.quantile requires float32 or float64, and q must match input dtype.
            # Cast both to float32 for compatibility with mixed precision/bfloat16.
            # [SOTA FIX] Detach to avoid graph retention in the safety threshold.
            thresh = torch.quantile(s.detach().float(), p[i].float())
            # Clamp outliers to the threshold
            x_out[i] = torch.clamp(x[i], min=-thresh, max=thresh)
            
        return x_out
