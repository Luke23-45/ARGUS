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
        Vectorized implementation (No Loops, No Syncs).
        
        x: Latent tensor [B, T, C]
        p: Effective percentile per sample [B], e.g. 0.99
        """
        B, T, C = x.shape
        # Flatten temporal/feature dims: [B, T*C]
        flat_x = x.abs().reshape(B, -1)
        
        # [v2026 SOTA] Vectorized Quantile via Sorting
        # Rationale: torch.quantile is slow in loops. Sorting is fast on GPU.
        # We find the value at the index corresponding to percentile p.
        sorted_x, _ = torch.sort(flat_x, dim=1) # Ascending
        
        # Calculate indices: idx = floor(p * N)
        # p is [B], N is T*C
        N = flat_x.shape[1]
        indices = (p * (N - 1)).long().clamp(0, N - 1) # [B]
        
        # Gather thresholds: [B, 1]
        thresh = sorted_x.gather(1, indices.unsqueeze(1))
        
        # Expand for broadcasting: [B, 1, 1]
        thresh = thresh.view(B, 1, 1)
        
        # Clamp entire batch at once
        x_out = torch.clamp(x, min=-thresh, max=thresh)
            
        return x_out
