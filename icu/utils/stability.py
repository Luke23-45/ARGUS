import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class DynamicThresholding(nn.Module):
    """
    [SOTA 2025] Dynamic Thresholding (Google Imagen Style).
    
    Prevents "Manifold Collapse" by rescaling latent vectors based on their 
    statistical distribution instead of hard clipping.
    """
    def __init__(self, percentile: float = 0.995, threshold: float = 3.0):
        super().__init__()
        self.percentile = percentile
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        abs_x = torch.abs(x)
        flat_abs = abs_x.view(B, -1)
        
        # Calculate s-th percentile
        s = torch.quantile(flat_abs, self.percentile, dim=1).view(B, 1, 1)
        
        # Scale factor
        s = torch.clamp(s, min=self.threshold)
        scale = self.threshold / s
        
        if x.dim() == 2:
            scale = scale.squeeze(-1)
            
        return x * scale

class ForensicStabilityAuditor(nn.Module):
    """
    [SOTA 2025] Forensic Stability Auditor (Pre-Clamp Validation).
    
    Unmasks "Metric Deception" by auditing raw clinical predictions 
    before the normalizer clamps them.
    """
    def __init__(self, guardian: Optional[Any] = None):
        super().__init__()
        self.guardian = guardian

    def audit_batch(
        self, 
        pred_clinical: torch.Tensor, 
        past_clinical: torch.Tensor,
        normalizer: Any
    ) -> Dict[str, float]:
        B, T, D = pred_clinical.shape
        device = pred_clinical.device
        
        # 1. Forensic Re-normalization (Ignore clamps to see true sigma)
        with torch.no_grad():
            s_min = normalizer.ts_stat_min.to(device).view(1, 1, -1)
            s_max = normalizer.ts_stat_max.to(device).view(1, 1, -1)
            l_mask = normalizer.log_mask.to(device).view(1, 1, -1)
            
            p_log = torch.log1p(torch.relu(pred_clinical))
            p_processed = torch.where(l_mask, p_log, pred_clinical)
            
            denom = (s_max - s_min).clamp(min=1e-3)
            true_sigma = 2.0 * (p_processed - s_min) / denom - 1.0
            
            phys_violations = (torch.abs(true_sigma) > 2.5).float().mean()
            max_sigma = torch.abs(true_sigma).max().item()

        # 2. Honest OOD Check
        ood_results = {}
        if self.guardian is not None:
            ood_results = self.guardian.check_trajectories(
                past_clinical, 
                pred_clinical, 
                force_clinical=True
            )
            
        return {
            "forensic/max_sigma": max_sigma,
            "forensic/phys_violation_rate": phys_violations.item(),
            "forensic/ood_rate": ood_results.get("ood_rate", 0.0),
            "forensic/safe_trajectories_avg": ood_results.get("safe_count", 0.0),
            "forensic/is_stable": float(max_sigma < 5.0)
        }
