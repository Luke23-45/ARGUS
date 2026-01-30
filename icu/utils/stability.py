import torch
import torch.distributed
import torch.nn as nn
from typing import Dict, Any, Optional

class DynamicThresholding(nn.Module):
    """
    [SOTA 2025] Dynamic Thresholding (Google Imagen Style).
    
    Prevents "Manifold Collapse" by rescaling latent vectors based on their 
    statistical distribution instead of hard clipping.
    
    [v27.0 FIX] Added EMA smoothing to reduce batch-to-batch scale variance.
    """
    def __init__(self, percentile: float = 0.995, threshold: float = 3.0, ema_decay: float = 0.99):
        super().__init__()
        self.percentile = percentile
        self.threshold = threshold
        self.ema_decay = ema_decay
        
        # [v27.0 FIX] EMA-smoothed percentile for gradient stability
        self.register_buffer("ema_s", torch.tensor(threshold))

    def forward(self, x: torch.Tensor, update_ema: bool = True) -> torch.Tensor:
        B = x.shape[0]
        abs_x = torch.abs(x)
        flat_abs = abs_x.view(B, -1)
        
        # [FIX] torch.quantile requires float32 or float64.
        # Calculate s-th percentile. Detach to avoid graph retention.
        batch_s = torch.quantile(flat_abs.detach().float(), self.percentile, dim=1).mean()
        
        # [v67.0 SOTA FIX] DDP Governance Consensus (Smoking Gun #67)
        # Rationale: EMA buffers MUST be identical across ranks to ensure 
        # consistent manifold scaling, otherwise gradients will 'fight' after AllReduce.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_s, op=torch.distributed.ReduceOp.SUM)
            batch_s /= torch.distributed.get_world_size()

        # [v27.0 FIX] EMA smoothing for gradient stability
        # [v2026 SOTA] Accumulation Guard: Only update on stepping batches during training.
        if update_ema and self.training:
            with torch.no_grad():
                self.ema_s.mul_(self.ema_decay).add_(batch_s * (1 - self.ema_decay))
        
        # Scale factor using smoothed percentile
        s = torch.clamp(self.ema_s, min=self.threshold)
        scale = self.threshold / s
        
        # Handle different tensor dimensions
        if x.dim() == 3:
            return x * scale.view(1, 1, 1)
        elif x.dim() == 2:
            return x * scale.view(1, 1)
        else:
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
