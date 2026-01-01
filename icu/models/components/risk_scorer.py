import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class PhysiologicalRiskScorer(nn.Module):
    """
    [Phase 1] Maps raw physiological vitals to a normalized Risk Coefficient [0, 1].
    
    Target Red Zones:
    - MAP < 65 (Shock)
    - Lactate > 2.0 (Tissue Hypoxia)
    
    Logic:
    Uses Sigmoid Soft-Cliffs to ensure the coefficient is differentiable and 
    saturates appropriately at extreme values.
    """
    def __init__(self, map_threshold: float = 65.0, lactate_threshold: float = 2.0):
        super().__init__()
        self.map_threshold = map_threshold
        self.lactate_threshold = lactate_threshold
        
    def forward(self, vitals: torch.Tensor, feature_indices: Dict[str, int]) -> torch.Tensor:
        """
        Args:
            vitals: [B, T, C] clinical vitals (denormalized)
            feature_indices: Map of feature names to indices
        Returns:
            risk_coef: [B] Scalar coefficient representing current instability
        """
        # 1. Extract raw features
        idx_map = feature_indices.get('map', 4)
        idx_lac = feature_indices.get('lactate', 7)
        
        map_seq = vitals[..., idx_map]
        lac_seq = vitals[..., idx_lac]
        
        # [NEW] NaN Robustness: Fill NaNs with 'Stable' defaults to prevent NaN propagation
        # MAP default = 80, Lactate default = 1.0
        map_seq = torch.where(torch.isnan(map_seq), torch.full_like(map_seq, 80.0), map_seq)
        lac_seq = torch.where(torch.isnan(lac_seq), torch.full_like(lac_seq, 1.0), lac_seq)
        
        # Risk from MAP (low is bad)
        map_risk = torch.sigmoid((self.map_threshold - map_seq) * 0.5)
        
        # Risk from Lactate (high is bad)
        lac_risk = torch.sigmoid((lac_seq - self.lactate_threshold) * 2.0)
        
        # Aggregate Risk
        map_risk_max, _ = map_risk.max(dim=1)
        lac_risk_max, _ = lac_risk.max(dim=1)
        
        risk_coef = torch.max(map_risk_max, lac_risk_max)
        
        return risk_coef
