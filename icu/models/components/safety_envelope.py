import torch
import torch.nn as nn
from typing import Dict, Tuple

class PhysiologicalSafetyEnvelope(nn.Module):
    """
    [Phase 2] Context-Aware Safety Boundaries.
    
    Replaces global sigma (2.5) with feature-specific clinical envelopes.
    Standardized per ICU-28 canonical specifications.
    
    Key Logic:
    1. Base Variance (σ): Defined per vital (e.g., SpO2 is tight, HR is wide).
    2. Risk Contraction: Bounds tighten as risk_coef (from Phase 1) increases.
    3. Normalization: Operates on denormalized clinical units for interpretability.
    """
    def __init__(self, feature_indices: Dict[str, int]):
        super().__init__()
        self.feature_indices = feature_indices
        
        # Define Canonical Clinical Bounds [Min, Max, Base_Sigma]
        # These are 'Loose' bounds for stable patients, to be contracted under risk.
        self.envelopes = {
            'hr':    (40.0, 180.0, 5.0),  # Wide
            'o2sat': (90.0, 100.0, 1.0),  # Ultra-Tight (Life-Critical)
            'sbp':   (90.0, 190.0, 4.0),
            'map':   (60.0, 130.0, 2.5),  # Standard
            'resp':  (8.0, 35.0, 2.0),
            'lactate': (0.5, 15.0, 1.5),  # Metabolic boundary
        }
        
    def get_bounds(self, risk_coef: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns dynamic [Min, Max, Effective_Sigma] for each feature.
        
        effective_sigma = base_sigma * (1.0 - risk_coef * 0.5)
        (Tightens by up to 50% under extreme risk)
        """
        B = risk_coef.shape[0]
        device = risk_coef.device
        
        dynamic_bounds = {}
        for feature, (f_min, f_max, base_sigma) in self.envelopes.items():
            # Contraction logic
            # risk_coef closer to 1 -> multiplier closer to 0.5
            multiplier = 1.0 - (risk_coef * 0.5) 
            
            # [B]
            eff_sigma = torch.full((B,), base_sigma, device=device) * multiplier
            low_bound = torch.full((B,), f_min, device=device)
            high_bound = torch.full((B,), f_max, device=device)
            
            dynamic_bounds[feature] = (low_bound, high_bound, eff_sigma)
            
        return dynamic_bounds

    def forward(self, vitals: torch.Tensor, risk_coef: torch.Tensor) -> torch.Tensor:
        """
        Computes the Violation Score.
        Violation = Mean( ReLU(x - Max) + ReLU(Min - x) )
        Weighted by 1/effective_sigma.
        """
        B, T, C = vitals.shape
        bounds = self.get_bounds(risk_coef)
        
        total_violation = torch.zeros(B, T, device=vitals.device)
        
        for feature, (low, high, sigma) in bounds.items():
            idx = self.feature_indices.get(feature)
            if idx is None or idx >= C: continue
            
            val = vitals[..., idx] # [B, T]
            
            # Broadcast bounds [B] -> [B, T]
            l = low.unsqueeze(1)
            h = high.unsqueeze(1)
            s = sigma.unsqueeze(1)
            
            # Calculate distance from 'Clinical Safe Zone'
            # Divided by sigma: sensitive features (SpO2) generate MUCH higher loss per unit violation.
            v_low = torch.relu(l - val) / (s + 1e-6)
            v_high = torch.relu(val - h) / (s + 1e-6)
            
            total_violation += (v_low + v_high)
            
        return total_violation.mean() # Combined violation scalar
