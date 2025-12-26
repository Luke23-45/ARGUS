"""
icu/utils/safety.py
--------------------------------------------------------------------------------
SOTA Safety Guardrails (NeurIPS 2024 Alignment).
Focus: Out-of-Distribution (OOD) Action Detection & Physiological Penalties.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional

class OODGuardian:
    """
    SOTA OOD Guardian (NeurIPS 2024 OGSRL style).
    Detects if the model is proposing trajectories that violate clinical norms.
    """
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        # Default SOTA thresholds (3-sigma or delta-based)
        self.thresholds = thresholds or {
            'sbp_drop_max': 40.0,   # Max SBP drop in 1h (mmHg)
            'hr_spike_max': 50.0,   # Max HR spike in 1h (bpm)
            'lactate_max': 10.0,    # Max Lactate (mmol/L) - extreme OOD
            'sofa_jump_max': 3.0    # Max SOFA jump in 1 window
        }

    def check_trajectories(
        self, 
        past_vitals: torch.Tensor, 
        pred_vitals: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Scans generated trajectories for medical anomalies.
        Args:
            past_vitals: [B, T_hist, D]
            pred_vitals: [B, T_pred, D]
        Returns:
            Dict containing OOD masks and severity scores.
        """
        B, T_p, D = pred_vitals.shape
        device = pred_vitals.device
        
        # [SAFETY CHECK] Unit Consistency
        # If max SBP is < 20, the data is likely normalized [-1, 1] 
        # and our absolute thresholds (e.g. 40 mmHg) will be invalid.
        if pred_vitals[..., 2].max() < 20.0:
            logger.warning("[OODGuardian] Input looks NORMALIZED. Precision safety checks may be invalid. Use Clinical Units!")
        
        # 1. Delta Checks (Dynamics Safety)
        # SBP = Index 2, HR = Index 0
        last_sbp = past_vitals[:, -1, 2]
        pred_sbp_first = pred_vitals[:, 0, 2]
        sbp_delta = torch.abs(pred_sbp_first - last_sbp)
        
        last_hr = past_vitals[:, -1, 0]
        pred_hr_first = pred_vitals[:, 0, 0]
        hr_delta = torch.abs(pred_hr_first - last_hr)
        
        # 2. Physiological Boundary Checks
        # Lactate = Index 7
        max_lac = pred_vitals[:, :, 7].max(dim=1)[0]
        
        # 3. Compute OOD Score
        is_ood_sbp = sbp_delta > self.thresholds['sbp_drop_max']
        is_ood_hr = hr_delta > self.thresholds['hr_spike_max']
        is_ood_lac = max_lac > self.thresholds['lactate_max']
        
        ood_mask = is_ood_sbp | is_ood_hr | is_ood_lac
        
        return {
            "ood_mask": ood_mask,           # [B] boolean
            "ood_rate": ood_mask.float().mean(),
            "sbp_delta_mean": sbp_delta.mean(),
            "lac_max_mean": max_lac.mean()
        }

def compute_sepsis3_violations(vitals: torch.Tensor) -> torch.Tensor:
    """
    Detects Sepsis-3 (2016) clinical violations.
    Used for Safety-Cost constraints in training.
    """
    # vitals: [B, T, D]
    # SBP (idx 2) <= 100 or Lactate (idx 7) > 2.0
    sbp = vitals[..., 2]
    lactate = vitals[..., 7]
    
    # Violation 1: Hypotension (Septic Shock core)
    # Use soft sigmoid to provide gradients
    v_sbp = torch.sigmoid((100.0 - sbp) / 5.0)
    
    # Violation 2: Metabolic failure (Lactate)
    v_lac = torch.sigmoid((lactate - 2.0) / 0.5)
    
    # Combined Safety Cost
    safety_cost = (v_sbp + v_lac) / 2.0
    return safety_cost
