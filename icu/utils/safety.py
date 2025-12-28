"""
icu/utils/safety.py
--------------------------------------------------------------------------------
APEX-MoE: SOTA Safety Guardrails (NeurIPS 2024 Alignment / OGSRL).
Focus: Out-of-Distribution (OOD) Action Detection, Stitching Consistency, 
       & Physiological Penalties.

Key Features:
1.  **Unit-Awareness**: Automatically detects if inputs are normalized and bypasses 
    absolute checks to prevent false positives, while logging warnings.
2.  **Stitching Consistency**: Measures the "jump" error between the last observed 
    vital and the first predicted vital (preventing hallucination gaps).
3.  **Sepsis-3 definitions**: Differentiable loss components based on the 2016 
    Consensus (MAP < 65 mmHg + Lactate > 2.0 mmol/L).
4.  **Dynamics Constraints**: Penalizes clinically impossible deltas (e.g., HR 
    spiking +100 bpm in 1 hour).

Author: APEX Research Team
Context: Life-Critical ICU Planning
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, Union

# Setup Logger
logger = logging.getLogger("APEX_Safety_Guardian")

# ==============================================================================
# 1. CLINICAL PHYSICS CONSTANTS (Sepsis-3 & Critical Care)
# ==============================================================================
# Aligned with icu.datasets.dataset.CANONICAL_COLUMNS
# Indices: 0=HR, 1=O2Sat, 2=SBP, 4=MAP, 7=Lactate

IDX_HR = 0
IDX_O2 = 1
IDX_SBP = 2
IDX_MAP = 4
IDX_LAC = 7

class SafetyConfig:
    """
    Clinical Thresholds for "Safe" Dynamics (1-hour window).
    Derived from PhysioNet 2019 Challenge & Sepsis-3 Guidelines.
    """
    # Max absolute jump between time steps (1h)
    MAX_DELTA_SBP = 40.0   # mmHg (Sudden hypotension/hypertension)
    MAX_DELTA_HR = 50.0    # bpm (Sudden tachycardia/bradycardia)
    MAX_DELTA_MAP = 30.0   # mmHg
    
    # Absolute Physiological Bounds (Life-Critical)
    # Violating these implies the model has hallucinated an impossible state.
    BOUNDS_HR = (20.0, 300.0)      # Asystole <-> Flutter
    BOUNDS_SBP = (30.0, 300.0)
    BOUNDS_MAP = (20.0, 250.0)     # Shock < 65
    BOUNDS_O2 = (20.0, 100.0)      # Hypoxia < 80
    BOUNDS_LAC = (0.0, 35.0)       # Normal < 2

    # Sepsis-3 Definitions (Septic Shock)
    THRESHOLD_SHOCK_MAP = 65.0     # mmHg
    THRESHOLD_SHOCK_LAC = 2.0      # mmol/L


class OODGuardian:
    """
    SOTA OOD Guardian (NeurIPS 2024 OGSRL style).
    Scans generated trajectories for:
    1.  Stitching Errors (Discontinuity from history).
    2.  Dynamics Violations (Impossible jumps).
    3.  Physiological Bounds (Hallucinations).
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cfg = SafetyConfig()
        self._warned_normalized = False  # [FIX: v14.0] Flag for "warn once" pattern

    def _is_normalized(self, tensor: torch.Tensor) -> bool:
        """
        Heuristic: Detects if data is in [-1, 1] or [0, 1] range.
        Clinical SBP is 80-180. Normalized SBP is 0.1-0.5.
        
        [FIX: v12.0] Check SBP max and also the range across all features.
        Clinical data has high variance and large absolute values.
        """
        if tensor.shape[-1] > IDX_SBP:
            sbp_max = tensor[..., IDX_SBP].max().item()
            # If SBP is < 10.0, it is 100% normalized (nobody lives with SBP 10)
            if sbp_max < 10.0: return True
            
            # If SBP is < 30.0 but variance is very small, likely normalized
            sbp_std = tensor[..., IDX_SBP].std().item()
            if sbp_max < 30.0 and sbp_std < 1.0: return True
            
        return False

    @torch.no_grad()
    def check_trajectories(
        self, 
        past_vitals: torch.Tensor, 
        pred_vitals: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Scans generated trajectories for medical anomalies.
        
        Args:
            past_vitals: [B, T_hist, D] (Observed History)
            pred_vitals: [B, T_pred, D] (Generated Future)
            src_mask: [B, T_hist] Optional mask for history
            
        Returns:
            Dict containing boolean OOD masks and continuous severity scores.
        """
        B, T_p, D = pred_vitals.shape
        device = pred_vitals.device
        
        # --- 0. Unit Consistency Guard ---
        if self._is_normalized(pred_vitals):
            if self.verbose and not self._warned_normalized:
                logger.warning("[OODGuardian] Input detected as NORMALIZED. Skipping absolute threshold checks.")
                self._warned_normalized = True

            return {
                "ood_mask": torch.zeros(B, dtype=torch.bool, device=device),
                "ood_rate": torch.tensor(0.0, device=device),
                "sbp_delta_mean": torch.tensor(0.0, device=device),
                "lac_max_mean": torch.tensor(0.0, device=device),
                "stitching_error": torch.tensor(0.0, device=device)
            }

        # --- 1. Stitching Check (Discontinuity) ---
        # "Did the model ignore the last observed state?"
        # [FIX: v12.0] Mask-Aware Last Observation
        if src_mask is not None:
             m = src_mask if src_mask.dim() == 2 else src_mask.any(dim=-1)
             last_obs = torch.zeros(B, D, device=device)
             for b in range(B):
                 valid_idx = torch.where(m[b])[0]
                 if len(valid_idx) > 0:
                     last_obs[b] = past_vitals[b, valid_idx[-1]]
                 else:
                     last_obs[b] = past_vitals[b, -1] # Fallback
        else:
            last_obs = past_vitals[:, -1, :]
            
        first_pred = pred_vitals[:, 0, :]
        
        stitch_diff = torch.abs(first_pred - last_obs)
        
        # HR Jump > 50, SBP Jump > 40
        stitch_err_hr = stitch_diff[:, IDX_HR] > self.cfg.MAX_DELTA_HR
        stitch_err_sbp = stitch_diff[:, IDX_SBP] > self.cfg.MAX_DELTA_SBP
        
        # --- 2. Dynamics Check (Intra-Trajectory Volatility) ---
        # "Is the trajectory physically jagged?"
        # Calculate max step-to-step delta within prediction
        if T_p > 1:
            deltas = torch.abs(pred_vitals[:, 1:] - pred_vitals[:, :-1])
            max_delta_sbp = deltas[..., IDX_SBP].max(dim=1)[0]
            max_delta_hr = deltas[..., IDX_HR].max(dim=1)[0]
        else:
            max_delta_sbp = torch.zeros(B, device=device)
            max_delta_hr = torch.zeros(B, device=device)
            
        is_jagged_sbp = max_delta_sbp > self.cfg.MAX_DELTA_SBP
        is_jagged_hr = max_delta_hr > self.cfg.MAX_DELTA_HR
        
        # --- 3. Boundary Check (Hallucination) ---
        # "Is the patient legally dead or exploding?"
        # Check min/max bounds across the horizon
        pred_sbp = pred_vitals[..., IDX_SBP]
        pred_map = pred_vitals[..., IDX_MAP]
        pred_o2 = pred_vitals[..., IDX_O2]
        pred_lac = pred_vitals[..., IDX_LAC]
        
        # [v14.1] Tolerance for Early Training Noise
        # Diffusion models often produce micro-noise (e.g. -0.01) around zero.
        # Strict < 0 checks flag these as "Impossible Physics", destroying valid batches.
        # [FIX] Add tolerance for Diffusion Noise. -0.001 is clinically 0.0.
        TOLERANCE = 1e-2

        is_ood_bounds = (
            (pred_sbp < self.cfg.BOUNDS_SBP[0] - TOLERANCE) | (pred_sbp > self.cfg.BOUNDS_SBP[1] + TOLERANCE) |
            (pred_map < self.cfg.BOUNDS_MAP[0] - TOLERANCE) | (pred_map > self.cfg.BOUNDS_MAP[1] + TOLERANCE) |
            (pred_o2 < self.cfg.BOUNDS_O2[0] - TOLERANCE)   | (pred_o2 > self.cfg.BOUNDS_O2[1] + TOLERANCE)   |
            (pred_lac < self.cfg.BOUNDS_LAC[0] - TOLERANCE) # Lactate < 0 often happens with noise
        ).any(dim=1) # [B]
        
        # --- 4. Sepsis-3 Specific OOD (Extreme Hyperlactatemia) ---
        # Lactate > 12.0 is usually pre-terminal
        max_lac = pred_lac.max(dim=1)[0]
        is_critical_lac = max_lac > 12.0
        
        # --- 5. Aggregation ---
        # Union of all failure modes
        ood_mask = (
            stitch_err_hr | stitch_err_sbp | 
            is_jagged_sbp | is_jagged_hr | 
            is_ood_bounds | is_critical_lac
        )
        
        return {
            "ood_mask": ood_mask,                       # [B] Boolean
            "ood_rate": ood_mask.float().mean(),        # Scalar
            "safe_count": (1.0 - ood_mask.float()).sum(), # [FIX] Scalar for metrics
            "stitching_error": stitch_diff.mean(),      # Scalar (Average discontinuity)
            "sbp_delta_mean": max_delta_sbp.mean(),     # Scalar
            "lac_max_mean": max_lac.mean(),             # Scalar
            "is_jagged": (is_jagged_sbp | is_jagged_hr).float().mean()
        }


def compute_sepsis3_violations(vitals: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Detects Sepsis-3 (2016) clinical violations.
    Used for Safety-Cost constraints in training (Differentiable).
    
    Definition of Septic Shock (Sepsis-3):
        1. Persisting Hypotension requiring vasopressors to maintain MAP >= 65 mmHg.
        2. Serum Lactate level > 2 mmol/L.
    
    We model the "Violation Cost" as the presence of these conditions.
    
    Args:
        vitals: [B, T, D] Tensor (Must be in Clinical Units).
        src_mask: [B, T] Optional validity mask.
        
    Returns:
        safety_cost: [B, T] Tensor (Higher = Worse).
    """
    # 1. Extract Channels (Canonical: MAP=4, Lactate=7)
    map_val = vitals[..., IDX_MAP]
    lactate = vitals[..., IDX_LAC]
    
    # 2. Hypotension Penalty (MAP < 65)
    # Using Soft Sigmoid for gradient flow:
    # If MAP = 65 -> 0.5. If MAP = 40 -> ~1.0. If MAP = 90 -> ~0.0.
    # Steepness divisor 5.0 ensures gradients exist around the threshold.
    v_map = torch.sigmoid((SafetyConfig.THRESHOLD_SHOCK_MAP - map_val) / 5.0)
    
    # 3. Metabolic Failure Penalty (Lactate > 2.0)
    # If Lactate = 2.0 -> 0.5. If Lactate = 6.0 -> ~1.0.
    v_lac = torch.sigmoid((lactate - SafetyConfig.THRESHOLD_SHOCK_LAC) / 1.0)
    
    # 4. Combined Safety Cost
    # "Soft OR" Logic: We want to penalize EITHER condition, but 
    # the combination is Septic Shock (worst case).
    # Since we want to prevent the *onset* of either, additive cost works best
    # for gradient descent (gradients flow from both terms independently).
    safety_cost = (v_map + v_lac)
    
    # 5. Apply Masking
    if src_mask is not None:
        # [FIX: v13.0] Zero out cost for padded regions
        if src_mask.dim() == 2:
            safety_cost = safety_cost * src_mask.float()
        else:
             safety_cost = safety_cost * src_mask.any(dim=-1).float()
    
    return safety_cost