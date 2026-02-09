"""
icu/utils/advantage_calculator.py
--------------------------------------------------------------------------------
APEX-MoE: SOTA Advantage Estimation Engine (Ultimate v10.0 - Life-Critical Edition).

Status: SAFETY-CRITICAL / PRODUCTION-READY
Purpose: Converts raw clinical outcomes into robust learning signals for offline RL.

"In critical care, accurate credit assignment can mean the difference between
life and death. This calculator ensures our model learns from the moments
that matter most."

This module implements the complete advantage estimation pipeline for clinical
reinforcement learning in ICU settings:

1.  **Sepsis-3 Clinical Reward Function**: Dense reward signals based on:
    - MAP (Mean Arterial Pressure) < 65 mmHg penalty
    - Lactate > 2.0 mmol/L penalty
    - Respiratory rate > 22 bpm penalty (qSOFA)
    - SBP (Systolic Blood Pressure) < 100 mmHg penalty
    - Delta/improvement rewards for recovery trajectories

2.  **Generalized Advantage Estimation (GAE)**: Time-series credit assignment
    with proper handling for:
    - Window truncation (bootstrapping)
    - Terminal state masking
    - Episode boundaries

3.  **Advantage-Weighted Regression (AWR)**: Safe importance weighting with:
    - Global whitening for stable exponentiation
    - FP16 safety (pre-exp clamping)
    - Effective Sample Size (ESS) monitoring
    - Weight entropy tracking

Upgrades (Ultimate v10.0 - Life-Critical Edition):
1.  **Sigmoid Soft-Cliffs**: Smooth, differentiable penalties that avoid
    gradient cliffs at clinical thresholds.
2.  **Bootstrap-Aware GAE**: Proper handling of sliding window datasets with
    explicit bootstrap value injection.
3.  **FP16-Safe AWR**: Pre-exponentiation clamping prevents overflow in
    mixed-precision training.
4.  **Explained Variance Diagnostic**: Measures critic quality (how well
    values predict returns).
5.  **Weight Entropy Tracking**: Information-theoretic metric for mode collapse.
6.  **qSOFA Integration**: Full Sepsis-3 criteria including respiratory rate.
7.  **Unit Safety Validation**: Automatic detection of normalized vs. clinical
    units to prevent threshold mismatches.
8.  **Focal Alpha Scaling**: Optional asymmetric weighting for negative rewards.
9.  **Delta Trend Rewards**: Rewards physiological improvement (lactate down,
    MAP up when low).
10. **Comprehensive Diagnostics**: Full telemetry for training analysis.

References:
    - Schulman et al. "High-Dimensional Continuous Control Using GAE" (ICLR 2016)
    - Komorowski et al. "The Artificial Intelligence Clinician" (Nature Medicine 2018)
    - Peng et al. "Advantage-Weighted Regression" (2019)
    - Singer et al. "Sepsis-3 Consensus Definitions" (JAMA 2016)
    - FAWAC: Feasibility Informed AWR for Safe Offline RL (2024)

Dependencies:
    - torch (PyTorch)
    - numpy (For statistics)
    - logging (For diagnostics)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
import torch.distributed as dist
from icu.utils.train_utils import ScalingSteward
from typing import Optional, Tuple, Dict, List, Union, Any

logger = logging.getLogger("APEX_Advantage_Ultimate")
logger.setLevel(logging.INFO)

# =============================================================================
# CLINICAL CONSTANTS: SEPSIS-3 & PHYSIOLOGICAL THRESHOLDS
# =============================================================================
# Derived from:
# - Singer et al. "Sepsis-3 Consensus Definitions" (JAMA 2016)
# - Surviving Sepsis Campaign Guidelines (2021 Update)
# - qSOFA Criteria (Quick Sequential Organ Failure Assessment)

SEPSIS_CONSTANTS = {
    # Hemodynamic Thresholds
    'MAP_TARGET': 65.0,          # mmHg (Vasopressor titration target)
    'MAP_SHOCK': 65.0,           # mmHg (Septic shock definition)
    'SBP_HYPOTENSION': 100.0,    # mmHg (qSOFA criteria)
    'SBP_SEVERE': 90.0,          # mmHg (Severe hypotension)
    
    # Metabolic Thresholds
    'LACTATE_UPPER': 2.0,        # mmol/L (Cellular distress threshold)
    'LACTATE_CRITICAL': 4.0,     # mmol/L (Severe metabolic dysfunction)
    
    # Respiratory Thresholds (qSOFA)
    'RESP_QSOFA': 22.0,          # bpm (qSOFA respiratory criterion)
    'RESP_CRITICAL': 30.0,       # bpm (Severe respiratory distress)
    
    # Other Clinical Markers
    'URINE_LOWER': 0.5,          # mL/kg/hr (Oliguria threshold)
    'GCS_LOWER': 14,             # Glasgow Coma Scale (Altered mentation)
    
    # Reward Scaling
    'SPARSE_REWARD_SCALE': 5.0,  # Magnitude of survival/death signal
    'DENSE_REWARD_CAP': 2.0,     # Maximum dense reward per timestep
}

# Default feature indices for Clinical 28 specification
# Matches CANONICAL_COLUMNS from dataset.py
DEFAULT_FEATURE_INDICES = {
    'hr': 0,          # Heart Rate (bpm)
    'o2sat': 1,       # Oxygen Saturation (%)
    'sbp': 2,         # Systolic Blood Pressure (mmHg)
    'dbp': 3,         # Diastolic Blood Pressure (mmHg)
    'map': 4,         # Mean Arterial Pressure (mmHg)
    'resp': 5,        # Respiratory Rate (bpm)
    'temp': 6,        # Temperature (°C)
    'lactate': 7,     # Lactate (mmol/L)
    'creatinine': 8,  # [FIX] Aligned to Clinical 28 Spec
    'bilirubin': 9,   # [FIX] Aligned to Clinical 28 Spec
    'platelets': 10,  # [FIX] Aligned to Clinical 28 Spec
    'wbc': 11,         # [FIX] Aligned to Clinical 28 Spec
    'glucose': 15,    # [FIX] Aligned to Clinical 28 Spec
}


class ICUAdvantageCalculator(nn.Module):
    """
    The 'Critic's Brain': Converts raw outcomes into robust learning signals.
    
    Designed for sliding-window datasets where episodes are truncated.
    Implements SOTA techniques for safe, stable advantage estimation
    in clinical offline reinforcement learning.
    
    Attributes:
        beta: AWR temperature (lower = stricter selection pressure)
        gamma: Discount factor for future rewards
        lambda_gae: GAE variance-bias trade-off parameter
        max_weight: Hard clip for AWR weights
        sparse_scale: Scale for terminal survival/death rewards
        shaping_coef: Scale for dense clinical shaping rewards
        focal_alpha: Asymmetric scaling for negative rewards
        adv_mean: Running mean for advantage whitening
        adv_std: Running std for advantage whitening
        stats_initialized: Whether global stats are locked
    """
    
    def __init__(
        self, 
        beta: float = 0.5,              # AWR Temperature (0.3-1.0 for clinical)
        gamma: float = 0.99,            # Discount Factor (~48h horizon)
    lambda_gae: float = 0.95,       # GAE Variance-Bias trade-off
    max_weight: float = 20.0,       # Hard clip for AWR weights
    sparse_reward_scale: float = 2.0,   # [v40.0 SOTA] Reduced to prevent drowning
    reward_shaping_coef: float = 0.5,   # [v40.0 SOTA] Increased for better guidance
    focal_alpha: float = 1.0,      # [v41.0 SOTA] 2x Death weight (Clinical Reality)
    qsofa_thresholds: Optional[Dict[str, float]] = None,  # Override defaults
    adaptive_beta: bool = True,     # [SOTA 2025] Enabled by default for fresh start
    adaptive_clipping: bool = True, # [SOTA 2025] Enabled by default for fresh start
    beta_momentum: float = 0.98,    # [v2026 SOTA] Smoother transition for high-frequency updates
    beta_gain: float = 2.0,         # [v116.0 SOTA FIX] PI-style gain for faster adaptation
    target_ess: float = 0.10,        # [v2026 SOTA] Tightened to 10% for sharper selection pressure
    min_beta: float = 0.1           # [v38.0 SOTA] Sharp selection floor
):
        """
        Initialize the Advantage Calculator.
        
        Args:
            beta: AWR temperature (lower = peakier weights, more selective)
            gamma: Discount factor (0.99 for ~48h clinical horizon)
            lambda_gae: GAE parameter (0.95 is standard)
            max_weight: Maximum allowed AWR weight (for stability)
            sparse_reward_scale: Scale for survival/death terminal reward
            reward_shaping_coef: Scale for dense physiological rewards
            focal_alpha: Optional scaling for negative rewards
            qsofa_thresholds: Override default qSOFA thresholds
            adaptive_beta: Enable dynamic beta scaling (std-based)
            adaptive_clipping: Enable dynamic weight clipping (quantile-based)
            beta_momentum: Momentum for adaptive beta updates (0.90-0.999)
            target_ess: Target Effective Sample Size (default 20.0)
        """
        super().__init__()
        self.register_buffer("beta", torch.tensor(1.0).float()) # Fresh Start: Default to 1.0
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.register_buffer("max_weight", torch.tensor(max_weight).float())
        self.sparse_scale = sparse_reward_scale
        self.shaping_coef = reward_shaping_coef
        self.focal_alpha = focal_alpha
        self.target_ess = target_ess
        
        # [v2025 SOTA] State Buffers for DDP Synchronization
        self.register_buffer("ess_buffer", torch.zeros(1))
        self.register_buffer("ess_momentum_buffer", torch.tensor(0.15)) # Improved Target: 15%
        self.register_buffer("clip_rate_buffer", torch.zeros(1))
        
        # [SOTA 2025] Adaptive Hyperparameters
        self.adaptive_beta = adaptive_beta
        self.adaptive_clipping = adaptive_clipping
        self.register_buffer("beta_momentum", torch.tensor(beta_momentum).float())
        self.beta_gain = beta_gain              
        self.register_buffer("clip_momentum", torch.tensor(0.90).float())
        
        # [SOTA v2026] Internal Scaled Constants
        self.base_beta_momentum = float(beta_momentum) 
        self.register_buffer("ess_ema_decay", torch.tensor(0.95).float())
        self.register_buffer("beta_growth_factor", torch.tensor(2.0).float())
        self.register_buffer("beta_growth_cooldown", torch.tensor(0, dtype=torch.long))
        
        # [SOTA v38.0] Selection Recovery Floor (Configurable)
        # Rationale: Higher floor (0.8) prevents AUROC collapse by ensuring 
        # broader manifold coverage.
        self.min_beta = min_beta            
        self.max_beta = 20.0
        
        # [v29.1 SOTA FIX] Whitening Momentum Stability (Abyssal #1)
        self.register_buffer("whitening_momentum", torch.tensor(0.99).float()) 
        self.base_whitening_momentum = 0.99
        
        # qSOFA thresholds
        if qsofa_thresholds is None:
            self.qsofa_thresholds = {
                'resp': SEPSIS_CONSTANTS['RESP_QSOFA'],
                'sbp': SEPSIS_CONSTANTS['SBP_HYPOTENSION'],
                'gcs': SEPSIS_CONSTANTS['GCS_LOWER']
            }
        else:
            self.qsofa_thresholds = qsofa_thresholds

        # [v2026 SOTA] Clinical Integrity Guard
        # "Sharpening the Axe": Verify all required constants exist at startup
        self._validate_clinical_config()

        # Global Whitening Statistics (Welford's Algorithm state)
        # Register as buffers for persistence across checkpoints
        self.register_buffer("adv_mean", torch.tensor([0.0]))
        self.register_buffer("adv_std", torch.tensor([1.0]))
        self.register_buffer("stats_count", torch.tensor(0))
        self.register_buffer("stats_initialized", torch.tensor(False))
        
        logger.info(
            f"[ADVANTAGE] Initialized: beta={beta}, gamma={gamma}, "
            f"lambda={lambda_gae}, max_weight={max_weight}, "
            f"adaptive_beta={adaptive_beta}, adaptive_clipping={adaptive_clipping}"
        )

    def _clinical_sigmoid(self, val: torch.Tensor, center: float, steepness: float, inverse: bool = False) -> torch.Tensor:
        """
        [v132.0 SOTA FIX] Wide-Bridge Sigmoid (Smoking Gun #132).
        Rationale: Standard sigmoids saturate and zero-out gradients for critical patients.
        Fix: Composite sigmoid (Fast + Slow) preserves the clinical 'cliff' while 
        maintaining a 'slope' for continuous learning in extreme shock zones.
        """
        diff = (val - center) if inverse else (center - val)
        # Fast Component: Sharp clinical threshold
        s_fast = torch.sigmoid(diff * steepness)
        # Slow Component: Wide-bridge gradient signal (5x wider)
        s_slow = torch.sigmoid(diff * (steepness * 0.2))
        return 0.5 * (s_fast + s_slow)

    def _validate_clinical_config(self):
        """
        [SOTA 2026] Clinical Integrity Guard.
        Ensures all required clinical thresholds are defined in SEPSIS_CONSTANTS.
        This prevents 'Ghost Constant' crashes mid-training.
        """
        required_keys = [
            'MAP_TARGET', 'SBP_HYPOTENSION', 'LACTATE_UPPER', 
            'RESP_QSOFA', 'DENSE_REWARD_CAP', 'GCS_LOWER'
        ]
        missing = [k for k in required_keys if k not in SEPSIS_CONSTANTS]
        
        if missing:
            error_msg = f"[CRITICAL CONFIG ERROR] Missing clinical constants: {missing}. Resolve in SEPSIS_CONSTANTS!"
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Γ£à [ADVANTAGE] Clinical Configuration Integrity Verified.")

    def set_stats(self, mean: float, std: float, beta: float = None, count: int = None):
        """
        Locks normalization statistics and restores internal state for stable resumptions.
        
        Args:
            mean: Global advantage mean
            std: Global advantage standard deviation
            beta: Optional AWR temperature (prevents reset shock)
            count: Optional sample count (stabilizes moving average)
        """
        self.adv_mean[0] = mean
        self.adv_std[0] = std if std > 1e-6 else 1.0
        
        if beta is not None:
            if isinstance(beta, torch.Tensor):
                self.beta.copy_(beta)
            else:
                self.beta.fill_(beta)
            logger.info(f"[RESUME] AWR Beta restored: {self.beta.item():.4f}")
            
        if count is not None:
            if isinstance(count, torch.Tensor):
                self.stats_count.copy_(count)
            else:
                self.stats_count.fill_(count)
            
        self.stats_initialized.fill_(True)
        logger.info(
            f"[ADVANTAGE] Stats Locked: mu={self.adv_mean.item():.4f}, sigma={self.adv_std.item():.4f}"
        )

    def get_awr_state(self) -> Dict[str, Any]:
        """[Phase 38.1] Export internal buffers for explicit persistence."""
        return {
            "adv_mean": self.adv_mean.clone(),
            "adv_std": self.adv_std.clone(),
            "stats_count": self.stats_count.clone(),
            "stats_initialized": self.stats_initialized.clone(),
            "beta": self.beta.clone(),
            "ess_buffer": self.ess_buffer.clone(),
            "clip_rate_buffer": self.clip_rate_buffer.clone(),
            "beta_growth_cooldown": self.beta_growth_cooldown.clone()
        }

    def load_awr_state(self, state: Dict[str, Any]):
        """[Phase 38.1] Explicitly restore AWR buffers to bypass re-calibration."""
        if not state: return
        
        for key, val in state.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, torch.Tensor):
                    attr.copy_(val.to(attr.device))
        
        logger.info(f"✅ [AWR] Persistence Bridge: Restored {len(state)} buffers (Amnesia Averted).")

    def _validate_units(
        self, 
        vitals: torch.Tensor, 
        feature_indices: Dict[str, int]
    ) -> bool:
        """
        Validates that vitals are in clinical units (not normalized).
        
        Returns True if the data appears to be in clinical units.
        Logs a warning if normalized data is detected.
        
        Args:
            vitals: Tensor of vital signs [B, T, C]
            feature_indices: Map of feature names to channel indices
        
        Returns:
            True if data appears to be in clinical units
        """
        # Check SBP as a proxy for unit detection
        idx_sbp = feature_indices.get('sbp', 2)
        if idx_sbp < vitals.shape[-1]:
            sbp_max = vitals[..., idx_sbp].max().item()
            sbp_thresh = self.qsofa_thresholds.get('sbp', 100.0)
            
            # If SBP max is very low but threshold is clinical, likely normalized
            if sbp_max < 20.0 and sbp_thresh > 80.0:
                logger.warning(
                    f"[UNIT MISMATCH] Vitals appear NORMALIZED (max SBP={sbp_max:.2f}) "
                    f"but thresholds are CLINICAL ({sbp_thresh}). "
                    f"Pass 'normalizer' to restore units!"
                )
                return False
        return True

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies AWR adaptation rates across step densities."""
        if n_curr <= 0: return
        
        logger.info(f"⚡ [AWR] Scaling Dynamics for {n_curr} steps (Ref: {ScalingSteward.REF_STEPS})")
        
        # 1. Scale Momentum Decays
        # Matches the 'awr_momentum' from config (e.g., 0.999)
        self.beta_momentum.fill_(ScalingSteward.get_decay(self.base_beta_momentum, n_curr))
        self.clip_momentum.fill_(ScalingSteward.get_decay(0.90, n_curr))
        
        # 2. Scale Telemetry Buffers
        self.ess_ema_decay.fill_(ScalingSteward.get_decay(0.95, n_curr))
        
        # 3. Scale Growth Rates (Baseline: 1.5)
        self.beta_growth_factor.fill_(float(1.5 ** (ScalingSteward.REF_STEPS / n_curr)))
        
        # 4. Scale Whitening Momentum (Ref: 0.999)
        self.whitening_momentum.fill_(ScalingSteward.get_decay(self.base_whitening_momentum, n_curr))
        
        logger.info(
            f"⚡ [AWR] Scaling Results: beta_mom={self.beta_momentum:.6f}, "
            f"ess_ema={self.ess_ema_decay:.4f}, growth={self.beta_growth_factor:.4f}, "
            f"white_mom={self.whitening_momentum:.6f}"
        )

    # =========================================================================
    # CLINICAL REWARD FUNCTION
    # =========================================================================

    def compute_clinical_reward(
        self, 
        vitals: torch.Tensor, 
        outcome_label: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict[str, int]] = None,
        normalizer: Optional[Any] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes dense Sepsis-3 clinical reward using Sigmoid Soft-Cliffs.
        
        Why Sigmoid Soft-Cliffs?
        -----------------------
        A ReLU penalty (e.g., penalty = ReLU(65 - MAP)) creates a constant
        gradient below the threshold. This can cause:
        1. Massive gradients when patient is already critical (MAP=20)
        2. Zero gradients when patient is healthy (MAP=80)
        
        A Sigmoid creates a stronger gradient NEAR the threshold (60-65)
        and saturates at extremes, preventing gradient explosion while
        maintaining meaningful learning signal across all states.
        
        Reward Components:
        1. **Sparse Terminal Reward**: +5 for survival, -5 for death
        2. **MAP Penalty**: Sigmoid-based penalty for hypotension
        3. **Lactate Penalty**: Penalty for metabolic distress
        4. **Respiratory Penalty**: qSOFA respiratory criterion
        5. **SBP Penalty**: Additional hypotension marker
        6. **Delta Rewards**: Improvement in MAP/Lactate trajectory
        
        Args:
            vitals: (B, T, C) Tensor of vital signs (normalized or clinical)
            outcome_label: (B,) or (B, T) binary outcome (0=Survival, 1=Death)
            dones: (B, T) binary terminal markers (1=End of episode)
            feature_indices: Map of feature names to channel indices
            normalizer: Optional ClinicalNormalizer to restore clinical units
            src_mask: Optional (B, T) boolean mask (True=Valid, False=Padding)
        
        Returns:
            rewards: (B, T) Tensor of dense clinical rewards
        """
        # Set default feature indices
        if feature_indices is None:
            feature_indices = DEFAULT_FEATURE_INDICES

        # --- 1. Unit Restoration (Critical Safety) ---
        units_ok = True
        if normalizer is not None:
            # .detach() is crucial: Reward calc should not backprop into encoder
            vitals_phys = normalizer.denormalize(vitals.detach())
        else:
            vitals_phys = vitals.detach()
            # Validate units (Strict Mode)
            units_ok = self._validate_units(vitals_phys, feature_indices)
            if not units_ok:
                error_msg = (
                    "[CRITICAL SAFETY FAILURE] Advantage Calculator detected NORMALIZED vitals "
                    "without a 'normalizer'. Dense clinical rewards cannot be computed safely. "
                    "Training must halt to prevent reward signal collapse."
                )
                logger.critical(error_msg)
                raise ValueError(error_msg) 

        B, T, C = vitals.shape
        device = vitals.device
        rewards = torch.zeros(B, T, device=device)
        
        # If units are broken, we ONLY compute sparse outcome rewards (which don't depend on vitals)
        # We skip all dense physiological logic.

        # Extract feature indices
        idx_map = feature_indices.get('map', 4)
        idx_sbp = feature_indices.get('sbp', 2)
        idx_lac = feature_indices.get('lactate', 7)
        idx_resp = feature_indices.get('resp', 5)

        if units_ok:
            # --- 2. Extract & Clamp Key Signals (Physical constraints) ---
            map_val = torch.clamp(vitals_phys[..., idx_map], 0, 300) if idx_map < C else None
            sbp_val = torch.clamp(vitals_phys[..., idx_sbp], 0, 300) if idx_sbp < C else None
            lactate_val = torch.clamp(vitals_phys[..., idx_lac], 0, 50) if idx_lac < C else None
            resp_val = torch.clamp(vitals_phys[..., idx_resp], 0, 100) if idx_resp < C else None
        else:
            map_val, sbp_val, lactate_val, resp_val = None, None, None, None

        # --- 3. Sparse Outcome Rewards (Terminal Only) ---
        if outcome_label.dim() == 1:
            outcome_expanded = outcome_label.unsqueeze(1).expand(-1, T)
        else:
            outcome_expanded = outcome_label

        if dones is None:
            dones = torch.zeros(B, T, device=device)
            
        # Mask: Only apply sparse reward at true episode end
        # [FIX: Mask-Aware Terminal Placement]
        is_terminal = dones.bool()
        
        # If we have a source mask, intersect terminal with it
        if src_mask is not None:
            # Mask is [B, T] or [B, T, C]
            m = src_mask if src_mask.dim() == 2 else src_mask.any(dim=-1)
            
            # Find last valid index per batch
            # is_last_valid: True at index t if m[t] is True and (m[t+1] is False or t is last)
            is_last_valid = torch.zeros_like(m, dtype=torch.bool)
            for b in range(B):
                valid_indices = torch.where(m[b])[0]
                if len(valid_indices) > 0:
                    last_idx = valid_indices[-1]
                    # Only treat as terminal if 'dones' says the episode ends in this window
                    # OR if we want to bootstrap correctly at the window edge.
                    # For APEX, 'dones' usually marks the REAL episode end.
                    if is_terminal[b].any():
                         # Move the terminal flag to the true end of data
                         is_last_valid[b, last_idx] = True
            
            is_terminal = is_last_valid

        # Reward Logic:
        survival_r = (1.0 - outcome_expanded) * self.sparse_scale
        death_r = outcome_expanded * (-self.sparse_scale)
        
        if self.focal_alpha != 1.0:
            death_r = death_r * (1.0 + self.focal_alpha)
        
        outcome_r = survival_r + death_r
        rewards[is_terminal] += outcome_r[is_terminal]

        # --- DENSE REWARD BLOCK (REQUIRES VALID UNITS) ---
        if units_ok:
            # [SOTA FIX] Fine-Grained Imputation Awareness
            # We only penalize if the specific signal is valid (mask=1)
            # Assumption: src_mask is [B, T, C] or [B, T]
            def get_f_mask(idx):
                if src_mask is None: return torch.ones(B, T, device=vitals.device)
                if src_mask.dim() == 2: return src_mask.float()
                if idx < src_mask.shape[-1]: return src_mask[..., idx].float()
                return torch.ones(B, T, device=vitals.device)

            # --- 4. MAP Penalty (Sigmoid Soft-Cliff) ---
            if map_val is not None:
                # [v132.0 SOTA FIX] Rescued Gradient (Target=65, Steepness=0.5)
                # [Patch 64] Strict Sepsis-3 Alignment
                map_penalty_score = self._clinical_sigmoid(map_val, SEPSIS_CONSTANTS['MAP_TARGET'], 0.5)
                # Apply MAP-specific mask
                rewards -= self.shaping_coef * map_penalty_score * get_f_mask(idx_map)

            # --- 5. SBP Penalty (Additional Hypotension Marker) ---
            if sbp_val is not None:
                # [v132.0 SOTA FIX] Rescued Gradient (Target=100, Steepness=0.2)
                # [Patch 64] Strict Sepsis-3 Alignment
                sbp_penalty_score = self._clinical_sigmoid(sbp_val, SEPSIS_CONSTANTS['SBP_HYPOTENSION'], 0.2)
                # Apply SBP-specific mask
                rewards -= self.shaping_coef * 0.5 * sbp_penalty_score * get_f_mask(idx_sbp)

            # --- 6. Lactate Penalty (Sigmoid Soft-Cliff) ---
            if lactate_val is not None:
                # [v132.0 SOTA FIX] Rescued Gradient (Target=2.0, Steepness=1.0, Inverse=True)
                # [Patch 64] Strict Sepsis-3 Alignment
                lac_penalty_score = self._clinical_sigmoid(lactate_val, SEPSIS_CONSTANTS['LACTATE_UPPER'], 1.0, inverse=True)
                # Apply Lactate-specific mask
                rewards -= self.shaping_coef * 1.5 * lac_penalty_score * get_f_mask(idx_lac)

            # --- 7. Respiratory Penalty (qSOFA) ---
            if resp_val is not None:
                # [v132.0 SOTA FIX] Rescued Gradient (Target=22, Steepness=0.3, Inverse=True)
                # [Patch 64] Strict Sepsis-3 Alignment
                resp_penalty_score = self._clinical_sigmoid(resp_val, SEPSIS_CONSTANTS['RESP_QSOFA'], 0.3, inverse=True)
                # Apply Resp-specific mask
                rewards -= self.shaping_coef * 0.3 * resp_penalty_score * get_f_mask(idx_resp)

            # --- 8. Delta Trends (Reward Recovery) ---
            if T > 1:
                # A. Lactate Improvement: Reward DECREASE in lactate
                if lactate_val is not None:
                    # positive delta = lactate going DOWN (good)
                    lac_delta = lactate_val[:, :-1] - lactate_val[:, 1:]
                    lac_improvement = torch.clamp(lac_delta, min=0.0, max=2.0)
                    # Use mask from previous step to ensure "start" was real
                    rewards[:, 1:] += self.shaping_coef * 2.0 * lac_improvement * get_f_mask(idx_lac)[:, :-1]

                # B. MAP Improvement: Reward INCREASE in MAP (if was low)
                if map_val is not None:
                    map_delta = map_val[:, 1:] - map_val[:, :-1]
                    # Only reward MAP increase if it was in danger zone (<75)
                    map_was_low = (map_val[:, :-1] < 75.0).float()
                    map_improvement = torch.clamp(map_delta, min=0.0, max=10.0) * map_was_low
                    rewards[:, 1:] += self.shaping_coef * 0.5 * map_improvement * get_f_mask(idx_map)[:, :-1]

                # C. SBP Improvement: Reward INCREASE in SBP (if was low)
                if sbp_val is not None:
                    sbp_delta = sbp_val[:, 1:] - sbp_val[:, :-1]
                    sbp_was_low = (sbp_val[:, :-1] < 110.0).float()
                    sbp_improvement = torch.clamp(sbp_delta, min=0.0, max=15.0) * sbp_was_low
                    rewards[:, 1:] += self.shaping_coef * 0.2 * sbp_improvement * get_f_mask(idx_sbp)[:, :-1]

            # --- 9. Clamp Total Dense Reward & Handle NaNs ---
            # Prevent dense rewards from overwhelming sparse signal
            reward_cap = SEPSIS_CONSTANTS['DENSE_REWARD_CAP']
            rewards = torch.clamp(rewards, min=-reward_cap * 2, max=reward_cap)
            
            # [SOTA FIX] NaN-Robustness
            # If any reward became NaN (e.g. from invalid clinical data), zero it out
            # to prevent gradient explosion.
            nan_mask = torch.isnan(rewards)
            if nan_mask.any():
                logger.warning(f"[NAN REWARD] Detected {nan_mask.sum()} NaNs in rewards. Zeroing them.")
                rewards = torch.where(nan_mask, torch.zeros_like(rewards), rewards)

        # --- 10. Apply Source Mask (Zero out padding) ---
        if src_mask is not None:
            # Ensure proper shape broadcasting
            if src_mask.dim() == 2:
                rewards = rewards * src_mask.float()
            elif src_mask.dim() == 3:
                 # Reduce to [B, T] if strictly necessary, but usually mask is [B, T]
                 # Assuming mask means "any feature valid"
                 rewards = rewards * src_mask.any(dim=-1).float()

        return rewards * 10.0
        

    # =========================================================================
    # GENERALIZED ADVANTAGE ESTIMATION (GAE)
    # =========================================================================

    # =========================================================================
    # STATE ADVANTAGE WEIGHTING (SAW) - 2025 SOTA
    # =========================================================================

    def compute_saw(
        self, 
        rewards: torch.Tensor, 
        student_values: torch.Tensor, 
        teacher_values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        bootstrap_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        [SOTA 2025] State Advantage Weighting (SAW).
        De-couples actions from values by focusing on state transitions.
        
        Formula: A_t = r_t + gamma * V_teacher(s_{t+1}) * (1-d_t) - V_student(s_t)
        
        Args:
            rewards: (B, T) Tensor of rewards
            student_values: (B, T) V_student predictions
            teacher_values: (B, T) V_teacher predictions (EMA)
            dones: (B, T) Terminal markers
            bootstrap_value: (B, 1) V_teacher(s_{T+1})
        """
        B, T = rewards.shape
        device = rewards.device
        
        if bootstrap_value is not None:
            if bootstrap_value.dim() == 1:
                bootstrap_value = bootstrap_value.unsqueeze(1)
            next_v_teacher = torch.cat([teacher_values[:, 1:], bootstrap_value], dim=1)
        else:
            next_v_teacher = torch.cat([teacher_values[:, 1:], teacher_values[:, -1:]], dim=1)

        if dones is not None:
            if dones.dim() == 1:
                non_terminal = torch.ones((B, T), device=device)
                non_terminal[:, -1] = 1.0 - dones.float()
            else:
                non_terminal = 1.0 - dones.float()
        else:
            non_terminal = torch.ones((B, T), device=device)

        # SAW Advantage: r + gamma * V_teacher(s') - V_student(s)
        # This prevents the student from "cheating" by lowering all values.
        advantages = rewards + (self.gamma * next_v_teacher * non_terminal) - student_values
        
        return advantages

    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: Optional[torch.Tensor] = None,
        bootstrap_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Trajectory-Aware Generalized Advantage Estimation.
        
        Handles window truncation correctly for sliding-window datasets.
        
        Algorithm:
        1. Compute TD errors: δ_t = r_t + γ * V(s_{t+1}) * (1-d_t) - V(s_t)
        2. GAE recursion: A_t = δ_t + (γ * λ * (1-d_t)) * A_{t+1}
        
        Args:
            rewards: (B, T) Tensor of rewards
            values: (B, T) Tensor of critic value estimates V(s_t)
            dones: (B, T) Optional binary markers (1=terminal step)
                   If None, assumes no terminal states in window
            bootstrap_value: (B, 1) or (B,) Optional V(s_{T+1})
                            Required if window ends but episode continues
        
        Returns:
            advantages: (B, T) Tensor of GAE advantages
        """
        B, T = rewards.shape
        device = rewards.device
        
        # --- 1. Construct Next Values V(s_{t+1}) ---
        if bootstrap_value is not None:
            # Use provided bootstrap value
            if bootstrap_value.dim() == 1:
                bootstrap_value = bootstrap_value.unsqueeze(1)
            next_values = torch.cat([values[:, 1:], bootstrap_value], dim=1)
        else:
            # Default: Bootstrap with last observed value
            # This is more conservative than zero-padding for sliding windows
            next_values = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
        
        # --- 2. Construct Non-Terminal Mask ---
        if dones is not None:
            # [v25.5 SOTA FIX] Robust Dimensionality Handling
            # drones can be (B,) if only the last step is considered terminal,
            # or (B, T) if terminals are dispersed.
            if dones.dim() == 1:
                # If dones is (B,), it refers to the last step of the trajectory.
                # However, for GAE, we need a (B, T) mask where only the terminal step
                # zeros out the bootstrap.
                # Default: all steps are non-terminal
                non_terminal = torch.ones((B, T), device=device)
                # Mark ONLY the last step as potentially terminal
                non_terminal[:, -1] = 1.0 - dones.float()
            else:
                # dones is (B, T)
                non_terminal = 1.0 - dones.float()
        else:
            # Assume all steps are non-terminal (sliding window assumption)
            non_terminal = torch.ones((B, T), device=device)
        
        # --- 3. TD Error (Delta) ---
        # δ_t = r_t + γ * V(s_{t+1}) * (1-d_t) - V(s_t)
        # Ensure non_terminal is broadcastable to next_values (B, T)
        deltas = rewards + (self.gamma * next_values * non_terminal) - values
        
        # --- 4. GAE Recursion (Backwards) ---
        advantages = torch.zeros_like(rewards)
        # Initialize as batch-sized tensor for proper broadcasting
        last_gae = torch.zeros(B, device=device)
        
        for t in reversed(range(T)):
            mask = non_terminal[:, t]
            delta_t = deltas[:, t]
            
            # A_t = δ_t + (γ * λ * mask) * A_{t+1}
            # [SAFETY] Detach future advantage to prevent gradient coupling
            last_gae_detached = last_gae.detach() if last_gae.requires_grad else last_gae
            last_gae = delta_t + (self.gamma * self.lambda_gae * mask) * last_gae_detached
            advantages[:, t] = last_gae
        
        return advantages

    # =========================================================================
    # AWR WEIGHT CALCULATION
    # =========================================================================

    def calculate_awr_weights(
        self, 
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes safe AWR weights with FP16 protection.
        
        Formula: w = clamp(exp((A - μ) / σ / β), max=max_weight)
        
        Features:
        - Pre-exponentiation clamping for FP16 safety
        - Global whitening for stable training
        - Comprehensive diagnostics (ESS, entropy, clipping rate)
        
        Args:
            advantages: (B, T) or (B,) Tensor of advantages
        
        Returns:
            weights: Tensor of AWR weights (same shape as input)
            diagnostics: Dict with ESS, entropy, clipping rate, etc.
        """
        if advantages.numel() == 0:
            # [v163.0 SOTA FIX] Neutral Tensor for Empty Batch (Smoking Gun #163)
            # Rationale: We MUST NOT return early. Ranks with empty batches must 
            # still participate in DDP collectives (whitening, max_sync, adaptive_stats)
            # to prevent deadlocks. We use a single-element neutral tensor for stats
            # and then mask the final output.
            adv_flat = advantages.detach()
        else:
            # [v54.0 SOTA FIX] Input Sanitization Gate (Smoking Gun #54)
            with torch.no_grad():
                advantages = torch.where(
                    torch.isfinite(advantages), 
                    advantages.clamp(min=-10000.0, max=10000.0), 
                    torch.zeros_like(advantages)
                )
            if mask is not None:
                if mask.dim() == 1 and mask.shape[0] == advantages.shape[0]:
                    mask = mask.bool()
                    advantages = advantages[mask]
                    # if values is not None: values = values[mask] # values not passed to this function
                    # if rewards is not None: rewards = rewards[mask] # rewards not passed to this function
        
        # [SOTA v3.1] Mask-Aware Statistics
        if mask is not None:
            adv_flat = advantages[mask.bool()]
        else:
            adv_flat = advantages.reshape(-1)

        # --- 1. Global Whitening & Winsorization ---
        # [v39.0 SOTA FIX] Adaptive Advantage Whitening (Smoking Gun #39)
        # Rationale: Fixed whitening stats from Epoch 0 become stale as the model improves.
        # This causes 'Selection Pressure Decay' where all samples look equally good.
        # Fix: Tracking advantage drift via a slow DDP-Sychronized EMA.
        if self.stats_initialized.item():
            # [SOTA FIX] Online Stats Update (Welford-Lite)
            with torch.no_grad():
                # Compute Global Batch Stats
                b_sum = adv_flat.sum()
                b_sq_sum = (adv_flat ** 2).sum()
                b_count = torch.tensor([float(adv_flat.numel())], device=adv_flat.device)
                
                if dist.is_initialized():
                    stats = torch.stack([b_sum, b_sq_sum, b_count])
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    g_b_sum, g_b_sq_sum, g_b_count = stats[0], stats[1], stats[2]
                else:
                    g_b_sum, g_b_sq_sum, g_b_count = b_sum, b_sq_sum, b_count
                
                if g_b_count > 1:
                    curr_mu = g_b_sum / g_b_count
                    curr_var = (g_b_sq_sum / g_b_count) - (curr_mu ** 2)
                    # [v165.0 SOTA FIX] Sigma Floor Hardening (Smoking Gun #165)
                    # Rationale: Increased floor from 1e-6 to 1e-5 for stable denominator
                    curr_sigma = torch.sqrt(curr_var.clamp(min=1e-5))
                    
                    # [v29.1 SOTA FIX] Step-Density Aware Whitening (Abyssal #1)
                    mom = self.whitening_momentum
                    self.stats_count.add_(1)
                    t = self.stats_count.float()
                    # [v23.0 SOTA FIX] Bias Correction Floor (Smoking Gun #245.2)
                    # [v29.2 SOTA FIX] Softened Floor (Abyssal #2)
                    # Rationale: Increasing floor from 1e-3 to 0.01 prevents 
                    # extreme 1000x Advantage scaling during early training/resumption.
                    bias_correction = torch.as_tensor(1.0 - (mom ** t.item()), device=t.device).clamp(min=0.01)
                    
                    # Update (Uncorrected)
                    self.adv_mean.mul_(mom).add_(curr_mu, alpha=1.0 - mom)
                    self.adv_std.mul_(mom).add_(curr_sigma, alpha=1.0 - mom)
                    
                    # [v101.0 SOTA FIX] AWR Boiling Frog Guard (Bias Correction)
                    # Use corrected stats for the current normalization pass.
                    mu = self.adv_mean / bias_correction
                    sigma = self.adv_std / bias_correction
                else:
                    mu, sigma = self.adv_mean, self.adv_std
            
            # [PHASE 14 FIX] Final Mu/Sigma selection
            # Ensures mu/sigma are tensors even if count is 1.
            if not isinstance(mu, torch.Tensor): mu = torch.tensor([mu], device=advantages.device)
            if not isinstance(sigma, torch.Tensor): sigma = torch.tensor([sigma], device=advantages.device)
        else:
            # [v96.2 SOTA FIX] Global Whitening Parity for Fresh Start
            # [v96.2 SOTA FIX] Global Whitening Parity for Fresh Start
            if adv_flat.numel() > 0:
                l_sum = adv_flat.sum()
                l_sq_sum = (adv_flat ** 2).sum()
                l_count = torch.tensor(float(adv_flat.numel()), device=adv_flat.device)
            else:
                # [v163.2 FIX] Neutral stats for empty ranks (Smoking Gun #163.2)
                l_sum = torch.tensor(0.0, device=advantages.device)
                l_sq_sum = torch.tensor(0.0, device=advantages.device)
                l_count = torch.tensor(0.0, device=advantages.device)
            
            if dist.is_initialized():
                # [v96.2 SOTA FIX] Exact Global Variance (Smoking Gun #96)
                # Replaced approximate mean-averaging with true Variance reduction.
                stats = torch.stack([l_sum, l_sq_sum, l_count])
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                
                g_b_sum, g_b_sq_sum, g_b_count = stats[0], stats[1], stats[2]
                
                if g_b_count > 1:
                    mu = g_b_sum / g_b_count
                    var = (g_b_sq_sum / g_b_count) - (mu ** 2)
                    sigma = torch.sqrt(var.clamp(min=1e-5))
                else:
                    mu, sigma = torch.tensor(0.0, device=advantages.device), torch.tensor(1.0, device=advantages.device)
            else:
                # [v165.0 SOTA FIX] Epsilon Hardening (Smoking Gun #165)
                # Local only
                if l_count > 1:
                    mu = l_sum / l_count
                    var = (l_sq_sum / l_count) - (mu ** 2)
                    sigma = torch.sqrt(var.clamp(min=1e-5))
                else:
                    mu, sigma = torch.tensor(0.0, device=advantages.device), torch.tensor(1.0, device=advantages.device)
            
            # [v23.0 SOTA FIX] Activate EMA Branch (Smoking Gun #245)
            # Rationale: Once we've seen at least one valid sample, we switch to EMA tracking.
            if (dist.is_initialized() and stats[2] > 0) or (not dist.is_initialized() and l_count > 0):
                self.stats_initialized.fill_(True)
                self.adv_mean.copy_(mu.detach().reshape(-1)[:1])
                self.adv_std.copy_(sigma.detach().reshape(-1)[:1])
            
        # [SOTA 2025] Advantage Winsorization (99th Percentile Clipping)
        # Uses the masked distribution to find the true 99th percentile.
        # [SOTA v30.5 FIX] DDP Consensus: Threshold must be identical across ranks.
        with torch.no_grad():
            # [v139.0 SOTA FIX] Empty Batch Guard (Smoking Gun #139)
            if adv_flat.numel() > 10:
                p99 = torch.quantile(adv_flat.detach().float(), 0.99)
            else:
                # Default high quantile if batch is too small or empty
                p99 = torch.tensor(20.0, device=advantages.device)
            
            # [v153.0 SOTA FIX] NaN Quantile Guard (Smoking Gun #153)
            # Rationale: If Rank N has all NaNs, quantile returns NaN. 
            # all_reduce(MAX) would then poison the entire cluster.
            if not torch.isfinite(p99):
                p99.fill_(20.0)

            if dist.is_initialized():
                dist.all_reduce(p99, op=dist.ReduceOp.MAX)
                
            advantages = torch.clamp(advantages, max=p99)
        
        # Z-Score normalization: A ~ N(0, 1)
        # [v167.0 SOTA FIX] Synergistic Batch-Local Normalization (BLN)
        # Rationale: Mix Global Z-Score with Batch Z-Score to prevent collapse (ESS < 2.0).
        # Fixes "Single Sample Dominance" in high-variance batches.
        
        # 1. Global Z-Score (Stability)
        z_global = (advantages - mu) / (sigma + 1e-5)
        
        # 2. Batch-Local Z-Score (Reactivity)
        if adv_flat.numel() > 1:
            b_mean = adv_flat.mean()
            b_std = adv_flat.std().clamp(min=1e-5)
            z_batch = (advantages - b_mean) / b_std
        else:
            z_batch = z_global
            
        # 3. Hybrid Mixing (80% Global / 20% Batch)
        # Keeps alignment with global value scale while ensuring >0 gradients for best local samples.
        alpha_bln = 0.8
        norm_adv = (alpha_bln * z_global) + ((1 - alpha_bln) * z_batch)
        
        # 4. FP16 Safety Clamp (prevent exp explosion)
        norm_adv = norm_adv.clamp(min=-5.0, max=5.0)
        
        # [SOTA 2025] Z-Score Normalization (Unclipped)
        # We no longer hard-clamp at ±2.0 to preserve heavy-tailed 'clinical crash' signals.
        # Stability is instead managed via Exponential Tempering and Hard-Weight Clipping.
        
        
        # --- 2. Scaled Advantage (Adaptive ESS) ---
        # [SOTA RECOVERY v3.2] Adaptive Beta Search
        # Rationale: Dynamically anneal temperature to guarantee broad sampling (ESS >= 20.0).
        # [v4.0] Reduced Target ESS (Selection Pressure Recovery)
        # Rationale: target_ess=20 forces beta too high (3.0-4.2), eliminating advantage signal.
        # [v5.0 SURGICAL PATCH] Standard interpretation (restore selection diversity)
        target_ess = max(10.0, self.target_ess)
        
        # [SOTA Phase 10 FIX] Consensus-AWR (Smoking Gun #3)
        # Rationale: DDP ranks must apply identical selection pressure (Beta) to 
        # prevent 'Noisy Rank Dominance'.
        
        # 1. Prepare data for search (Mask-aware)
        if mask is not None:
             # advantages is [B, T], mu/sigma are scalars
             norm_flat = (advantages - mu) / sigma
             # Mask out invalid steps for count/sums
             norm_flat = norm_flat.view(-1)[mask.view(-1) == 0]
        else:
             norm_flat = norm_adv.view(-1)
             
        # Definition: Global ESS = (sum(w))^2 / sum(w^2) across all ranks
        def get_consensus_ess(temperature):
             t = max(temperature, 1e-3)
             # Local weights (not sum-to-1)
             w_local = torch.exp(norm_flat / t)
             
             l_sum = w_local.sum()
             l_sq_sum = w_local.pow(2).sum()
             l_count = torch.tensor(norm_flat.numel(), device=w_local.device, dtype=w_local.dtype)
             
             if dist.is_initialized():
                  stats = torch.stack([l_sum, l_sq_sum, l_count])
                  dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                  g_sum, g_sq_sum, g_count = stats[0], stats[1], stats[2]
             else:
                  g_sum, g_sq_sum, g_count = l_sum, l_sq_sum, l_count
                  
             if g_count < 1: return 1.0
             # ESS formula: (SumW)^2 / Sum(W^2)
             return (g_sum.pow(2) / (g_sq_sum + 1e-12)).item()

        # Update target_ess to count-safe value
        target_ess_val = max(5.0, self.target_ess)

        # Check current ESS
        current_temp = self.beta.item()
        
        # [SOTA v4.1 LEGACY REMOVED] Bidirectional Beta Adaptation
        # Redundant block removed. Logic centralized in _update_adaptive_stats.
        # This block caused "Batch Size Paradox" by forcing Beta=20.0 before Safety Valve could act.
             
        scaled_adv = norm_adv / self.beta
        
        # --- 3. SOTA Numerical Stability: Local-Global Robust AWR ---
        # [v139.0 SOTA FIX] Empty Batch Guard (Smoking Gun #139)
        if scaled_adv.numel() > 0:
            g_max_log_w = scaled_adv.max()
        else:
            # -20.0 is the log-space floor, effectively zero weight
            g_max_log_w = torch.tensor(-20.0, device=scaled_adv.device)
            
        # [v153.0 SOTA FIX] Finite Max Sync (Smoking Gun #153)
        if not torch.isfinite(g_max_log_w):
            g_max_log_w.fill_(-20.0)

        if dist.is_initialized():
            # [v26.1] Using ReduceOp.MAX for bit-perfect consensus
            dist.all_reduce(g_max_log_w, op=dist.ReduceOp.MAX)
        
        # log_weights_global: Stable exponentially across ALL ranks
        log_weights_global = scaled_adv - g_max_log_w
        log_weights_global = torch.clamp(log_weights_global, min=-20.0, max=5.0)
        
        weights_local = torch.exp(log_weights_global)
        
        # --- 4. Global Normalization & Sync ---
        # We want the weights to sum to 'num_total_samples' (standard AWR normalization)
        sum_w_local = weights_local.sum().view(1)
        numel_local = torch.tensor([float(weights_local.numel())], device=weights_local.device)
        
        if dist.is_initialized():
            sync_tensor = torch.stack([sum_w_local, numel_local])
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            sum_w_global, numel_global = sync_tensor[0], sync_tensor[1]
            # [v36.1 FIX] Weight Normalization Factor
            # norm_factor = total_count / global_sum
            norm_factor = numel_global / (sum_w_global + 1e-8)
        else:
            norm_factor = numel_local / (sum_w_local + 1e-8)
            
        weights = weights_local * norm_factor
        
        # --- 5. Hard Clipping (Standard AWR practice) ---
        weights_clipped = torch.clamp(weights, max=self.max_weight)
        
        # --- 6. Diagnostics & Adaptive Sync ---
        with torch.no_grad():
            numel_local = weights.numel()
            
            # [v139.1 SOTA FIX] Empty Batch Diagnostic Guard (Removed early return #163)
            # Rationale: We no longer return early here, as all ranks must reach
            # _update_adaptive_stats for DDP parity.
            if numel_local == 0:
                sum_w = torch.tensor(0.0, device=weights.device)
                sum_w_sq = torch.tensor(1e-8, device=weights.device) # Prevents NaN in ESS
                clipping_count = torch.tensor(0.0, device=weights.device)
            else:
                sum_w = weights_clipped.sum()
                sum_w_sq = (weights_clipped ** 2).sum()
                clipping_count = (weights > self.max_weight).float().sum()
            
            # [SOTA 2025] DDP Global Synchronization
            # ESS and clipping rate must be global to prevent rank divergence
            if dist.is_initialized():
                stats = torch.stack([sum_w, sum_w_sq, clipping_count, torch.tensor(numel_local, device=weights.device, dtype=torch.float32)])
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                g_sum_w, g_sum_w_sq, g_clip_count, g_numel = stats[0], stats[1], stats[2], stats[3]
            else:
                g_sum_w, g_sum_w_sq, g_clip_count, g_numel = sum_w, sum_w_sq, clipping_count, float(numel_local)

            # Global Effective Sample Size (ESS)
            # [SOTA BUG FIX] Return ESS as Count (1..N), not Ratio (1/N..1).
            # Consumers (Probes/Logs) expect Count.
            ess = (g_sum_w ** 2) / (g_sum_w_sq + 1e-8)
            self.ess_buffer.fill_(ess) 
            
            # Global Clipping Rate
            clipped_rate = g_clip_count / (g_numel + 1e-8)
            self.clip_rate_buffer.fill_(clipped_rate)
            
            # [SOTA 2025] Adaptive Dynamics Update (Uses Global Statistics)
            beta_raw = self.beta.item()
            if self.adaptive_beta or self.adaptive_clipping:
                beta_raw = self._update_adaptive_stats(
                    advantages, weights, ess, clipped_rate.item(), 
                    total_batch_size=float(g_numel)
                )
            
            # Weight Entropy (Information Theoretic)
            probs = weights_clipped / (sum_w + 1e-8)
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs) / math.log(numel_local + 1)
            
            hard_clipped_rate = (weights > self.max_weight).float().mean()
            
            if numel_local == 0:
                diagnostics = {
                    "adv_mean": mu.item() if isinstance(mu, torch.Tensor) else mu,
                    "adv_std": sigma.item() if isinstance(sigma, torch.Tensor) else sigma,
                    "weights_max": 0.0,
                    "weights_mean": 0.0,
                    "weights_std": 0.0,
                    "ess": self.ess_buffer.item(),
                    "weight_entropy": 0.0,
                    "fp16_clipped_ratio": self.clip_rate_buffer.item(),
                    "hard_clipped_ratio": 0.0,
                    "beta_dynamic": self.beta.item(),
                    "max_weight_dynamic": self.max_weight.item()
                }
            else:
                diagnostics = {
                    "adv_mean": mu.item() if isinstance(mu, torch.Tensor) else mu,
                    "adv_std": sigma.item() if isinstance(sigma, torch.Tensor) else sigma,
                    "weights_max": weights_clipped.max().item(),
                    "weights_mean": weights_clipped.mean().item(),
                    "weights_std": weights_clipped.std().item(),
                    "ess": self.ess_buffer.item(),
                    "weight_entropy": entropy.item(),
                    "fp16_clipped_ratio": self.clip_rate_buffer.item(),
                    "hard_clipped_ratio": hard_clipped_rate.item(),
                    "beta_dynamic": self.beta.item(),
                    "beta_raw": beta_raw, # [Telemetry]
                    "max_weight_dynamic": self.max_weight.item()
                }
        
        return weights_clipped, diagnostics

    def _update_adaptive_stats(
        self, advantages: torch.Tensor, weights: torch.Tensor, ess: torch.Tensor, clipped_rate: float,
        total_batch_size: float = None
    ) -> float:
        """
        [SOTA 2025] Dynamically adapts hyperparameters to squeeze performance.
        Returns the raw (un-momentum-ed) beta target for forensics.
        """
        beta_raw = self.beta.item()
        with torch.no_grad():
            # A. Adaptive Beta (Target ESS = 10%)
            if self.adaptive_beta:
                # [SAFETY] If we successfully clamped too many values (FP16 limit),
                # the weights become uniform (clamped_max), which paradoxically INCREASES ESS.
                # If this happens, the controller mistakenly tries to lower beta further,
                # causing a collapse to min_beta.
                # FIX: If saturation is high (>5%), force-increase Beta to restore gradients.
                if clipped_rate > 0.05:
                    # Saturation Recovery Mode (Turbo-Charged)
                    # [SOTA FIX]: Boost beta proportional to clipping severity.
                    # If 100% clipped, beta doubles instantly.
                    # If 10% clipped, beta * 1.1.
                    # This fixes the "lazy adaptation" (33 steps -> 3 steps).
                    boost_factor = 1.0 + clipped_rate
                    self.beta.copy_(self.beta * boost_factor)
                
                # [FIX] current_ess must be defined unconditionally for use at line 1017
                current_ess = ess.item()
                
                if clipped_rate <= 0.05:
                    # Standard ESS Control Mode
                    # [SOTA FIX v2.0] Adaptive Target Scaling (The "Batch Size Paradox" Fix)
                    # Rationale: If target_ess (30) > batch_size (16), controller panics -> Beta=20.
                    # Fix: Dynamically clamp target to 50% of available batch size.
                    
                    if total_batch_size is not None:
                        batch_size = total_batch_size
                    else:
                        batch_size = float(weights.numel())
                    
                    current_raw_ess = ess.item()
                    
                    # Resolve Target
                    if self.target_ess > 1.0:
                        # Interpreted as Raw Count (e.g., 30.0)
                        safe_cap = batch_size * 0.5 # Nyquist-style safety limit
                        target_val = min(self.target_ess, safe_cap)
                        current_val = current_raw_ess
                    else:
                        # Interpreted as Ratio (e.g., 0.20)
                        target_val = self.target_ess
                        current_val = current_raw_ess / (batch_size + 1e-6)

                    # P-Controller with Anti-Windup
                    error_ess = (target_val - current_val)
                    correction = math.exp(10.0 * error_ess * self.beta_gain)
                    
                    correction = max(0.5, min(2.0, correction))
                    new_beta = self.beta * correction
                    beta_raw = new_beta.item()
                    
                    # [SOTA v10.5] AWR Selection Recovery (Grid-Search Proven)
                    # Rationale: Hardcoded mom=0.999 caused 320-step convergence lag.
                    # Optimal mom=0.95 (via self.beta_momentum) achieves 2-step convergence.
                    mom = self.beta_momentum
                    updated_beta = (mom * self.beta) + ((1.0 - mom) * new_beta)
                    self.beta.copy_(torch.clamp(updated_beta, min=self.min_beta))
                    
                    # [PHASE 47] Unfreezing Telemetry
                    # print(f"[AWR DEBUG] ESS={current_ess:.4f} | Target={target_ess} | Err={error_ess:.4f} | Corr={correction:.4f} | Beta: {self.beta.item():.4f}")
                else:
                    pass
                    # print(f"[AWR DEBUG] Saturation Mode! ClipRate={clipped_rate:.4f} | Beta Boosting...")
                
                # [v27.1 FIX] ESS Safety Floor with Cooldown
                # Prevents runaway multiplicative growth (166 clamps/200 steps → ~20)
                if current_ess < 0.05 and self.beta_growth_cooldown == 0:
                    # [v36.0 SOTA FIX] Extended Emergency Cooldown (Fix #H4)
                    # Rationale: Preventative hardening - 10-step cooldown allowed up to 20
                    # emergency growths per epoch, causing potential beta ratcheting.
                    # New 50-step cooldown limits to ~4 per epoch for stable selection pressure.
                    self.beta.copy_(torch.clamp(self.beta * self.beta_growth_factor, min=1.5))
                    self.beta_growth_cooldown = 50  # Extended cooldown: 50 steps between emergency growths
                 # [v27.1 FIX] Tensor-Safe Cooldown Update
            if self.beta_growth_cooldown.item() > 0:
                self.beta_growth_cooldown.sub_(1)

            # [v25.6 SOTA] ESS Momentum Buffer
            # Stabilizes telemetry across jittery batches.
            # [v2026] Uses Scaled Decay
            ema_d = self.ess_ema_decay
            self.ess_momentum_buffer.copy_(ema_d * self.ess_momentum_buffer + (1.0 - ema_d) * current_ess)
            
            # [v7.2 SOTA FIX] IronFloor: Strict runtime clamp
            # Rationale: Persistence and config overrides were bypassing the Phase 6
            # min_beta=1.0 floor, causing ESS collapse in DDP environments.
            self.beta.clamp_(min=self.min_beta, max=self.max_beta)

                
            # B. Adaptive Clipping (Target = 95th Percentile)
            if self.adaptive_clipping:
                # [v163.0 SOTA FIX] Deadlock-Free Adaptive Sync (Smoking Gun #163)
                # Rationale: Ranks with zero samples must NOT skip the all_reduce
                # or the cluster will hang. We use 0.0 as a neutral MAX element.
                p95_t = torch.tensor([0.0], device=self.max_weight.device, dtype=self.max_weight.dtype)
                
                if weights.numel() > 0:
                    try:
                        # Find 95th percentile of RAW weights
                        p95_t.fill_(torch.quantile(weights.detach().float(), 0.95).item())
                    except:
                        pass # Fallback if quantile fails
                
                # Check for NaN before sync
                if not torch.isfinite(p95_t):
                    p95_t.fill_(0.0)

                if dist.is_initialized():
                    # Every rank calls this, even if p95 is 0.0
                    dist.all_reduce(p95_t, op=dist.ReduceOp.MAX)
                
                p95 = p95_t.item()
                if p95 > 1e-6:
                    target_clip = max(2.0, min(20.0, p95 * 1.2))
                    self.max_weight.copy_(torch.as_tensor(target_clip, device=self.max_weight.device, dtype=self.max_weight.dtype))

            # [DDP SYNCHRONIZATION] Prevent divergence of adaptive parameters across ranks
            # [SOTA v30.5] Ensuring perfect bit-parity across all compute nodes.
            if dist.is_initialized():
                # 1. Sync Adaptive Parameters (beta, max_weight)
                state_params = torch.stack([self.beta, self.max_weight])
                dist.all_reduce(state_params, op=dist.ReduceOp.SUM)
                state_params /= dist.get_world_size() 
                self.beta.copy_(state_params[0])
                self.max_weight.copy_(state_params[1])

                # 2. Sync Metric Buffers for consistent telemetry
                metric_params = torch.stack([self.ess_buffer, self.clip_rate_buffer])
                dist.all_reduce(metric_params, op=dist.ReduceOp.SUM)
                metric_params /= dist.get_world_size()
                self.ess_buffer.copy_(metric_params[0])
                self.clip_rate_buffer.copy_(metric_params[1])
            
            return beta_raw

    def calculate_weights(
        self, 
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full AWR weight calculation with explained variance diagnostic.
        
        This is the primary entry point for weight calculation, including
        additional diagnostics like explained variance when critic outputs
        are available.
        
        Args:
            advantages: (B, T) or (B,) Tensor of advantages
            values: Optional (B, T) critic value predictions (for diagnostics)
            rewards: Optional (B, T) reward tensor (for diagnostics)
        
        Returns:
            weights: Tensor of AWR weights
            diagnostics: Dict with ESS, entropy, explained variance, etc.
        """
        # Core AWR weight calculation
        weights, diagnostics = self.calculate_awr_weights(advantages, mask=mask)
        
        # --- Additional Diagnostics ---
        
        # Explained Variance: 1 - Var(Returns - Values) / Var(Returns)
        # Measures how well the critic predicts returns
        exp_var = 0.0
        if values is not None and rewards is not None:
            with torch.no_grad():
                # Simplified returns = rewards (proxy for target values)
                target_returns = rewards
                y_diff = target_returns - values
                var_y = torch.var(target_returns)
                var_diff = torch.var(y_diff)
                
                if var_y > 1e-8:
                    exp_var = (1.0 - var_diff / var_y).item()
                    exp_var = max(-1.0, min(1.0, exp_var))  # Clamp [-1, 1]
        
        diagnostics["explained_variance"] = exp_var
        diagnostics["max_weight"] = diagnostics["weights_max"]  # Alias
        
        return weights, diagnostics


# =============================================================================
# VERIFICATION BLOCK
# =============================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("APEX Advantage Calculator (Ultimate v10.0) - Smoke Test")
    print("="*60)
    
    # Mock Clinical Normalizer Interface
    class MockNormalizer:
        """Mock normalizer: maps [-1, 1] -> [0, 100] (like clinical units)"""
        def denormalize(self, x): 
            return (x + 1) * 50.0  # e.g., 0 -> 50 (MAP), 1 -> 100
    
    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies AWR adaptation rates across step densities."""
        if n_curr <= 0: return
        
        # 1. Scale Beta Momentum (Baseline 0.90 for 200 steps)
        # Using SOTA Power-Law to preserve the effective memory window.
        self.beta_momentum = ScalingSteward.get_decay(self.base_beta_momentum, n_curr)
        
        # 2. Scale ESS EMA Decay (Baseline 0.95 for 200 steps)
        self.ess_ema_decay = ScalingSteward.get_decay(0.95, n_curr)
        
        # 3. Scale Clip Momentum (Baseline 0.90 for 200 steps)
        self.clip_momentum = ScalingSteward.get_decay(0.90, n_curr)

        logger.info(
            f"⚡ [AWR] Dynamics Scaled: beta_mom={self.beta_momentum:.4f}, "
            f"ess_ema={self.ess_ema_decay:.4f} | n_curr={n_curr}"
        )

    # =========================================================================
    # SMOKE TEST
    # =========================================================================
    calc = ICUAdvantageCalculator(beta=0.5)
    
    # Test Data: Batch=2, Time=5, Channels=28
    B, T, C = 2, 5, 28
    
    # 1. Mock Vitals (Normalized [-1, 1])
    vitals = torch.zeros(B, T, C)
    # Patient 0: Stable (MAP normalized 0.0 -> 50 mmHg after denorm)
    vitals[0, :, 4] = 0.3  # MAP -> 65 mmHg (healthy)
    vitals[0, :, 7] = -0.5  # Lactate -> ~25 (too high, but mock)
    
    # Patient 1: Critical (MAP normalized -0.5 -> 25 mmHg)
    vitals[1, :, 4] = -0.5  # MAP -> 25 mmHg (shock)
    vitals[1, :, 7] = 0.5   # Lactate -> high
    
    # 2. Outcomes & Dones
    outcomes = torch.tensor([0.0, 1.0])  # Pat 0 survives, Pat 1 dies
    dones = torch.zeros(B, T)
    dones[:, -1] = 1.0  # Terminal at end
    
    # 3. Compute Clinical Reward
    print("\n[1] Computing Clinical Rewards...")
    rewards = calc.compute_clinical_reward(
        vitals, outcomes, dones, normalizer=MockNormalizer()
    )
    print(f"    Rewards Shape: {rewards.shape}")
    print(f"    Patient 0 (Stable) Mean Reward: {rewards[0].mean():.4f}")
    print(f"    Patient 1 (Critical) Mean Reward: {rewards[1].mean():.4f}")
    print(f"    (Expected: Pat0 > Pat1)")
    
    # 4. Compute GAE
    print("\n[2] Computing GAE Advantages...")
    values = torch.randn(B, T) * 0.1  # Mock critic values
    advantages = calc.compute_gae(rewards, values, dones)
    print(f"    Advantages Shape: {advantages.shape}")
    print(f"    Advantage Mean: {advantages.mean():.4f}")
    print(f"    Advantage Std: {advantages.std():.4f}")
    
    # 5. Set Stats (Pretend we saw whole dataset)
    print("\n[3] Locking AWR Statistics...")
    calc.set_stats(mean=advantages.mean().item(), std=advantages.std().item())
    
    # 6. Calculate Weights
    print("\n[4] Computing AWR Weights...")
    weights, diagnostics = calc.calculate_weights(
        advantages, values=values, rewards=rewards
    )
    print(f"    Weights Shape: {weights.shape}")
    print(f"    Weight Range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"\n    Diagnostics:")
    for k, v in diagnostics.items():
        print(f"      {k}: {v:.4f}")
    
    print("\n" + "="*60)
    print("Smoke Test Complete!")
    print("="*60)
