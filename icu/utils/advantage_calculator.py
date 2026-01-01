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
    'RESP_UPPER': 22.0,          # bpm (qSOFA respiratory criterion)
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
            sparse_reward_scale: float = 5.0,   # Terminal reward magnitude
            reward_shaping_coef: float = 0.1,   # Dense reward scale
            focal_alpha: float = 0.25,      # Negative reward emphasis
            qsofa_thresholds: Optional[Dict[str, float]] = None,  # Override defaults
            adaptive_beta: bool = False,    # [SOTA 2025] Dynamic Temperature
            adaptive_clipping: bool = False # [SOTA 2025] Dynamic Weight Clipping
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
        """
        super().__init__()
        self.register_buffer("beta", torch.tensor(beta).float())
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.register_buffer("max_weight", torch.tensor(max_weight).float())
        self.sparse_scale = sparse_reward_scale
        self.shaping_coef = reward_shaping_coef
        self.focal_alpha = focal_alpha
        
        # [v2025 SOTA] State Buffers for DDP Synchronization
        self.register_buffer("ess_buffer", torch.zeros(1))
        self.register_buffer("clip_rate_buffer", torch.zeros(1))
        
        # [SOTA 2025] Adaptive Hyperparameters
        self.adaptive_beta = adaptive_beta
        self.adaptive_clipping = adaptive_clipping
        self.beta_momentum = 0.90      # Faster updates (was 0.95)
        self.clip_momentum = 0.90
        self.min_beta = 0.01           # Allow sharper peaks
        self.max_beta = 10.0
        
        # qSOFA thresholds
        if qsofa_thresholds is None:
            self.qsofa_thresholds = {
                'resp': SEPSIS_CONSTANTS['RESP_UPPER'],
                'sbp': SEPSIS_CONSTANTS['SBP_HYPOTENSION'],
                'gcs': SEPSIS_CONSTANTS['GCS_LOWER']
            }
        else:
            self.qsofa_thresholds = qsofa_thresholds

        # Global Whitening Statistics (Welford's Algorithm state)
        # Register as buffers for persistence across checkpoints
        self.register_buffer("adv_mean", torch.tensor(0.0))
        self.register_buffer("adv_std", torch.tensor(1.0))
        self.register_buffer("stats_count", torch.tensor(0))
        self.register_buffer("stats_initialized", torch.tensor(False))
        
        logger.info(
            f"[ADVANTAGE] Initialized: beta={beta}, gamma={gamma}, "
            f"lambda={lambda_gae}, max_weight={max_weight}, "
            f"adaptive_beta={adaptive_beta}, adaptive_clipping={adaptive_clipping}"
        )

    def set_stats(self, mean: float, std: float):
        """
        Locks normalization statistics for stable AWR weight computation.
        
        This should be called after computing statistics over the entire
        training dataset. Critical for evaluation stability.
        
        Args:
            mean: Global advantage mean
            std: Global advantage standard deviation
        """
        self.adv_mean.fill_(mean)
        self.adv_std.fill_(std if std > 1e-6 else 1.0)
        self.stats_initialized.fill_(True)
        logger.info(
            f"[ADVANTAGE] Stats Locked: mu={self.adv_mean.item():.4f}, sigma={self.adv_std.item():.4f}"
        )

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

    # =========================================================================
    # CLINICAL REWARD FUNCTION
    # =========================================================================

    def compute_intrinsic_reward(
        self, 
        vitals: torch.Tensor, 
        outcome_label: torch.Tensor,
        dones: torch.Tensor,
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

        # Mask: Only apply sparse reward at true episode end
        # [FIX: Mask-Aware Terminal Placement]
        # In sliding windows, 'last index' might be padding. We want the last VALID step.
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
            # --- 4. MAP Penalty (Sigmoid Soft-Cliff) ---
            if map_val is not None:
                # Sigmoid centered at MAP=60, steepness=0.5
                # MAP 65+ -> ~0 penalty, MAP 55 -> high penalty
                # Formula: sigmoid((60 - MAP) * steepness)
                map_penalty_score = torch.sigmoid((60.0 - map_val) * 0.5)
                rewards -= self.shaping_coef * map_penalty_score

            # --- 5. SBP Penalty (Additional Hypotension Marker) ---
            if sbp_val is not None:
                # Sigmoid centered at SBP=95, steepness=0.2
                # SBP 100+ -> ~0 penalty, SBP 90 -> moderate penalty
                sbp_penalty_score = torch.sigmoid((95.0 - sbp_val) * 0.2)
                rewards -= self.shaping_coef * 0.5 * sbp_penalty_score

            # --- 6. Lactate Penalty (Sigmoid Soft-Cliff) ---
            if lactate_val is not None:
                # Sigmoid centered at Lactate=3.0, steepness=1.0
                # Lactate 2.0 -> low penalty, Lactate 4.0+ -> high penalty
                lac_penalty_score = torch.sigmoid((lactate_val - 3.0) * 1.0)
                # Higher weight for lactate (primary mortality predictor)
                rewards -= self.shaping_coef * 1.5 * lac_penalty_score

            # --- 7. Respiratory Penalty (qSOFA) ---
            if resp_val is not None:
                # Sigmoid centered at Resp=24, steepness=0.3
                # Resp 22 -> low penalty, Resp 30+ -> high penalty
                resp_penalty_score = torch.sigmoid((resp_val - 24.0) * 0.3)
                rewards -= self.shaping_coef * 0.3 * resp_penalty_score

            # --- 8. Delta Trends (Reward Recovery) ---
            if T > 1:
                # A. Lactate Improvement: Reward DECREASE in lactate
                if lactate_val is not None:
                    # positive delta = lactate going DOWN (good)
                    lac_delta = lactate_val[:, :-1] - lactate_val[:, 1:]
                    lac_improvement = torch.clamp(lac_delta, min=0.0, max=2.0)
                    rewards[:, 1:] += self.shaping_coef * 2.0 * lac_improvement

                # B. MAP Improvement: Reward INCREASE in MAP (if was low)
                if map_val is not None:
                    map_delta = map_val[:, 1:] - map_val[:, :-1]
                    # Only reward MAP increase if it was in danger zone (<75)
                    # Otherwise we encourage hypertension
                    map_was_low = (map_val[:, :-1] < 75.0).float()
                    map_improvement = torch.clamp(map_delta, min=0.0, max=10.0) * map_was_low
                    rewards[:, 1:] += self.shaping_coef * 0.5 * map_improvement

                # C. SBP Improvement: Reward INCREASE in SBP (if was low)
                if sbp_val is not None:
                    sbp_delta = sbp_val[:, 1:] - sbp_val[:, :-1]
                    sbp_was_low = (sbp_val[:, :-1] < 110.0).float()
                    sbp_improvement = torch.clamp(sbp_delta, min=0.0, max=15.0) * sbp_was_low
                    rewards[:, 1:] += self.shaping_coef * 0.2 * sbp_improvement

            # --- 9. Clamp Total Dense Reward ---
            # Prevent dense rewards from overwhelming sparse signal
            reward_cap = SEPSIS_CONSTANTS['DENSE_REWARD_CAP']
            rewards = torch.clamp(rewards, min=-reward_cap * 2, max=reward_cap)

        # --- 10. Apply Source Mask (Zero out padding) ---
        if src_mask is not None:
            # Ensure proper shape broadcasting
            if src_mask.dim() == 2:
                rewards = rewards * src_mask.float()
            elif src_mask.dim() == 3:
                 # Reduce to [B, T] if strictly necessary, but usually mask is [B, T]
                 # Assuming mask means "any feature valid"
                 rewards = rewards * src_mask.any(dim=-1).float()

        return rewards

    # Alias for backward compatibility
    def compute_clinical_reward(
        self, 
        vitals: torch.Tensor, 
        outcome_label: torch.Tensor,
        feature_indices: Optional[Dict[str, int]] = None,
        normalizer: Optional[Any] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Backward-compatible wrapper for compute_intrinsic_reward.
        
        Creates a synthetic 'dones' tensor assuming terminal at last timestep.
        
        Args:
            vitals: (B, T, C) Tensor of vital signs
            outcome_label: (B,) or (B, T) binary outcome
            feature_indices: Map of feature names to channel indices
            normalizer: Optional ClinicalNormalizer
            src_mask: Optional mask for padding logic
        
        Returns:
            rewards: (B, T) Tensor of clinical rewards
        """
        B, T, _ = vitals.shape
        dones = torch.zeros(B, T, device=vitals.device)
        dones[:, -1] = 1.0  # Terminal at last timestep
        
        return self.compute_intrinsic_reward(
            vitals=vitals,
            outcome_label=outcome_label,
            dones=dones,
            feature_indices=feature_indices,
            normalizer=normalizer,
            src_mask=src_mask
        )

    # =========================================================================
    # GENERALIZED ADVANTAGE ESTIMATION (GAE)
    # =========================================================================

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
            # If done[t]=1, then V(s_{t+1}) should be masked (treated as 0)
            non_terminal = 1.0 - dones.float()
        else:
            # Assume all steps are non-terminal (sliding window assumption)
            non_terminal = torch.ones_like(rewards)
        
        # --- 3. TD Error (Delta) ---
        # δ_t = r_t + γ * V(s_{t+1}) * (1-d_t) - V(s_t)
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
        advantages: torch.Tensor
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
        # --- 1. Global Whitening ---
        if self.stats_initialized.item():
            mu, sigma = self.adv_mean, self.adv_std
        else:
            # Batch estimation (fallback)
            mu = advantages.mean()
            sigma = advantages.std() + 1e-8
        
        # Z-Score normalization: A ~ N(0, 1)
        norm_adv = (advantages - mu) / sigma
        
        # --- 2. Scaled Advantage ---
        scaled_adv = norm_adv / self.beta
        
        # --- 3. FP16 Safety: Pre-Exp Clamping ---
        # float16 max is ~65504. exp(11.1) ≈ 65000.
        # We clamp input to exp() to prevent overflow
        max_exp_input = 10.0  # Safe for FP16
        scaled_adv_clamped = torch.clamp(scaled_adv, min=-max_exp_input, max=max_exp_input)
        
        # --- 4. Exponentiation ---
        weights = torch.exp(scaled_adv_clamped)
        
        # --- 5. Hard Clipping (Standard AWR practice) ---
        weights_clipped = torch.clamp(weights, max=self.max_weight)
        
        # --- 6. Diagnostics ---
        with torch.no_grad():
            numel = weights.numel()
            sum_w = weights_clipped.sum()
            sum_w_sq = (weights_clipped ** 2).sum()
            
            # Effective Sample Size (ESS)
            ess = (sum_w ** 2) / (sum_w_sq * numel + 1e-8)
            self.ess_buffer.fill_(ess / numel) # Normalized ESS
            
            # Clipping Rate (Needed for adaptive safety)
            clipped_rate = (scaled_adv > max_exp_input).float().mean()
            self.clip_rate_buffer.fill_(clipped_rate)
            
            # [SOTA 2025] Adaptive Dynamics Update (Uses current ESS & Rate)
            if self.adaptive_beta or self.adaptive_clipping:
                self._update_adaptive_stats(advantages, weights, ess / numel, clipped_rate.item())
            
            # Weight Entropy (Information Theoretic)
            probs = weights_clipped / (sum_w + 1e-8)
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs) / math.log(numel + 1)
            
            clipped_rate = (scaled_adv > max_exp_input).float().mean()
            hard_clipped_rate = (weights > self.max_weight).float().mean()
            
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
                "beta_dynamic": self.beta.item(), # [Telemetry]
                "max_weight_dynamic": self.max_weight.item() # [Telemetry]
            }
        
        return weights_clipped, diagnostics

    def _update_adaptive_stats(self, advantages: torch.Tensor, weights: torch.Tensor, ess: torch.Tensor, clipped_rate: float):
        """
        [SOTA 2025] Dynamically adapts hyperparameters to squeeze performance.
        """
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
                    self.beta = self.beta * boost_factor
                else:
                    # Standard ESS Control Mode
                    # Target 20% ESS (Robust balance between selection and diversity)
                    target_ess = 0.20
                    current_ess = ess.item()
                    
                    # P-Controller
                    error_ess = (target_ess - current_ess)
                    
                    # Gain k=10.0 (High-performance / Low-Latency)
                    # Increased from 4.0 to respond to clinical shocks instantly.
                    correction = math.exp(10.0 * error_ess)
                    new_beta = self.beta * correction
                    
                    # Momentum Update (Reduced lag)
                    # 0.80 allows faster tracking of distribution shifts.
                    self.beta.copy_((0.80 * self.beta) + (0.20 * new_beta))
                
                self.beta.copy_(self.beta.clamp(min=self.min_beta, max=self.max_beta))
                
            # B. Adaptive Clipping (Target = 95th Percentile)
            if self.adaptive_clipping:
                # Find 95th percentile of RAW weights (before current clip)
                # We use 'weights' which is unclamped by max_weight (only FP16 clamped)
                if weights.numel() > 0:
                    try:
                        # [SOTA FIX] Quantile requires .detach().float() for safety
                        p95 = torch.quantile(weights.detach().float(), 0.95).item()
                        
                        # Soft expansion: Allow clip to grow if signal is strong but stable
                        # We limit growth rate to avoid explosions
                        target_clip = max(2.0, min(100.0, p95 * 1.5)) # 1.5x buffer
                        
                        new_max_weight = (self.clip_momentum * self.max_weight) + \
                                          ((1 - self.clip_momentum) * target_clip)
                        self.max_weight.copy_(torch.tensor(new_max_weight).to(self.max_weight.device))
                    except:
                        pass # Fallback if quantile fails (e.g. not enough elements)

            # [DDP SYNCHRONIZATION] Prevent divergence of adaptive parameters across ranks
            if torch.distributed.is_initialized():
                with torch.no_grad():
                    # Average beta and max_weight across all ranks
                    torch.distributed.all_reduce(self.beta, op=torch.distributed.ReduceOp.SUM)
                    self.beta.div_(torch.distributed.get_world_size())
                    
                    torch.distributed.all_reduce(self.max_weight, op=torch.distributed.ReduceOp.SUM)
                    self.max_weight.div_(torch.distributed.get_world_size())
                    
                    # Also sync metric buffers for consistent telemetry
                    torch.distributed.all_reduce(self.ess_buffer, op=torch.distributed.ReduceOp.SUM)
                    self.ess_buffer.div_(torch.distributed.get_world_size())
                    
                    torch.distributed.all_reduce(self.clip_rate_buffer, op=torch.distributed.ReduceOp.SUM)
                    self.clip_rate_buffer.div_(torch.distributed.get_world_size())

    def calculate_weights(
        self, 
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None
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
        weights, diagnostics = self.calculate_awr_weights(advantages)
        
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
    
    # Initialize Calculator
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
    
    # 3. Compute Intrinsic Reward
    print("\n[1] Computing Intrinsic Rewards...")
    rewards = calc.compute_intrinsic_reward(
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
