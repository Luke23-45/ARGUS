"""
icu/utils/advantage_calculator.py
--------------------------------------------------------------------------------
SOTA Advantage Engine for Offline Clinical Reinforcement Learning.

This module implements:
1.  **Generalized Advantage Estimation (GAE)**: TD(lambda) for time-series credit assignment.
2.  **Global Whitening**: Standardizing advantages to N(0,1) for stable exponentiation.
3.  **ESS Monitoring**: Tracking Effective Sample Size to prevent mode collapse.
4.  **Clinical Physics Rewards**: qSOFA and Lactate-driven dense reward signals.

Reference: 
    - Peng et al. "Advantage-Weighted Regression: Simple and Scalable Off-Policy RL"
    - Schulman et al. "High-Dimensional Continuous Control Using GAE"
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, List, Union, Dict, Any

logger = logging.getLogger(__name__)

class ICUAdvantageCalculator:
    """
    State-of-the-Art Advantage Calculator for ICU Trajectories.
    
    Why this matters:
    Standard MSE loss treats all time-steps as equal. Clinical outcomes are 
    determined by critical turning points. This calculator identifies those
    pivotal moments and weights the gradient accordingly.
    """
    def __init__(
        self, 
        beta: float = 0.5,           # Temperature (lower = peakier weights)
        gamma: float = 0.99,         # Discount factor
        lambda_gae: float = 0.95,    # GAE variance control (0.95 is standard)
        max_weight: float = 20.0,    # Clipping to prevent gradient explosions
        qsofa_thresholds: Dict[str, float] = {'resp': 22.0, 'sbp': 100.0}
    ):
        self.beta = beta
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.max_weight = max_weight
        self.qsofa_thresholds = qsofa_thresholds

        # Global Statistics for Whitening (Running Mean/Std)
        # We use Welford's algorithm logic or static global set
        self.adv_mean = 0.0
        self.adv_std = 1.0
        self.stats_initialized = False

    def set_stats(self, mean: float, std: float):
        """
        Manually sets the normalization statistics.
        Usually called after a pass over the entire dataset (Offline RL).
        """
        self.adv_mean = mean
        self.adv_std = std if std > 1e-8 else 1.0
        self.stats_initialized = True
        logger.info(f"[AWR] Stats Initialized: Mean={self.adv_mean:.4f}, Std={self.adv_std:.4f}")

    def compute_clinical_reward(
        self, 
        vitals: torch.Tensor, 
        outcome_label: torch.Tensor,
        # CRITICAL: Indices MUST match CANONICAL_COLUMNS in dataset.py
        # HR=0, O2Sat=1, SBP=2, DBP=3, MAP=4, Resp=5, Temp=6, Lactate=7
        feature_indices: Dict[str, int] = {'sbp': 2, 'resp': 5, 'lactate': 7},
        normalizer: Optional[Any] = None  # <--- PATCH: Unit Awareness
    ) -> torch.Tensor:
        """
        Computes a Dense Physiological Reward signal.
        
        CRITICAL SAFETY:
        This function requires REAL CLINICAL UNITS (mmHg, bpm).
        If 'vitals' are normalized [-1, 1], you MUST pass the 'normalizer' object
        to restore the physical units. Otherwise, thresholds (SBP < 100) will break.
        """
        # --- 1. Unit Safety Logic (The Patch) ---
        if normalizer is not None:
            # Safely revert to clinical units (mmHg/bpm)
            # This handles the [-1, 1] -> [0, 300] conversion
            vitals = normalizer.denormalize(vitals)
        
        # Heuristic Safety Check:
        # If we didn't unnormalize, but the data looks normalized (Max SBP < 20),
        # and our threshold is absolute (> 80), this is likely a bug.
        idx_sbp_check = feature_indices.get('sbp', 2)
        if idx_sbp_check < vitals.shape[-1]:
            sbp_max = vitals[..., idx_sbp_check].max()
            sbp_thresh = self.qsofa_thresholds.get('sbp', 100.0)
            if sbp_max < 20.0 and sbp_thresh > 80.0:
                logger.warning(
                    f"CRITICAL: Reward inputs look Normalized (Max SBP={sbp_max:.2f}) "
                    f"but Threshold is Absolute ({sbp_thresh}). "
                    f"Pass 'normalizer' or use Raw Vitals!"
                )
        # ----------------------------------------

        B, T, D = vitals.shape
        device = vitals.device
        
        # 1. Base Sparse Reward (Outcome)
        rewards = torch.zeros(B, T, device=device)
        terminal_reward = (1.0 - outcome_label) * 5.0 + (outcome_label * -5.0)
        rewards[:, -1] += terminal_reward

        # 2. Sepsis-3 Safety Guardrails (2016 Consensus updates)
        idx_sbp = feature_indices.get('sbp', 2)
        idx_resp = feature_indices.get('resp', 5) # Corrected to CANONICAL index 5
        
        if idx_sbp < D and idx_resp < D:
            sbp = vitals[..., idx_sbp]
            resp = vitals[..., idx_resp]
            
            # Sepsis-3: SBP <= 100 mmHg is a critical marker for Shock/Organ Failure
            # We use a steeper sigmoid centered at 100 to enforce the SOTA boundary
            pen_sbp = torch.sigmoid((100.0 - sbp) / 5.0) 
            
            # Resp > 22 bpm (qSOFA criteria)
            pen_resp = torch.sigmoid((resp - 22.0) / 3.0)
            
            # Weighting: SBP is more critical for Shock detection
            rewards -= 0.2 * pen_sbp + 0.05 * pen_resp
            
            # [SOTA 2025] Proactive Improvement Reward (Delta)
            # Reward increase in SBP if it was low, or decrease in Lactate
            if T > 1:
                # SBP Delta: positive if increasing
                sbp_delta = sbp[:, 1:] - sbp[:, :-1]
                # Only reward SBP increase if it's below 110 (prevent hypertension)
                rewards[:, 1:] += 0.01 * torch.clamp(sbp_delta, min=0) * (sbp[:, 1:] < 110.0).float()

        # 3. Lactate Penalty (The Metabolic Safety Boundary)
        idx_lac = feature_indices.get('lactate', 7)
        if idx_lac < D:
            lactate = vitals[..., idx_lac]
            # Sepsis-3: Lactate > 2.0 mmol/L defines Metabolic Dysfunction
            pen_lac = F.relu(lactate - 2.0) 
            rewards -= 0.3 * pen_lac  
            
            # [SOTA 2025] Metabolic Recovery Reward
            if T > 1:
                lac_delta = lactate[:, 1:] - lactate[:, :-1]
                # Reward Lactate DECREASE (negative delta)
                rewards[:, 1:] += 0.05 * torch.clamp(-lac_delta, min=0)

        return rewards


    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized Generalized Advantage Estimation (GAE).
        
        Calculates:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + (gamma * lambda) * A_{t+1}
            
        Args:
            rewards: [B, T]
            values: [B, T] (Predicted by Value Head)
        
        Returns:
            advantages: [B, T]
        """
        B, T = rewards.shape
        device = rewards.device
        
        # Create "Next Values" by shifting.
        # [SAFETY FIX] Bootstrap with last value instead of zero
        # Original bug: Zero-padding assumed terminal state at every window end,
        # causing the critic to predict a "crash" at the end of each window.
        # Bootstrapping is more conservative for sliding windows of ongoing stays.
        next_values = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
        
        # 1. Temporal Difference Error (Delta)
        # delta = r + gamma * V_next - V_curr
        deltas = rewards + (self.gamma * next_values) - values
        
        # 2. GAE Recursion (Vectorized loop for T steps)
        # We calculate backward from T-1 to 0
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0.0
        
        # We must loop because A_t depends on A_{t+1}
        # This is fast for T=30 (typical window)
        for t in reversed(range(T)):
            # GAE formula: delta_t + (gamma * lambda) * previous_advantage
            # [SAFETY FIX] Detach the future advantage to treat it as a static target
            # for the current timestep's weight.
            is_tensor = isinstance(last_gae_lam, torch.Tensor)
            prev_adv = last_gae_lam.detach() if is_tensor else last_gae_lam
            
            advantages[:, t] = deltas[:, t] + (self.gamma * self.lambda_gae) * prev_adv
            last_gae_lam = advantages[:, t]
            
        return advantages

    def calculate_weights(
        self, 
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the safe AWR weights and gathers diagnostics.
        
        Formula: weight = exp( (A - mu) / sigma / beta )
        
        Returns:
            weights: [B, T] Tensor (clamped)
            diagnostics: Dict of metrics (ESS, Max Weight, Explained Var)
        """
        # 1. Global Whitening (Standardization)
        if self.stats_initialized:
            mu = self.adv_mean
            sigma = self.adv_std
        else:
            mu = advantages.mean()
            sigma = advantages.std() + 1e-8

        normalized_adv = (advantages - mu) / sigma

        # 2. Exponentiation with Temperature
        raw_weights = torch.exp(normalized_adv / self.beta)
        
        # 3. Safety Clipping
        weights = torch.clamp(raw_weights, max=self.max_weight)
        
        # 4. Diagnostics: Effective Sample Size (ESS)
        sum_w = weights.sum()
        sum_w_sq = (weights ** 2).sum()
        
        if sum_w_sq == 0:
            ess_val = 0.0
        else:
            # Normalized ESS in [0, 1]
            ess_val = (sum_w ** 2) / (sum_w_sq * weights.numel())
            
        # 5. Explained Variance
        # Formula: 1 - Var(Returns - Values) / Var(Returns)
        exp_var = 0.0
        if values is not None and rewards is not None:
            # Simple approximation of returns (Reward-to-go)
            # In AWR, returns = target values for the critic
            # We can use the simple sum of rewards as a proxy for the 'Y' in ESS formula
            # Y = rewards (discounted)
            target_returns = rewards # Simplification for diagnostic
            y_diff = target_returns - values
            var_y = torch.var(target_returns)
            var_diff = torch.var(y_diff)
            if var_y > 1e-8:
                exp_var = (1.0 - var_diff / var_y).item()

        diagnostics = {
            "ess": ess_val.item(),
            "max_weight": weights.max().item(),
            "explained_variance": exp_var
        }
            
        return weights, diagnostics

# ==============================================================================
# Smoke Test / Example Usage
# ==============================================================================
if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(level=logging.INFO)
    
    # 1. Initialize
    calc = ICUAdvantageCalculator(beta=1.0)
    
    # 2. Mock Data (Clinical Units)
    B, T, D = 4, 30, 28
    
    # Unnormalized Vitals (e.g. SBP=120, SBP=80 for shock patient)
    vitals = torch.randn(B, T, D) * 10 + 100 
    vitals[0, :, 2] = 120.0  # Healthy SBP
    vitals[1, :, 2] = 80.0   # Shock SBP
    
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0]) # 2 stable, 2 shock
    
    # 3. Predicted Baseline (Values from model)
    pred_values = torch.randn(B, T) # The model's guess of "how good is this state?"
    
    # 4. Pipeline
    # A. Compute Reward
    rewards = calc.compute_clinical_reward(vitals, labels)
    print(f"Rewards Shape: {rewards.shape}")
    print(f"Reward Mean (Stable vs Shock): {rewards[0].mean():.2f} vs {rewards[1].mean():.2f}")
    
    # B. Compute GAE
    advantages = calc.compute_gae(rewards, pred_values)
    print(f"Advantages Shape: {advantages.shape}")
    
    # C. Fit Stats (Pretend we saw the whole dataset)
    calc.set_stats(mean=advantages.mean().item(), std=advantages.std().item())
    
    # D. Get Weights
    weights, ess = calc.calculate_weights(advantages)
    
    print(f"AWR Weights Shape: {weights.shape}")
    print(f"Effective Sample Size: {ess:.2%}")
    print(f"Weight Range: {weights.min():.4f} - {weights.max():.4f}")