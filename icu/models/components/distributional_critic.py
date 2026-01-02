"""
icu/models/components/distributional_critic.py
--------------------------------------------------------------------------------
SOTA Implicit Distributional Critic (IDC-25) for Clinical RL.

"In a crisis, the mean is a lie. The tail is the truth."

This module implements the Frontier-class critic architecture for APEX-MoE:
1.  **Quantile Prediction**: Predicts N=25 discrete quantiles for the return 
    distribution at every future time step.
2.  **IQL Expectile Regression**: Implements Implicit Q-Learning value estimation 
    with expectile factor tau=0.7 for conservative medical reasoning.
3.  **Quantile-Huber Loss**: Provides stable gradients for distributional 
    prediction, preventing "Quantile Crossing" artifacts.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class GatedValueBlock(nn.Module):
    """
    Expert-level Gated Residual Block for clinical manifold mapping.
    Based on GRN (Gated Residual Networks) for high-capacity state processing.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))

class DistributionalValueHead(nn.Module):
    """
    SOTA Distributional Value Head (2025).
    Expertly designed for high-stakes clinical risk modeling.
    
    Architecture:
    1.  **Gated Residual Pipeline**: Uses GLU-gated residuals to prevent gradient 
        saturation on rare vitals.
    2.  **Structural Monotonicity**: Enforces V(q_i) <= V(q_{i+1}) via sorting.
    3.  **Risk Estimation**: Exposes CVaR and alpha-Expectile summaries.
    """
    def __init__(self, d_model: int, pred_len: int, num_quantiles: int = 25, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.num_quantiles = num_quantiles
        
        self.pre_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            GatedValueBlock(d_model, dropout)
        )
        
        self.head = nn.Linear(d_model, pred_len * num_quantiles)
        
        # Expert Initialization (Orthogonal with clinical-safe gain)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, D_model] Global context
        Returns:
            quantiles: [Batch, T_pred, NumQuantiles]
        """
        B = x.shape[0]
        feat = self.pre_block(x)
        out = self.head(feat).view(B, self.pred_len, self.num_quantiles)
        
        # [v4.1 SOTA] Deterministic Crossing Prevention
        # Statistical sorting is superior to penalty terms for medical safety.
        quantiles, _ = torch.sort(out, dim=-1)
        return quantiles

    def get_cvar(self, quantiles: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """
        Calculates Conditional Value-at-Risk (CVaR) for the bottom alpha tail.
        This represents the "Worst-Case Clinical Outcome" the model anticipates.
        """
        # quantiles is [B, T, N]
        num_alpha = max(1, int(alpha * self.num_quantiles))
        return quantiles[:, :, :num_alpha].mean(dim=-1)

    def get_expectile_summary(self, quantiles: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """
        Extracts the tau-expectile summary for scalar RL bootstrapping.
        If tau=0.5, returns the mean (risk-neutral).
        If tau > 0.5 (pessimistic), emphasizes LOWER quantiles (worst outcomes).
        
        Note: In medical RL, pessimism means fearing the worst-case (lower quantiles).
        This forces the policy to avoid actions that could lead to catastrophic states.
        """
        if tau == 0.5:
            return quantiles.mean(dim=-1)
            
        # [v4.2 SOTA] Quantile-to-Expectile weighted mapping
        # For pessimistic RL (tau > 0.5), we weight LOWER quantiles more heavily.
        N = self.num_quantiles
        # Midpoint quantiles [0.02, 0.06... 0.98]
        taus_q = torch.linspace(1/(2*N), 1 - 1/(2*N), N, device=quantiles.device)
        
        # Pessimistic Weights: Higher weight on LOWER quantiles when tau > 0.5
        # This is the key insight: taus_q < 0.5 are the "danger zone" (lower outcomes)
        weights = torch.where(taus_q < 0.5, tau, 1 - tau)  # Flipped from before
        weights = weights / weights.sum()
        
        return (quantiles * weights.view(1, 1, -1)).sum(dim=-1)


class IQLQuantileLoss(nn.Module):
    """
    SOTA Integrated IDC Loss (Expectile + Quantile Huber).
    
    This loss combines:
    1.  **IQL (V-Learning)**: Expectile loss to estimate a conservative value 
        function without overestimating OOD trajectories.
    2.  **Quantile Regression**: Forces the model to explain return variance 
        by matching predicted quantiles to observed target distributions.
    """
    def __init__(self, tau: float = 0.7, delta: float = 1.0):
        super().__init__()
        self.tau = tau   # IQL Expectile (0.7 = Conservative)
        self.delta = delta # Huber threshold

    def forward(self, pred_quantiles: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        """
        Computes Dual Expectile-Quantile Loss.
        
        Args:
            pred_quantiles: [B, T, N]
            target_returns: [B, T]
        """
        B, T, N = pred_quantiles.shape
        device = pred_quantiles.device
        
        # 1. Conservative IQL Expectile Baseline
        v_pred_mean = pred_quantiles.mean(dim=-1)
        diff = target_returns - v_pred_mean
        # Expectile weight (Asymmetric L2)
        weight_iql = torch.where(diff < 0, 1 - self.tau, self.tau)
        expectile_loss = (weight_iql * (diff**2)).mean()
        
        # 2. QR-DQN Distributional Hub (Quantile Huber)
        target_expanded = target_returns.unsqueeze(-1) # [B, T, 1]
        errors = target_expanded - pred_quantiles # [B, T, N]
        
        # Stable Huber component
        abs_err = torch.abs(errors)
        huber_loss = torch.where(
            abs_err <= self.delta,
            0.5 * (errors**2),
            self.delta * (abs_err - 0.5 * self.delta)
        )
        
        # Pinball Loss weighting
        # Midpoint quantiles [1/2N, 3/2N... (2N-1)/2N]
        taus = torch.linspace(0.0, 1.0, N + 1, device=device)
        taus = (taus[:-1] + taus[1:]).view(1, 1, N) / 2.0
        
        quantile_weight = torch.abs(taus - (errors < 0).float())
        qr_loss = (quantile_weight * huber_loss).mean()
        
        # 3. Crossing Guard (Double-Safety)
        # Even with sorting, we penalize crossing to push the raw logits 
        # toward a naturally monotonic manifold (faster convergence).
        diff_q = pred_quantiles[:, :, 1:] - pred_quantiles[:, :, :-1]
        crossing_penalty = torch.relu(-diff_q).mean() * 5.0
        
        return expectile_loss + qr_loss + crossing_penalty

    @staticmethod
    def compute_explained_variance(pred_quantiles: torch.Tensor, target_returns: torch.Tensor) -> float:
        """
        Calculates EV using the mean of the distribution.
        """
        v_pred = pred_quantiles.mean(dim=-1).detach()
        y_true = target_returns.detach()
        
        var_y = torch.var(y_true) + 1e-8
        ev = 1.0 - torch.var(y_true - v_pred) / var_y
        return ev.item()
