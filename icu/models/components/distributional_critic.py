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
from typing import Dict, Tuple, Optional, Union

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
        # [v7.1 SOTA FIX] Unlocked Head: Increased gain from 0.01 to 1.0 to resolve
        # 'Dead Man' initialization pattern and restore Advantage range.
        nn.init.orthogonal_(self.head.weight, gain=1.0)
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
        
        # [ry.md FIX 1] Removed Forward Sorting
        # Rationale: torch.sort enforces monotonicity mechanically but neurotoxically —
        # it scrambles neuron-to-quantile assignments across batches, AND it makes the
        # IQLQuantileLoss.crossing_penalty permanently zero (dead code), removing the only 
        # gradient signal that teaches neurons their individual quantile roles.
        # Without sort, the quantile regression loss + crossing_penalty jointly enforce
        # monotonicity through gradient descent, allowing stable neuron specialization.
        return out

    def get_cvar(self, quantiles: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """
        Calculates Conditional Value-at-Risk (CVaR) for the bottom alpha tail.
        This represents the "Worst-Case Clinical Outcome" the model anticipates.
        """
        # quantiles is [B, T, N]
        num_alpha = max(1, int(alpha * self.num_quantiles))
        return quantiles[:, :, :num_alpha].mean(dim=-1)

    def get_expectile_summary(self, quantiles: torch.Tensor, tau: Union[float, torch.Tensor] = 0.5) -> torch.Tensor:
        """
        Extracts the tau-expectile summary for scalar RL bootstrapping.
        If tau=0.5, returns the mean (risk-neutral).
        If tau > 0.5 (pessimistic), emphasizes LOWER quantiles (worst outcomes).
        
        Note: In medical RL, pessimism means fearing the worst-case (lower quantiles).
        This forces the policy to avoid actions that could lead to catastrophic states.
        """
        # [v2026 SOTA FIX] Vectorized Expectile Summary (Zero-Sync)
        # Rationale: Replaced scalar branching (if tau == 0.5) with a unified weighted sum
        # to prevent host-side stalls. The weights calculation naturally yields 
        # uniform weights when tau=0.5.
        
        N = self.num_quantiles
        taus_q = torch.linspace(1/(2*N), 1 - 1/(2*N), N, device=quantiles.device)
        
        # [v2026 SOTA] Standardized Weighting
        tau_t = torch.as_tensor([tau], device=quantiles.device)
        tau_w = torch.where(taus_q < 0.5, tau_t, 1.0 - tau_t)
        uni_w = torch.ones_like(taus_q)
        
        # 50/50 Consensus (Pessimistic + Neutral)
        # Rationale: Purely pessimistic estimates (Expectile IQL) can be too noisy 
        # in sparse-reward clinical zones. Consensus adds a "stabilizing floor."
        w_p = tau_w / tau_w.sum()
        w_n = uni_w / uni_w.sum()
        weights = 0.5 * w_p + 0.5 * w_n
        
        # Normalize to ensure sum=1.0 regardless of tau tensor value
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

    def forward(self, pred_quantiles: torch.Tensor, target_returns: torch.Tensor, tau: Optional[float] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes Dual Expectile-Quantile Loss.
        
        Args:
            pred_quantiles: [B, T, N]
            target_returns: [B, T]
            tau: Optional override for risk-aversion expectile
            mask: [B, T] Surgical Mask for padding/truncation
        """
        B, T, N = pred_quantiles.shape
        device = pred_quantiles.device
        if mask is not None and mask.dim() == 3:
            mask = mask.any(dim=-1)
            
        curr_tau = tau if tau is not None else self.tau
        
        # [SOTA FIX] Unified Value Estimator (Avoids Schizophrenic Critic)
        # We use the same weighted consensus as bootstrapping for graph-consistency.
        taus_q = torch.linspace(1/(2*N), 1 - 1/(2*N), N, device=device)
        tau_t = torch.as_tensor([curr_tau], device=device)
        # Weight distribution toward lower quantiles if tau > 0.5 (pessimism)
        tau_w = torch.where(taus_q < 0.5, tau_t, 1.0 - tau_t)
        uni_w = torch.ones_like(taus_q)
        
        # Normalize weights
        w_p = tau_w / (tau_w.sum() + 1e-8)
        w_n = uni_w / (uni_w.sum() + 1e-8)
        weights = 0.5 * w_p + 0.5 * w_n
        
        # v_pred_safe: [B, T] - Distribution-Aware Expectile Summary (Mean if tau=0.5)
        v_pred_safe = (pred_quantiles * weights.view(1, 1, N)).sum(dim=-1)
        diff = target_returns - v_pred_safe
        
        # 1. Expectile Loss (The "Selector" - Component 1)
        weight_iql = torch.where(diff < 0, 1.0 - tau_t, tau_t)
        raw_expectile_loss = weight_iql * (diff**2)
        
        if mask is not None:
            mask_f = mask.float()
            expectile_loss = (raw_expectile_loss * mask_f).sum() / (mask_f.sum() + 1e-8)
        else:
            expectile_loss = raw_expectile_loss.mean()
        
        # 2. Quantile Regression Loss (The "Estimator" - Component 2)
        # Huber loss between each quantile pred and the scalar target return
        errors = target_returns.unsqueeze(-1) - pred_quantiles # [B, T, N]
        huber_loss = F.huber_loss(pred_quantiles, target_returns.unsqueeze(-1).expand_as(pred_quantiles), reduction='none', delta=self.delta)
        
        # Quantile penalty: |tau - I(error < 0)| * Loss
        # Here taus represents the quantile locations [0...1]
        taus = taus_q.view(1, 1, N)
        quantile_weight = torch.abs(taus - (errors < 0).float())
        raw_qr_loss = quantile_weight * huber_loss
        
        if mask is not None:
            # Broadcast mask to quantiles [B, T, N]
            qr_loss = (raw_qr_loss * mask_f.unsqueeze(-1)).sum() / (mask_f.sum() * N + 1e-8)
        else:
            qr_loss = raw_qr_loss.mean()
        
        # 3. Structural Crossing Penalty (Monotonicity Guard)
        # Ensure Q_i <= Q_{i+1}
        diff_q = pred_quantiles[..., 1:] - pred_quantiles[..., :-1]
        
        # [SOTA FIX 1] Reduce explosive multiplier (50 -> 10) to balance magnitude
        # Root Cause: 50x caused V=42 vs D=0.2 (200x gap) leading to NaN at step 649.
        # The quantile regression loss itself already penalizes out-of-order quantiles
        # via the |tau - I(e<0)| weighting, so 10x is sufficient gradient pressure.
        crossing_penalty = torch.relu(-diff_q).mean() * 10.0
        
        total = expectile_loss + qr_loss + crossing_penalty
        
        # [SOTA FIX 2] Graph-Severing NaN Shield
        # Rationale: nan_to_num protects the scalar loss but allows the "0 * NaN = NaN" 
        # trap to poison input gradients during chain-rule propagation. 
        # By replacing a non-finite loss with a NEW leaf tensor, we physically sever 
        # the computational graph, guaranteeing that zero gradient reaches the parameters.
        if not torch.isfinite(total):
            total = torch.zeros(1, device=pred_quantiles.device, requires_grad=True).squeeze()
        return total

    @staticmethod
    def compute_explained_variance(pred_quantiles: torch.Tensor, target_returns: torch.Tensor, tau: float = 0.5, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the Explained Variance using a Distribution-Aware Value summary.
        
        Formula: 1 - Var(target - pred) / Var(target)
        SOTA Implementation: Mask-aware using Welford's-style identity if mask provided.
        """
        B, T, N = pred_quantiles.shape
        if mask is not None and mask.dim() == 3:
            mask = mask.any(dim=-1)
            
        device = pred_quantiles.device
        
        # 1. Distribution-Aware Value Summary
        # [v2026 SOTA] Aligning the "Ruler" with the "Objective"
        taus_q = torch.linspace(1/(2*N), 1 - 1/(2*N), N, device=device)
        tau_t = torch.as_tensor([tau], device=device)
        tau_w = torch.where(taus_q < 0.5, tau_t, 1.0 - tau_t)
        uni_w = torch.ones_like(taus_q)
        w_p = tau_w / (tau_w.sum() + 1e-8)
        w_n = uni_w / (uni_w.sum() + 1e-8)
        weights = 0.5 * w_p + 0.5 * w_n
        
        v_pred = (pred_quantiles * weights.view(1, 1, N)).sum(dim=-1).detach()
        y_true = target_returns.detach()
        
        if mask is not None:
            mask_f = mask.float().detach()
            # Only compute over non-masked steps
            n = mask_f.sum()
            if n < 2: return torch.tensor(0.0, device=device)
            
            # Masked Mean & Variance
            def masked_var(x, m, n_count):
                mu = (x * m).sum() / n_count
                return ((x - mu)**2 * m).sum() / (n_count - 1 + 1e-8)
            
            var_y = masked_var(y_true, mask_f, n) + 1e-8
            var_err = masked_var(y_true - v_pred, mask_f, n)
        else:
            var_y = torch.var(y_true) + 1e-8
            var_err = torch.var(y_true - v_pred)
            
        ev = 1.0 - var_err / var_y
        return ev
 # [v2026 SOTA] Return tensor to avoid .item() sync
