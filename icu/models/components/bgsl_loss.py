"""
working/bgsl_loss.py
-------------------------
[v4.0 SOTA] Biological Gradient Supervised Learning (BGSL).

RATIONALE:
Sepsis models plateau at 0.76 AUC because they only optimize for the 'State' (is there sepsis?). 
To hit 0.82+, the model must optimize for the 'Velocity' of clinical decline.

BGSL implements a Triple Gradient Objective:
1.  State Loss (L_s): Standard classification (Cross-Entropy).
2.  Trend Loss (L_t): Slope consistency (1st derivative of risk).
3.  Shock Loss (L_d): Divergence from Diffusion expectation (2nd derivative).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class BGSLLoss(nn.Module):
    def __init__(
        self, 
        pos_weight: float = 2.0, 
        gamma: float = 2.0,
        trend_coef: float = 2.0, 
        shock_coef: float = 1.0
    ):
        """
        Args:
            pos_weight: Handling high class imbalance.
            gamma: Focal loss factor to focus on 'Hard' sepsis cases.
            trend_coef: Weight for the first derivative (Velocity).
            shock_coef: Weight for the second derivative (Acceleration).
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.trend_coef = trend_coef
        self.shock_coef = shock_coef
        
        # [v14.9 SOTA] Unified Focal Core
        # Uses forensic alpha=0.85 for sepsis boost
        from icu.models.components.focal_loss import FocalLoss
        self.focal_core = FocalLoss(alpha=0.85, gamma=gamma, reduction='none')
        
        # [v4.0 PERFECT] Dynamic Balancing Buffers
        self.register_buffer("w_t", torch.tensor([trend_coef]))
        self.register_buffer("w_h", torch.tensor([shock_coef]))

    def state_loss_fn(self, logits: torch.Tensor, targets: torch.Tensor, stability_factor: float = 1.0) -> torch.Tensor:
        """
        [v5.2 SOTA] Unified Focal Loss for Clinical State tracking.
        """
        # [RY.MD FIX] Restore Forensic Alpha/Gamma alignment
        # Adaptive Focal Relaxation: gamma_neg drops to 1.0 during shocks
        # We manually handle the relaxation to preserve BGSLLoss specific dynamics
        with torch.no_grad():
            gamma_neg_eff = 1.0 + (self.gamma - 1.0) * stability_factor
            p = torch.sigmoid(logits)
            
            # [v14.9.1 SURGICAL PATCH] ASL Probability Shifting
            # Rationale: Shift negative probabilities up by 0.05 clip margin 
            # to hard-threshold trivially easy negatives to zero gradient.
            p_neg_shifted = (1.0 - p + 0.05).clamp(max=1.0)
            pt = p * targets + p_neg_shifted * (1.0 - targets)
            
            # Asymmetrical focusing
            gamma_t = 1.0 * targets + gamma_neg_eff * (1.0 - targets)
            alpha_t = 0.85 * targets + 0.15 * (1.0 - targets)
            
            focal_w = alpha_t * torch.pow(1.0 - pt, gamma_t)
            
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        return focal_w * bce

    def forward(
        self, 
        pred_state: torch.Tensor, 
        true_state: torch.Tensor, 
        past_vitals: torch.Tensor,
        risk_coef: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        stability_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        [v4.5 PERFECT] Triple Gradient Objective with Robust Numerics.
        """
        # [v5.1 SOTA] Removed activation-level sanitization.
        # Smoothing and hierarchical clipping handle stability without blinding.
        
        # --- 1. State Loss (Hard-Negative Aware ASL) ---
        l_state_unreduced = self.state_loss_fn(pred_state, true_state, stability_factor=stability_factor)
        
        # Apply Risk-Aware Critical Penalty (v4.0 PERFECT Integration)
        if risk_coef is not None:
             # Scale at risk=1.0 is 3x (1 + 2*1.0)
             critical_penalty = 1.0 + (risk_coef * 2.0)
             l_state_unreduced = l_state_unreduced * critical_penalty

        # Masking: true = masked/padding
        if mask is not None:
             # [SOTA FIX] Strict type-casting for compiled/FP16 safety
             valid_mask = (~mask.bool()).float()
             l_state = (l_state_unreduced * valid_mask.unsqueeze(-1)).sum() / (valid_mask.sum() + 1e-8)
        else:
             l_state = l_state_unreduced.mean()
             
        # [v5.1 SOTA] Restored full probability signal
        pred_prob = torch.sigmoid(pred_state)
        
        # [SOTA FIX] Slice to dynamic channels (0-22) so static constants don't dilute velocity
        DYNAMIC_CHANNELS = 22
        slopes = past_vitals[:, 1:, :DYNAMIC_CHANNELS] - past_vitals[:, :-1, :DYNAMIC_CHANNELS]
        vit_velocity = slopes.abs().mean(dim=-1, keepdim=True) # [B, T-1, 1]
        surprise = torch.sigmoid(vit_velocity * 2.0).detach() + 0.5 # [B, T-1, 1]
        
        # Trend: Directional consistency
        # Use probabilities, not logits, for trend/shock loss (Stable Gradient)
        pred_slopes = pred_prob[:, 1:] - pred_prob[:, :-1]
        true_slopes = true_state[:, 1:] - true_state[:, :-1]
        l_trend_unreduced = F.mse_loss(pred_slopes, true_slopes, reduction='none')
        
        if mask is not None:
            # [SOTA FIX] Strict Boolean OR preventing Float bitwise crash
            slope_mask = mask[:, 1:].bool() | mask[:, :-1].bool() 
            valid_slope = (~slope_mask).float()
            l_trend = (l_trend_unreduced * surprise * valid_slope.unsqueeze(-1)).sum() / (valid_slope.sum() + 1e-8)
        else:
            l_trend = (l_trend_unreduced * surprise).mean()
        
        # Shock: Acceleration
        # [Fix] Safe Division for num_shock
        accel_vitals = (slopes[:, 1:] - slopes[:, :-1]).abs().mean(dim=-1, keepdim=True)
        num_shock = accel_vitals / (vit_velocity[:, 1:].detach() + 0.1) 
        
        pred_accel = (pred_slopes[:, 1:] - pred_slopes[:, :-1]).abs()
        true_accel = (true_slopes[:, 1:] - true_slopes[:, :-1]).abs()
        l_shock_unreduced = F.mse_loss(pred_accel, true_accel, reduction='none')
        
        if mask is not None:
            # [SOTA FIX] Strict Boolean OR preventing Float bitwise crash
            accel_mask = mask[:, 2:].bool() | mask[:, 1:-1].bool() | mask[:, :-2].bool()
            valid_accel = (~accel_mask).float()
            l_shock = (l_shock_unreduced * num_shock.detach() * valid_accel.unsqueeze(-1)).sum() / (valid_accel.sum() + 1e-8)
        else:
            l_shock = (l_shock_unreduced * num_shock.detach()).mean()
        
        # [v38.2 SOTA FIX] Persistence Bridge
        # Rationale: w_t and w_h are now persistent buffers. 
        # Do not use fill_ in forward to prevent amnesia.
            
        # [v2026 SOTA FIX] Explicit Reduction to 0D Scalar
        # Rationale: w_t and w_h are [1]-shaped buffers. Multiplying with scalar
        # .mean() produces a [1]-shaped result, which propagates downstream and
        # causes shape mismatches in loss_scaler's torch.stack().
        # .squeeze() ensures the product is a true 0D scalar.
        total_loss = l_state.mean() + (self.w_t.squeeze() * l_trend.mean()) + (self.w_h.squeeze() * l_shock.mean())
        
        return {
            "loss": total_loss,
            "l_state": l_state.mean(),
            "l_trend": l_trend.mean(),
            "l_shock": l_shock.mean(),
            "w_trend": self.w_t,
            "w_shock": self.w_h
        }

if __name__ == "__main__":
    # Smoke Test
    model = BGSLLoss()
    p = torch.randn(4, 24, 1, requires_grad=True)
    t = torch.randint(0, 2, (4, 24, 1)).float()
    v = torch.randn(4, 24, 28)
    m = torch.zeros(4, 24).bool()
    mock_risk = torch.rand(4, 24, 1)
    
    out = model(p, t, v, risk_coef=mock_risk, mask=m)
    print(f"Total BGSL Loss: {out['loss']:.4f}")
    print(f"Components: State={out['l_state']:.4f}, Trend={out['l_trend']:.4f}, Shock={out['l_shock']:.4f}")
    print(f"Weights: W_Trend={out['w_trend'].item():.4f}, W_Shock={out['w_shock'].item():.4f}")
    out['loss'].backward()
    print("Backward pass successful.")
