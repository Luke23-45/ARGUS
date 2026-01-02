"""
working/bypass_context.py
-------------------------
[v4.0 SOTA] Lateral Expert Bypass (Contextual Skip-Connection).

RATIONALE:
Standard Transformers are 'Denoising Engines' that tend to smooth out sharp, 
short-term physiological spikes (e.g., sudden tachycardia or blood pressure drops)
to produce a clear 'Mean Trajectory' for the Diffusion Planner. 

This Bypass module ensures the Sepsis Expert Head receives a 'Dirty' (High-Frequency)
version of the data alongside the 'Clean' (Smoothed) output from the Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SwiGLU(nn.Module):
    """[SOTA] Swish-Gated Linear Unit as used in NTH-Encoder."""
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * F.silu(self.w2(x))

class PhysiologicalSEBlock(nn.Module):
    """[SOTA] Channel calibration based on global sequence context."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        y = self.avg_pool(x).view(B, C)
        weights = self.fc(y).view(B, C, 1)
        return x * weights

class SymmetryGate(nn.Module):
    """[SOTA] Gated residual fusion for physiological correlations."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        p = self.proj(x)
        return self.norm(g * p + (1 - g) * x)

class ClinicalInceptionBlock(nn.Module):
    """[v4.0 PERFECT] Multi-Scale Clinical Feature Extractor with Symmetry Gating."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = out_dim // 4
        # Multi-scale paths
        self.b1 = nn.Sequential(nn.Conv1d(in_dim, mid, 3, padding=1), nn.BatchNorm1d(mid), nn.SiLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_dim, mid, 5, padding=2), nn.BatchNorm1d(mid), nn.SiLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_dim, mid, 7, padding=3), nn.BatchNorm1d(mid), nn.SiLU())
        self.b4 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_dim, mid, 1), nn.BatchNorm1d(mid), nn.SiLU())
        
        self.se = PhysiologicalSEBlock(mid * 4)
        self.gate = SymmetryGate(mid * 4)
        self.dropout = nn.Dropout1d(dropout)
        self.proj = nn.Conv1d(mid * 4, out_dim, 1)
        self.res = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.res(x)
        # Cat paths: [B, mid*4, T]
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        out = self.se(out)
        
        # Apply Symmetry Gate in [B, T, D] space
        out = out.transpose(1, 2)
        out = self.gate(out)
        out = out.transpose(1, 2)
        
        out = self.dropout(out)
        out = self.proj(out)
        
        # Final Residual + Norm
        out = out.transpose(1, 2)
        if identity.shape != out.shape:
             identity = identity.transpose(1, 2)
        return self.norm(out + identity).transpose(1, 2)

class VolatilityAwareGate(nn.Module):
    """
    [v4.0 PERFECT] Gated fusion that automatically 'opens' for physiological spikes.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Deeper gating network
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.volatility_gate = nn.Parameter(torch.tensor(1.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, smooth_ctx: torch.Tensor, raw_ctx: torch.Tensor, raw_past: torch.Tensor) -> torch.Tensor:
        # Calculate volatility: |x_t - x_{t-1}|
        x_pad = F.pad(raw_past[:, :-1, :], (0, 0, 1, 0)) 
        delta = (raw_past - x_pad).abs().mean(dim=-1, keepdim=True) # [B, T, 1]
        
        combined = torch.cat([smooth_ctx, raw_ctx], dim=-1)
        semantic_gate = self.gate_proj(combined)
        
        g = self.sigmoid(semantic_gate + (delta * self.volatility_gate))
        return g

class LateralBypass(nn.Module):
    def __init__(self, input_dim: int, d_model: int, hemo_dim: int = 7, labs_dim: int = 11, elec_dim: int = 4, static_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        # [v4.2.1 SOTA] Dynamic Groups
        self.hemo_dim = hemo_dim
        self.labs_dim = labs_dim
        self.elec_dim = elec_dim
        self.static_dim = static_dim
        
        self.hemo_proj = nn.Linear(hemo_dim, d_model // 4)
        self.labs_proj = nn.Linear(labs_dim, d_model // 4)
        self.elec_proj = nn.Linear(elec_dim, d_model // 4)
        self.other_proj = nn.Linear(static_dim, d_model // 4)
        
        self.group_gate = SymmetryGate(d_model)
        self.feat_extractor = ClinicalInceptionBlock(d_model, d_model, dropout=dropout)
        self.gate = VolatilityAwareGate(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_past: torch.Tensor, encoder_ctx: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = raw_past.shape
        L = encoder_ctx.shape[1]
        
        # [v4.0 PERFECT] Handle Static Token Alignment
        # The encoder prepends a static token (L = T + 1)
        if L == T + 1:
            static_token = encoder_ctx[:, :1, :]
            temporal_ctx = encoder_ctx[:, 1:, :]
            # [v4.0 FIX] Also split the mask if it matches ctx length
            if mask is not None and mask.size(1) == L:
                static_mask = mask[:, :1]
                temporal_mask = mask[:, 1:]
            else:
                static_mask = None
                temporal_mask = mask
        else:
            static_token = None
            temporal_ctx = encoder_ctx
            static_mask = None
            temporal_mask = mask

        # 1. Grouped Projection [v4.2.1 SOTA Dynamic]
        idx_hemo = self.hemo_dim
        idx_labs = self.hemo_dim + self.labs_dim
        idx_elec = self.hemo_dim + self.labs_dim + self.elec_dim
        
        z_hemo = self.hemo_proj(raw_past[:, :, :idx_hemo])
        z_labs = self.labs_proj(raw_past[:, :, idx_hemo:idx_labs])
        z_elec = self.elec_proj(raw_past[:, :, idx_labs:idx_elec])
        z_other = self.other_proj(raw_past[:, :, idx_elec:])
        z_raw = torch.cat([z_hemo, z_labs, z_elec, z_other], dim=-1)
        
        # [SOTA] Gated multi-modal fusion
        z_raw = self.group_gate(z_raw)
        
        # 2. Multi-Scale extraction [B, D, T] -> [B, T, D]
        x_bypass_latent = self.feat_extractor(z_raw.transpose(1, 2)).transpose(1, 2)
        x_bypass_latent = self.dropout(x_bypass_latent)
        
        if temporal_mask is not None:
            # Mask temporal segments: [B, T]
            x_bypass_latent = x_bypass_latent.masked_fill(temporal_mask.unsqueeze(-1), 0.0)
        
        # 3. Volatility-Aware Gating (Applied only to temporal context)
        gate_weights = self.gate(temporal_ctx, x_bypass_latent, raw_past)
        
        # 4. Expert Manifold Fusion
        sharp_temporal = temporal_ctx + (gate_weights * x_bypass_latent)
        
        if temporal_mask is not None:
            sharp_temporal = sharp_temporal.masked_fill(temporal_mask.unsqueeze(-1), 0.0)
        
        # Support for static token mask if needed
        if static_token is not None and static_mask is not None:
             static_token = static_token.masked_fill(static_mask.unsqueeze(-1), 0.0)
        
        # Recombine with static token if present
        if static_token is not None:
            sharp_ctx = torch.cat([static_token, sharp_temporal], dim=1)
        else:
            sharp_ctx = sharp_temporal
            
        return sharp_ctx

if __name__ == "__main__":
    # Smoke Test
    model = LateralBypass(input_dim=28, d_model=512)
    mock_raw = torch.randn(16, 24, 28)
    mock_enc = torch.randn(16, 24, 512)
    
    out = model(mock_raw, mock_enc)
    print(f"Bypass Output Shape: {out.shape}") # Expected [16, 24, 512]
    assert out.shape == (16, 24, 512)
    print("LateralBypass: SOTA v4.0 Validation Passed.")
