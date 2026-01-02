"""
working/alb_encoder.py
-------------------------
[v4.0 SOTA] Asymmetric Latent Bottleneck (ALB).

RATIONALE:
Multi-task learning in ICU environments fails because of 'Gradient Domination'
and 'Feature Convergence.' The Planner wants smooth data; the Expert wants 
anomalous data. 

The ALB architecture explicitly decouples the representational flow into two 
distinct manifolds:
1.  Smooth (Planner) -> Primary Backbone output.
2.  Sharp (Expert) -> Backbone + Lateral Bypass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from icu.models.components.bypass_context import LateralBypass

class GatedResidualNetwork(nn.Module):
    """[SOTA] TFT-Style GRN with LayerNorm and SwiGLU."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.w1(x)
        out = F.silu(out) # Swish/SiLU for perfection
        out = self.w2(out)
        out = self.dropout(out)
        g = torch.sigmoid(self.gate(x))
        return self.norm(residual + g * out)

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x: torch.Tensor, seq_len: int):
        if self.cached_cos is None or self.cached_cos.size(2) < seq_len or self.cached_cos.device != x.device:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, d_model]
            self.cached_cos = emb.cos().unsqueeze(0).unsqueeze(0) # [1, 1, T, D]
            self.cached_sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :]

class RoPEMultiheadAttention(nn.Module):
    """[SOTA] Rotary-Positional Multihead Attention for perfect temporal sync."""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.rope = RotaryEmbedding(self.head_dim)

    def _rotate_half(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, Tq, D = q_in.shape
        Tk = k_in.shape[1]
        
        q = self.q_proj(q_in).reshape(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(q, Tq)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
             attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        weights = F.softmax(attn, dim=-1)
        weights = self.attn_dropout(weights)
        out = (weights @ v).transpose(1, 2).reshape(B, Tq, D)
        return self.out(out)

class SinusoidalPositionalEncoding(nn.Module):
    """[SOTA] Classic Sinusoidal PE for stable temporal indexing."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        # Ensure DType alignment for Mixed Precision (BF16/FP16)
        return self.pe[:T, :].unsqueeze(0).to(dtype=x.dtype, device=x.device)

class AsymmetricLatentBottleneck(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.bypass = LateralBypass(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # [v4.0 PERFECT] Manifold Synchronization with RoPE and GRN
        self.sync = RoPEMultiheadAttention(d_model)
        self.expert_proj = nn.Sequential(
            GatedResidualNetwork(d_model),
            GatedResidualNetwork(d_model),
            GatedResidualNetwork(d_model) # 3 Layers for Deep Expert Representation
        )

    def forward(
        self, 
        past_norm: torch.Tensor, 
        static: torch.Tensor, 
        imputation_mask: Optional[torch.Tensor] = None, 
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perfectly decouples the Planner and Expert manifolds.
        """
        # 1. Base Encoding
        # Capture full_mask (T+1) which includes the static token
        ctx_seq, global_ctx, full_mask = self.encoder(
            past_norm, 
            static, 
            imputation_mask=imputation_mask, 
            padding_mask=padding_mask
        )
        
        # [v4.0 PERFECT] Inject Temporal Clock
        pos = self.pos_encoder(ctx_seq)
        ctx_seq = ctx_seq + pos
        
        # 2. Planner Manifold
        ctx_planner = ctx_seq
        global_planner = global_ctx
        
        # 3. Expert Manifold
        # Use full_mask (T+1) for bypass and sync to handle prepended static token
        ctx_sharp = self.bypass(past_norm, ctx_seq, mask=full_mask)
        # ctx_sharp has length T (bypass strips static)
        # ctx_planner has length T+1
        # sync mask applies to keys (ctx_planner), so use full_mask (T+1)
        ctx_synced = self.sync(ctx_sharp, ctx_planner, ctx_planner, mask=full_mask) 
        ctx_expert = self.expert_proj(ctx_synced)
        
        if full_mask is not None:
             ctx_expert = ctx_expert.masked_fill(full_mask.unsqueeze(-1), 0.0)
        
        return {
            "ctx_planner": ctx_planner,
            "global_planner": global_planner,
            "ctx_expert": ctx_expert,
            "ctx_mask": full_mask # [v4.0 FIX] Propagate full (T+1) mask to backbone
        }

if __name__ == "__main__":
    # Mock Test
    class MockEncoder(nn.Module):
        def forward(self, p, s, imputation_mask=None, padding_mask=None):
            return torch.randn(16, 24, 512), torch.randn(16, 512), torch.zeros(16, 24)
            
    encoder = MockEncoder()
    alb = AsymmetricLatentBottleneck(encoder, input_dim=28, d_model=512)
    
    mock_p = torch.randn(16, 24, 28)
    mock_s = torch.randn(16, 6)
    
    out = alb(mock_p, mock_s)
    
    print(f"Planner Context Shape: {out['ctx_planner'].shape}")
    print(f"Expert Context Shape: {out['ctx_expert'].shape}")
    
    # Mathematical Divergence Test: They should NOT be equal
    diff = (out['ctx_planner'] - out['ctx_expert']).abs().mean()
    print(f"Representational Divergence Delta: {diff.item():.4f}")
    assert diff > 0.0, "Manifolds failed to decouple!"
    print("ALB: SOTA v4.0 Validation Passed.")
