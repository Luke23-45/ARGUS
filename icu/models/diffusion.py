"""
icu/models/diffusion_icu.py
--------------------------------------------------------------------------------
State-of-the-Art Unified Diffusion Planner for ICU Forecasting (SOTA v3.0).

Status: PRODUCTION-READY / SAFETY-CRITICAL
Architectural Pillars:
1.  **DiT-1D Backbone**: AdaLN-Zero conditioning for stable diffusion generation.
2.  **Causal RoPE**: Unified temporal coordinate system (History: 0-23, Future: 24-29).
3.  **Mask-Aware Physics**: Attention and Pooling layers strictly ignore padding.
    (Prevents "Zero-Shock" hallucinations where missing data is interpreted as cardiac arrest).
4.  **SwiGLU & Flash Attention**: High-performance compute primitives.
5.  **APEX-MoE Ready**: Exposes Auxiliary Heads and Latent Hooks for Phase 2.

References:
    - Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT)
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - OpenAI, "Improved Denoising Diffusion Probabilistic Models"
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# [PATCH 3] Import gradient checkpointing handler
from torch.utils.checkpoint import checkpoint

# Setup Logger
logger = logging.getLogger("ICU_Diffusion_SOTA")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class ICUConfig:
    """
    Hyperparameter Configuration for APEX-MoE Diffusion.
    """
    # Dimensions (Frontier 28 Specification)
    # CRITICAL: input_dim MUST match CANONICAL_COLUMNS (28 channels)
    # The full [T, 28] matrix includes static context at indices 22-27
    input_dim: int = 28     # [Hemodynamic(7) + Labs(11) + Electrolytes(4) + Static(6)]
    static_dim: int = 6     # [Age, Gender, Unit1, Unit2, AdmTime, LOS]
    history_len: int = 24   # T_obs
    pred_len: int = 6       # T_pred

    # Transformer Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6       # DiT Depth
    encoder_layers: int = 4 # History Encoder Depth
    ffn_dim_ratio: int = 4
    dropout: float = 0.1

    # SOTA Components
    use_rope: bool = True        # Rotary Embeddings (Vital for Time Series)
    use_swiglu: bool = True      # SwiGLU > GELU
    use_flash_attn: bool = True  # SDPA
    gradient_checkpointing: bool = False

    # Diffusion Physics
    timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    use_ddim_sampling: bool = True

    # APEX-MoE Specifics
    use_auxiliary_head: bool = True
    num_phases: int = 3  # Tri-Phase: Stable(0) -> Pre-Shock(1) -> Shock(2)

# =============================================================================
# 2. LOW-LEVEL PRIMITIVES (Mask-Aware & Robust)
# =============================================================================

class MaskedAvgPool1d(nn.Module):
    """
    Critically important for Time-Series.
    Standard AvgPool averages zeros (padding), diluting the signal.
    This module sums valid tokens and divides by the valid count.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] Boolean (True = Pad, False = Keep) OR (True = Keep) based on convention.
                  Here we assume: True = Padding/Invalid (Standard PyTorch convention).
        """
        if mask is None:
            return x.mean(dim=1)

        # Invert mask: We want 1.0 for VALID data, 0.0 for PADDING
        # Assuming input mask is "True if padding" (e.g., key_padding_mask)
        valid_mask = (~mask).float().unsqueeze(-1) # [B, T, 1]
        
        # Zero out padding (just in case)
        x_masked = x * valid_mask
        
        # Sum valid items
        sum_x = x_masked.sum(dim=1) # [B, D]
        
        # Count valid items
        count_valid = valid_mask.sum(dim=1) # [B, 1]
        count_valid = torch.clamp(count_valid, min=1e-6) # Prevent div/0
        
        return sum_x / count_valid

class RMSNorm(nn.Module):
    """LlaMA-style RMSNorm (No center, just scale). Stable training."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer 2020)."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

# --- SOTA: Unified Time RoPE ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            offset: The starting global time index. 
                    History starts at 0. Future starts at HistoryLen.
        """
        # Auto-expand cache if needed
        req_len = seq_len + offset
        if self.cos_cached is None or self.cos_cached.shape[-2] < req_len:
            t = torch.arange(max(self.max_seq_len, req_len), device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(device)[None, None, :, :]
            self.sin_cached = emb.sin().to(device)[None, None, :, :]
            
        return (
            self.cos_cached[:, :, offset : offset + seq_len, :],
            self.sin_cached[:, :, offset : offset + seq_len, :]
        )

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rot * sin

# --- Mask-Aware Attention ---
def robust_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, dropout: float,
    # RoPE Arguments
    cos_q: Optional[torch.Tensor] = None, sin_q: Optional[torch.Tensor] = None,
    cos_k: Optional[torch.Tensor] = None, sin_k: Optional[torch.Tensor] = None,
    # Masking
    key_padding_mask: Optional[torch.Tensor] = None # [B, L_k] True=Pad
) -> torch.Tensor:
    """
    Robust SDPA Wrapper. Handles RoPE injection + Padding Masks correctly.
    """
    B, L_q, D = q.shape
    _, L_k, _ = k.shape
    head_dim = D // n_heads

    # [B, L, H, D_h] -> [B, H, L, D_h]
    q = q.view(B, L_q, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, L_k, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, L_k, n_heads, head_dim).transpose(1, 2)

    # 1. Apply RoPE
    if cos_q is not None: q = apply_rotary_pos_emb(q, cos_q, sin_q)
    if cos_k is not None: k = apply_rotary_pos_emb(k, cos_k, sin_k)

    # 2. Prepare Mask for SDPA
    # SDPA expects attention mask to be added to scores (float) or boolean (True=Keep?)
    # PyTorch documentation is inconsistent across versions. 
    # Safest: Use causal masking if needed, or manual mask expansion.
    # For padding: We want to MASK OUT True values in key_padding_mask.
    
    attn_mask = None
    if key_padding_mask is not None:
        # key_padding_mask is [B, L_k] where True is BAD (Pad).
        # We need to broadcast to [B, 1, L_q, L_k].
        # In SDPA, if we pass a boolean mask, it expects True = MASK OUT (drop)? No, check docs.
        # Actually safest is to construct the additive mask manually if unsure of PyTorch version.
        # However, to use Flash Kernel, we shouldn't supply explicit dense masks if possible.
        # Ideally, we pass it via `key_padding_mask` argument if SDPA wrapped supported it, but F.sdpa doesn't.
        # We must reshape.
        
        # Reshape [B, L_k] -> [B, 1, 1, L_k]
        mask_expanded = key_padding_mask.view(B, 1, 1, L_k).expand(-1, n_heads, L_q, -1)
        
        # Additive Mask: 0 for keep, -inf for drop
        # But this prevents Flash Attn kernel usage in older PyTorch. 
        # Modern Torch: Boolean mask supported? 
        # For Robustness: We will use the standard torch.where.
        # To enable optimizations, we rely on PyTorch's backend to fuse this.
        # Note: If memory is tight, this expansion is costly. 
        
        # Fallback Strategy:
        # If no mask, attn_mask=None.
        pass # Logic handled below

    if key_padding_mask is not None:
        # Create broadcastable mask [B, 1, 1, L_k]
        # True means PAD.
        attn_bias = torch.zeros_like(key_padding_mask, dtype=q.dtype).masked_fill(key_padding_mask, float("-inf"))
        attn_bias = attn_bias.view(B, 1, 1, L_k)
        # Note: Adding bias forces non-flash path in some versions, but correctness > speed here.
    else:
        attn_bias = None

    # 3. Attention
    out = F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=attn_bias, 
        dropout_p=dropout if dropout > 0 else 0.0
    )

    return out.transpose(1, 2).contiguous().view(B, L_q, D)

# =============================================================================
# 3. TEMPORAL ENCODER (The History Reader)
# =============================================================================

class TemporalFusionEncoder(nn.Module):
    """
    Encodes Past Vitals + Static Context into a robust latent vector.
    Fixes:
        - Explicit handling of padding masks in Attention.
        - Masked Global Pooling (No averaging zeros).
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        
        # Projects
        # [CRITICAL FIX] vitals_proj now only processes DYNAMIC features (22 channels)
        # Static features (6 channels) are processed separately via static_proj
        self.vitals_proj = nn.Linear(22, cfg.d_model)  # Was: cfg.input_dim (28)
        self.static_proj = nn.Linear(cfg.static_dim, cfg.d_model)
        
        # RoPE
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.history_len + 24)
        
        # Layers
        self.layers = nn.ModuleList([
            EncoderBlock(cfg) for _ in range(cfg.encoder_layers)
        ])
        
        # Robust Pooling
        self.pool = MaskedAvgPool1d()
        self.out_norm = RMSNorm(cfg.d_model)

    def forward(
        self, 
        past_vitals: torch.Tensor, 
        static_context: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: # [PATCH] Return Mask
        
        B, T, _ = past_vitals.shape
        
        # [CRITICAL FIX] Static Redundancy Prevention
        # The input vitals tensor has shape [B, T, 28] where:
        # - Indices 0-21: Dynamic features (Hemodynamic + Labs + Electrolytes)
        # - Indices 22-27: Static features (Age, Gender, Unit1, Unit2, AdmTime, ICULOS)
        # 
        # We MUST slice to [B, T, 22] to prevent "Demographic Drowning":
        # The static features are already processed separately via static_proj.
        # Including them in vitals_proj causes the model to see them at EVERY timestep,
        # drowning out the dynamic physiological signals.
        past_vitals_dynamic = past_vitals[..., :22]  # [B, T, 22]
        
        # 1. Embeddings
        x_seq = self.vitals_proj(past_vitals_dynamic) # [B, T, D]
        x_static = self.static_proj(static_context).unsqueeze(1) # [B, 1, D]
        
        # 2. Concat Static (Prepend)
        # Sequence: [Static, H_0, H_1, ..., H_23]
        x = torch.cat([x_static, x_seq], dim=1)
        
        # [PATCH] Centralized Mask Logic
        if src_mask is not None:
            # src_mask is [B, T]. We need [B, T+1].
            # Prepend 'False' (Valid) for the static token.
            valid_static = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([valid_static, src_mask], dim=1)
        else:
            full_mask = None
            
        # 4. RoPE (Global Time 0..23)
        cos, sin = self.rope(x.shape[1], x.device, offset=0)
        
        # 5. Encoding
        for layer in self.layers:
            if self.cfg.gradient_checkpointing:
                x = checkpoint(layer, x, cos, sin, full_mask)
            else:
                x = layer(x, cos, sin, full_mask)
        
        x = self.out_norm(x)
        
        # 6. Masked Pooling -> Global Context
        global_ctx = self.pool(x, full_mask) # [B, D]
        
        # [PATCH] Return the authoritative mask to the Planner
        return x, global_ctx, full_mask

class EncoderBlock(nn.Module):
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        
        if cfg.use_swiglu:
            hidden_dim = int(cfg.d_model * cfg.ffn_dim_ratio * 2 / 3)
            self.ffn = SwiGLU(cfg.d_model, hidden_dim, cfg.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * cfg.ffn_dim_ratio),
                nn.GELU(),
                nn.Linear(cfg.d_model * cfg.ffn_dim_ratio, cfg.d_model),
                nn.Dropout(cfg.dropout)
            )
            
    def forward(self, x, cos, sin, mask=None):
        # Self-Attention with Pre-Norm (Q, K, V all from normalized input)
        h = self.norm1(x)
        attn = robust_flash_attention(
            h, h, h,  # FIXED: All three use normalized input
            self.cfg.n_heads, self.cfg.dropout, 
            cos_q=cos, sin_q=sin, cos_k=cos, sin_k=sin,
            key_padding_mask=mask
        )
        x = x + self.dropout(attn)
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# =============================================================================
# 4. DIFFUSION BACKBONE (DiT with Cross-Attention)
# =============================================================================

class AdaLNZero(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_model, 6 * d_model, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        chunk = self.linear(self.silu(condition))
        return chunk.chunk(6, dim=1)

class DiTBlock1D(nn.Module):
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.norm3 = RMSNorm(cfg.d_model)
        self.adaLN = AdaLNZero(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # FFN
        if cfg.use_swiglu:
            hidden_dim = int(cfg.d_model * cfg.ffn_dim_ratio * 2 / 3)
            self.ffn = SwiGLU(cfg.d_model, hidden_dim, cfg.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 4),
                nn.GELU(),
                nn.Linear(cfg.d_model * 4, cfg.d_model)
            )

    def forward(self, x, t_cond, context, cos_q, sin_q, cos_k, sin_k, ctx_mask=None):
        """
        x: Noisy Future [B, T_pred, D]
        context: Encoded History [B, T_hist, D]
        ctx_mask: Mask for History [B, T_hist]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(t_cond)
        
        # 1. Self-Attention (Future-Future)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn = robust_flash_attention(
            h, h, h, self.cfg.n_heads, self.cfg.dropout,
            cos_q=cos_q, sin_q=sin_q, cos_k=cos_q, sin_k=sin_q
            # No mask needed for future self-attention (always full length usually)
        )
        x = x + self.dropout(gate_msa.unsqueeze(1) * attn)
        
        # 2. Cross-Attention (Future-History)
        # CRITICAL: Pass ctx_mask to ignore padded history
        h = self.norm2(x)
        cross = robust_flash_attention(
            h, context, context, self.cfg.n_heads, self.cfg.dropout,
            cos_q=cos_q, sin_q=sin_q, cos_k=cos_k, sin_k=sin_k,
            key_padding_mask=ctx_mask
        )
        x = x + self.dropout(cross)
        
        # 3. FFN
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.ffn(h)
        x = x + self.dropout(gate_mlp.unsqueeze(1) * ffn_out)
        
        return x

class DiffusionActionHead(nn.Module):
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.pred_len, cfg.d_model) * 0.02)
        
        # Global RoPE Generator
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=128)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )
        
        self.blocks = nn.ModuleList([DiTBlock1D(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.input_dim)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def get_time_emb(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal Embeddings
        half_dim = self.cfg.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

    def forward(
        self, 
        noisy_x: torch.Tensor, 
        t: torch.Tensor, 
        context_seq: torch.Tensor, 
        global_ctx: torch.Tensor, 
        ctx_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        B, T_pred, _ = noisy_x.shape
        T_hist = context_seq.shape[1]
        
        # 1. Embed Future
        x = self.in_proj(noisy_x) + self.pos_emb
        
        # 2. Time Condition
        t_emb = self.get_time_emb(t)
        cond = self.time_mlp(t_emb) + global_ctx
        
        # 3. RoPE (Unified Coordinates)
        # History: [0, ..., T_hist-1]
        # Future:  [T_hist, ..., T_hist + T_pred - 1]
        # This relative shift helps the model understand "x comes after context"
        cos_hist, sin_hist = self.rope(T_hist, x.device, offset=0)
        cos_fut, sin_fut = self.rope(T_pred, x.device, offset=T_hist)
        
        # 4. DiT Stack
        for block in self.blocks:
            if self.cfg.gradient_checkpointing:
                x = checkpoint(
                    block, x, cond, context_seq, 
                    cos_fut, sin_fut, cos_hist, sin_hist, ctx_mask
                )
            else:
                x = block(
                    x, cond, context_seq, 
                    cos_fut, sin_fut, cos_hist, sin_hist, ctx_mask
                )
        
        return self.out_head(self.final_norm(x))

# =============================================================================
# 5. NOISE SCHEDULER
# =============================================================================
class NoiseScheduler(nn.Module):
    def __init__(self, timesteps: int = 100):
        super().__init__()
        self.timesteps = timesteps
        
        # Squared Cosine Cap v2 Schedule
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
        betas = betas.float()
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # [PATCH] Precompute shifted alphas for O(1) step logic
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        s1 = self.sqrt_alphas_cumprod[t][:, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return s1 * x_start + s2 * noise, noise

    def step(self, pred_noise, t, x_t, use_ddim=True):
        # [PATCH] Fix Indexing: buffer is pre-shifted/padded, so 't' indices correspond correctly.
        # Original Bug: torch.clamp(t-1, min=0) caused t=0 and t=1 to fetch the same value.
        alpha_bar_t = self.alphas_cumprod[t][:, None, None]
        alpha_bar_prev = self.alphas_cumprod_prev[t][:, None, None]
        
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        
        if use_ddim:
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        else:
            beta_t = self.betas[t][:, None, None]
            # Handle noise for t=0 safely
            noise = torch.randn_like(x_t)
            # Mask out noise for t=0 (standard DDPM logic)
            noise_mask = (t > 0).float()[:, None, None]
            
            x_prev = (x_t - beta_t * pred_noise / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(1 - beta_t)
            x_prev = x_prev + torch.sqrt(beta_t) * noise * noise_mask
            
        return x_prev

# =============================================================================
# 6. UNIFIED ORCHESTRATOR
# =============================================================================
class ICUUnifiedPlanner(nn.Module):
    """
    The Safety-Critical Host.
    Connects Encoder, Backbone, and Scheduler with Strict Unit Handling.
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = TemporalFusionEncoder(cfg)
        self.backbone = DiffusionActionHead(cfg)
        self.scheduler = NoiseScheduler(cfg.timesteps)
        
        # APEX Heads
        if cfg.use_auxiliary_head:
            self.aux_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.d_model // 2, cfg.num_phases)
            )
            
        # [PATCH 2A] Dense Value Head for GAE-Lambda
        # Projects global context to a Value Sequence of length T_pred
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.SiLU(),
            # Output dim is now cfg.pred_len (e.g., 6), not 1.
            # This provides a baseline V(s_t) for every step t in the future.
            nn.Linear(cfg.d_model // 2, cfg.pred_len) 
        )
        
        # Hooks for Normalizer
        from icu.datasets.normalizer import ClinicalNormalizer
        self.normalizer = ClinicalNormalizer(ts_channels=cfg.input_dim, static_channels=cfg.static_dim)

    # [PATCH] Tuple-Safe Wrapper for ClinicalNormalizer
    def normalize(self, x_ts: torch.Tensor, x_static: Optional[torch.Tensor] = None):
        # The ClinicalNormalizer returns (ts_norm, static_norm)
        norm_ts, norm_static = self.normalizer.normalize(x_ts, x_static)
        # If input static was None, norm_static is None.
        return norm_ts, norm_static

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(x)

    def forward(self, batch: Dict[str, torch.Tensor], reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        # Unpack
        past = batch["observed_data"]
        fut = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # [PATCH] Demographic Drowning Fix: Normalize Static Context
        past_norm, static_norm = self.normalize(past, static)
        fut_norm, _ = self.normalize(fut, None) 
        
        # [PATCH] Encoder returns the authoritative mask
        ctx_seq, global_ctx, ctx_mask = self.encoder(past_norm, static_norm, src_mask)
        
        # --- 4. Forward Diffusion ---
        B = past.shape[0]
        t = torch.randint(0, self.cfg.timesteps, (B,), device=past.device)
        noisy_fut, noise_eps = self.scheduler.add_noise(fut_norm, t)
        
        pred_noise = self.backbone(noisy_fut, t, ctx_seq, global_ctx, ctx_mask)
        
        # --- 6. Loss & Heads ---
        if reduction == 'none':
            diff_loss = F.mse_loss(pred_noise, noise_eps, reduction='none').mean(dim=[1, 2])
            
            if self.cfg.use_auxiliary_head and "phase_label" in batch:
                logits = self.aux_head(global_ctx)
                aux_loss = F.cross_entropy(logits, batch["phase_label"].long(), reduction='none')
            else:
                aux_loss = torch.zeros(B, device=past.device)
        else:
            diff_loss = F.mse_loss(pred_noise, noise_eps)
            aux_loss = torch.tensor(0.0, device=past.device)
            if self.cfg.use_auxiliary_head and "phase_label" in batch:
                logits = self.aux_head(global_ctx)
                
                # SOTA: Apply class weighting for the 3.1% Sepsis Imbalance
                # Expected Order: [Stable, Pre-Shock, Shock]
                # Default weight of 32.6 corresponds to the User's Research Findings.
                w = batch.get("aux_weight") 
                aux_loss = F.cross_entropy(logits, batch["phase_label"].long(), weight=w)
                
        # [PATCH] Dead Critic Fix: Include Value Loss if Target Provided
        # Get Dense Value Sequence [B, pred_len]
        pred_val = self.value_head(global_ctx)
        value_loss = torch.tensor(0.0, device=past.device)
        
        if "clinical_reward" in batch: # Look for pre-computed rewards from AdvantageCalculator
            # MSE between Dense Value Head and Dense Rewards
            value_loss = F.mse_loss(pred_val, batch["clinical_reward"])
        
        # Sum into total loss (Value weight 0.5 is standard for AWR baselines)
        total = diff_loss + 0.1 * aux_loss + 0.5 * value_loss
        
        return {
            "loss": total,
            "diffusion_loss": diff_loss,
            "aux_loss": aux_loss,
            "value_loss": value_loss,
            "pred_value": pred_val
        }

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], num_steps: Optional[int] = None) -> torch.Tensor:
        past = batch["observed_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        B = past.shape[0]
        # [PATCH] Normalize both
        past_norm, static_norm = self.normalize(past, static)
        
        # [PATCH] Use authoritative mask
        ctx_seq, global_ctx, ctx_mask = self.encoder(past_norm, static_norm, src_mask)
        

        x_t = torch.randn(B, self.cfg.pred_len, self.cfg.input_dim, device=past.device)
        steps = num_steps or self.cfg.timesteps
        
        for i in reversed(range(steps)):
            t = torch.full((B,), i, dtype=torch.long, device=past.device)
            out = self.backbone(x_t, t, ctx_seq, global_ctx, ctx_mask)
            x_t = self.scheduler.step(out, t, x_t, use_ddim=self.cfg.use_ddim_sampling)
            
        return self.unnormalize(x_t)