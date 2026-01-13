"""
icu/models/diffusion_icu.py
--------------------------------------------------------------------------------
APEX-MoE: SOTA Diffusion Planner (Unified Architecture v11.0).

Status: PRODUCTION-READY / SAFETY-CRITICAL
"The engine of survival."

This module defines the neural architecture for the clinical diffusion planner.
It acts as the 'Donor' model for the APEX-MoE specialist system.

Architectural Pillars (Unified from v3.0 + v10.0):
1.  **DiT-1D Backbone**: AdaLN-Zero conditioning for stable diffusion generation.
2.  **Causal RoPE**: Unified temporal coordinate system (History: 0-23, Future: 24-29).
3.  **Mask-Aware Physics**: Attention and Pooling layers strictly ignore padding.
    (Prevents "Zero-Shock" hallucinations where missing data is interpreted as cardiac arrest).
4.  **SwiGLU & Flash Attention**: High-performance compute primitives.
5.  **APEX-MoE Ready**: Exposes Auxiliary Heads and Latent Hooks for Phase 2.
6.  **Physics-Guided Sampling (PGS)**: Implements active gradient steering 
    inside the generation loop to enforce biological constraints in real-time.
7.  **Analog Bits (Self-Conditioning)**: The architecture explicitly routes 
    recurrent x0 estimates into the input stream, allowing the model to 
    'read its own thoughts' and correct hallucinations.
8.  **Temporal Attention Pooling**: A learnable mechanism to detect transient 
    critical events (e.g., a 5-minute hypotensive drop) that mean pooling misses.
9.  **Deep Stochasticity**: DropPath and Flash Attention v2 integration for 
    training stability on long-horizon ICU data.

References:
    - Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT)
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - OpenAI, "Improved Denoising Diffusion Probabilistic Models"
    - Jaegle et al., "Perceiver IO" (2021) [Attention Pooling]
    - Chen et al., "Analog Bits: Generating Discrete Data" (2022) [Self-Cond]
    - Hong et al., "Physics-Guided Diffusion" (2023) [Gradient Guidance]
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Gradient checkpointing for memory efficiency
from torch.utils.checkpoint import checkpoint

# SOTA Components
from icu.models.components.geometric_projector import GeometricProjector
from icu.models.components.temporal_sampler import TemporalSampler
from icu.models.components.nth_encoder import NTHEncoderBlock
from icu.models.components.sequence_aux_head import SequenceAuxHead
from icu.models.components.loss_scaler import UncertaintyLossScaler

# [PHASE 4-5] Agentic Evolution Components
from icu.models.components.risk_scorer import PhysiologicalRiskScorer
from icu.models.components.adaptive_sampler import StateAwareSampler
from icu.models.components.clinical_governor import ConfidenceAwareGovernor
from icu.utils.stability import DynamicThresholding

# Setup Logger
logger = logging.getLogger("ICU_Diffusion_v11")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class ICUConfig:
    """
    Hyperparameter Configuration for APEX-MoE Diffusion.
    Unified from v3.0 and v10.0 specifications.
    """
    # Dimensions (Frontier 28 Specification)
    # CRITICAL: input_dim MUST match CANONICAL_COLUMNS (28 channels)
    # The full [T, 28] matrix includes static context at indices 22-27
    input_dim: int = 28     # [Hemodynamic(7) + Labs(11) + Electrolytes(4) + Static(6)]
    static_dim: int = 6     # [Age, Gender, Unit1, Unit2, AdmTime, LOS]
    history_len: int = 24   # T_obs (Input Window)
    pred_len: int = 6       # T_pred (Prediction Horizon)

    # Transformer Architecture (DiT)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6       # DiT Depth
    encoder_layers: int = 4 # History Encoder Depth
    ffn_dim_ratio: int = 4
    dropout: float = 0.1
    stochastic_depth_prob: float = 0.1 # [v10.0] DropPath for regularization

    # SOTA Components
    use_rope: bool = True        # Rotary Embeddings (Vital for Time Series)
    use_swiglu: bool = True      # SwiGLU > GELU (Shazeer 2020)
    use_flash_attn: bool = True  # SDPA (Scales better than manual attention)
    gradient_checkpointing: bool = False
    
    # [v10.0] Architectural Flags
    use_attention_pooling: bool = True  # TimeAttentionPooling vs MaskedAvgPool
    use_self_conditioning: bool = True  # Critical for 'Analog Bits' refinement
    use_imputation_masks: bool = True   # [v12.0] Ingest imputation masks as features

    # Diffusion Physics
    timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"  # Superior to linear schedule
    prediction_type: str = "epsilon"          # Predict noise (standard)
    use_ddim_sampling: bool = True            # Deterministic sampling
    physics_guidance_scale: float = 2.0       # [v10.0] Strength of bio-constraints

    # APEX-MoE Specifics
    use_auxiliary_head: bool = True
    num_phases: int = 3  # Tri-Phase: Stable(0) -> Pre-Shock(1) -> Shock(2)
    aux_loss_scale: float = 0.1 # [v11.1] Configurable Aux Loss Scale

    # Stable Sampling [v18.0]
    use_dynamic_thresholding: bool = True
    
    # [v11.0] Agentic Evolution
    min_sampling_steps: int = 25
    base_safety_percentile: float = 0.99
    min_safety_percentile: float = 0.90

# =============================================================================
# 2. LOW-LEVEL PRIMITIVES (Mask-Aware & Robust)
# =============================================================================

class DropPath(nn.Module):
    """
    [v10.0] Stochastic Depth (DropPath) regularization.
    Randomly drops residual branches during training to improve generalization.
    
    Reference: Huang et al., "Deep Networks with Stochastic Depth" (2016)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, broadcasting over batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class TimeAttentionPooling(nn.Module):
    """
    [v10.0] Learnable Pooling via Cross-Attention.
    Captures transient spikes (e.g., sudden MAP drops) that averaging misses.
    Acts as a 'Summary Expert' for the history encoder.
    
    Unlike mean pooling, this learns to weight timesteps by clinical importance.
    For example, a 5-minute hypotensive episode gets higher attention weight
    than stable readings around it.
    
    Reference: Jaegle et al., "Perceiver IO" (2021)
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        # Learnable query vector (the "What to summarize" expert)
        self.summary_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, SeqLen, D_model]
            mask: [Batch, SeqLen] (True = Padding, to be IGNORED)
        
        Returns:
            [Batch, D_model] - Single summary vector per batch
        """
        B = x.shape[0]
        # Expand query for batch
        query = self.summary_query.expand(B, -1, -1)
        
        # Standard Multihead Attention
        # Query = Summary (what we want), Key/Value = History (what we have)
        # key_padding_mask expects True for positions to IGNORE
        out, _ = self.attn(
            query, x, x,
            key_padding_mask=mask,
            need_weights=False
        )
        # Squeeze to [Batch, D_model]
        return self.norm(out.squeeze(1))


class MaskedAvgPool1d(nn.Module):
    """
    [v3.0] Critically important for Time-Series.
    Standard AvgPool averages zeros (padding), diluting the signal.
    This module sums valid tokens and divides by the valid count.
    
    Example Problem Solved:
        Patient A has 24 hours of data, Patient B has 12 hours + 12 hours padding.
        Without masked pooling, Patient B's summary would be "half as strong"
        because we'd average in zeros for the padded hours.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] Boolean (True = Pad/Invalid, False = Keep/Valid).
                  Here we assume: True = Padding/Invalid (Standard PyTorch convention).
        
        Returns:
            [B, D] - Mean over valid (non-padded) timesteps
        """
        if mask is None:
            return x.mean(dim=1)

        # Invert mask: We want 1.0 for VALID data, 0.0 for PADDING
        # Assuming input mask is "True if padding" (e.g., key_padding_mask)
        valid_mask = (~mask).float().unsqueeze(-1)  # [B, T, 1]
        
        # Zero out padding (just in case it has garbage values)
        x_masked = x * valid_mask
        
        # Sum valid items
        sum_x = x_masked.sum(dim=1)  # [B, D]
        
        # Count valid items
        count_valid = valid_mask.sum(dim=1)  # [B, 1]
        count_valid = torch.clamp(count_valid, min=1e-6)  # Prevent div/0
        
        return sum_x / count_valid


class RMSNorm(nn.Module):
    """
    [LLaMA-style] Root Mean Square Layer Normalization.
    No mean-centering, just scale. More stable for deep transformers.
    
    Advantage over LayerNorm: 
        - Fewer operations (no mean subtraction)
        - Empirically more stable for very deep networks
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU FFN (Shazeer 2020).
    GLU Variants Improve Transformer performance.
    
    Architecture: 
        out = SiLU(W1 @ x) * (W2 @ x)
        out = W3 @ out
    
    This is the SOTA choice for FFN in modern LLMs (LLaMA, PaLM).
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# =============================================================================
# 3. ROTARY POSITION EMBEDDING (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE: Encodes relative positions via rotation.
    
    Key Insight: Instead of adding position embeddings, we ROTATE queries and keys.
    This allows the model to naturally encode relative positions through
    the dot product geometry.
    
    For ICU Forecasting:
        - History tokens get positions 0..23
        - Future tokens get positions 24..29 (offset by history_len)
        - This tells the model "future comes after history"
    
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
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
            seq_len: Number of positions to generate
            device: Target device
            offset: The starting global time index. 
                    History starts at 0. Future starts at HistoryLen.
        
        Returns:
            cos, sin: [1, 1, seq_len, dim] tensors for rotation
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
    """
    Apply rotary position embeddings to input tensor.
    
    The rotation is applied in 2D blocks:
        [x1, x2] -> [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    
    This is equivalent to rotating the vector in the complex plane.
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rot * sin


# =============================================================================
# 4. ATTENTION MECHANISM
# =============================================================================

def robust_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_heads: int, dropout: float,
    # RoPE Arguments
    cos_q: Optional[torch.Tensor] = None, sin_q: Optional[torch.Tensor] = None,
    cos_k: Optional[torch.Tensor] = None, sin_k: Optional[torch.Tensor] = None,
    # Masking
    key_padding_mask: Optional[torch.Tensor] = None  # [B, L_k] True=Pad
) -> torch.Tensor:
    """
    Robust SDPA Wrapper. Handles RoPE injection + Padding Masks correctly.
    
    This wrapper:
    1. Reshapes Q, K, V for multi-head attention
    2. Applies RoPE to Q and K
    3. Converts padding mask to attention bias (compatible with Flash Attention)
    4. Calls PyTorch's scaled_dot_product_attention (enables hardware acceleration)
    
    Mask Convention:
        key_padding_mask[b, t] = True means position t in batch b is PADDING
        These positions should be IGNORED (get -inf attention scores)
    """
    B, L_q, D = q.shape
    _, L_k, _ = k.shape
    head_dim = D // n_heads

    # Reshape: [B, L, D] -> [B, H, L, D_h]
    q = q.view(B, L_q, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, L_k, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, L_k, n_heads, head_dim).transpose(1, 2)

    # 1. Apply RoPE (if provided)
    if cos_q is not None: 
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
    if cos_k is not None: 
        k = apply_rotary_pos_emb(k, cos_k, sin_k)

    # 2. Prepare Mask for SDPA
    # We use an additive attention bias: 0 for keep, -inf for drop
    # This is the most robust approach across PyTorch versions
    attn_bias = None
    if key_padding_mask is not None:
        # key_padding_mask is [B, L_k] where True is BAD (Pad).
        # We need to broadcast to [B, 1, 1, L_k] for attention scores
        attn_bias = torch.zeros_like(key_padding_mask, dtype=q.dtype).masked_fill(
            key_padding_mask, float("-inf")
        )
        attn_bias = attn_bias.view(B, 1, 1, L_k)
        # Note: Adding explicit bias may disable Flash kernel in some versions,
        # but correctness > speed for safety-critical applications

    # 3. Scaled Dot Product Attention
    out = F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=attn_bias, 
        dropout_p=dropout if dropout > 0 else 0.0
    )

    # Reshape back: [B, H, L, D_h] -> [B, L, D]
    return out.transpose(1, 2).contiguous().view(B, L_q, D)


# =============================================================================
# 5. ENCODER BLOCKS
# =============================================================================

class EncoderBlock(nn.Module):
    """
    Single Encoder Block for the History Encoder.
    Uses Pre-Norm (norm before attention) for training stability.
    Includes DropPath for stochastic depth regularization.
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # [v10.0] Stochastic Depth
        self.drop_path = DropPath(cfg.stochastic_depth_prob) if cfg.stochastic_depth_prob > 0 else nn.Identity()
        
        # FFN
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
            h, h, h,  # CRITICAL: All three use normalized input
            self.cfg.n_heads, self.cfg.dropout, 
            cos_q=cos, sin_q=sin, cos_k=cos, sin_k=sin,
            key_padding_mask=mask
        )
        x = x + self.drop_path(self.dropout(attn))
        
        # FFN with Pre-Norm
        x = x + self.drop_path(self.dropout(self.ffn(self.norm2(x))))
        return x


class TemporalFusionEncoder(nn.Module):
    """
    Encodes Past Vitals + Static Context into a robust latent vector.
    
    Key Design Decisions:
    1. Static context is prepended as a single token (not repeated)
    2. Padding masks are carefully propagated to attention
    3. Either TimeAttentionPooling (v10.0) or MaskedAvgPool (v3.0) is used
    
    Fixes from v3.0:
        - Explicit handling of padding masks in Attention.
        - Masked Global Pooling (No averaging zeros).
        - [CRITICAL] Static Redundancy Prevention (slice to 22 dynamic features)
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        
        # Projects
        # [SOTA Phase 1] Geometric Projector
        self.vitals_proj = GeometricProjector(cfg.d_model, use_imputation_masks=cfg.use_imputation_masks)
        self.static_proj = nn.Linear(cfg.static_dim, cfg.d_model)
        
        # [SOTA Phase 1] Temporal Sampler
        self.temporal_sampler = TemporalSampler(cfg.d_model)

        # RoPE for history sequence (Legacy: NTHEncoder handles its own RoPE now, but kept for compatibility if mixed)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.history_len + 24)
        
        # Encoder Layers (NTH Architecture)
        self.layers = nn.ModuleList([
            NTHEncoderBlock(cfg.d_model, cfg.n_heads, hidden_dim=cfg.d_model * 2) 
            for _ in range(cfg.encoder_layers)
        ])
        
        # Pooling Strategy (v10.0 adds TimeAttentionPooling option)
        if cfg.use_attention_pooling:
            self.pool = TimeAttentionPooling(cfg.d_model, cfg.n_heads)
        else:
            self.pool = MaskedAvgPool1d()
            
        self.out_norm = RMSNorm(cfg.d_model)

    def forward(
        self, 
        past_vitals: torch.Tensor, 
        static_context: torch.Tensor, 
        imputation_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            past_vitals: [B, T, 28] - Normalized clinical timeseries
            static_context: [B, 6] - Static patient features
            imputation_mask: [B, T, 28] - (Optional) 1.0=Real, 0.0=Imputed
            padding_mask: [B, T] - (Optional) True=Padding (to be IGNORED)
        
        Returns:
            x: [B, T+1, D] - Full sequence representation (Static + History)
            global_ctx: [B, D] - Pooled context vector
            full_mask: [B, T+1] - Mask including static token
        """
        B, T, _ = past_vitals.shape
        
        # [CRITICAL FIX v3.0] Static Redundancy Prevention
        # The input vitals tensor has shape [B, T, 28] where:
        # - Indices 0-21: Dynamic features (Hemodynamic + Labs + Electrolytes)
        # - Indices 22-27: Static features (Age, Gender, Unit1, Unit2, AdmTime, ICULOS)
        # 
        # We MUST slice to [B, T, 22] to prevent "Demographic Drowning":
        # The static features are already processed separately via static_proj.
        # Including them in vitals_proj causes the model to see them at EVERY timestep,
        # drowning out the dynamic physiological signals.
        past_vitals_dynamic = past_vitals[..., :22]  # [B, T, 22]
        
        # [v12.0] Imputation Awareness (Feature Conditioning)
        if self.cfg.use_imputation_masks:
            if imputation_mask is not None:
                # Slice mask to dynamic features only
                mask_dynamic = imputation_mask[..., :22]
                
                # [CRITICAL FIX v12.1] Geometric Group Alignment
                # GeometricProjector expects: [Hemo_V, Hemo_M, Lab_V, Lab_M, Elec_V, Elec_M]
                # Default "cat" produces: [All_V, All_M] which misaligns the groups.
                
                # 1. Hemodynamics (Indices 0-6)
                hemo_v = past_vitals_dynamic[..., 0:7]
                hemo_m = mask_dynamic[..., 0:7]
                hemo_grp = torch.cat([hemo_v, hemo_m], dim=-1) # 14 channels
                
                # 2. Labs (Indices 7-17)
                labs_v = past_vitals_dynamic[..., 7:18]
                labs_m = mask_dynamic[..., 7:18]
                labs_grp = torch.cat([labs_v, labs_m], dim=-1) # 22 channels
                
                # 3. Electrolytes (Indices 18-21)
                elec_v = past_vitals_dynamic[..., 18:22]
                elec_m = mask_dynamic[..., 18:22]
                elec_grp = torch.cat([elec_v, elec_m], dim=-1) # 8 channels
                
                # Final Interleaved Input
                network_input = torch.cat([hemo_grp, labs_grp, elec_grp], dim=-1)
            else:
                # Fallback: Assume all real (ones) if no mask provided but expected
                mask_dynamic = torch.ones_like(past_vitals_dynamic)
                # Apply same interleaving fallback
                hemo_v = past_vitals_dynamic[..., 0:7]
                hemo_m = mask_dynamic[..., 0:7]
                hemo_grp = torch.cat([hemo_v, hemo_m], dim=-1)
                
                labs_v = past_vitals_dynamic[..., 7:18]
                labs_m = mask_dynamic[..., 7:18]
                labs_grp = torch.cat([labs_v, labs_m], dim=-1)
                
                elec_v = past_vitals_dynamic[..., 18:22]
                elec_m = mask_dynamic[..., 18:22]
                elec_grp = torch.cat([elec_v, elec_m], dim=-1)
                
                network_input = torch.cat([hemo_grp, labs_grp, elec_grp], dim=-1)
        else:
            network_input = past_vitals_dynamic

        # 1. Embeddings
        x_seq = self.vitals_proj(network_input)  # [B, T, D]
        x_static = self.static_proj(static_context).unsqueeze(1)  # [B, 1, D]
        
        # 2. Concat Static (Prepend) -- MOVED BELOW SAMPLER
        # Sequence: [Static, H_0, H_1, ..., H_23]
        # x = torch.cat([x_static, x_seq], dim=1)
        
            
        # 4. RoPE (Global Time 0..T) - Generated for consistency but NTH uses internal
        # Note: We generate this AFTER concat to get full length embeddings if needed
        
        # 4b. Temporal Sampling (Volatility Aware)
        # [SOTA REFINEMENT]: Apply only to dynamic history (x_seq), NOT static.
        # Volatility between Static->Time0 is meaningless.
        x_seq, sample_weights = self.temporal_sampler(x_seq, padding_mask)

        # 2. Concat Static (Prepend)
        # Sequence: [Static, H_0, H_1, ..., H_23]
        x = torch.cat([x_static, x_seq], dim=1)
        
        # 3. Centralized Mask Logic (Attention Mask)
        # [v12.0] FIX: Use padding_mask for attention, NOT imputation_mask.
        # padding_mask is [B, T]. We need [B, T+1] (Static token is always valid).
        
        valid_static = torch.zeros(B, 1, dtype=torch.bool, device=x.device) # False = Valid
        
        if padding_mask is not None:
            full_mask = torch.cat([valid_static, padding_mask], dim=1)
        else:
            # If no padding mask, assume entire sequence is valid (standard for fixed window)
            # Create a [B, T] mask of Falses
            valid_seq = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([valid_static, valid_seq], dim=1)

        # 5. Encoding (NTH-Attention)
        for layer in self.layers:
            # NTHEncoderBlock signature: (x, mask=None)
            x = layer(x, mask=full_mask) 
            # Note: We ignore 'cos', 'sin' here as NTHEncoderBlock has internal RoPE.
            # If we wanted to use global RoPE, we'd need to modify NTHEncoder.
            
        x = self.out_norm(x)
        
        # 6. Masked Pooling -> Global Context
        global_ctx = self.pool(x, full_mask)  # [B, D]
        
        # Return the authoritative mask to the Planner
        return x, global_ctx, full_mask


# =============================================================================
# 6. DIFFUSION BACKBONE (DiT with Cross-Attention)
# =============================================================================

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero.
    Conditioning mechanism for DiT (Diffusion Transformers).
    
    Regresses 6 parameters from the timestep/context embedding:
        - shift_msa, scale_msa, gate_msa (for self-attention)
        - shift_mlp, scale_mlp, gate_mlp (for FFN)
    
    The "Zero" refers to the zero-initialization, which makes the block
    start as an identity function and gradually learn to modulate.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_model, 6 * d_model, bias=True)
        # Initialize to zero so the block is an identity function at start of training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        chunk = self.linear(self.silu(condition))
        return chunk.chunk(6, dim=1)


class DiTBlock1D(nn.Module):
    """
    Diffusion Transformer Block.
    Processes noisy future sequence while attending to history context.
    
    Architecture:
        1. Self-Attention (Future <-> Future) with AdaLN conditioning
        2. Cross-Attention (Future <-> History) with mask for padding
        3. FFN with AdaLN conditioning
        
    All branches use DropPath for stochastic depth regularization.
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.norm3 = RMSNorm(cfg.d_model)
        self.adaLN = AdaLNZero(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # [v10.0] Stochastic Depth
        self.drop_path = DropPath(cfg.stochastic_depth_prob) if cfg.stochastic_depth_prob > 0 else nn.Identity()
        
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
        Args:
            x: Noisy Future [B, T_pred, D]
            t_cond: Time + Context embedding [B, D]
            context: Encoded History [B, T_hist, D]
            cos_q, sin_q: RoPE for future positions
            cos_k, sin_k: RoPE for history positions
            ctx_mask: Mask for History [B, T_hist] (True = Pad)
        """
        # Regress conditioning parameters from timestep
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(t_cond)
        
        # 1. Self-Attention (Future-Future, Time Mixing)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn = robust_flash_attention(
            h, h, h, self.cfg.n_heads, self.cfg.dropout,
            cos_q=cos_q, sin_q=sin_q, cos_k=cos_q, sin_k=sin_q
            # No mask needed for future self-attention (no padding in future)
        )
        x = x + self.drop_path(self.dropout(gate_msa.unsqueeze(1) * attn))
        
        # 2. Cross-Attention (Future-History, Conditioning)
        # CRITICAL: Pass ctx_mask to ignore padded history tokens
        h = self.norm2(x)
        cross = robust_flash_attention(
            h, context, context, self.cfg.n_heads, self.cfg.dropout,
            cos_q=cos_q, sin_q=sin_q, cos_k=cos_k, sin_k=sin_k,
            key_padding_mask=ctx_mask
        )
        x = x + self.drop_path(self.dropout(cross))
        
        # 3. FFN
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.ffn(h)
        x = x + self.drop_path(self.dropout(gate_mlp.unsqueeze(1) * ffn_out))
        
        return x


class DiffusionActionHead(nn.Module):
    """
    The Denoising Backbone.
    Predicts noise (epsilon) given noisy input, timestep, and history context.
    
    [v10.0] Self-Conditioning (Analog Bits):
        If enabled, the input dimension doubles (concat noisy_x + prev_x0)
        This allows the model to "read its own thoughts" and refine predictions.
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        
        # [v10.0] Self-Conditioning Support
        # If enabled, input dimension doubles (concat noisy_x + prev_x0)
        self.in_channels = cfg.input_dim * 2 if cfg.use_self_conditioning else cfg.input_dim
        
        self.in_proj = nn.Linear(self.in_channels, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.pred_len, cfg.d_model) * 0.02)
        
        # Global RoPE Generator (shared for unified timeline)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=128)
        
        # Timestep Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )
        
        self.blocks = nn.ModuleList([DiTBlock1D(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.input_dim)
        
        # Zero-init output layer for training stability
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def get_time_emb(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embeddings (standard in diffusion models)."""
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
        ctx_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noisy_x: [Batch, T_pred, InputDim] - Noisy future vitals
            t: [Batch] - Diffusion timestep
            context_seq: [Batch, T_hist, D_model] - Encoded history
            global_ctx: [Batch, D_model] - Pooled history summary
            ctx_mask: [Batch, T_hist] - Padding mask for history
            self_cond: [Batch, T_pred, InputDim] - (Optional) Previous x0 estimate
        
        Returns:
            [Batch, T_pred, InputDim] - Predicted noise (epsilon)
        """
        B, T_pred, _ = noisy_x.shape
        T_hist = context_seq.shape[1]
        
        # [v10.0] Self-Conditioning Logic
        if self.cfg.use_self_conditioning:
            if self_cond is None:
                # Cold start: Zero padding (standard initialization)
                self_cond = torch.zeros_like(noisy_x)
            x_input = torch.cat([noisy_x, self_cond], dim=-1)
        else:
            x_input = noisy_x
        
        # 1. Embed Future
        x = self.in_proj(x_input) + self.pos_emb
        
        # 2. Time Condition (fuse with global context)
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
                    cos_fut, sin_fut, cos_hist, sin_hist, ctx_mask,
                    use_reentrant=False
                )
            else:
                x = block(
                    x, cond, context_seq, 
                    cos_fut, sin_fut, cos_hist, sin_hist, ctx_mask
                )
        
        return self.out_head(self.final_norm(x))


# =============================================================================
# 7. NOISE SCHEDULER
# =============================================================================

class NoiseScheduler(nn.Module):
    """
    DDPM/DDIM Noise Scheduler.
    Manages the forward diffusion process (adding noise) and the backward steps.
    
    Uses the "Squared Cosine Cap v2" schedule, which is superior to linear
    for lower-resolution data like clinical time series.
    """
    def __init__(self, timesteps: int = 100):
        super().__init__()
        self.timesteps = timesteps
        
        # Squared Cosine Cap v2 Schedule (OpenAI Improved DDPM)
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
        
        # Precompute shifted alphas for O(1) step logic
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to x_start according to timestep t.
        
        Returns:
            noisy_x: x_start with noise added
            noise: the noise that was added (ground truth for training)
        """
        noise = torch.randn_like(x_start)
        s1 = self.sqrt_alphas_cumprod[t][:, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return s1 * x_start + s2 * noise, noise

    def step(
        self, 
        pred_noise: torch.Tensor, 
        t: torch.Tensor, 
        x_t: torch.Tensor, 
        use_ddim: bool = True
    ) -> torch.Tensor:
        """
        Single denoising step.
        
        Args:
            pred_noise: Model's predicted noise
            t: Current timestep
            x_t: Current noisy sample
            use_ddim: If True, deterministic (DDIM). If False, stochastic (DDPM).
        
        Returns:
            x_{t-1}: Denoised sample
        """
        alpha_bar_t = self.alphas_cumprod[t][:, None, None]
        alpha_bar_prev = self.alphas_cumprod_prev[t][:, None, None]
        
        # 1. Predict x0
        sqrt_alpha_t = torch.sqrt(alpha_bar_t).clamp(min=1e-3)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / sqrt_alpha_t
        
        if use_ddim:
            # Deterministic Step (DDIM)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        else:
            # Stochastic Step (DDPM)
            beta_t = self.betas[t][:, None, None]
            noise = torch.randn_like(x_t)
            # Mask out noise for t=0 (standard DDPM logic)
            noise_mask = (t > 0).float()[:, None, None]
            
            x_prev = (x_t - beta_t * pred_noise / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(1 - beta_t)
            x_prev = x_prev + torch.sqrt(beta_t) * noise * noise_mask
            
        return x_prev


# =============================================================================
# 8. PHYSIOLOGICAL SAFETY
# =============================================================================

class PhysiologicalConsistencyLoss(nn.Module):
    """
    [v10.0] Enforces biological plausibility.
    Penalizes values that drift outside the 2.5 sigma 'Hard Deck'.
    
    In normalized space, valid clinical values should be within [-2.5, 2.5].
    Values outside this range indicate physiologically implausible predictions
    (e.g., a MAP of -500 or a heart rate of 1000).
    
    This loss is used both during training and during Physics-Guided Sampling.
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.bounds = 2.5 

    def forward(self, x_pred: torch.Tensor) -> torch.Tensor:
        """
        Delegates to SOTA implementation in icu.core.robust_losses.
        """
        # [2025 ALIGNMENT] Use System-Wide implementation (Squared Penalty)
        # Import internally to avoid circular deps if any, or just strictly link it.
        from icu.core.robust_losses import physiological_violation_loss
        return physiological_violation_loss(x_pred, bounds=self.bounds, weight=self.weight)


# =============================================================================
# 9. UNIFIED ORCHESTRATOR
# =============================================================================

class ClinicalResidualHead(nn.Module):
    """
    SOTA Clinical Residual Head (2025).
    Rare signals (Sepsis/Death) get smoothed over in the latent space.
    This Residual Block acts as a "Translator," allowing the backbone to keep
    its smooth representation while the head extracts the sharp risk signal.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Capacity Injection (Maintain dimension, don't shrink yet)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(), # SOTA choice for non-linearity in clinical 2025 benchmarks
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )
        
        # 2. Output Projection
        self.norm_out = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, output_dim)
        
        # 3. Initialization Safety (Zero-init final layer for stable start)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # Residual Connection: x + Block(x)
        # This is the "Gradient Highway" that prevents signal loss
        feat = x + self.block(x)
        return self.head(self.norm_out(feat))


class ICUUnifiedPlanner(nn.Module):
    """
    The Safety-Critical Host.
    Combines Encoder, DiT Backbone, and Specialized Heads.
    
    This is the main model class that orchestrates:
    1. History Encoding (via TemporalFusionEncoder)
    2. Diffusion Training (via DiffusionActionHead + NoiseScheduler)
    3. Sepsis Risk Prediction (via aux_head)
    4. Value Estimation for AWR (via value_head)
    5. Physics-Guided Sampling (via PhysiologicalConsistencyLoss)
    """
    def __init__(self, cfg: ICUConfig):
        super().__init__()
        self.cfg = cfg
        
        # Core Components
        self.encoder = TemporalFusionEncoder(cfg)
        self.backbone = DiffusionActionHead(cfg)
        self.scheduler = NoiseScheduler(cfg.timesteps)
        
        # APEX Auxiliary Heads (Hardened)
        if cfg.use_auxiliary_head:
            # Replaced with 2025 SOTA SequenceAuxHead
            self.aux_head = SequenceAuxHead(
                d_model=cfg.d_model, 
                num_classes=cfg.num_phases, # Usually 3 phases? Or binary?
                # SequenceAuxHead supports multi-class output projection.
                # However, Asymmetric Loss is typically binary/multi-label.
                # If num_phases=3 (Stable/Pre/Shock), it's multi-class.
                # NTH SequenceAuxHead output shape is [B, num_classes].
                num_layers=2,
                n_heads=4
            )
            
            # [SOTA Phase 1] Uncertainty Loss Scaler
            self.loss_scaler = UncertaintyLossScaler(num_tasks=4)
            
        # Dense Value Head for GAE-Lambda (AWR)
        # Replaced with 2025 SOTA ClinicalResidualHead
        # Critical for accurate AWR weights in high-variance clinical data
        self.value_head = ClinicalResidualHead(
            input_dim=cfg.d_model, 
            output_dim=cfg.pred_len,
            dropout=0.1
        )
        
        # [v10.0] Physics Loss for both training and PGS
        self.phys_loss = PhysiologicalConsistencyLoss()
        
        # [v13.0 SOTA FIX] Dynamic Manifold Governance
        # Replaces hard clamps with Google Imagen-style thresholding
        self.governance = DynamicThresholding(percentile=0.995, threshold=3.0)
        
        # =====================================================================
        # [NEW] AGENTIC EVOLUTION CORE (Phases 4-5)
        # =====================================================================
        self.risk_scorer = PhysiologicalRiskScorer()
        # Phase 4: Adaptive Compute
        self.adaptive_sampler = StateAwareSampler(
            min_steps=cfg.min_sampling_steps, 
            max_steps=cfg.timesteps
        )
        # Phase 5: Teacher Governance
        self.clinical_governor = ConfidenceAwareGovernor(
            base_p=cfg.base_safety_percentile,
            min_p=cfg.min_safety_percentile
        )
        self.clinical_feat_idx = {'hr': 0, 'o2sat': 1, 'sbp': 2, 'map': 4, 'lactate': 7, 'resp': 5}

        # Hooks for Normalizer
        logger.info(f"[APEX PLANNER] Initialized v11.0 Agentic: {self.cfg.d_model}d, {self.cfg.n_layers}L")
        from icu.datasets.normalizer import ClinicalNormalizer
        self.normalizer = ClinicalNormalizer(ts_channels=cfg.input_dim, static_channels=cfg.static_dim)

    # --- Tuple-Safe Wrapper for ClinicalNormalizer ---
    def normalize(self, x_ts: torch.Tensor, x_static: Optional[torch.Tensor] = None):
        """Normalize clinical data to model space."""
        norm_ts, norm_static = self.normalizer.normalize(x_ts, x_static)
        return norm_ts, norm_static

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert model output back to clinical units."""
        return self.normalizer.denormalize(x)

    def forward(self, batch: Dict[str, torch.Tensor], reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """
        Standard training forward pass.
        
        Note: The Wrapper (wrapper_generalist.py) handles the complex 
        Two-Pass Self-Conditioning logic. This method is a fallback for 
        simple supervised training.
        
        Args:
            batch: Dictionary containing:
                - observed_data: [B, T_obs, 28]
                - future_data: [B, T_pred, 28]
                - static_context: [B, 6]
                - src_mask: [B, T_obs] (optional)
                - phase_label: [B] (optional)
                - aux_weight: [num_phases] (optional)
                - clinical_reward: [B, T_pred] (optional)
            reduction: 'mean' or 'none'
        """
        # Unpack
        past = batch["observed_data"]
        fut = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # Normalize
        past_norm, static_norm = self.normalize(past, static)
        fut_norm, _ = self.normalize(fut, None) 
        
        # 1. Encode History
        # [v12.1] Robust Padding Detection:
        # src_mask is [B, T, 28] where 0.0 = Missing.
        # A timestep is PADDING if ALL channels are missing.
        # [v16.0] FIX: Disable unsafe padding inference. Fixed window dataset has no padding.
        padding_mask = batch.get("padding_mask", None) 
        
        ctx_seq, global_ctx, ctx_mask = self.encoder(
            past_norm, static_norm, 
            imputation_mask=src_mask, 
            padding_mask=padding_mask # Successfully propagates to Attention
        )
        
        # 2. Forward Diffusion (add noise)
        B = past.shape[0]
        t = torch.randint(0, self.cfg.timesteps, (B,), device=past.device)
        noisy_fut, noise_eps = self.scheduler.add_noise(fut_norm, t)
        
        # 3. Two-Pass Self-Conditioning ("Analog Bits")
        # [v12.1] Parity Fix: Match the 50/50 logic in wrapper_generalist.py
        # This makes the Planner robust as a standalone module.
        self_cond_tensor = None
        if self.cfg.use_self_conditioning:
            self_cond_tensor = torch.zeros_like(noisy_fut)
            
            # 50% probability of using a preliminary x0 estimate
            if self.training and (torch.rand(1).item() < 0.5):
                with torch.no_grad():
                    # Pass 1: "Guess" noisy epsilon
                    guess_eps = self.backbone(
                        noisy_fut, t, ctx_seq, global_ctx, ctx_mask, self_cond=self_cond_tensor
                    )
                    # Reconstruct x0 estimate
                    alpha_bar = self.scheduler.alphas_cumprod[t][:, None, None]
                    sqrt_alpha_clamped = torch.sqrt(alpha_bar).clamp(min=1e-3)

                    # 2. Calculate
                    guess_x0 = (noisy_fut - torch.sqrt(1 - alpha_bar) * guess_eps) / sqrt_alpha_clamped

                    # 3. Manifold Constraint (SOTA Governance)
                    # Replaces guess_x0.clamp(-3.0, 3.0) with dynamic thresholding
                    self_cond_tensor = self.governance(guess_x0).detach()

        # Pass 2: Final Denoising with Conditioning
        pred_noise = self.backbone(noisy_fut, t, ctx_seq, global_ctx, ctx_mask, self_cond=self_cond_tensor)
        
        # 4. Loss Computation
        if reduction == 'none':
            # Mean over valid prediction horizon
            diff_loss = F.mse_loss(pred_noise, noise_eps, reduction='none').mean(dim=[1, 2])
            
            if self.cfg.use_auxiliary_head and "phase_label" in batch:
                # [SOTA Upgrade] SequenceAuxHead takes (x_seq, mask, targets)
                # We use ctx_seq from Encoder (result of TemporalSampler + NTH)
                # ctx_seq: [B, T', D] 
                # ctx_mask: [B, T']
                # Targets: phase_label [B]
                
                # Note: SequenceAuxHead returns (logits, loss).
                logits, sota_aux_loss = self.aux_head(
                    ctx_seq, 
                    mask=ctx_mask, 
                    targets=batch["phase_label"].long() if batch["phase_label"] is not None else None
                )
                
                # If sota_aux_loss is returned, use it directly (includes Asymmetric Logic)
                if sota_aux_loss is not None:
                    aux_loss = sota_aux_loss 
                else:
                    # Fallback (shouldn't happen if targets provided)
                    aux_loss = F.cross_entropy(logits, batch["phase_label"].long(), reduction='none')
            else:
                aux_loss = torch.zeros(B, device=past.device)
        else:
            diff_loss = F.mse_loss(pred_noise, noise_eps)
            if self.cfg.use_auxiliary_head and "phase_label" in batch:
                # Scaler handles reduction usually, but here we return scalar
                 logits, sota_aux_loss = self.aux_head(
                    ctx_seq, 
                    mask=ctx_mask, 
                    targets=batch["phase_label"].long()
                )
                 aux_loss = sota_aux_loss
            else:
                aux_loss = torch.tensor(0.0, device=past.device)
            
        # 5. Value Prediction (Critic for AWR)
        pred_val = self.value_head(global_ctx)
        value_loss = torch.tensor(0.0, device=past.device)
        
        if "clinical_reward" in batch:
            # [v11.1] Masked MSE for Value Head
            # Ignore padding in the prediction horizon
            target_val = batch["clinical_reward"]
            
            # Try to get explicit future mask
            f_mask = batch.get("future_mask")
            if f_mask is None:
                # Fallback: Inference from target zeros? No, risky. 
                # Use src_mask if it matches length? 
                f_mask = torch.ones_like(target_val, dtype=torch.float32)
                
                loss_ele = F.smooth_l1_loss(pred_val, target_val, beta=1.0, reduction='none')
                value_loss = (loss_ele * f_mask).sum() / (f_mask.sum() + 1e-8)
            else:
                # Ensure mask matches shape
                if f_mask.dim() == 3: f_mask = f_mask.any(dim=-1) # [B, T]
                f_mask = f_mask.float()
                
                # Check alignment
                if f_mask.shape[1] > pred_val.shape[1]:
                    f_mask = f_mask[:, :pred_val.shape[1]]
                
                # Weighted MSE
                sq_err = (pred_val - target_val) ** 2
                masked_mse = (sq_err * f_mask).sum() / (f_mask.sum() + 1e-8)
                value_loss = masked_mse
        
        # Total loss (Value weight 0.5 is standard for AWR baselines)
        # Total loss (Value weight 0.5 is standard for AWR baselines)
        # Total loss (Value weight 0.5 is standard for AWR baselines)
        # [SOTA Upgrade] Use Uncertainty Scaler if available
        if hasattr(self, 'loss_scaler') and self.cfg.use_auxiliary_head:
            losses = {'diffusion': diff_loss, 'aux': aux_loss}
            total, logs = self.loss_scaler(losses)
            total = total + 0.5 * value_loss
        else:
            total = diff_loss + self.cfg.aux_loss_scale * aux_loss + 0.5 * value_loss
            logs = {}
        
        # Compute aux_logits for return (needed by callbacks)
        # Use logits computed earlier (from SequenceAuxHead)
        aux_logits = logits if self.cfg.use_auxiliary_head else None
        
        ret = {
            "loss": total,
            "diffusion_loss": diff_loss,
            "aux_loss": aux_loss,
            "aux_logits": aux_logits,
            "value_loss": value_loss,
            "pred_value": pred_val
        }
        ret.update(logs) # Add sigma/weight logs
        return ret


    @torch.no_grad()
    def sample(self, 
               batch: Dict[str, torch.Tensor], 
               num_steps: Optional[int] = None, 
               teacher_model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        [v10.0] Physics-Guided Sampling Loop.
        
        Actively steers the trajectory using physiological gradients AND
        uses Self-Conditioning to refine predictions iteratively.
        
        This is the inference-time generation method.
        
        Returns:
            [B, T_pred, 28] - Generated future vitals (in clinical units)
        """
        past = batch["observed_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # [PHASE 4] Adaptive Compute & Risk Scoring
        risk_coef = self.risk_scorer(past, self.clinical_feat_idx)
        if num_steps is None:
            steps_batch = self.adaptive_sampler.calculate_steps(risk_coef)
            steps = int(steps_batch.max().item())
            logger.info(f"[Agentic Sample] Adaptive Horizon: {steps} steps (Risk range: {risk_coef.min():.2f}-{risk_coef.max():.2f})")
        else:
            steps = num_steps
        
        B = past.shape[0]
        
        # 1. Perception (Encode history)
        past_norm, static_norm = self.normalize(past, static)
        
        # [v12.1] Robust Padding Detection (Inference)
        # [v12.1] Robust Padding Detection (Inference)
        # Check batch for mask, default to None if missing
        padding_mask = batch.get("padding_mask", None)
        
        # [Phase 2 Audit] Fallback: If no explicit padding mask, infer from src_mask (Standard logic)
        if padding_mask is None and src_mask is not None:
             # src_mask is [B, T, 28]. If all channels are 0/Nan -> Padding.
             if src_mask.dim() == 3:
                 padding_mask = (src_mask.sum(dim=-1) == 0)
             
        ctx_seq, global_ctx, ctx_mask = self.encoder(
            past_norm, static_norm, 
            imputation_mask=src_mask,
            padding_mask=padding_mask # [FIX] Pass padding mask
        )
        
        # 2. Initialize from pure noise
        x_t = torch.randn(B, self.cfg.pred_len, self.cfg.input_dim, device=past.device)
        # steps already calculated above
        
        # [v10.0] Initialize Self-Conditioning Buffer (Analog Bits)
        x_self_cond = torch.zeros_like(x_t)
        
        # 3. Denoising Loop
        for i in reversed(range(steps)):
            t = torch.full((B,), i, dtype=torch.long, device=past.device)
            
            # --- A. Physics Guidance Step (Active Steering) ---
            # We temporarily enable gradients for x_t to compute d(Loss)/dx_t
            if self.cfg.physics_guidance_scale > 0:
                with torch.enable_grad():
                    x_t_in = x_t.detach().requires_grad_(True)
                    
                    # Estimate x0 (approx) with current self_cond
                    out_eps = self.backbone(x_t_in, t, ctx_seq, global_ctx, ctx_mask, self_cond=x_self_cond)
                    
                    # Reconstruct x0 (DDIM equation)
                    alpha_bar = self.scheduler.alphas_cumprod[t][:, None, None]

                    sqrt_alpha = torch.sqrt(alpha_bar).clamp(min=1e-3)
                    x0_approx = (x_t_in - torch.sqrt(1 - alpha_bar) * out_eps) / sqrt_alpha

                    # 2. DO NOT DENORMALIZE. 
                    # The PhysLoss (Bounds=2.5) expects Normalized (Sigma) units.
                    # If you feed it Clinical Units (e.g. 120), it will crush the values to 2.5.
                    loss = self.phys_loss(x0_approx)
                    
                    # Compute Gradient
                    grad = torch.autograd.grad(loss, x_t_in)[0]
                    
                # [v11.1] Per-Sample Steering Normalization
                # Prevents one patient's artifact from suppressing the batch
                grad_norm = grad.norm(dim=(1, 2), keepdim=True)
                
                # 2. USE GRAD_NORM HERE: Normalize the gradient
                # This ensures the steering 'direction' is preserved but magnitude is controlled
                grad = grad / (grad_norm + 1e-8)
                
                # 3. USE CLAMP: Final safety guard against high-frequency noise
                # Since the grad is now normalized to ~1.0, a clamp of 0.1 is very safe
                grad = torch.clamp(grad, -0.1, 0.1) 
                    
                # 4. Apply steering using the scaled, normalized, and clamped gradient
                # x_t = x_t - (Force * Direction)
                x_t = x_t - self.cfg.physics_guidance_scale * grad.detach()


            # --- B. Standard Diffusion Step ---
            out_student = self.backbone(x_t, t, ctx_seq, global_ctx, ctx_mask, self_cond=x_self_cond)
            
            # [PHASE 5] Teacher Governance (Audit)
            distrust = None
            if teacher_model is not None:
                # Use teacher's encoder or just the backbone if encoders are shared
                _, t_global_ctx, _ = teacher_model.encoder(past_norm, static_norm, padding_mask=padding_mask)
                out_teacher = teacher_model.backbone(x_t, t, ctx_seq, t_global_ctx, ctx_mask, self_cond=x_self_cond)
                
                distrust = self.clinical_governor.calculate_distrust(out_student, out_teacher, risk_coef)
                p_eff = self.clinical_governor.get_dynamic_percentile(distrust)
            else:
                p_eff = torch.full((B,), self.clinical_governor.base_p, device=x_t.device)

            # Apply Confidence-Aware Governance
            out = self.clinical_governor.apply_governance(out_student, p_eff)
            
            # [v10.0] Update Self-Conditioning for next step
            # We need x0 estimate to condition the next step.
            alpha_bar = self.scheduler.alphas_cumprod[t][:, None, None]
            sqrt_alpha = torch.sqrt(alpha_bar).clamp(min=1e-3)
            x_self_cond_raw = (x_t - torch.sqrt(1 - alpha_bar) * out) / sqrt_alpha
            
            # [v13.0 SOTA FIX] Dynamic Manifold Governance (Sampler)
            x_self_cond = self.governance(x_self_cond_raw).detach()

            
            # Step (DDIM or DDPM)
            x_t = self.scheduler.step(out, t, x_t, use_ddim=self.cfg.use_ddim_sampling)
            
            # --- C. SOTA Stability Patches ---
            # [v13.0] Final Manifold Projection
            # We apply governance to the final x_t estimate
            x_t = self.governance(x_t).detach()

        # Final denormalization with one last safety check
        return self.unnormalize(self.governance(x_t))
