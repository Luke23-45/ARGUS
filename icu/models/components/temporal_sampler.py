import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            pe: [1, T, D]
        """
        T = x.size(1)
        # [v12.8 SOTA FIX] DType Alignment
        # Ensure the positional encoding buffer matches the input (BF16/FP16)
        return self.pe[:T, :].unsqueeze(0).to(dtype=x.dtype, device=x.device)

class VolatilityGate(nn.Module):
    """
    [2025 SOTA] Computes an importance score for each timestep based on:
    1. Learnable semantic importance (MLP)
    2. Explicit volatility (Delta magnitude)
    3. Positional Context (Time-Awareness) - [NEW]
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Energy Projection now sees Position-Enhanced features
        self.energy_proj = nn.Linear(d_model, 1)
        self.volatility_scale = nn.Parameter(torch.tensor(1.0))
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        self.mixer = nn.Linear(d_model * 2, d_model) # Mix Content + Pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            scores: [B, T, 1] - Importance scores (unnormalized logits)
        """
        # 0. Add Positional Context for Scoring only
        # We don't change the OUTPUT x, just the scoring input
        pos = self.pos_encoder(x) # [1, T, D]
        
        # Mix Content + Position (Concatenate and Project)
        # Allows model to learn "Spike at T=0 is fine, Spike at T=24 is bad"
        # Cheap mixer: element-wise add or concat?
        # Let's simple add for now to preserve dimensions for energy_proj
        # Or better: MLP scorer sees both.
        
        # SOTA: Add PE to X before scoring
        x_pos = x + pos
        
        # 1. Semantic Energy (What the model thinks is important)
        semantic_scores = self.energy_proj(x_pos) # [B, T, 1]

        # 2. Volatility Energy (Actual change magnitude)
        # Delta = |x_t - x_{t-1}|
        # Pad first stats with 0
        x_pad = F.pad(x[:, :-1, :], (0, 0, 1, 0)) 
        delta = (x - x_pad).abs().mean(dim=-1, keepdim=True) # [B, T, 1]
        
        # Combine
        total_score = semantic_scores + (delta * self.volatility_scale)
        return total_score

class TemporalSampler(nn.Module):
    """
    [Step 2] Volatility-Aware Temporal Sampler.
    
    Purpose: 
    Prevention of "Signal Dilution". In standard attention, stable periods (normal vitals)
    can dominate the softmax, washing out brief but critical spike events.
    
    Mechanism:
    1. Calulates an Importance Score (Semantic + Volatility + Position).
    2. Returns EITHER:
       a) Soft-Weighted sequence (re-weighted by importance).
       b) Top-K sampled sequence (hard selection) - *Configurable*.
       
    Default behavior for T=24 (Hourly) is Soft-Weighting to preserve gradients,
    but "hard" sampling is available for higher-frequency experiments.
    """
    def __init__(self, d_model: int, top_k: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.scorer = VolatilityGate(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] - True if padding (to be ignored)
            
        Returns:
            x_sampled: [B, T, D] (weighted) OR [B, K, D] (sampled)
            importance_mask: [B, T] or [B, K] - The computed importance weights
        """
        # 1. Calculate Scores
        scores = self.scorer(x) # [B, T, 1]
        
        # Mask out padding (set score to -inf)
        if mask is not None:
            # mask is True for Padding.
            scores = scores.masked_fill(mask.unsqueeze(-1), -1e9)
            
        # 2. Calculate Attention/Weights
        weights = F.softmax(scores, dim=1) # [B, T, 1] -> Soft importance
        
        # 3. Apply Weighting (Soft "Sampling")
        # We multiply the input by its normalized importance relative to the uniform average.
        # This amplifies spikes and suppresses flatlines.
        T = x.shape[1]
        x_weighted = x * (weights * T) # Scale so mean is 1.0 (approximating identity if uniform)
        
        if self.top_k is not None and self.top_k < T:
            # Hard Top-K Selection
            # Find indices
            top_scores, top_indices = torch.topk(scores.squeeze(-1), self.top_k, dim=1)
            
            # Sort indices to preserve temporal order! (Critical for Transformers)
            top_indices, _ = torch.sort(top_indices, dim=1)
            
            # Gather
            batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(-1).expand(-1, self.top_k)
            x_out = x_weighted[batch_indices, top_indices, :]
            mask_out = mask[batch_indices, top_indices] if mask is not None else None
            
            return self.norm(x_out), weights
            
        else:
            # Return full sequence, just re-weighted
            return self.norm(x_weighted), weights
