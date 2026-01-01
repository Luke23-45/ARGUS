import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SwiGLU(nn.Module):
    """
    [2025 SOTA] Swish-Gated Linear Unit.
    Superior to standard GLU/ReGLU for Transformers (PaLM, LLaMA).
    Output = (xW + b) * Swish(xV + c)
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear_act = nn.Linear(input_dim, output_dim, bias=bias)
        self.linear_gate = nn.Linear(input_dim, output_dim, bias=bias)
        self.silu = nn.SiLU() # Swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_path = self.linear_act(x)
        gate_path = self.linear_gate(x)
        return act_path * self.silu(gate_path)

class GatedResidualNetwork(nn.Module):
    """
    [TFT-Style] Gated Residual Network (GRN) with SwiGLU.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = SwiGLU(d_model, d_model) # Upgraded to SwiGLU
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Processing
        residual = x
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # 2. Gating
        x = self.glu(x)
        
        # 3. Residual + Norm
        return self.norm(residual + x)

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x: torch.Tensor, seq_len: int):
        if self.cached_cos is None or self.cached_cos.size(0) < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, d_model]
            self.cached_cos = emb.cos().unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, d_model]
            self.cached_sin = emb.sin().unsqueeze(0).unsqueeze(0)
            
        return self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [1, 1, T, D]
    
    # split q into q1, q2
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    
    # rotate
    q_rot = torch.cat((-q2, q1), dim=-1)
    k_rot = torch.cat((-k2, k1), dim=-1)
    
    q_out = (q * cos) + (q_rot * sin)
    k_out = (k * cos) + (k_rot * sin)
    return q_out, k_out

class RoPEMultiheadAttention(nn.Module):
    """
    Custom MHA with Rotary Embeddings.
    Uses scaled dot-product attention.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryEmbedding(self.head_dim) # Apply to head dimension
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, T, _ = query.shape
        
        # Projections [B, T, D]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape [B, T, H, D_h] -> [B, H, T, D_h]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = self.rope(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        # scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
             # attn_mask [T, T] or [B*H, T, T]
             # Simplify: expect inputs to handle dimensions or broadcast
             scores = scores + attn_mask
             
        if key_padding_mask is not None:
            # key_padding_mask [B, T] -> True means ignore
            # Expand to [B, 1, 1, T] for broadcast
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, -1e9)
            
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v) # [B, H, T, D_h]
        
        # Reassemble
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.out_proj(output), weights

class NTHAttention(nn.Module):
    """
    [Phase 2] Hierarchical Attention (Neighborhood + Global).
    Uses RoPE-enhanced Multihead Attention.
    """
    def __init__(self, d_model: int, n_heads: int, local_window: int = 4):
        super().__init__()
        self.d_model = d_model
        assert n_heads % 2 == 0, "n_heads must be even for NTH split"
        self.mid_heads = n_heads // 2
        
        # Upgraded to RoPE Attention
        self.local_attn = RoPEMultiheadAttention(d_model, self.mid_heads)
        self.global_attn = RoPEMultiheadAttention(d_model, self.mid_heads)
        
        self.local_window = local_window
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        B, T, D = x.shape
        
        # --- 1. Local Neighborhood Attention ---
        # Create Local Mask [T, T]
        indices = torch.arange(T, device=x.device)
        dist = indices.unsqueeze(0) - indices.unsqueeze(1)
        # 0 for keep, -inf for mask
        local_mask_2d = (dist.abs() > self.local_window) # True to mask
        local_mask_float = torch.zeros((T, T), device=x.device)
        local_mask_float = local_mask_float.masked_fill(local_mask_2d, -1e9)
        
        # Local Branch
        h_local, _ = self.local_attn(x, x, x, key_padding_mask=mask, attn_mask=local_mask_float)
        
        # --- 2. Global Trend Attention ---
        # Global Branch (Full Context, no extra mask)
        h_global, _ = self.global_attn(x, x, x, key_padding_mask=mask)
        
        # --- 3. Fuse ---
        h_fused = (h_local + h_global) / 2.0
        out = self.out_proj(h_fused)
        
        return self.norm(residual + out)

class NTHEncoderBlock(nn.Module):
    """
    [Step 3] The Combined NTH/TFT Encoder Block.
    1. Processing: Gated Residual Network (SwiGLU) for feature extraction.
    2. Mixing: NTH Attention (Local + Global, RoPE) for temporal mixing.
    """
    def __init__(self, d_model: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.grn = GatedResidualNetwork(d_model, hidden_dim)
        self.attn = NTHAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Feature Processing (Time-distributed)
        x = self.grn(x)
        
        # 2. Temporal Mixing
        x = self.attn(x, mask=mask)
        
        return x
