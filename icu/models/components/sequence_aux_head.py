import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AsymmetricLoss(nn.Module):
    """
    [SOTA 2025] Asymmetric Loss for Medical Diagnosis.
    Unlike Focal Loss which just handles down-weighting easy negatives,
    Asymmetric Loss allows us to explicitly PENALIZE False Negatives more than False Positives.
    Crucial for Sepsis: Missing a case (FN) is worse than a false alarm (FP).
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        x: logits
        y: targets (multi-label binarized vector)
        """
        # Calculate probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy
        # For positives: log(p)
        # For negatives: log(1-p)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        # Down-weight easy negatives (gamma_neg > gamma_pos)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t=1 else 1-p
                pt = (pt0 + pt1).detach()
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            loss = -one_sided_w * (los_pos + los_neg)
        else:
            loss = -(los_pos + los_neg)
            
        return loss.mean()

# --- Shared SOTA Components (Duplicated from nth_encoder.py for independence) ---
class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear_act = nn.Linear(input_dim, output_dim, bias=bias)
        self.linear_gate = nn.Linear(input_dim, output_dim, bias=bias)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_act(x) * self.silu(self.linear_gate(x))

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
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos().unsqueeze(0).unsqueeze(0)
            self.cached_sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat((-q2, q1), dim=-1)
    k_rot = torch.cat((-k2, k1), dim=-1)
    q_out = (q * cos) + (q_rot * sin)
    k_out = (k * cos) + (k_rot * sin)
    return q_out, k_out

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.shape
        q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None: scores = scores + attn_mask
        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, -1e9)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(output), weights

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.size(-1) ** -0.5)
        return self.scale * x / (rms_x + self.eps)

class SotaTransformerBlock(nn.Module):
    """
    [2025 SOTA] Pre-RMSNorm + RoPE + SwiGLU Block.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model) # FFN usually projects up?
        # Standard Transformer FFN: d -> 4d -> d
        # SwiGLU handles internal projection.
        # Let's verify standard SwiGLU FFN: 
        # Usually: Gate(d->4d), Val(d->4d) -> Output(4d->d)
        # My SwiGLU(d, d) above is simple. Let's make a proper FFN wrapper.
        self.ffn_net = nn.Sequential(
            SwiGLU(d_model, d_model * 4), # Expands to 4x
            nn.Linear(d_model * 4, d_model) # Projects back
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Pre-Norm Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        
        # 2. Pre-Norm FFN (SwiGLU)
        x_norm = self.norm2(x)
        ffn_out = self.ffn_net(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x

class SequenceAuxHead(nn.Module):
    """
    [Step 4] Sequence-Aware Classification Head - SOTA Version.
    Features: CLS Token, RoPE Attention, SwiGLU FFN, RMSNorm, Asymmetric Loss.
    """
    def __init__(self, d_model: int, num_classes: int = 1, num_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # SOTA Stack
        self.blocks = nn.ModuleList([
            SotaTransformerBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        
        # Final Projection
        self.head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_classes)
        )
        
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = x.shape[0]
        # 1. Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, x], dim=1)
        
        # 2. Adjust Mask
        if mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)
            seq_mask = torch.cat([cls_mask, mask], dim=1) 
        else:
            seq_mask = None
            
        # 3. Process Sequence
        for block in self.blocks:
            x_seq = block(x_seq, mask=seq_mask)
        
        # 4. Extract CLS
        cls_out = x_seq[:, 0, :]
        
        # 5. Predict
        logits = self.head(cls_out)
        
        # 6. Loss
        loss = None
        if targets is not None:
            num_classes = logits.shape[-1]
            if num_classes > 1:
                # Multi-Class: Expect Long indices, convert to One-Hot
                if targets.ndim == 1 and (targets.dtype == torch.long or targets.dtype == torch.int):
                    targets = F.one_hot(targets, num_classes=num_classes).float()
                elif targets.ndim == 1:
                     # Float but flat? Unsafe. Assume indices if >1 class.
                     targets = F.one_hot(targets.long(), num_classes=num_classes).float()
            else:
                # Binary: [B] -> [B, 1]
                if targets.ndim == 1:
                    targets = targets.float().unsqueeze(-1)
                    
            loss = self.criterion(logits, targets)
            
        return logits, loss
