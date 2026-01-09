import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class AsymmetricLoss(nn.Module):
    """
    [SOTA 2025] Asymmetric Loss for Medical Diagnosis.
    Unlike Focal Loss which just handles down-weighting easy negatives,
    Asymmetric Loss allows us to explicitly PENALIZE False Negatives more than False Positives.
    Crucial for Sepsis: Missing a case (FN) is worse than a false alarm (FP).
    
    [v13.0 PATCH] Tuned gamma values based on data analysis:
    - Data shows 98.24% normal, 1.76% sepsis at timestep level
    - gamma_neg=6: Aggressively down-weight easy negatives (was 4)
    - gamma_pos=0: Don't down-weight any positives - they're precious (was 1)
    """
    def __init__(self, gamma_neg=6, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
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

class EvidentialLoss(nn.Module):
    """
    [SOTA 2025] Evidential Loss (Type II Maximum Likelihood).
    Minimizes the "Bayes Risk" with respect to the Dirichlet prior.
    
    Components:
    1. Negative Log Likelihood (NLL): Fit the data.
    2. KL Divergence: Regularize towards uniform distribution (vacuous prior) to prevent overconfidence.
    """
    def __init__(self, num_classes: int = 2, annealing_step: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.epoch_num = 0

    def forward(self, alpha: torch.Tensor, y: torch.Tensor, epoch_num: int = None) -> torch.Tensor:
        """
        alpha: [B, C] Dirichlet concentration parameters (alpha = evidence + 1)
        y: [B, C] One-hot target labels
        """
        if epoch_num is not None:
            self.epoch_num = epoch_num
            
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 1. Expected Mean Squared Error (Risk)
        # A = E[p] = alpha / S
        # Loss = (y - A)^2 + Var(p)
        A = alpha / S
        m = alpha / S
        
        # Log Likelihood of the Dirichlet (Type 2 ML)
        # L = sum( y * (log(S) - log(alpha)) )
        nll = torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        
        # 2. KL Divergence Regularizer (Penalty for being confident but wrong)
        # Drives distribution towards uniform Dirichlet [1, 1, ...] when evidence is low/wrong.
        # annealed_weight = min(1, epoch / 10)
        annealing_coef = min(1, max(self.epoch_num / self.annealing_step, 0))
        
        # KL(Dir(alpha) || Dir([1,1,...]))
        # Approximate: alpha_tilde = y + (1-y)*alpha
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        # KL term
        kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(self.num_classes, device=alpha.device)) \
             - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
             + torch.sum((alpha_tilde - 1) * (torch.digamma(S_tilde) - torch.digamma(torch.tensor(self.num_classes, device=alpha.device))), dim=1, keepdim=True)
             
        # Combine
        loss = nll + annealing_coef * kl
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

class DropPath(nn.Module):
    """
    [v10.0] Stochastic Depth (DropPath) regularization.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

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
    def __init__(self, d_model: int, n_heads: int, drop_path_prob: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn_net = nn.Sequential(
            SwiGLU(d_model, d_model * 4), 
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(0.1)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Pre-Norm Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.drop_path(self.dropout(attn_out))
        
        # 2. Pre-Norm FFN (SwiGLU)
        x_norm = self.norm2(x)
        ffn_out = self.ffn_net(x_norm)
        x = x + self.drop_path(self.dropout(ffn_out))
        
        return x

class SequenceAuxHead(nn.Module):
    """
    [Step 4] Sequence-Aware Classification Head - SOTA Version.
    Features: CLS Token, RoPE Attention, SwiGLU FFN, RMSNorm, Asymmetric Loss.
    """
    def __init__(self, d_model: int, num_classes: int = 1, num_layers: int = 2, n_heads: int = 4, drop_path_prob: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # SOTA Stack
        self.blocks = nn.ModuleList([
            SotaTransformerBlock(d_model, n_heads, drop_path_prob=drop_path_prob) for _ in range(num_layers)
        ])
        
        # Final Projection
        self.head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_classes)
        )
        
        # [v15.0 SOTA] Prior-Aware Initialization
        # Constraint: Sepsis prevalence is 1.76%.
        # Standard init assumes 50/50 (binary) or Uniform (multi-class), causing massive initial gradient shock.
        # Fix: Hardcode bias to log(odds) of prevalence.
        final_layer = self.head[-1]
        
        # Standard logic for Imbalanced Classification (Works for Softmax/Sigmoid and EDL)
        # Target: P(Sepsis) approx 0.0176
        bias_val = -4.02 # log(0.0176 / 0.9824)
        
        if num_classes > 1:
            # Multi-class Case (Stable vs Pre-Shock vs Shock)
            # Class 0 (Stable) is dominant (~98%) -> Bias 0 (Reference)
            # Classes > 0 are rare (~2%) -> Bias -4.02
            nn.init.zeros_(final_layer.bias)
            with torch.no_grad():
                final_layer.bias[1:].fill_(bias_val)
        else:
            # Binary Case
            nn.init.constant_(final_layer.bias, bias_val)
        

        
        # [v14.0 PATCH] Replaced AsymmetricLoss with EvidentialLoss
        # This allows the model to output *uncertainty* alongside probability.
        # Critical for safety when inputs are 90% imputed.
        self.criterion = EvidentialLoss(num_classes=num_classes, annealing_step=10)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, epoch_num: int = None, return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """
        [SOTA 2025] Evidential Forward Pass.
        Returns:
            Dict containing 'logits', 'alpha' (Dirichlet), 'uncertainty', and 'loss' (if targets)
        """
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
        
        # 4. Predict
        if return_sequence:
            seq_out = x_seq[:, 1:, :] 
            logits = self.head(seq_out)
        else:
            cls_out = x_seq[:, 0, :]
            logits = self.head(cls_out)
        
        # 5. [SOTA 2025] Evidential Deep Learning (EDL)
        # alpha = evidence + 1. We use Softplus for evidence to ensure non-negativity.
        evidence = F.softplus(logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        # Vacuous Uncertainty: Lower means the model is more confident in the distribution
        uncertainty = logits.shape[-1] / S 
        
        # 6. Loss
        loss = None
        if targets is not None and not return_sequence:
            num_classes = logits.shape[-1]
            if num_classes > 1:
                # [SOTA FIX] Multi-Class One-Hot Conversion
                if targets.ndim == 1:
                    targets_oh = F.one_hot(targets.long(), num_classes=num_classes).float()
                else:
                    targets_oh = targets.float()
            else:
                targets_oh = targets.float().unsqueeze(-1) if targets.ndim == 1 else targets.float()
                    
            # [v14.0 PATCH] Use Evidential Loss on alphas
            # We pass 'alpha' (Dirichlet params) instead of 'logits'
            # Note: The loss needs the current epoch for KL annealing. 
            loss = self.criterion(alpha, targets_oh, epoch_num=epoch_num)
            
        return {
            "logits": logits,
            "alpha": alpha,
            "uncertainty": uncertainty,
            "loss": loss
        }
