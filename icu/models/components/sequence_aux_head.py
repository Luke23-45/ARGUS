import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

# Modular Imports (Unified Primitives from nth_encoder)
from icu.models.components.nth_encoder import SwiGLU, DropPath, RotaryEmbedding, RoPEMultiheadAttention, RMSNorm, SotaTransformerBlock

class AsymmetricLoss(nn.Module):
    """
    [SOTA 2025] Asymmetric Loss for Medical Diagnosis.
    Unlike Focal Loss which just handles down-weighting easy negatives,
    Asymmetric Loss allows us to explicitly PENALIZE False Negatives more than False Positives.
    Crucial for Sepsis: Missing a case (FN) is worse than a false alarm (FP).
    
    [v13.0 PATCH] Tuned gamma values based on data analysis:
    - Data shows 98.24% normal, 1.76% sepsis at timestep level
    - gamma_neg=2: Moderate down-weighting (was 6, which caused gradient starvation)
    - gamma_pos=0: Don't down-weight any positives - they're precious (was 1)
    """
    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
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

class SequenceAuxHead(nn.Module):
    """
    [Step 4] Sequence-Aware Classification Head - SOTA Version.
    Features: CLS Token, RoPE Attention, SwiGLU FFN, RMSNorm, AsymmetricLoss (v14.1).
    
    [v14.1 FORENSIC FIX] Reverted from EvidentialLoss to AsymmetricLoss.
    Rationale: EvidentialLoss KL-term forces uniform distribution, directly
    conflicting with Prior-Aware Initialization and causing collapse on skewed data (1.76% positive).
    AsymmetricLoss correctly handles the 98% easy negatives without fighting the prior.
    """
    def __init__(
        self, 
        d_model: int, 
        num_classes: int = 1, 
        num_layers: int = 4, 
        n_heads: int = 4, 
        drop_path_prob: float = 0.1,
        # [v14.1 PATCH] Configurable ASL hyperparameters
        gamma_neg: float = 2.0,  # [v33.0 SOTA FIX] Corrected from 6.0 to 2.0 (Prevents Grad Starvation)
        gamma_pos: float = 0.0,  # No down-weighting of precious positives
        clip: float = 0.05,      # Asymmetric clipping
        prevalence: float = 0.0176 # [v33.0 SOTA FIX] Dynamic Prior (Sepsis-3 Baseline)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes  # [v14.1] Store for activation selection
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
        # Rationale: Standard init assumes 50/50, causing massive initial gradient shock.
        # Fix: Hardcode bias to log(odds) of prevalence.
        final_layer = self.head[-1]
        
        # [v33.0 SOTA FIX] Dynamically calculated logit bias
        # Target: P(Positive) approx prevalence
        # bias = log(p / (1-p))
        bias_val = math.log(prevalence / (1.0 - prevalence))
        
        if num_classes > 1:
            # Multi-class Case (Stable vs Pre-Shock vs Shock)
            # Class 0 (Stable) is dominant -> Bias 0 (Reference)
            # Classes > 0 are rare -> Bias mapped to logit space
            nn.init.zeros_(final_layer.bias)
            with torch.no_grad():
                final_layer.bias[1:].fill_(bias_val)
        else:
            # Binary Case
            nn.init.constant_(final_layer.bias, bias_val)
        
        # [v14.1 FORENSIC FIX] Revert to Asymmetric Loss (Smoking Gun #470)
        self.criterion = AsymmetricLoss(
            gamma_neg=gamma_neg, 
            gamma_pos=gamma_pos, 
            clip=clip
        )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None, 
        epoch_num: int = None,  # [API COMPAT] Unused with ASL
        return_sequence: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        [SOTA 2025] Sequence-Aware Forward Pass (ASL v14.1).
        """
        B = x.shape[0]
        
        # 1. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, x], dim=1)
        
        # 2. Adjust Mask for CLS token
        if mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)
            seq_mask = torch.cat([cls_mask, mask], dim=1) 
        else:
            seq_mask = None
            
        # 3. Process Sequence through Transformer blocks
        for block in self.blocks:
            x_seq = block(x_seq, mask=seq_mask)
        
        # 4. Predict from CLS token or full sequence
        if return_sequence:
            seq_out = x_seq[:, 1:, :]  # Exclude CLS
            logits = self.head(seq_out)
        else:
            cls_out = x_seq[:, 0, :]  # CLS token only
            logits = self.head(cls_out)
            
        # [v89.0] Logit Clamping (Safety) - prevents gradient explosion
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        # 5. [v14.1] Probabilities - activation based on num_classes
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)  # Binary classification
        else:
            probs = torch.softmax(logits, dim=-1)  # Multi-class classification
        
        # 6. [v14.1 API COMPAT] Compute Predictive Entropy as Uncertainty Surrogate
        eps = 1e-12
        if self.num_classes == 1:
            p = probs.clamp(eps, 1.0 - eps)
            entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        else:
            p = probs.clamp(eps, 1.0 - eps)
            entropy = -(p * p.log()).sum(dim=-1, keepdim=True)
        
        # Normalize entropy to [0, 1]
        max_entropy = float(torch.log(torch.tensor(max(self.num_classes, 2), dtype=torch.float32)))
        uncertainty = (entropy / max_entropy).clamp(0.0, 1.0)
        
        if uncertainty.ndim == 1:
            uncertainty = uncertainty.unsqueeze(-1)
        if uncertainty.shape[-1] != 1:
            uncertainty = uncertainty.mean(dim=-1, keepdim=True)
        
        # 7. Loss Calculation
        loss = None
        if targets is not None and not return_sequence:
            num_classes = logits.shape[-1]
            if num_classes > 1:
                if targets.ndim == 1:
                    targets_oh = F.one_hot(targets.long(), num_classes=num_classes).float()
                else:
                    targets_oh = targets.float()
            else:
                targets_oh = targets.float().unsqueeze(-1) if targets.ndim == 1 else targets.float()
                    
            loss = self.criterion(logits, targets_oh)
        
        return {
            "logits": logits,
            "probs": probs,
            "alpha": None,  # Dirichlet not available with ASL
            "uncertainty": uncertainty,
            "loss": loss
        }
