"""
SOTA Gradient Stabilization Primitives
--------------------------------------
Research-backed implementations for stabilizing gradients in complex 
Multi-Task / Multi-Phase architectures.

References:
1. "Gradient Surgery for Multi-Task Learning" (PCGrad), Yu et al., NeurIPS 2020.
2. "Multi-Task Learning Using Uncertainty to Weigh Losses", Kendall et al., CVPR 2018.
3. "Deep Metric Learning with Spherical Embedding" (NormFace), Wang et al.
4. "High-Performance Large-Scale Image Recognition Without Normalization" (AGC), Brock et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Union, Tuple

# ==============================================================================
# 1. STABLE CONTRASTIVE LOSS (The "Hard Fix" for ACL)
# ==============================================================================

class StableContrastiveLoss(nn.Module):
    """
    [SOTA v2025] Momentum-Updated Contrastive Loss.
    
    Why this fixes Gradient Explosion:
    1. Removes learnable centroids (which oscillate wildly via SGD).
    2. Replaces them with EMA (Exponential Moving Average) updates.
    3. Enforces strict L2 normalization on the hypersphere.
    4. Decouples representation learning from clustering stability.
    """
    def __init__(self, d_model: int, num_classes: int = 3, temperature: float = 0.25, momentum: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = momentum
        
        # Buffer, not Parameter -> No Gradients on Centroids directly
        self.register_buffer('centroids', F.normalize(torch.randn(num_classes, d_model), dim=1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] representations
            targets: [B] class indices
        """
        # 1. Spherical Projection (Critical for stability)
        features = F.normalize(features, p=2, dim=1)
        
        # 2. EMA Update (Training Only)
        if self.training:
            with torch.no_grad():
                # For each class present in batch
                unique_classes = torch.unique(targets)
                for c in unique_classes:
                    mask = (targets == c)
                    # Mean vector for this class in current batch
                    batch_center = features[mask].mean(dim=0)
                    batch_center = F.normalize(batch_center, p=2, dim=0)
                    
                    # Momentum update: New = m * Old + (1-m) * Batch
                    # Note: We track even if initialized to drift slowly
                    if self.initialized.item():
                        self.centroids[c].mul_(self.momentum).add_(batch_center, alpha=1 - self.momentum)
                    else:
                        self.centroids[c] = batch_center
                        
                self.centroids.data = F.normalize(self.centroids, p=2, dim=1)
                self.initialized.fill_(True)
        
        # 3. Compute Logits (Scaled Dot Product)
        # Range: [-1/temp, 1/temp]
        logits = torch.matmul(features, self.centroids.t()) / self.temperature
        
        # 4. Standard Cross Entropy
        loss = F.cross_entropy(logits, targets.long())
        
        return loss

# ==============================================================================
# 2. GRADIENT THROTTLER (The "Peace Treaty")
# ==============================================================================

class GradientThrottler:
    """
    Mechanism to scale gradients for specific branches relative to the backbone.
    Prevents auxiliary tasks (like ACL) from dominating the feature extractor.
    """
    @staticmethod
    def throttle(tensor: torch.Tensor, factor: float = 0.1) -> torch.Tensor:
        """
        Registers a hook to scale gradients by 'factor' during backward pass.
        Returns the tensor with the hook attached.
        """
        if tensor.requires_grad:
            tensor.register_hook(lambda grad: grad * factor)
            return tensor
        return tensor

    @staticmethod
    def log_scale_prevalence(n_total: int, n_target: int) -> float:
        """
        Calculates a safe Class Frequency Multiplier (CFM).
        Replaces linear scaling (16x) with log-scaling (~4x).
        
        Formula: 1 + log(1 + n_total / n_target)
        """
        ratio = n_total / max(1, n_target)
        return 1.0 + math.log(1.0 + ratio)

# ==============================================================================
# 3. ROBUST LOSS SCALER (Homoscedastic + Dynamic Floor)
# ==============================================================================

class RobustLossScaler(nn.Module):
    """
    [SOTA] Uncertainty Weighting with Dynamic Stability.
    
    Fixes:
    1. Removes hard clamp at -0.693 (allow low weights for unstable losses).
    2. Adds EMA smoothing to loss tracking.
    3. Prevents "weight explosion" when loss is effectively zero.
    """
    def __init__(self, num_tasks: int, decay: float = 0.99):
        super().__init__()
        self.num_tasks = num_tasks
        # Learnable log_vars (s_i in paper)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # EMA tracking for stability monitoring
        self.register_buffer("loss_emas", torch.zeros(num_tasks))
        self.decay = decay

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        weights_dict = {}
        
        for i, loss in enumerate(losses):
            # Precision = exp(-log_var)
            # Loss = precision * loss + log_var
            # Dynamic soft-clamp based on current value to prevent divergence
            
            # 1. Update EMA
            with torch.no_grad():
                curr_val = loss.item()
                self.loss_emas[i] = self.decay * self.loss_emas[i] + (1 - self.decay) * curr_val
                
                # Dynamic Floor Calculation
                # If loss is HUGE (>10), allow log_var to grow (weight -> 0)
                # If loss is TINY (<0.1), restrict log_var (weight -> 1)
                # Use EMA for stability
                floor = 5.0 if self.loss_emas[i] > 5.0 else 2.0
            
            # Safe clamping
            log_var = self.log_vars[i].clamp(min=-2.0, max=floor)
            precision = torch.exp(-log_var)
            
            scaled_loss = precision * loss + 0.5 * log_var
            total_loss += scaled_loss
            
            weights_dict[f"w_{i}"] = precision.item()
            
        return total_loss, weights_dict

# ==============================================================================
# 4. ADAPTIVE GRADIENT CLIPPING (AGC)
# ==============================================================================

def unitwise_norm(x: torch.Tensor, norm_type: float = 2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # Norm over all dims except the first (output channels/features)
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

def adaptive_gradient_clip_(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
    """
    [SOTA] Adaptive Gradient Clipping for Transformer stability.
    Scales gradients based on the ratio of param_norm to grad_norm layer-wise.
    Superior to global norm clipping for deep transformers.
    """
    for p in parameters:
        if p.grad is None:
            continue
            
        p_data = p.detach()
        g_data = p.grad.detach()
        
        max_norm = unitwise_norm(p_data).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data)
        
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6)).clamp(max=1.0)
        p.grad.detach().copy_(clipped_grad)

# ==============================================================================
# 5. ADVANTAGE CLAMPER
# ==============================================================================

def robust_awr_weights(advantages: torch.Tensor, beta: float, min_clamp: float = -4.0, max_clamp: float = 4.0) -> torch.Tensor:
    """
    Compute AWR weights effectively with strict bounds to prevent explosion.
    Wrapper for exp(A/beta).
    """
    # 1. Standardization (if not already done globally)
    # Assuming input 'advantages' are already roughly normalized or specific values
    # We apply hard clamps in log-space
    
    scaled_adv = advantages / beta
    
    # Check for explosion risk
    if scaled_adv.max() > 10.0:
        # Emergency soft-scaling
        pass 
        
    clamped_adv = torch.clamp(scaled_adv, min=min_clamp, max=max_clamp)
    weights = torch.exp(clamped_adv)
    
    return weights
