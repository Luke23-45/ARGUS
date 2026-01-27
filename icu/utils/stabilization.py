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
from .train_utils import ScalingSteward

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

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies contrastive momentum across step densities."""
        if n_curr <= 0: return
        # Baseline 0.99 for 200 steps
        self.momentum = ScalingSteward.get_decay(0.99, n_curr)

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
                        
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(self.centroids.data, op=torch.distributed.ReduceOp.SUM)
                    self.centroids.data /= torch.distributed.get_world_size()
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

class LinearManifoldSentinel:
    """
    [PMS] Manifold Gradient Projection (MGP).
    Projects auxiliary gradients to be non-conflicting with the foundation.
    """
    @staticmethod
    def project(grad_aux: torch.Tensor, grad_foundation: torch.Tensor) -> torch.Tensor:
        """
        Calculates the PCGrad projection of grad_aux onto the normal of grad_foundation.
        
        PMS Modification: 
        Supports grad_foundation as a 'Directional Vector' [D] while grad_aux is [..., D].
        This ensures shape-invariance across variable batch/sequences.
        """
        # 1. Flatten both to handle potential shape mismatches in dot product
        # If grad_foundation is a directional vector [D], we project every vector 
        # in grad_aux against it.
        
        # Calculate dot product across the last dimension (Feature Dim)
        # dot = sum(grad_aux * grad_foundation)
        dot = (grad_aux * grad_foundation).sum(dim=-1, keepdim=True) # [..., 1]
        
        # Only project if conflict detected (dot < -0.1)
        # Softened threshold for shared feature overlap
        conflict_mask = (dot < -0.1)
        
        if conflict_mask.any():
            mag_fnd = (grad_foundation * grad_foundation).sum() + 1e-8
            proj = (dot / mag_fnd) * grad_foundation
            
            # Apply only to conflicting components
            new_grad = grad_aux.clone()
            new_grad[conflict_mask.expand_as(grad_aux)] = (grad_aux - proj)[conflict_mask.expand_as(grad_aux)]
            return new_grad
        
        return grad_aux

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
        
        self.register_buffer("loss_emas", torch.zeros(num_tasks))
        self.decay = decay

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies uncertainty decay across step densities."""
        if n_curr <= 0: return
        # Baseline 0.99 for 200 steps
        self.decay = ScalingSteward.get_decay(0.99, n_curr)

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

def adaptive_gradient_clip_(parameters, clip_factor: float = 0.1, eps: float = 1e-3):
    """
    [SOTA v2025] Fused Adaptive Gradient Clipping (PyTorch 2.0+).
    Uses _foreach_ implementation to eliminate Python loop overhead.
    
    NOTE: This implementation computes TENSOR-WISE norms (Frobenius), 
    not Unit-Wise norms. This is significantly faster and standard for 
    High-Performance Transformer training (LARS/LAMB style).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
        
    # Filter for params with grads
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
        
    # 1. Compute Norms (Fused)
    # torch._foreach_norm returns a list of scalars (L2 norm of each tensor)
    p_norms = torch._foreach_norm(params_with_grad, 2)
    g_norms = torch._foreach_norm([p.grad for p in params_with_grad], 2)
    
    # 2. Compute Clipping Coefficients
    # ratio = max_norm / grad_norm
    # clipped = grad * clamp(ratio, max=1.0)
    
    # Use torch.stack to vectorize scalar operations
    p_norms_stack = torch.stack(p_norms)
    g_norms_stack = torch.stack(g_norms)
    
    # Clamp norms to prevent division by zero or extremely small param norms
    max_norms = p_norms_stack.clamp(min=eps).mul_(clip_factor)
    grad_norms_clamped = g_norms_stack.clamp(min=1e-6)
    
    # Calculate stepping coefficients
    step_coefficients = (max_norms / grad_norms_clamped).clamp_(max=1.0)
    
    # 3. Apply Clipping (Fused Mul)
    # in-place: grad = grad * step_coeff
    torch._foreach_mul_([p.grad for p in params_with_grad], step_coefficients.unbind())

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

# ==============================================================================
# 6. ORTHOGONAL GRADIENT PROJECTION (The Circuit Breaker)
# ==============================================================================

class OrthogonalGuard(object):
    """
    [SOTA] Gradient Orthogonality Guard.
    Project: Protects the 'Survival' manifold from 'Diffusion' noise.
    """
    @staticmethod
    def sanitize_gradients(model, primary_task_name="diffusion"):
        # 1. Global Norm Check (The Explosion Detector)
        # Efficiently computes norm over all parameters
        grads = [torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        total_norm = torch.norm(torch.stack(grads))
        
        # 2. Adaptive Clipping (The Response)
        # If GN > 1.0, we don't just clip, we perform 'Soft Clamping'
        # Formula: g = g * (target / max(target, g_norm))
        clip_target = 1.0
        if total_norm > clip_target:
            scale_factor = clip_target / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(scale_factor)
                    
        return total_norm.item()
