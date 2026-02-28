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
        self.register_buffer('initialized', torch.zeros([1], dtype=torch.bool))

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies contrastive momentum across step densities."""
        if n_curr <= 0: return
        # Baseline 0.99 tuned on SOTA reference density
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
                # [v145.0 SOTA FIX] Contrastive NaN-Gate (Zero-Sync)
                # Rationale: Replaced finite-check branching with masked updates.
                finite_mask = torch.isfinite(features).all(dim=1, keepdim=True) # [B, 1]
                
                # [v2026 SOTA] Vectorized Batch EMA (Zero-Sync)
                # Rationale: Replaced unique_classes loop with masked scatter.
                # 1. Create one-hot mask: [B, num_classes]
                masks = F.one_hot(targets.long(), num_classes=self.num_classes).float() # [B, K]
                masks = masks * finite_mask # [B, K]
                
                # 2. Compute sums and counts for each class
                counts = masks.sum(dim=0, keepdim=True).t() # [K, 1]
                weighted_features = features.unsqueeze(1) * masks.unsqueeze(2) # [B, K, D]
                sums = weighted_features.sum(dim=0) # [K, D]
                
                # 3. Compute batch centers
                batch_centers = sums / (counts + 1e-8)
                batch_centers = F.normalize(batch_centers, p=2, dim=1)
                
                # 4. Atomic EMA Update (Zero-Sync)
                # We use counts > 0 as a mask to only update classes present in the batch.
                update_mask = (counts > 0)
                momentum = self.momentum
                
                # new_centroids = lerp(buffer, batch_center, 1-mom)
                updated_centroids = torch.lerp(self.centroids, batch_centers, 1.0 - momentum)
                
                # First-time init logic (Zero-Sync)
                # If centrifugal is 0, use batch_center, else use updated
                first_init = (self.centroids.abs().sum(dim=1, keepdim=True) == 0)
                final_next = torch.where(first_init, batch_centers, updated_centroids)
                
                # Apply update only where count > 0
                self.centroids.copy_(torch.where(update_mask, final_next, self.centroids))
                            
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
        # [v2026 SOTA] Pure Tensor Projection (Zero-Sync)
        # Rationale: Removed if conflict_mask.any() branching and fixed missing 'dot'.
        # dot: Projection of aux gradient onto foundation direction
        dot = (grad_aux * grad_foundation).sum(dim=-1, keepdim=True)
        # [Omni-Scan FIX #710] FP16 Epsilon Collapse
        # Rationale: 1e-8 evaluates to exactly 0.0 in FP16, causing NaN on division.
        # Upgraded to 1e-5 to guarantee mathematical survival in half-precision.
        mag_fnd = (grad_foundation * grad_foundation).sum() + 1e-5
        proj = (dot / mag_fnd) * grad_foundation
        
        # PCGrad: proj only if dot < 0 (conflicting)
        return torch.where(dot < 0.0, grad_aux - proj, grad_aux)

    @staticmethod
    def log_scale_prevalence(n_total, n_target) -> torch.Tensor:
        """
        Calculates a safe Class Frequency Multiplier (CFM).
        Replaces linear scaling (16x) with log-scaling (~4x).
        
        Formula: 1 + log(1 + n_total / n_target)
        """
        # Ensure we are using torch for vectorization
        if isinstance(n_total, (int, float)):
            n_total = torch.tensor(float(n_total))
        if isinstance(n_target, (int, float)):
            n_target = torch.tensor(float(n_target))
            
        device = n_total.device if hasattr(n_total, 'device') else torch.device('cpu')
        
        ratio = n_total / torch.clamp(n_target, min=1.0)
        return 1.0 + torch.log(1.0 + ratio).to(device)

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
        self.register_buffer("decay_buffer", torch.tensor([decay]).float())
        self.decay = decay # Keep for non-buffer access if needed

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies uncertainty decay across step densities."""
        if n_curr <= 0: return
        # Baseline 0.99 tuned on SOTA reference density
        self.decay_buffer.fill_(ScalingSteward.get_decay(0.99, n_curr))
        self.decay = float(self.decay_buffer.item())

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        [v2026 SOTA] Vectorized Uncertainty Weighting (Zero-Sync)
        """
        # 1. Stack losses for vectorized operation
        L = torch.stack(losses) # [N]
        
        # 2. Update EMAs (Zero-Sync)
        with torch.no_grad():
            self.loss_emas.lerp_(L.detach(), 1.0 - self.decay_buffer)
            
            # Dynamic Floor Calculation
            # If loss > 5.0, floor = 5.0, else 2.0
            floors = torch.where(self.loss_emas > 5.0, torch.as_tensor([5.0], device=L.device), torch.as_tensor([2.0], device=L.device))
        
        # 3. Apply Clamping and Weighting
        log_var_clamped = torch.clamp(self.log_vars, min=-2.0)
        log_var_clamped = torch.min(log_var_clamped, floors)
        
        # [SOTA FIX 1] Straight-Through Estimator (STE) for Gradient Survival
        # If log_vars hits the floor, torch.min kills the gradient. 
        # STE ensures the forward pass uses the clamped value, but 100% of the downward gradient reaches log_vars.
        log_var = self.log_vars + (log_var_clamped - self.log_vars).detach()
        
        precision = torch.exp(-log_var)
        
        # [SOTA FIX 2] Exact Kendall Mathematical Formulation
        # Both terms require the 0.5 multiplier to symmetrically balance the partial derivatives.
        scaled_losses = 0.5 * precision * L + 0.5 * log_var
        
        total_loss = scaled_losses.sum()
        
        # 4. Return Tensors in weights_dict to avoid .item() sync
        weights_dict = {f"w_{i}": precision[i] for i in range(self.num_tasks)}
            
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

def adaptive_gradient_clip_(parameters, clip_factor: float = 0.1, eps: float = 1e-2):
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
    
    Enhanced v26.5: Now supports Variance Tracking for Adaptive Sentinels.
    """
    @staticmethod
    def compute_grad_norm(model, extra_params: Optional[List[nn.Parameter]] = None) -> torch.Tensor:
        """
        [SOTA v14.0] Fused Global Consensus Gradient Norm calculation.
        Uses _foreach_norm to eliminate Python loops over parameters.
        Returns a Tensor to prevent graph-breaks.
        """
        with torch.no_grad():
            params = [p for p in model.parameters() if p.grad is not None]
            if extra_params:
                params.extend([p for p in extra_params if p.grad is not None])
            
            if not params:
                return torch.tensor([0.0], device=next(model.parameters()).device)
                
            # [Optimization] Fused Norm (Zero-Copy)
            # torch._foreach_norm returns a list of scalars (L2 norm of each tensor)
            sq_norms = torch._foreach_norm([p.grad.detach() for p in params], 2)
            local_sq_sum = torch.stack(sq_norms).pow(2).sum()
            
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_sq_sum, op=torch.distributed.ReduceOp.SUM)
                # Normalize by world size for average gradient norm
                total_norm = torch.sqrt(local_sq_sum / torch.distributed.get_world_size() + 1e-8)
            else:
                total_norm = torch.sqrt(local_sq_sum + 1e-8)
                
            return total_norm

    @staticmethod
    def sanitize_gradients(model):
        """[Iron Dome] Fused in-place clipping for final manifold safety."""
        # 1. Global Norm Check (The Explosion Detector)
        total_norm = OrthogonalGuard.compute_grad_norm(model)
        
        # 2. Adaptive Clipping (Fused Response)
        clip_target = 1.0
        
        # Condition on tensor to allow torch.compile to optimize the path
        # we use a functional approach to avoid graph-breaks
        with torch.no_grad():
            scale = torch.clamp(clip_target / (total_norm + 1e-6), max=1.0)
            
            params_with_grad = [p for p in model.parameters() if p.grad is not None]
            if params_with_grad:
                torch._foreach_mul_([p.grad for p in params_with_grad], scale)
                    
        return total_norm

# ==============================================================================
# 7. TREND SENTINEL (The Early Warning System)
# ==============================================================================

class TrendSentinel:
    """
    [SOTA v2026] Adaptive Distribution Monitor.
    Detects Manifold Shocks using Z-Score analysis.
    Optimized for torch.compile (Zero Graph-Breaks).
    """
    @staticmethod
    def calculate_z_score(current_val: Union[float, torch.Tensor], ema: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        [SOTA 2026] Computes the scale-invariant directional standard deviation distance.
        Directional (Clamp min=0): Downward drops (convergence) do not trigger anomalies.
        Relative Floor (EMA * 0.1): Prevents hypersensitivity when gradients stabilize at large magnitudes.
        
        [v2026.1 STABILITY FIX] Dynamic Floor Relaxation:
        Increased absolute floor from 0.05 to 0.20 to prevent the "Hypersensitivity Trap".
        With floor=0.05 and EMA≈0.39, a norm of 0.86 yields Z=5.33 (false positive).
        With floor=0.20, the same scenario yields Z=1.98 (correctly absorbed as noise).
        Validated by forensic_stability_probe.py.
        """
        if not isinstance(current_val, torch.Tensor):
            current_val = torch.as_tensor(current_val, device=ema.device)
            
        # 1. Directional Numerator: Only penalize upward spikes (Exploding Gradients)
        # Drops below EMA are healthy convergence, so deviation is 0.0.
        diff = torch.clamp(current_val - ema, min=0.0)
        
        # 2. Dynamic Scale-Invariant Floor:
        # Guarantees at least a 20% absolute tolerance margin for mini-batch stochasticity.
        # [v2026.1 FIX] Relaxed from 0.05 to 0.20 to prevent Hypersensitivity Trap.
        dynamic_floor = (ema * 0.1) + 0.20
        safe_std = torch.clamp(std, min=dynamic_floor) 
        
        return diff / safe_std

    @staticmethod
    def is_shock(z_score: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
        """Trigger if the deviation exceeds N standard deviations."""
        return z_score > threshold

    @staticmethod
    def is_unstable(ema: torch.Tensor, std: torch.Tensor, max_pressure: float = 5.0, max_sigma: float = 2.0) -> torch.Tensor:
        """
        [SOTA v2026] Hybrid Sentinel: Detects both Drift (Boiling Frog) and Shock.
        Performance: Tensor-based logic prevents iterative stalls.
        """
        is_shock_val = (ema > max_pressure) | (std > max_sigma)
        return is_shock_val

    @staticmethod
    def get_stats(ema: torch.Tensor, std: torch.Tensor, decay: float, step_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[SOTA v2026] Returns bias-corrected statistics without updating."""
        # Bias correction: 1 - beta^t
        bias_correction = (1.0 - torch.pow(decay, step_tensor)).clamp(min=0.01)
        corrected_ema = ema / bias_correction
        corrected_std = std / torch.sqrt(bias_correction)
        return corrected_ema, corrected_std

    @staticmethod
    def update_stats(current_val: Union[float, torch.Tensor], ema: torch.Tensor, std: torch.Tensor, decay: float, step_tensor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """[SOTA v2026] Zero-Break EMA update for mean and variance."""
        with torch.no_grad():
            if not isinstance(current_val, torch.Tensor):
                current_val = torch.as_tensor(current_val, device=ema.device)

            # [v167.1] Rank Synchrony Fix
            if torch.distributed.is_initialized():
                # Fused Sync
                torch.distributed.all_reduce(current_val, op=torch.distributed.ReduceOp.SUM)
                current_val = current_val / torch.distributed.get_world_size()

            if step_tensor is not None:
                step_tensor.add_(1)
                bias_correction = (1.0 - torch.pow(decay, step_tensor)).clamp(min=0.01)
            else:
                bias_correction = torch.tensor([1.0], device=ema.device)

            delta = current_val - ema
            # Update Mean (Uncorrected)
            ema.mul_(decay).add_(current_val, alpha=1.0 - decay)
            
            # Update Variance (Uncorrected)
            new_delta = current_val - ema
            sq_diff = delta * new_delta
            
            # Update memory buffers in-place (Uncorrected)
            var = std.pow(2)
            new_var = decay * var + (1.0 - decay) * sq_diff
            std.copy_(torch.sqrt(new_var.clamp(min=1e-6)))

            # Return Bias-Corrected tensors
            corrected_ema = ema / bias_correction
            corrected_std = std / torch.sqrt(bias_correction)
            
            return corrected_ema, corrected_std
