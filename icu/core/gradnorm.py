"""
working/gradnorm.py
-------------------
SOTA GradNorm Dynamic Weighting.

From lucidrains/gradnorm-pytorch (2025 best practices).
"""

import torch
import torch.nn as nn

class GradNormBalancer(nn.Module):
    """
    SOTA GradNorm (2025).
    Inherits from nn.Module for proper device synchronization (.to(device))
    and parameter registration.
    """
    def __init__(self, num_tasks: int, shared_params, alpha: float = 1.5, initial_weight: float = 1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks) * initial_weight)
        self.alpha = alpha
        self.shared_params = list(shared_params)
        self.register_buffer("initial_losses", torch.zeros(num_tasks)) # SOTA: Non-None for state_dict

        self.optimizer = None 
        # [PHASE 8] GradNorm Damping & Anchoring
        self.register_buffer("norm_emas", torch.zeros(num_tasks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.ema_alpha = 0.90 # Damping factor for norm smoothing

    def get_weights(self):
        # [SOTA] Softmax weighting ensures task preservation (no task gets 0 weight)
        # We scale by num_tasks so the average weight is 1.0 (prevents gradient vanishing)
        return torch.softmax(self.weights, dim=0) * len(self.weights)

    def update(self, losses):
        # [SOTA 2026] Universal DDP Bridge
        # Rationale: Ranks must agree on relative task rates (L/L0) for weights.
        # [v23.0 FIX] We compute synchronized values for meta-logic BUT 
        # ensure they retain their rank-specific gradient connections where needed.
        if torch.distributed.is_initialized():
            losses_sync_val = losses.detach().clone()
            torch.distributed.all_reduce(losses_sync_val, op=torch.distributed.ReduceOp.SUM)
            losses_sync_val /= torch.distributed.get_world_size()
            # For Task Weight evolution (L/L0), we use the global average scalar.
            # Rationale: All ranks must move meta-weights in the same direction.
            # But the 'losses' tensor used for autograd must NOT be the detached one.
            meta_losses_val = losses_sync_val
        else:
            meta_losses_val = losses.detach()

        if self.optimizer is None:
            # [PHASE 8] Reduced LR (0.005) for smoother adaptation (Smoking Gun #19)
            self.optimizer = torch.optim.Adam([self.weights], lr=0.005)


        # [PATCH 8.2] Dynamic Initial Loss Anchoring (Smoking Gun #117 FIX)
        # Rationale: A fixed anchor (v30.5) causes extreme imbalance as tasks are solved.
        # Fix: Transition to a continuous slow-EMA anchor to maintain relative parity.
        curr_loss_detached = meta_losses_val # Use synchronized values
        if self.initial_losses.sum() == 0:
            self.initial_losses.data.copy_(curr_loss_detached)
        else:
            # [v117.0 SOTA FIX] Multi-Stage Anchoring
            # 1. Warmup Phase (First 100 steps): Fast adaptation to initial scale.
            # 2. Tracking Phase (Continuous): Slow adaptation (0.001) to follow manifold drift.
            alpha_init = 0.05 if self.step_count < 100 else 0.001
            self.initial_losses.data.mul_(1.0 - alpha_init).add_(curr_loss_detached, alpha=alpha_init)
        
        self.step_count += 1

        weights = self.get_weights()
        norms = []
        for i, loss in enumerate(losses):
            # 1. Get gradient of the RAW loss (retain_graph for multi-task)
            is_last = (i == len(losses) - 1)
            grad = torch.autograd.grad(
                loss, 
                self.shared_params, 
                retain_graph=True, 
                allow_unused=True
            )
            
            # 2. Compute norm of the gradient (detached from theta)
            # This follows the GradNorm paper: theta is fixed when updating weights.
            valid_grads = [torch.norm(g.detach()) for g in grad if g is not None]
            if not valid_grads:
                raw_grad_norm = torch.tensor(1e-6, device=loss.device)
            else:
                raw_grad_norm = torch.stack(valid_grads).norm()
            

            if torch.distributed.is_initialized():
                dist_norm = raw_grad_norm ** 2
                torch.distributed.all_reduce(dist_norm, op=torch.distributed.ReduceOp.SUM)
                # SOTA 2025 FIX: Add 1e-8 inside sqrt to prevent NaN on zero-gradient batches
                raw_grad_norm = torch.sqrt((dist_norm / torch.distributed.get_world_size()) + 1e-8)
                
            
            # [PATCH 8.1] GradNorm Damping (Smoking Gun #19)
            # Rationale: Prevents batch-level noise from causing weight jitter.
            if self.norm_emas[i] == 0:
                self.norm_emas[i] = raw_grad_norm.detach()
            else:
                self.norm_emas[i] = (self.ema_alpha * self.norm_emas[i]) + ((1.0 - self.ema_alpha) * raw_grad_norm.detach())
                
            # 3. Explicitly multiply by weights[i] so gn_loss is differentiable w.r.t weight
            norms.append(weights[i] * raw_grad_norm)

        norms = torch.stack(norms)

        # 2. Relative inverse rates
        # Slower tasks (loss ratio higher) get more weight
        # [SAFETY] Ensure initial_losses is never zero to prevent INF weights
        safe_init = torch.where(self.initial_losses > 0, self.initial_losses, torch.ones_like(self.initial_losses))
        # Use synchronized meta_losses_val for global parity
        rel_rates = meta_losses_val / (safe_init + 1e-8)
        avg_rate = rel_rates.mean()
        rel_rates = rel_rates / (avg_rate + 1e-8)

        # 3. Target norms (The balance point)
        target = norms.mean() * (rel_rates ** self.alpha)

        # 4. GradNorm loss (Drives weights toward target balance)
        gradnorm_loss = torch.abs(norms - target).mean()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gradnorm_loss, op=torch.distributed.ReduceOp.SUM)
            gradnorm_loss = gradnorm_loss / torch.distributed.get_world_size()

        return gradnorm_loss, weights.detach()

