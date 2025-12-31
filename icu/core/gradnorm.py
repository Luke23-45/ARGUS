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

        # [SOTA 2025] Lazy Initialization for Task Weight Optimizer
        # This ensures parameters are on the correct device before optimizer creation.
        self.optimizer = None 

    def get_weights(self):
        # [SOTA] Softmax weighting ensures task preservation (no task gets 0 weight)
        # We scale by num_tasks so the average weight is 1.0 (prevents gradient vanishing)
        return torch.softmax(self.weights, dim=0) * len(self.weights)

    def update(self, losses):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam([self.weights], lr=0.025)


        if self.initial_losses.sum() == 0:
            init_loss = losses.detach().clone()
            if torch.distributed.is_initialized():
                # Strictly use Rank 0 as the source of truth for L(0)
                torch.distributed.broadcast(init_loss, src=0)
            self.initial_losses.data.copy_(init_loss)

        weights = self.get_weights()
        norms = []
        for i, loss in enumerate(losses):
            # 1. Get gradient of the RAW loss (retain_graph for multi-task)
            is_last = (i == len(losses) - 1)
            grad = torch.autograd.grad(
                loss, 
                self.shared_params, 
                retain_graph=not is_last, 
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
                
            
            # 3. Explicitly multiply by weights[i] so gn_loss is differentiable w.r.t weight
            norms.append(weights[i] * raw_grad_norm)

        norms = torch.stack(norms)

        # 2. Relative inverse rates
        # Slower tasks (loss ratio higher) get more weight
        # [SAFETY] Ensure initial_losses is never zero to prevent INF weights
        safe_init = torch.where(self.initial_losses > 0, self.initial_losses, torch.ones_like(self.initial_losses))
        rel_rates = losses.detach() / (safe_init + 1e-8)
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

