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
        self.register_buffer("initial_losses", torch.zeros(num_tasks)) 

        # [v2026 SOTA FIX] Eager Optimizer Initialization (Smoking Gun #Amnesia)
        # Rationale: Initializing in __init__ allows the wrapper to discover 
        # the optimizer and save its state_dict in the main checkpoint.
        self.optimizer = torch.optim.Adam([self.weights], lr=0.005)
        
        # [PHASE 8] GradNorm Damping & Anchoring
        self.register_buffer("norm_emas", torch.zeros(num_tasks))
        self.register_buffer("step_count", torch.tensor([0], dtype=torch.long))
        self.ema_alpha = 0.90 # Damping factor for norm smoothing

    def get_weights(self):
        # [SOTA] Softmax weighting ensures task preservation
        return torch.softmax(self.weights, dim=0) * len(self.weights)

    def get_gradnorm_state(self) -> dict:
        """[v2026] Export meta-optimizer state for persistence."""
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "initial_losses": self.initial_losses.clone(),
            "norm_emas": self.norm_emas.clone(),
            "step_count": self.step_count.clone()
        }

    def load_gradnorm_state(self, state: dict):
        """[v2026] Restore meta-optimizer state (Amnesia Guard)."""
        if not state: return
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.initial_losses.copy_(state["initial_losses"])
        self.norm_emas.copy_(state["norm_emas"])
        self.step_count.copy_(state["step_count"])

    def update(self, losses, scaler=None):
        """
        [SOTA 2026] Universal DDP Bridge with Meta-Optimizer Step.
        
        Args:
            losses: List or Tensor of task losses (requires_grad=True).
            scaler: Optional PyTorch GradScaler for AMP-safe meta-gradients.
        """
        if torch.distributed.is_initialized():
            losses_sync_val = losses.detach().clone()
            torch.distributed.all_reduce(losses_sync_val, op=torch.distributed.ReduceOp.SUM)
            losses_sync_val /= torch.distributed.get_world_size()
            meta_losses_val = losses_sync_val
        else:
            meta_losses_val = losses.detach()

        # 2. Dynamic Initial Loss Anchoring
        curr_loss_detached = meta_losses_val.flatten() 
        if self.initial_losses.sum() == 0:
            self.initial_losses.data.copy_(curr_loss_detached)
        else:
            alpha_init = 0.05 if self.step_count < 100 else 0.001
            self.initial_losses.data.mul_(1.0 - alpha_init).add_(curr_loss_detached, alpha=alpha_init)
        
        self.step_count += 1

        # 3. Compute Gradient Norms per Task
        weights = self.get_weights()
        norms = []
        for i, loss in enumerate(losses):
            # [v2026 SOTA FIX] AMP-Safe Task Gradients (Smoking Gun #Ghosting)
            # Rationale: In FP16, unscaled loss gradients often underflow to zero.
            # We use the provided scaler (if any) to ensure the meta-gradients survive.
            scaled_loss = scaler.scale(loss) if scaler is not None else loss
            
            grad = torch.autograd.grad(
                scaled_loss, 
                self.shared_params, 
                retain_graph=True, 
                allow_unused=True
            )
            
            if scaler is not None and grad is not None:
                # Unscale the gradients back to the original range for norm calculation
                inv_scale = 1.0 / (scaler.get_scale() + 1e-8)
                grad = [g * inv_scale if g is not None else None for g in grad]

            valid_grads = [torch.norm(g.detach()) for g in grad if g is not None]
            if not valid_grads:
                raw_grad_norm = torch.tensor(1e-6, device=loss.device)
            else:
                raw_grad_norm = torch.stack(valid_grads).norm()

            if torch.distributed.is_initialized():
                dist_norm = raw_grad_norm ** 2
                torch.distributed.all_reduce(dist_norm, op=torch.distributed.ReduceOp.SUM)
                raw_grad_norm = torch.sqrt((dist_norm / torch.distributed.get_world_size()) + 1e-8)
                
            # Damping for stability
            if self.norm_emas[i] == 0:
                self.norm_emas[i] = raw_grad_norm.detach()
            else:
                self.norm_emas[i] = (self.ema_alpha * self.norm_emas[i]) + ((1.0 - self.ema_alpha) * raw_grad_norm.detach())
            
            # [v2026 SOTA] Bias Correction (Smoking Gun #ColdStart)
            # Rationale: EMA is weighted by (1 - alpha^t) to prevent zero-bias in early steps.
            bc_factor = 1.0 - (self.ema_alpha ** self.step_count.float())
            norm_bc = self.norm_emas[i] / (bc_factor + 1e-8)
                
            norms.append(weights[i] * norm_bc)

        norms = torch.stack(norms)

        # 4. Relative Inverse Rates (Loss Balancing)
        # [v2026 SOTA] Hardened epsilon for FP16 stability
        safe_init = torch.where(self.initial_losses > 1e-6, self.initial_losses, torch.ones_like(self.initial_losses) * 1e-6)
        rel_rates = meta_losses_val / (safe_init + 1e-6)
        avg_rate = rel_rates.mean()
        rel_rates = rel_rates / (avg_rate + 1e-6)

        # 5. Target Norms & Meta-Loss
        target = norms.mean() * (rel_rates ** self.alpha)
        gradnorm_loss = torch.abs(norms - target).mean()
        
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gradnorm_loss, op=torch.distributed.ReduceOp.SUM)
            gradnorm_loss = gradnorm_loss / torch.distributed.get_world_size()

        # [v2026 SOTA FIX] The Meta-Optimizer "Heartbeat" (Smoking Gun #Frozen)
        # Rationale: If we don't call backward/step, weights NEVER change.
        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        
        # [ROBUSTNESS] Hard clip task-weight gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_([self.weights], 1.0)
        
        self.optimizer.step()

        return gradnorm_loss.detach(), weights.detach()

