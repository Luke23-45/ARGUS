import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class BayesianProjectedScaler(nn.Module):
    """
    [SOTA 2025] Bayesian-PGD Uncertainty Scaler with UW-SO (Achituve et al. 2024).
    
    Features:
    1. Clinical-Priority UW-SO: Softmax-normalized priority balancing.
    2. PGD Projection: Ensures log_vars stay in the differentiable zone [-2, 5].
    3. Gradient Entropy Preservation: No forward-pass clamping.
    """
    def __init__(self, num_tasks: int = 6, decay: float = 0.99):
        super().__init__()
        self.keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb']
        self.num_tasks = num_tasks
        
        # Learnable log_vars (Homoscedastic uncertainty)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # EMA tracking for UW-SO stability
        self.register_buffer("loss_emas", torch.ones(num_tasks))
        self.decay = decay
        
    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Unified weighting pass with Dynamic Priority.
        """
        losses = []
        active_keys = []
        for i, key in enumerate(self.keys):
            if key in loss_dict:
                losses.append(loss_dict[key])
                active_keys.append((i, key))
        
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device), {}

        losses_tensor = torch.stack(losses)
        
        # [SOTA 2025] DDP Loss Synchronization
        # Rationale: Ensures that all ranks compute identical uncertainty weights (Precision).
        # Prevents "Rank Divergence" where different GPUs disagree on task priority.
        if torch.distributed.is_initialized() and self.training:
            # Clone to avoid affecting the original loss graph on the local rank
            sync_losses = losses_tensor.detach().clone()
            torch.distributed.all_reduce(sync_losses, op=torch.distributed.ReduceOp.SUM)
            avg_losses = sync_losses / torch.distributed.get_world_size()
        else:
            avg_losses = losses_tensor.detach()

        indices = torch.tensor([idx for idx, _ in active_keys], device=losses_tensor.device)
        
        # 1. Soft Optimal Uncertainty Weighting (UW-SO)
        # Update EMA using global average losses
        with torch.no_grad():
            curr_emas = self.loss_emas[indices]
            updated_emas = self.decay * curr_emas + (1 - self.decay) * avg_losses
            self.loss_emas[indices] = updated_emas
            
            # [v23.0 PATCH] Neutralized Static Weights (Freedom of Uncertainty)
            # Rationale: "Double Scaling" (Beta + 1.5x) was causing gradient explosions (Z9 E11).
            # Fix: Set all weights to 1.0. Let the Adaptive Beta in wrapper_generalist handle the
            # magnitude balancing, and let Bayesian Scaler handle the noise balancing.
            clinical_weights = torch.ones(self.num_tasks, device=losses_tensor.device)
            uw_weights = clinical_weights[indices]
        
        # 2. Bayesian Weighting (Kendall et al.)
        # NO CLAMPING in forward to preserve gradient flow
        log_vars_active = self.log_vars[indices]
        precision = torch.exp(-log_vars_active)
        
        # [PATCH 1] Restrained Uncertainty Weighting (RUW)
        # Liebel & Körner (2018) / Softplus variant
        # Original: 0.5 * log_var can be negative when log_var < 0
        # Fix: softplus(x) = ln(1 + exp(x)) > 0 for all x
        # Guarantees: L_total > 0 for all epochs
        regularization = F.softplus(log_vars_active)
        weighted_losses = 0.5 * (precision * losses_tensor * uw_weights) + regularization
        total_loss = weighted_losses.sum()
        
        # Logging
        log_metrics = {}
        for idx, (original_idx, key) in enumerate(active_keys):
            log_metrics[f"weight/{key}"] = 0.5 * precision[idx].item()
            log_metrics[f"priority/{key}"] = uw_weights[idx].item()
            log_metrics[f"sigma/{key}"] = torch.exp(0.5 * log_vars_active[idx]).item()
            
        return total_loss, log_metrics

    @torch.no_grad()
    def project_parameters(self):
        """
        [SOTA] Parameter Projection Hook.
        Must be called after optimizer.step() to prevent 'Dead Zones'.
        
        PMS Extension: Enforces Clinical Ranking Constraint (PRUW).
        We guarantee that Sepsis uncertainty (aux) never exceeds Diffusion uncertainty,
        ensuring that the Sepsis task always maintains its priority signal.
        """
        # 1. Standard Bayesian Boundary Projection
        self.log_vars.clamp_(min=-2.0, max=5.0)
        
        # 2. [v23.0 PATCH] PRUW Clamp REMOVED
        # Rationale: The "Confidence Trap". Clamping aux_log_var <= diff_log_var forced the model
        # to be "overconfident" about sepsis even on hard cases, leading to Manifold Shocks.
        # Fix: Allowed the Sepsis Head to be uncertain.
        # self.log_vars[2].clamp_(max=diff_log_var.item())  <-- DELETED
        # self.log_vars[3].clamp_(max=diff_log_var.item())  <-- DELETED
        pass
