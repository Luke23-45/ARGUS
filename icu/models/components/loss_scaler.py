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
        indices = torch.tensor([idx for idx, _ in active_keys], device=losses_tensor.device)
        
        # 1. Soft Optimal Uncertainty Weighting (UW-SO)
        # Update EMA
        with torch.no_grad():
            curr_emas = self.loss_emas[indices]
            updated_emas = self.decay * curr_emas + (1 - self.decay) * losses_tensor.detach()
            self.loss_emas[indices] = updated_emas
            
            # [PATCH 3] Fixed Clinical Priority Weights
            # Original: Softmax priority creates zero-sum game where one task spike starves others
            # Evidence: A dropped from 0.043 (E0) to 0.0007 (E8)
            # Fix: Fixed weights based on clinical importance
            # Primary tasks (aux, acl): Higher weight for sepsis detection
            # Secondary tasks (diff, critic): Lower weight for generative quality
            clinical_weights = torch.tensor(
                [0.5, 0.5, 1.5, 1.5, 1.0, 1.0],  # [diff, critic, aux, acl, bgsl, tcb]
                device=losses_tensor.device
            )
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
        """
        self.log_vars.clamp_(min=-2.0, max=5.0)
