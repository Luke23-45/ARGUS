import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class UncertaintyLossScaler(nn.Module):
    """
    [SOTA 2025] Multi-Task Loss Balancer using Homoscedastic Uncertainty.
    
    Instead of manually tuning `aux_loss_scale` (e.g., finding that 5.0 works best),
    we learn the optimal balance dynamically.
    
    Theory (Kendall et al.):
        L_total = 1/(2*sigma_1^2) * L_diffusion + log(sigma_1) + 
                  1/(2*sigma_2^2) * L_aux       + log(sigma_2)
                  
    This allows the model to "admit" when one task is too noisy/hard and down-weight it 
    temporarily, preventing gradient fighting.
    """
    def __init__(self, num_tasks: int = 2, init_scales: list = [1.0, 1.0]):
        super().__init__()
        # We learn log_var (s) for numerical stability. 
        # sigma^2 = exp(s)
        # s is initialized to log(init_scale)
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # Initialize custom scales if provided
        # e.g. if we want to start with aux_scale=5.0, we affect the initial "s"
        # However, typically we start at 0.0 (sigma=1.0) and let it drift.
        
    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            loss_dict: Dictionary of raw losses. 
                       Must contain keys matching the implicit order: 
                       0: 'diffusion', 1: 'aux', etc.
        Returns:
            total_loss: Scalar tensor for backprop
            log_dict: Dict of weights/sigmas for logging
        """
        # Ensure consistent ordering
        # Task 0: Diffusion (Generative)
        # Task 1: Aux (Discriminative)
        
        # Default keys
        keys = ['diffusion', 'aux']
        
        total_loss = 0.0
        log_metrics = {}
        
        for i, key in enumerate(keys):
            if key in loss_dict:
                loss = loss_dict[key]
                
                # Get learned parameter s_i
                log_var = self.log_vars[i]
                
                # Weight = 1 / (2 * exp(s))
                # We use precision weighting
                precision = torch.exp(-log_var)
                
                # L_weighted = precision * loss + log_var/2
                # Note: The + log_var/2 term prevents the model from diverging to sigma -> infinity
                weighted_loss = 0.5 * precision * loss + 0.5 * log_var
                
                total_loss += weighted_loss
                
                # Logging metrics (Keep as tensors to avoid graph breaks)
                log_metrics[f"weight/{key}"] = 0.5 * precision
                log_metrics[f"sigma/{key}"] = torch.exp(0.5 * log_var)
                
        return total_loss, log_metrics
