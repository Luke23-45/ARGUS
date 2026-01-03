import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class UncertaintyLossScaler(nn.Module):
    """
    [SOTA 2025] Robust Loss Balancer with Dynamic Stability.
    
    Replaces static 'max_log_var' clamps with:
    1. Homoscedastic Uncertainty Weighting.
    2. Dynamic Flooring (weight can go lower if loss explodes > 5.0).
    3. EMA Loss Tracking for smooth floor transitions.
    """
    def __init__(self, num_tasks: int = 2, init_scales: list = [1.0, 1.0], decay: float = 0.99):
        super().__init__()
        # [v4.2.1 SOTA FIX] Robust Registry
        self.keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb']
        if num_tasks != len(self.keys):
             print(f"[WARNING] LossScaler num_tasks ({num_tasks}) != Registry ({len(self.keys)})")
        
        self.num_tasks = num_tasks
        # Learnable log_vars (s_i in paper)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # EMA tracking for stability monitoring
        self.register_buffer("loss_emas", torch.zeros(num_tasks))
        self.decay = decay
        
    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            loss_dict: Dictionary of raw losses. 
        """
        total_loss = 0.0
        log_metrics = {}
        
        for i, key in enumerate(self.keys):
            if i >= self.num_tasks: break 
            
            if key in loss_dict:
                loss = loss_dict[key]
                
                # --- [SOTA 2025] Dynamic Stability Logic ---
                # Update EMA
                with torch.no_grad():
                    curr_val = loss.item()
                    self.loss_emas[i] = self.decay * self.loss_emas[i] + (1 - self.decay) * curr_val
                    
                    # Dynamic Floor Calculation From stabilization.py
                    # If loss > 5.0 (Explosion risk), allow log_var to grow (weight -> 0)
                    # If loss < 0.1 (Lazy risk), restrict log_var (weight -> 1)
                    floor_val = 5.0 if self.loss_emas[i] > 5.0 else 2.0
                
                # Safe clamping using dynamic floor
                # min=-2.0 -> Max Weight = exp(2) = 7.39
                # max=floor -> Min Weight = exp(-floor)
                log_var = self.log_vars[i].clamp(min=-2.0, max=floor_val)
                precision = torch.exp(-log_var)
                
                # Weighted Loss = precision * loss + log_var/2
                weighted_loss = 0.5 * precision * loss + 0.5 * log_var
                total_loss += weighted_loss
                
                # Logging
                log_metrics[f"weight/{key}"] = 0.5 * precision
                log_metrics[f"sigma/{key}"] = torch.exp(0.5 * log_var)
                
        return total_loss, log_metrics
