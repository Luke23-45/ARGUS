import torch
import torch.nn as nn
import math

class StateAwareSampler(nn.Module):
    """
    [Phase 4] Dynamic Compute Allocation.
    
    Varies the number of diffusion denoising steps based on clinical risk.
    Higher Risk = More Steps = Better Precision for critical interventions.
    
    Logic:
    N_steps = floor(min_steps + risk_coef * (max_steps - min_steps))
    """
    def __init__(self, min_steps: int = 50, max_steps: int = 250):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        
    def calculate_steps(self, risk_coef: torch.Tensor) -> torch.Tensor:
        """
        Maps risk [B] to step counts [B].
        """
        steps = self.min_steps + risk_coef * (self.max_steps - self.min_steps)
        return steps.floor().long()

    def get_time_schedule(self, steps: int, device: str = "cpu"):
        """
        Generates a standard linear schedule sub-sampled to N steps.
        Essentially a DDIM-style sub-sampling of the 1000-step training noise.
        """
        # Training usually has 1000 steps
        training_steps = 1000
        schedule = torch.linspace(training_steps - 1, 0, steps, device=device).long()
        return schedule
