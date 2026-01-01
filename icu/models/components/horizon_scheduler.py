import math
import torch
import torch.nn as nn

class ClinicalHorizonScheduler(nn.Module):
    """
    [Phase 3] Longitudinal Convergence Engine.
    
    Dynamically adjusts the discount factor (gamma) to create a 'Long-Horizon' curriculum.
    
    Logic:
    - Early training (Short Horizon): Gamma is small (e.g., 0.8), focusing the 
      advantage engine on immediate physiological responses (e.g., MAP response to fluids).
    - Late training (Long Horizon): Gamma ramps to 0.99, enabling the agent to 
      understand long-term survival credit assignment.
    """
    def __init__(self, 
                 start_gamma: float = 0.80, 
                 end_gamma: float = 0.99, 
                 warmup_epochs: int = 10,
                 ramp_epochs: int = 40):
        super().__init__()
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        
    def get_gamma(self, current_epoch: int) -> float:
        """
        Calculates the current discount factor based on training progress.
        """
        if current_epoch < self.warmup_epochs:
            return self.start_gamma
        
        if current_epoch >= (self.warmup_epochs + self.ramp_epochs):
            return self.end_gamma
            
        # Linear Ramp
        progress = (current_epoch - self.warmup_epochs) / self.ramp_epochs
        gamma = self.start_gamma + (self.end_gamma - self.start_gamma) * progress
        
        return gamma

    def get_foresight_hours(self, gamma: float, timestep_mins: int = 60) -> float:
        """
        Utility to calculate the 'Effective Foresight' in hours.
        Effective Horizon H = 1 / (1 - gamma)
        """
        horizon_steps = 1.0 / (1.0 - gamma + 1e-6)
        horizon_hours = (horizon_steps * timestep_mins) / 60.0
        return horizon_hours
