import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger("ClinicalHorizonScheduler")

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
        
        logger.info(f"Initialized v30.0 SOTA HorizonScheduler: Gamma={start_gamma}->{end_gamma}")
        
    def get_gamma(self, current_epoch: int) -> float:
        """Calculates gamma based on epoch (Legacy)."""
        # [v29.5] Redirect to step-based logic using a default n_batches=200
        return self.get_gamma_step(current_epoch * 200)

    def get_gamma_step(self, total_steps: int) -> float:
        """
        [v29.5 SOTA FIX] Step-Invariant Horizon Ramp (Abyssal #5).
        Uses ScalingSteward reference steps to ensure identical ramps across densities.
        """
        # Ref: 200 steps = 1 epoch
        warmup_steps = self.warmup_epochs * 200
        ramp_steps = self.ramp_epochs * 200
        
        if total_steps < warmup_steps:
            return self.start_gamma
        
        if total_steps >= (warmup_steps + ramp_steps):
            return self.end_gamma
            
        progress = (total_steps - warmup_steps) / ramp_steps
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
