import math
import torch
import torch.nn as nn
import logging
from icu.utils.train_utils import ScalingSteward

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
        # [SOTA FIX - DYNAMIC BUDGET] Use initialized dynamic parameter or tuning baseline
        spe = getattr(self, 'steps_per_epoch', ScalingSteward.SOTA_REF_STEPS)
        return self.get_gamma_step(current_epoch * spe)

    def scale_dynamics(self, steps_per_epoch: int):
        """[SOTA v2026] Unifies horizon ramps across step densities."""
        if steps_per_epoch <= 0: return
        self.steps_per_epoch = steps_per_epoch
        logger.info(f"[Horizon] Scaling Dynamics: {steps_per_epoch} steps/epoch")

    def get_gamma_step(self, total_steps: int) -> float:
        """
        [v29.5 SOTA FIX] Step-Invariant Horizon Ramp (Abyssal #5).
        Uses current steps_per_epoch to ensure identical ramps across densities.
        """
        # [SOTA FIX - DYNAMIC BUDGET] Dynamic Epoch Bound: 1 epoch = self.steps_per_epoch
        # Fallback to SOTA reference if scale_dynamics hasn't been called
        spe = getattr(self, 'steps_per_epoch', ScalingSteward.SOTA_REF_STEPS)
        
        warmup_steps = self.warmup_epochs * spe
        ramp_steps = self.ramp_epochs * spe
        
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
