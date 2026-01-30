import torch
import torch.nn as nn
import logging
from icu.core.gradnorm import GradNormBalancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AP")

def test_initial_loss_drift():
    logger.info("Simulating Initial Loss Stagnation (Smoking Gun #20)...")
    
    shared_params = [nn.Parameter(torch.randn(10, 10))]
    balancer = GradNormBalancer(num_tasks=2, shared_params=shared_params)
    
    # Init Loss: Dependent on shared_params to allow gradients
    p_norm = (shared_params[0] ** 2).sum()
    init_losses = torch.stack([p_norm * 10.0, p_norm * 0.01])
    
    # 1. First update captures init_losses
    balancer.update(init_losses)
    logger.info(f"Captured Initial Losses: {balancer.initial_losses.tolist()}")
    
    # 2. Distribution shifts: Both tasks should now be equal scale (e.g. 1.0)
    current_losses = torch.stack([p_norm * 1.0, p_norm * 1.0])
    
    # rel_rates = current / initial
    # Task 0: 1.0 / 10.0 = 0.1 (Too easy)
    # Task 1: 1.0 / 0.01 = 100.0 (Too hard)
    
    # Prediction: GradNorm will massively overweight Task 1 even though both are equal in the current regime.
    _, weights = balancer.update(current_losses)
    logger.info(f"Regime Shift Weights: {weights.tolist()}")
    
    if weights[1] > weights[0] * 5:
        logger.error("❌ SMOKING GUN #20: GradNorm is stuck in an 'Initial Loss Trap'.")
        logger.error(f"Task 1 weight ({weights[1]:.4f}) is {weights[1]/weights[0]:.1f}x higher than Task 0 despite identical current scale.")
    else:
        logger.info("✅ GradNorm adapted correctly (Unexpected in buggy version).")

if __name__ == "__main__":
    test_initial_loss_drift()
