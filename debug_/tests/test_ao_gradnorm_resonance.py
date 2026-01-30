import torch
import torch.nn as nn
import logging
from icu.core.gradnorm import GradNormBalancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AO")

def simulate_gradnorm_resonance():
    logger.info("Simulating GradNorm Harmonic Resonance (Smoking Gun #19)...")
    
    # Mock shared parameters
    shared_params = [nn.Parameter(torch.randn(10, 10))]
    
    # 2 tasks
    balancer = GradNormBalancer(num_tasks=2, shared_params=shared_params, alpha=1.5)
    
    # Simulation: 
    # Task 0 loss decreases linearly
    # Task 1 loss increases linearly
    # We want to see if weights track smoothly or start oscillating
    
    weights_history = []
    
    for step in range(100):
        # Losses that are functions of shared_params to allow gradients
        # We multiply by sin/cos to create oscillating gradient magnitudes
        loss_scale_0 = 1.0 + 0.5 * torch.sin(torch.tensor(step * 0.5))
        loss_scale_1 = 1.0 + 0.5 * torch.cos(torch.tensor(step * 0.5))
        
        l0 = (shared_params[0] ** 2).sum() * loss_scale_0
        l1 = (shared_params[0] ** 2).sum() * loss_scale_1
        
        # Fake gradients
        shared_params[0].grad = torch.randn_like(shared_params[0])
        
        losses = torch.stack([l0, l1])
        gn_loss, weights = balancer.update(losses)
        
        # Manual optimizer step for balancer
        balancer.optimizer.step()
        balancer.optimizer.zero_grad()
        
        weights_history.append(weights.clone())
        
        if step % 20 == 0:
            logger.info(f"Step {step} | Weights: {weights.tolist()}")

    # Check for oscillation magnitude
    w0_vals = [w[0].item() for w in weights_history]
    max_w0 = max(w0_vals)
    min_w0 = min(w0_vals)
    logger.info(f"Weight 0 Range: [{min_w0:.4f}, {max_w0:.4f}]")
    
    if max_w0 / (min_w0 + 1e-8) > 2.0:
        logger.warning("❌ SMOKING GUN #19: GradNorm weights exhibit high oscillation/resonance.")
    else:
        logger.info("✅ GradNorm weights stayed within reasonable bounds (No resonance detected in this mock).")

if __name__ == "__main__":
    simulate_gradnorm_resonance()
