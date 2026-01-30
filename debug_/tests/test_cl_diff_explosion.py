import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CL")

def test_diffusion_gradient_explosion():
    logger.info("Verifying Bug #175: Diffusion Hallucination Gradient Explosion...")
    
    # 1. Setup simulated tensors
    B, T, D = 4, 24, 28
    pred_noise = torch.randn(B, T, D, requires_grad=True)
    noisy_fut = torch.randn(B, T, D)
    
    # 2. Simulate large timestep t (where alpha_bar is very small)
    # In DDPM, alpha_bar decays from 1.0 to ~0.0001 or smaller.
    # Let's say we are at a very noisy step.
    alpha_t = torch.tensor([1e-12], device=pred_noise.device).view(1, 1, 1) # Extremely noisy
    
    # 3. x0 reconstruction formula (Analog Bits / DDPM)
    # x0_approx = (noisy_fut - sqrt(1 - alpha_t) * pred_noise) / sqrt(alpha_t)
    sqrt_alpha = torch.sqrt(alpha_t).clamp(min=1e-5) # Standard clamp in wrapper_generalist.py
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)
    
    x0_approx = (noisy_fut - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
    
    # 4. Simulate a simple physics loss (e.g. ReLU on some boundary violation)
    # If the hallucinated x0 violates a boundary, we compute loss
    loss = torch.relu(x0_approx - 5.0).mean() # Violation of +5.0 threshold
    
    # 5. Compute Gradient
    loss.backward()
    
    grad_norm = pred_noise.grad.norm().item()
    logger.info(f"Gradient Norm of pred_noise: {grad_norm:.2f}")
    
    # Analysis
    # The multiplier is 1 / sqrt_alpha = 1 / 1e-5 = 100,000
    # Any small violation in x0_approx will result in a HUGE gradient for pred_noise.
    
    if grad_norm > 1000.0:
        logger.error(f"❌ SMOKING GUN #175 CONFIRMED: Physics loss on hallucianted x0 caused Gradient Explosion! (Norm={grad_norm:.2f})")
        logger.info("Rationale: The 1/sqrt(alpha) term in the x0 reconstruction multiplier acts as a 100,000x gradient amplifier for large t.")
    else:
        logger.info("✅ Gradient Norm within stable bounds.")

if __name__ == "__main__":
    test_diffusion_gradient_explosion()
