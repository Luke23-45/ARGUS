import torch
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CF")

def test_awr_sigma_collapse():
    logger.info("Verifying Hypothesis #159: AWR Sigma Collapse...")
    
    calc = ICUAdvantageCalculator(beta=0.1) # Use selective beta to amplify issue
    
    # 1. Simulate nearly uniform advantages (Sigma Collapse)
    # Advantage mean = 1.0, std = extremely small
    # We add a tiny perturbation to one element
    B, T = 1, 10
    advantages = torch.full((B, T), 1.0)
    advantages[0, 0] += 1e-7 # Tiny noise
    
    # Current code uses: mu, sigma = advantages.mean(), advantages.std() + 1e-8
    # mu = 1.00000001
    # var = (1e-7)^2 * (9/10) / 9? No.
    # torch.std is ~3.16e-8
    
    # 2. Run calculation
    weights, diagnostics = calc.calculate_weights(advantages)
    
    w_max = weights.max().item()
    w_std = weights.std().item()
    
    logger.info(f"Advantages Stats: Mean={advantages.mean():.9f}, Std={advantages.std():.9e}")
    logger.info(f"Weights Stats: Max={w_max:.4f}, Std={w_std:.4f}")
    
    # If sigma is effectively 1e-8, and advantages-mu is 1e-7.
    # scaled_adv = (1e-7 / 1e-8) / 0.1 = 100
    # exp(100) is massive, should be saved by log_weights_global clamp (5.0)
    # exp(5.0) = 148.4
    
    if w_max > 20.0: # 20.0 is a reasonable upper limit for a balanced batch
        logger.error(f"❌ Hypothesis #159 CONFIRMED! Weights exploded to {w_max:.4f} due to Sigma Collapse.")
    else:
        logger.info("✅ AWR Weights remained stable despite Sigma Collapse.")

if __name__ == "__main__":
    test_awr_sigma_collapse()
