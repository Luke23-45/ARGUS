import torch
import torch.nn as nn
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AM")

def simulate_beta_feedback_loop():
    logger.info("Simulating AWR Beta Feedback Loop (Saturation Trap)...")
    
    # Initialize calculator
    calc = ICUAdvantageCalculator(beta=0.5, adaptive_beta=True)
    calc.beta_momentum = 0.5 # Faster for simulation
    
    # Simulation: 
    # High noise environment where a few samples have massive advantages.
    # We want to see if beta COLLAPSES to 0.01 (min_beta)
    
    for step in range(50):
        # Advantages: 1 huge outlier, others zero
        adv = torch.zeros(100)
        adv[0] = 50.0 # Outlier
        
        # Calculate weights
        weights, diag = calc.calculate_weights(adv)
        
        current_beta = calc.beta.item()
        ess = diag['ess']
        clip_rate = diag['fp16_clipped_ratio']
        
        logger.info(f"Step {step:02d} | Beta: {current_beta:.4f} | ESS: {ess:.4f} | Clip: {clip_rate:.4f}")
        
        # THE BUG TRIGGER:
        # If ess is low (it will be, dominated by 1 sample), controller LOWERS beta.
        # But lowering beta just makes that 1 sample's weight hit max_weight faster.
        # If clip_rate < 0.05, we don't hit the Recovery Mode (L831).
        if current_beta < 0.02:
            logger.error("\u274c SMOKING GUN #16: Beta collapsed to near-min in only 50 steps!")
            break
            
    if calc.beta.item() > 0.02:
        logger.info("\u2705 Beta stayed stable or recovered.")

if __name__ == "__main__":
    simulate_beta_feedback_loop()
