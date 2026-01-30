import torch
import torch.nn as nn
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AL")

def test_awr_sigma_collapse():
    logger.info("Simulating AWR Variance Collapse (Zero Sigma)...")
    
    # Initialize calculator
    calc = ICUAdvantageCalculator(beta=0.5)
    
    # Create identical advantages (Zero Variance)
    # This happens if the critic is flat and rewards are uniform in a batch
    advantages = torch.ones(10) * 5.0
    
    logger.info(f"Advantages: {advantages}")
    
    try:
        # The Buggy Implementation:
        # sigma = advantages.std() -> 0.0
        # norm_adv = (advantages - mu) / sigma -> NaN
        weights, diag = calc.calculate_weights(advantages)
        
        logger.info(f"Weights: {weights}")
        logger.info(f"Diagnostics: {diag}")
        
        if torch.isnan(weights).any():
            logger.error("\u274c SMOKING GUN #15: AWR weights became NaN due to zero-variance advantages!")
        elif torch.isinf(weights).any():
            logger.error("\u274c SMOKING GUN #15: AWR weights became Inf due to zero-variance advantages!")
        else:
            logger.info("\u2705 AWR handled zero-variance (Unexpected in buggy version).")
            
    except Exception as e:
        logger.error(f"\u274c AWR crashed on zero-variance: {e}")

if __name__ == "__main__":
    test_awr_sigma_collapse()
