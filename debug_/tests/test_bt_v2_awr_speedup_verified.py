import torch
import math
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_BT_v2")

def test_awr_beta_speedup_verification():
    logger.info("Verifying AWR PI-Controller Speedup (#116)...")
    
    # Initialize with new default SOTA parameters: momentum=0.90, gain=2.0
    calc = ICUAdvantageCalculator(
        adaptive_beta=True, 
        beta_momentum=0.90, 
        beta_gain=2.0
    )
    
    # Start Beta at 1.0
    calc.beta.fill_(1.0)
    target_ess = 0.20
    
    # Simulation: Extreme selection pressure (ESS drops to 0.05)
    # We want to see how quickly Beta increases to relax the pressure.
    low_ess = torch.tensor(0.05)
    
    beta_history = []
    logger.info(f"Step | Beta | Correction | Momentum")
    logger.info("-" * 40)
    
    for step in range(30):
        old_beta = calc.beta.item()
        # Simulate update
        calc._update_adaptive_stats(
            advantages=torch.zeros(10), 
            weights=torch.zeros(10), 
            ess=low_ess, 
            clipped_rate=0.0
        )
        new_beta = calc.beta.item()
        beta_history.append(new_beta)
        
        if step % 5 == 0 or step < 5:
            logger.info(f"{step:4d} | {new_beta:7.4f} | {new_beta/old_beta:10.4f} | {calc.beta_momentum:.2f}")

    # Analysis
    growth = beta_history[-1] / beta_history[0]
    logger.info(f"Total Beta Growth after 30 steps: {growth:.2f}x")
    
    # SUCCESS CRITERIA:
    # In the sluggish version (0.99 momentum), after 30 steps it only reached ~1.4x.
    # With 0.90 momentum and 2.0 gain, it should be much higher (> 3x).
    if growth > 3.0:
        logger.info("✅ Verification SUCCESS! AWR Beta is now highly responsive.")
    else:
        logger.error(f"❌ Verification FAILED! Beta growth too slow ({growth:.2f}x).")

if __name__ == "__main__":
    test_awr_beta_speedup_verification()
