import torch
import torch.nn as nn
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AN")

def test_p95_signal_erasure():
    logger.info("Simulating P95 Clinical Signal Erasure...")
    
    # Initialize calculator
    calc = ICUAdvantageCalculator(beta=0.5)
    
    # Batch size 100, 1 critical outlier (advantage 50.0), 99 baseline (0.0)
    adv = torch.zeros(100)
    adv[0] = 50.0
    
    logger.info(f"Raw Max Advantage: {adv.max().item()}")
    
    # Calculate weights
    # Internal logic will clamp to P95.
    # P95 of [50, 0, 0... (99 times)] is 0.
    weights, diag = calc.calculate_weights(adv)
    
    logger.info(f"Whitened Mean: {diag['adv_mean']}")
    logger.info(f"Whitened Std: {diag['adv_std']}")
    logger.info(f"AWR Weight[0] (Expert): {weights[0].item()}")
    logger.info(f"AWR Weight[1] (Baseline): {weights[1].item()}")
    
    if weights[0] == weights[1]:
        logger.error("\u274c SMOKING GUN #18: The Expert outlier was erased! Weights are uniform.")
        logger.error("Rationale: Quantile(0.95) for 1% prevalence is 0.0. The signal is suppressed.")
    else:
        logger.info("\u2705 Signal preserved.")

if __name__ == "__main__":
    test_p95_signal_erasure()
