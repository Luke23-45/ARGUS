import torch
from icu.utils.advantage_calculator import ICUAdvantageCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BA")

def test_stale_whitening():
    logger.info("Simulating Stale Advantage Whitening (Smoking Gun #39)...")
    
    # 1. Calibrate at Epoch 0 (Large residuals)
    calc = ICUAdvantageCalculator(beta=0.5)
    init_advantages = torch.randn(1000) * 10.0 # High volatility
    calc.set_stats(mean=0.0, std=10.0) # sigma=10.0
    
    # 2. Simulate Epoch 100 (Model improved, residuals shrunk)
    # Advantages represent prediction error. As model gets better, error drops.
    shrunk_advantages = torch.randn(256) * 1.0 # Volatility dropped 10x
    # One sample is still "twice as good" as others (relative to current dist)
    shrunk_advantages[0] = 5.0 
    
    # 3. Calculate weights using ADAPTIVE sigma
    # Run a loop to let EMA catch up
    for _ in range(2000):
        weights, diag = calc.calculate_awr_weights(shrunk_advantages)
    
    logger.info(f"Updated Sigma: {diag['adv_std']:.2f}")
    logger.info(f"Weights Max: {weights.max().item():.4f}")
    logger.info(f"Weights Min: {weights.min().item():.4f}")
    
    # Selection Ratio: How much more do we learn from the 'best' sample?
    ratio = weights.max() / weights.mean()
    logger.info(f"Selection Ratio (Peak/Mean): {ratio:.4f}")
    
    # If ratio is near 1.0, the model is treating the 'best' sample 
    # as no better than a 'random' sample due to stale whitening.
    if ratio < 1.5:
        logger.error(f"❌ Selection Pressure Decay! Fixed whitening is blinding the model to relative improvements.")
    else:
        logger.info("✅ Selection pressure remains active.")

if __name__ == "__main__":
    test_stale_whitening()
