import torch
from icu.utils.advantage_calculator import ICUAdvantageCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AX")

def test_awr_ess_collapse():
    logger.info("Simulating AWR ESS Collapse (Smoking Gun #33)...")
    
    calc = ICUAdvantageCalculator(beta=0.1) # Strict beta
    
    batch_size = 256
    # 1. Simulate a batch with one massive sepsis outlier
    # Most samples have low advantage (stable)
    advantages = torch.randn(batch_size, 24) * 0.1
    # One sample is a "Miracle Survival" - massive advantage relative to others
    advantages[0, :] = 100.0 
    
    # 2. Calculate weights
    weights, diag = calc.calculate_awr_weights(advantages)
    
    logger.info(f"ESS (Global): {diag['ess']:.6f}")
    logger.info(f"Max Weight: {weights.max().item():.2f}")
    logger.info(f"Min Weight: {weights.min().item():.2f}")
    
    # Analyze weight distribution
    weight_sum = weights.sum().item()
    max_contribution = weights[0].sum().item() / weight_sum
    logger.info(f"Max Weight Contribution to Batch: {max_contribution*100:.2f}%")
    
    # If 1 sample contributes > 90% of the learning signal, we have ESS collapse
    if diag['ess'] < (1.0 / batch_size) * 10: # ESS is normalized to [0,1]?
        # ICUAdvantageCalculator line 788: ess = (g_sum_w ** 2) / (g_sum_w_sq * g_numel + 1e-8)
        # For 1 non-zero weight w, sum_w = w, sum_w_sq = w^2.
        # ess = w^2 / (w^2 * 256) = 1/256 = 0.0039
        logger.error(f"❌ ESS Collapse Detected! Advantage outliers are starving the batch.")
    else:
        logger.info("✅ ESS remains healthy.")

if __name__ == "__main__":
    test_awr_ess_collapse()
