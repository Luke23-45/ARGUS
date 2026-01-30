import torch
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_CB_v2")

def test_awr_empty_batch_robustness():
    logger.info("Verifying AWR Empty Batch Guard (#139)...")
    
    calc = ICUAdvantageCalculator()
    
    # Simulate an empty advantage tensor
    advantages = torch.zeros(0, 6) 
    values = torch.zeros(0, 6)
    rewards = torch.zeros(0, 6)
    mask = torch.zeros(0, 6)
    
    try:
        logger.info("Attempting calculate_weights with empty input...")
        weights, diagnostics = calc.calculate_weights(
            advantages, 
            values=values, 
            rewards=rewards, 
            mask=mask
        )
        
        # Verify diagnostics are populated
        if "weights_max" in diagnostics:
            logger.info(f"✅ Logic passed. Weights Max: {diagnostics['weights_max']}")
            logger.info("✅ Verification SUCCESS! AWR is robust to empty batches.")
        else:
            logger.error("❌ Diagnostics missing 'weights_max'.")
            
    except Exception as e:
        logger.error(f"❌ Verification FAILED! AWR still crashed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_awr_empty_batch_robustness()
