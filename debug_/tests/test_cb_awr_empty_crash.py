import torch
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CB")

def test_awr_empty_batch_crash():
    logger.info("Verifying Smoking Gun #139: AWR Empty Batch Crash...")
    
    calc = ICUAdvantageCalculator()
    
    # Simulate an empty advantage tensor (Rank receiving no valid samples)
    advantages = torch.zeros(0, 6) # Empty batch
    rewards = torch.zeros(0, 6)
    values = torch.zeros(0, 6)
    mask = torch.zeros(0, 6)
    
    try:
        logger.info("Attempting calculate_weights with empty input...")
        weights, diagnostics = calc.calculate_weights(
            advantages, 
            values=values, 
            rewards=rewards, 
            mask=mask
        )
        logger.info("✅ No crash (unexpected).")
    except RuntimeError as e:
        logger.error(f"❌ Smoking Gun #139 CONFIRMED! AWR Crashed: {e}")
    except Exception as e:
        logger.error(f"❌ Smoking Gun #139 CONFIRMED! AWR Crashed with unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_awr_empty_batch_crash()
