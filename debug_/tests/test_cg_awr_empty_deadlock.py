import torch
import torch.nn as nn
import logging
from unittest.mock import patch
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CG")

def test_awr_empty_deadlock_fix():
    logger.info("Verifying Fix for Smoking Gun #163: AWR Empty Batch Deadlock...")
    
    # Initialize two identical calculators
    calc_r0 = ICUAdvantageCalculator(beta=1.0)
    calc_r1 = ICUAdvantageCalculator(beta=1.0)
    
    # Mock distributed
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_reduce") as mock_all_reduce:
        
        # Rank 0 has data. Rank 1 has NONE.
        adv_r0 = torch.randn(8, 24)
        adv_r1 = torch.zeros(0, 24)
        
        logger.info("Running Rank 0 (with data)...")
        calc_r0.calculate_awr_weights(adv_r0)
        num_syncs_r0 = mock_all_reduce.call_count
        
        # Reset count for R1
        mock_all_reduce.reset_mock()
        
        logger.info("Running Rank 1 (EMPTY)...")
        calc_r1.calculate_awr_weights(adv_r1)
        num_syncs_r1 = mock_all_reduce.call_count
        
        logger.info(f"Sync calls for Rank 0: {num_syncs_r0}")
        logger.info(f"Sync calls for Rank 1: {num_syncs_r1}")
        
        # Standard AWR forward pass with adaptive stats should have 8 sync points
        if num_syncs_r0 == num_syncs_r1 and num_syncs_r0 == 8:
            logger.info(f"✅ Fix for Smoking Gun #163 VERIFIED! Identical sync points (8/8).")
        else:
            logger.error(f"❌ Fix FAILED! Sync call mismatch: R0={num_syncs_r0}, R1={num_syncs_r1}")

if __name__ == "__main__":
    test_awr_empty_deadlock_fix()
