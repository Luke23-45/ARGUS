import torch
import logging
import math
from unittest.mock import patch
from icu.utils.stabilization import TrendSentinel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CJ")

def test_sentinel_synchrony_fix():
    logger.info("Verifying Fix for Smoking Gun #167: TrendSentinel Rank Synchrony...")
    
    ema_r0 = torch.tensor(1.0)
    std_r0 = torch.tensor(0.5)
    step_r0 = torch.tensor(10, dtype=torch.long)
    
    ema_r1 = torch.tensor(1.0)
    std_r1 = torch.tensor(0.5)
    step_r1 = torch.tensor(10, dtype=torch.long)
    
    decay = 0.99
    
    # Mock distributed
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_reduce") as mock_all_reduce:
        
        # Simulated asymmetric values
        val_r0 = 10.0
        val_r1 = 1.0
        
        # Simulation: Synchronous average should be (10+1)/2 = 5.5
        def mock_all_reduce_side_effect(tensor, op=None):
            tensor.fill_(11.0) # Sum of 10.0 and 1.0
            return None
        
        mock_all_reduce.side_effect = mock_all_reduce_side_effect
        
        TrendSentinel.update_stats(val_r0, ema_r0, std_r0, decay, step_r0)
        TrendSentinel.update_stats(val_r1, ema_r1, std_r1, decay, step_r1)
        
        # Check for Parity
        diff_ema = abs(ema_r0.item() - ema_r1.item())
        diff_std = abs(std_r0.item() - std_r1.item())
        
        logger.info(f"EMA Difference: {diff_ema:.6e}")
        logger.info(f"STD Difference: {diff_std:.6e}")
        
        if diff_ema < 1e-6 and diff_std < 1e-6:
            logger.info("✅ Fix for Smoking Gun #167 VERIFIED! Sentinel EMAs matched across ranks.")
        else:
            logger.error(f"❌ Fix FAILED! Sentinel divergence: EMA_diff={diff_ema:.6e}, STD_diff={diff_std:.6e}")

if __name__ == "__main__":
    test_sentinel_synchrony_fix()
