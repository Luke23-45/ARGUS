"""
Test BO: AWR Rank Divergence Verification (Call Path Check)
-----------------------------------------------------------
Verifies that ICUAdvantageCalculator invokes distributed synchronization
for statistics, confirming the fix for Smoking Gun #96/#97.
"""
import torch
import unittest
from unittest.mock import patch, MagicMock
from icu.utils.advantage_calculator import ICUAdvantageCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BO_CallCheck")

class TestAWRDivergence(unittest.TestCase):
    @patch('icu.utils.advantage_calculator.dist')
    def test_sync_call_path(self, mock_dist):
        logger.info("Verifying AWR Synchronization Call Path...")
        
        # Setup Calculator
        calc = ICUAdvantageCalculator(beta=0.5)
        adv = torch.tensor([1.0, 2.0, 3.0])
        
        # Configure Mock to behave like a real distributed environment
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 1
        mock_dist.ReduceOp.SUM = "SUM"
        mock_dist.ReduceOp.MAX = "MAX"
        
        # We don't want all_reduce to return a Mock that infects tensors
        # It should return None (in-place modification)
        mock_dist.all_reduce.return_value = None
        
        # Execute
        calc.calculate_awr_weights(adv)
        
        # Verify
        # We expect all_reduce to be called to sync Sum/SqSum/Count in calculate_awr_weights
        # AND adaptive params in _update_adaptive_stats
        if mock_dist.all_reduce.called:
            logger.info("✅ SUCCESS: dist.all_reduce was called correctly.")
            logger.info(f"Total all_reduce calls: {mock_dist.all_reduce.call_count}")
        else:
            logger.error("❌ FAILURE: dist.all_reduce was NOT called.")
            self.fail("Synchronization missing")

if __name__ == "__main__":
    unittest.main()
