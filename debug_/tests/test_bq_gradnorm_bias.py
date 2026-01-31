"""
Test BQ: GradNorm Accumulation Bias Verification (Bayesian Scaler)
------------------------------------------------------------------
Verifies that BayesianProjectedScaler correctly accumulates losses across
sub-batches before performing DDP synchronization and weight updates.

This prevents "Tail Bias" where the final sub-batch of an accumulation 
cycle would disproportionately influence task weights.
"""
import torch
import unittest
from unittest.mock import patch, MagicMock
from icu.models.components.loss_scaler import BayesianProjectedScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BQ_Verification")

class TestGradNormBiasVerification(unittest.TestCase):
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.all_reduce')
    def test_accumulation_parity(self, mock_all_reduce, mock_is_init):
        logger.info("Verifying Patch #107: GradNorm Accumulation Parity...")
        
        # 1. Setup Scaler
        scaler = BayesianProjectedScaler(num_tasks=2)
        acc_batches = 4
        
        # Force initial emas to be 1.0
        scaler.loss_emas.fill_(1.0)
        
        # SCENARIO (Same as original demo):
        # Batches 1-3: Task A is high, Task B is low.
        # Batch 4: Task B spikes (Outlier), Task A is low.
        
        losses_history = [
            {'diffusion': torch.tensor(10.0, requires_grad=True), 'critic': torch.tensor(1.0, requires_grad=True)}, # B1
            {'diffusion': torch.tensor(11.0, requires_grad=True), 'critic': torch.tensor(1.1, requires_grad=True)}, # B2
            {'diffusion': torch.tensor(10.5, requires_grad=True), 'critic': torch.tensor(0.9, requires_grad=True)}, # B3
            {'diffusion': torch.tensor(1.0, requires_grad=True), 'critic': torch.tensor(50.0, requires_grad=True)}, # B4 (Outlier spike)
        ]
        
        # Expected Average Losses:
        # Task A: (10+11+10.5+1)/4 = 32.5/4 = 8.125
        # Task B: (1+1.1+0.9+50)/4 = 53/4 = 13.25
        
        # 2. Simulate Accumulation Cycle
        for i in range(acc_batches):
            is_acc = (i < acc_batches - 1)
            scaler.forward(losses_history[i], batch_size=1, is_accumulating=is_acc)
            
            if is_acc:
                # all_reduce should NOT be called during accumulation
                self.assertEqual(mock_all_reduce.call_count, 0)
        
        # 3. Verify AllReduce state on the 'Step' batch (B4)
        mock_all_reduce.assert_called_once()
        
        # Extract the buffer passed to all_reduce
        sync_buffer = mock_all_reduce.call_args[0][0]
        
        # sync_buffer structure in BayesianProjectedScaler:
        # [loss_accumulator (num_tasks), task_counters (num_tasks), batch_counter (1)]
        
        accumulated_losses = sync_buffer[:2]
        accumulated_counts = sync_buffer[2:4]
        
        logger.info(f"Accumulated Sum Losses: {accumulated_losses.tolist()}")
        logger.info(f"Accumulated Task Counts: {accumulated_counts.tolist()}")
        
        # Check parity with expected sums
        self.assertAlmostEqual(accumulated_losses[0].item(), 32.5, places=2)
        self.assertAlmostEqual(accumulated_losses[1].item(), 53.0, places=2)
        self.assertEqual(accumulated_counts[0].item(), 4.0)
        self.assertEqual(accumulated_counts[1].item(), 4.0)
        
        logger.info("✅ SUCCESS: BayesianProjectedScaler correctly accumulated all losses.")
        logger.info("This confirms the weighted average will be mathematically perfect.")

if __name__ == "__main__":
    unittest.main()
