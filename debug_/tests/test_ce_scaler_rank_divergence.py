import torch
import torch.nn as nn
import logging
from unittest.mock import patch
from icu.models.components.loss_scaler import BayesianProjectedScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CE")

def test_bayesian_rank_divergence_fix():
    logger.info("Verifying Fix for Smoking Gun #160: Bayesian Rank Divergence...")
    
    # 1. Initialize two identical scalers
    scaler_r0 = BayesianProjectedScaler(num_tasks=7, decay=0.9)
    scaler_r1 = BayesianProjectedScaler(num_tasks=7, decay=0.9)
    
    # Mock distributed to simulate 2 ranks
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_reduce") as mock_all_reduce:
        
        # We need to simulate the sync_buffer behavior from forward()
        # Rank 0 has tasks [0, 2]. Rank 1 has task [0].
        # Global Avg: Diff=1.0, Aux=0.5 (if Rank 0 had 1.0 and Rank 1 had none)
        
        # Simulated global buffer: [Diff_Avg, Critic_Avg, Aux_Avg, ..., BatchSizeTotal]
        # For simplicity, let's assume global avg losses are [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        # and total batch size is 2.
        
        def mock_all_reduce_side_effect(buffer, op=None):
            # Fill buffer with global averages multiplied by total batch size (2)
            # indices: 0, 1, 2, 3, 4, 5, 6, 7(batch_size)
            buffer[0] = 2.0 # 1.0 * 2
            buffer[1] = 0.0
            buffer[2] = 1.0 # 0.5 * 2
            buffer[3] = 0.0
            buffer[4] = 0.0
            buffer[5] = 0.0
            buffer[6] = 0.0
            buffer[7] = 2.0
            return None

        mock_all_reduce.side_effect = mock_all_reduce_side_effect
        
        # Rank 0 sees tasks [0, 2]
        loss_dict_r0 = {'diffusion': torch.tensor(1.0), 'aux': torch.tensor(1.0)}
        scaler_r0.train()
        scaler_r0(loss_dict_r0, batch_size=1)
        
        # Rank 1 sees task [0] only
        loss_dict_r1 = {'diffusion': torch.tensor(1.0)}
        scaler_r1.train()
        scaler_r1(loss_dict_r1, batch_size=1)
        
        # 3. Check for Divergence
        # In the FIXED version, both ranks should update ALL EMAs using the synced avg_losses_all
        # Even if Rank 1 didn't see Task 2, its EMA[2] should have updated to (0.9*1.0 + 0.1*0.5) = 0.95
        
        diff_ema0 = torch.abs(scaler_r0.loss_emas[0] - scaler_r1.loss_emas[0]).item()
        diff_ema2 = torch.abs(scaler_r0.loss_emas[2] - scaler_r1.loss_emas[2]).item()
        
        logger.info(f"Diff EMA[0] (Shared Task): {diff_ema0:.6e}")
        logger.info(f"Diff EMA[2] (Asymmetric Task): {diff_ema2:.6e}")
        
        if diff_ema2 < 1e-6:
            logger.info("✅ Fix for Smoking Gun #160 VERIFIED! Bayesian EMAs matched across ranks.")
        else:
            logger.error(f"❌ Fix FAILED! Divergence detected: {diff_ema2:.6e}")

if __name__ == "__main__":
    test_bayesian_rank_divergence_fix()
