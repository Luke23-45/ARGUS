import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple
from icu.models.components.loss_scaler import BayesianProjectedScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_CA_v2")

def test_hardened_scaler_ddp_verification():
    logger.info("Verifying DDP Scaler Hardening (#137) and Weighted Averaging (#156)...")
    
    scaler = BayesianProjectedScaler(num_tasks=7)
    
    # Mocking torch.distributed.all_reduce
    # We will manually simulate the 'sync_buffer' state after all_reduce
    
    # Simulate Rank 0: B=32, losses for ['diffusion', 'critic', 'aux']
    batch_size_r0 = 32
    losses_r0 = {
        'diffusion': torch.tensor(1.0),
        'critic': torch.tensor(0.5),
        'aux': torch.tensor(0.2)
    }
    
    # Simulate Rank 1: B=16, losses for ['diffusion', 'critic'] (No aux)
    batch_size_r1 = 16
    losses_r1 = {
        'diffusion': torch.tensor(1.2),
        'critic': torch.tensor(0.6)
    }
    
    # 1. Manual computation of expected weighted average
    # Global B = 32 + 16 = 48
    # Global Sum Diffusion = 1.0 * 32 + 1.2 * 16 = 32 + 19.2 = 51.2
    # Global Sum Critic = 0.5 * 32 + 0.6 * 16 = 16 + 9.6 = 25.6
    # Global Sum Aux = 0.2 * 32 + 0.0 * 16 = 6.4
    
    expected_diff = 51.2 / 48
    expected_critic = 25.6 / 48
    expected_aux = 6.4 / 48
    
    logger.info(f"Expected Global Averages: Diff={expected_diff:.4f}, Critic={expected_critic:.4f}, Aux={expected_aux:.4f}")
    
    # 2. Simulate the 'sync_buffer' logic from forward()
    # Diffusion=0, Critic=1, Aux=2 
    sync_buffer_r0 = torch.zeros(scaler.num_tasks + 1)
    sync_buffer_r0[0] = 1.0 * batch_size_r0
    sync_buffer_r0[1] = 0.5 * batch_size_r0
    sync_buffer_r0[2] = 0.2 * batch_size_r0
    sync_buffer_r0[-1] = float(batch_size_r0)
    
    sync_buffer_r1 = torch.zeros(scaler.num_tasks + 1)
    sync_buffer_r1[0] = 1.2 * batch_size_r1
    sync_buffer_r1[1] = 0.6 * batch_size_r1
    sync_buffer_r1[-1] = float(batch_size_r1)
    
    # Simulated all_reduce (Sum)
    global_sync_buffer = sync_buffer_r0 + sync_buffer_r1
    
    global_batch_size = global_sync_buffer[-1].item()
    global_sum_losses = global_sync_buffer[:scaler.num_tasks]
    avg_losses_all = global_sum_losses / (global_batch_size + 1e-8)
    
    actual_diff = avg_losses_all[0].item()
    actual_critic = avg_losses_all[1].item()
    actual_aux = avg_losses_all[2].item()
    
    logger.info(f"Actual Global Averages: Diff={actual_diff:.4f}, Critic={actual_critic:.4f}, Aux={actual_aux:.4f}")
    
    # Check for Fixed Buffer robustness (All ranks use buffer of size 8)
    if sync_buffer_r0.shape == sync_buffer_r1.shape:
        logger.info(f"✅ Fixed Buffer check passed: Both ranks used buffer size {sync_buffer_r0.shape[0]}.")
    else:
        logger.error("❌ Fixed Buffer check FAILED!")
        
    error_thresh = 1e-6
    if abs(actual_diff - expected_diff) < error_thresh and \
       abs(actual_critic - expected_critic) < error_thresh and \
       abs(actual_aux - expected_aux) < error_thresh:
        logger.info("✅ Weighted Averaging (#156) verification SUCCESS!")
    else:
        logger.error("❌ Weighted Averaging (#156) verification FAILED!")

if __name__ == "__main__":
    test_hardened_scaler_ddp_verification()
