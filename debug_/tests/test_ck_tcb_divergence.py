import torch
import torch.nn as nn
import logging
from unittest.mock import patch
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CK")

def test_tcb_rank_divergence():
    logger.info("Verifying Bug #171: TCB Rank Divergence...")
    
    d_model = 64
    capacity = 100
    tcb_r0 = TemporalContrastiveBuffer(d_model=d_model, capacity=capacity)
    tcb_r1 = TemporalContrastiveBuffer(d_model=d_model, capacity=capacity)
    
    # Initialize both with same zero queue
    tcb_r0.queue.zero_()
    tcb_r1.queue.zero_()
    tcb_r0.queue_filled.fill_(0)
    tcb_r1.queue_filled.fill_(0)
    
    # Simulate Rank 0 and Rank 1 having DIFFERENT data (Standard DDP)
    q_r0 = torch.randn(8, d_model)
    k_r0 = torch.randn(8, d_model)
    
    q_r1 = torch.randn(8, d_model)
    k_r1 = torch.randn(8, d_model)
    
    # Mock distributed for the shuffle
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.broadcast") as mock_broadcast:
        
        # Rank 0 runs
        tcb_r0(q_r0, k_r0)
        
        # Rank 1 runs
        tcb_r1(q_r1, k_r1)
        
        # Check if queues are the same
        # They should be different because they enqueued different k's
        diff = torch.abs(tcb_r0.queue - tcb_r1.queue).sum().item()
        logger.info(f"Queue Difference after 1 step: {diff:.6f}")
        
        # Check filled count
        logger.info(f"Rank 0 Filled: {tcb_r0.queue_filled.item()}")
        logger.info(f"Rank 1 Filled: {tcb_r1.queue_filled.item()}")
        
        if diff > 1e-4:
            logger.error("❌ SMOKING GUN #171 CONFIRMED: TCB Memory Banks diverged across ranks!")
            
            # Now show how this affects loss
            # Next step: Use same query on both ranks, but they have different queues
            q_shared = torch.randn(1, d_model)
            k_shared = torch.randn(1, d_model)
            
            out_r0 = tcb_r0(q_shared, k_shared)
            out_r1 = tcb_r1(q_shared, k_shared)
            
            loss_diff = abs(out_r0['loss'].item() - out_r1['loss'].item())
            logger.info(f"Loss Difference on identical query: {loss_diff:.6f}")
            
            if loss_diff > 1e-4:
                logger.error("❌ BUG #171 IMPACT: Inconsistent TCB losses across ranks will cause Gradient Divergence.")
        else:
            logger.info("✅ TCB Queues are identical (Bug #171 not present).")

if __name__ == "__main__":
    test_tcb_rank_divergence()
