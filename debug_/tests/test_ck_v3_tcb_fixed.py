import torch
import torch.nn as nn
import logging
from unittest.mock import patch, MagicMock
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CK_v2")

def test_tcb_parity_fix():
    logger.info("Verifying Fix for Bug #171: TCB Rank Parity...")
    
    d_model = 64
    capacity = 100
    tcb_r0 = TemporalContrastiveBuffer(d_model=d_model, capacity=capacity)
    tcb_r1 = TemporalContrastiveBuffer(d_model=d_model, capacity=capacity)
    
    # Initialize same
    tcb_r0.queue.zero_()
    tcb_r1.queue.zero_()
    tcb_r0.queue_filled.fill_(0)
    tcb_r1.queue_filled.fill_(0)
    
    # Simulate rank-local data
    k_r0 = torch.randn(4, d_model)
    k_r1 = torch.randn(6, d_model) # Different counts!
    
    # Mock distributed environment
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = True
    mock_dist.get_world_size.return_value = 2
    mock_dist.ReduceOp.MIN = "MIN"
    
    def mock_all_gather(tensor_list, tensor):
        # Find which rank's tensor this is by checking size
        if tensor.shape[0] == 4: # Rank 0's local count or padded keys
             rank = 0
        elif tensor.shape[0] == 6: # Rank 1's local count
             rank = 1
        elif tensor.shape[0] == 1: # Count tensor
             rank = 0 if tensor.item() == 4 else 1
        else: # Padded keys (both size 6)
             # This is tricky in a serial mock. 
             # Let's assume Rank 0 calls it first, then Rank 1.
             pass

    # We need a more sophisticated mock for the asymmetric gather
    # Rank 0 call:
    #   all_gather counts -> [4, 6]
    #   all_gather padded -> [padded_r0, padded_r1]
    
    # Instead of complex mocks, let's just manually run the logic for Rank 0 and Rank 1
    # and verify the bit-parity of the queue.
    
    import torch.distributed as dist
    
    def simulate_ddp_update(module, local_keys, rank, world_size, all_keys_list):
        # Manually perform the work that _dequeue_and_enqueue does
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=world_size), \
             patch("torch.distributed.all_gather", side_effect=lambda l, t: [l[i].copy_(all_keys_list[i]) for i in range(world_size)]), \
             patch("torch.distributed.broadcast") as mock_b:
             
             module._dequeue_and_enqueue(local_keys)
             
    # Prepare the "global" state for the mock
    k_r0_padded = torch.zeros(6, d_model); k_r0_padded[:4] = k_r0
    k_r1_padded = k_r1.clone()
    counts = [torch.tensor([4]), torch.tensor([6])]
    paddeds = [k_r0_padded, k_r1_padded]

    def mock_all_gather_impl(tensor_list, tensor):
        if tensor.shape == (1,):
            # Simulation of counts gather
            tensor_list[0].copy_(counts[0])
            tensor_list[1].copy_(counts[1])
        else:
            # Simulation of keys gather (padded)
            tensor_list[0].copy_(paddeds[0])
            tensor_list[1].copy_(paddeds[1])

    shared_broadcasts = []

    def simulate_ddp_update(module, local_keys, rank, world_size):
        broadcast_ptr = [0]
        
        def mock_broadcast_impl(tensor, src):
            if rank == src:
                if len(shared_broadcasts) <= broadcast_ptr[0]:
                    shared_broadcasts.append(tensor.clone())
                else:
                    shared_broadcasts[broadcast_ptr[0]] = tensor.clone()
            else:
                tensor.copy_(shared_broadcasts[broadcast_ptr[0]])
            broadcast_ptr[0] += 1

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=world_size), \
             patch("torch.distributed.all_gather", side_effect=mock_all_gather_impl), \
             patch("torch.distributed.all_reduce", side_effect=lambda t, op: None), \
             patch("torch.distributed.broadcast", side_effect=mock_broadcast_impl):
             
             # Call forward to trigger both bank update and uniformity sampling
             q = torch.randn(local_keys.shape[0], d_model)
             module(q, local_keys)

    # Simulate Rank 0
    simulate_ddp_update(tcb_r0, k_r0, rank=0, world_size=2)
    # Simulate Rank 1
    simulate_ddp_update(tcb_r1, k_r1, rank=1, world_size=2)

    # Verify Bit-Parity
    diff = torch.abs(tcb_r0.queue - tcb_r1.queue).sum().item()
    logger.info(f"Queue Parity Check: Diff = {diff:.6f}")
    
    if diff < 1e-6:
        logger.info("✅ Fix for Smoking Gun #171 VERIFIED! Queues are identical across ranks.")
        logger.info(f"Common Queue Filled Count: {tcb_r0.queue_filled.item()}")
    else:
        logger.error("❌ Fix for Smoking Gun #171 FAILED! Queues diverged.")

if __name__ == "__main__":
    test_tcb_parity_fix()
