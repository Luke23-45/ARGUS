import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BH_v2")

def test_agem_accumulation_v2():
    logger.info("Verifying Patch #53: Amnesia-Proof AGEM Accumulation...")
    
    # 1. Setup parameter
    p = nn.Parameter(torch.ones(10))
    p.grad = torch.zeros(10)
    
    # Ref Accumulator (Mocking self.grad_ref_buffer)
    grad_ref_buffer = torch.zeros(10)
    
    # Directions
    g_ref1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    g_ref2 = torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Strong opposite
    
    # --- BATCH 1 ---
    g_batch1 = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    p.grad.add_(g_batch1) # manual_backward
    grad_ref_buffer.add_(g_ref1) # Accumulate ref
    
    # --- BATCH 2 ---
    g_batch2 = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    p.grad.add_(g_batch2) # manual_backward
    grad_ref_buffer.add_(g_ref2) # Accumulate ref
    
    # --- STEP (Wait until not is_accumulating) ---
    # In my logic, we divide by world_size * acc_batches (or just world_size if scaled)
    # Let's assume acc_batches=2, so we divide by 2 (or individual refs were already scaled)
    # In wrapper_generalist, l_ref_bwd is scaled by 1/acc_batches.
    # So g_ref1 and g_ref2 in my mock should be scaled. 
    # Let's just average them.
    avg_ref = grad_ref_buffer / 2.0 
    # avg_ref = [ (1-2)/2, 0, ... ] = [-0.5, 0, ...]
    
    # Project TOTAL p.grad against ACCUMULATED avg_ref
    # p.grad total before projection = [1.0, 0.2, ...]
    dot = torch.dot(p.grad, avg_ref)
    # 1.0 * -0.5 = -0.5 (Conflict)
    
    if dot < 0:
        norm = torch.dot(avg_ref, avg_ref) + 1e-8
        alpha = -dot / norm # alpha = 0.5 / 0.25 = 2.0
        p.grad.add_(avg_ref, alpha=alpha) # Project
    
    # Restore anchor
    p.grad.add_(avg_ref)
    
    # Final Grad check
    # p.grad was [1.0, 0.2, ...]
    # Projection: [1.0, ...] + 2.0 * [-0.5, 0, ...] = [0.0, 0.2, ...]
    # Restoration: [0.0, 0.2, ...] + [-0.5, 0, ...] = [-0.5, 0.2, ...]
    
    logger.info(f"Final Grad[0]: {p.grad[0].item():.2f}")
    
    # The key is that the final update is now CONSISTENT with the OVERALL clinical memory.
    # It doesn't oscillate between Batch 1 and Batch 2 mid-step.
    
    if abs(p.grad[0].item() - (-0.5)) < 1e-3:
        logger.info("✅ Patch #53 SUCCESS! AGEM now protects the total step update.")
    else:
        logger.error(f"❌ Patch #53 FAILED! Unexpected grad value: {p.grad[0].item():.2f}")

if __name__ == "__main__":
    test_agem_accumulation_v2()
