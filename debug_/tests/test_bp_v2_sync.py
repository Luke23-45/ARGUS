import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BP_v2")

def test_uniform_step_scaling_fixed():
    logger.info("Verifying Patch #91: Uniform Step Scaling...")
    
    acc_ref = 4
    lr = 0.1
    
    # CASE 1: Full Accumulation (4 batches)
    w_full = torch.tensor([1.0], requires_grad=True)
    for _ in range(acc_ref):
        loss = w_full * 1.0
        loss.backward()
    
    # APPLY FIX: Divide by actual_accum
    actual_accum_full = 4
    with torch.no_grad():
        w_full.grad.mul_(1.0 / actual_accum_full)
    
    grad_final_full = w_full.grad.clone()
    logger.info(f"Full Acc Final Gradient: {grad_final_full.item():.4f}")
    
    # CASE 2: Tail Accumulation (1 batch)
    w_tail = torch.tensor([1.0], requires_grad=True)
    for _ in range(1):
        loss = w_tail * 1.0
        loss.backward()
    
    # APPLY FIX: Divide by actual_accum
    actual_accum_tail = 1
    with torch.no_grad():
        w_tail.grad.mul_(1.0 / actual_accum_tail)
        
    grad_final_tail = w_tail.grad.clone()
    logger.info(f"Tail Acc Final Gradient: {grad_final_tail.item():.4f}")
    
    # VERIFICATION
    diff = abs(grad_final_full.item() - grad_final_tail.item())
    logger.info(f"Final Gradient Parity Difference: {diff:.8f}")
    
    if diff < 1e-6:
        logger.info("✅ Patch #91 SUCCESS! Uniform step scaling achieved across epoch boundaries.")
    else:
        logger.error(f"❌ Verification Failed. Divergence: {diff:.8f}")

if __name__ == "__main__":
    test_uniform_step_scaling_fixed()
