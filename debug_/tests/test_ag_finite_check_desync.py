import torch
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_AG")

def simulate_finite_desync():
    logger.info("Simulating Finite Check Desync (DDP 2 Ranks)...")
    
    # [BUGGY] Local Checks
    w0_buggy = torch.tensor([1.0], requires_grad=True)
    w1_buggy = torch.tensor([1.0], requires_grad=True)
    
    # [PATCHED] Global Heartbeat (Universal Bridge Pillar 2)
    w0_patched = torch.tensor([1.0], requires_grad=True)
    w1_patched = torch.tensor([1.0], requires_grad=True)
    
    grad_norm = torch.tensor([0.1])
    grad_inf = torch.tensor([float('inf')])
    
    # Step 1: Inf occurs on Rank 0
    logger.info("Simulating INF gradient on Rank 0...")
    
    # --- 1. BUGGY PASS ---
    should0_buggy = torch.isfinite(grad_inf)
    should1_buggy = torch.isfinite(grad_norm)
    
    if should0_buggy: w0_buggy.data -= grad_inf
    if should1_buggy: w1_buggy.data -= grad_norm
    
    # --- 2. PATCHED PASS (Global Consensus) ---
    is_finite0 = torch.isfinite(grad_inf)
    is_finite1 = torch.isfinite(grad_norm)
    
    # Heartbeat: should_apply = all_reduce(is_finite, Op.AND)
    global_should_apply = is_finite0 and is_finite1 # Logic on both ranks is identical
    
    if global_should_apply:
        w0_patched.data -= grad_inf
        w1_patched.data -= grad_norm
    else:
        logger.warning("Heartbeat Consensus: Inf detected on any rank. ALL ranks skip step. [SAFE]")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS: GLOBAL HEARTBEAT")
    logger.info("="*60)
    buggy_diff = torch.abs(w0_buggy - w1_buggy).item()
    patched_diff = torch.abs(w0_patched - w1_patched).item()
    
    logger.info(f"Buggy Weight Divergence:   {buggy_diff:.4f}")
    logger.info(f"Patched Weight Divergence: {patched_diff:.4f} [Consensus]")

    if patched_diff < 1e-6:
        logger.info("\u2705 TEST AG PASSED: Global Heartbeat prevents State Fracture.")
    else:
        logger.error("\u274c TEST AG: State fracture still occurred in Patched mode.")
            
    return patched_diff

if __name__ == "__main__":
    simulate_finite_desync()
