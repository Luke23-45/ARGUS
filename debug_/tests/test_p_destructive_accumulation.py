import torch
import torch.nn as nn
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_P")

def sanitize_gradients_mock(params, clip_target=1.0):
    """In-place clipping logic from stabilization.py:326"""
    with torch.no_grad():
        grads = [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        if not grads: return 0.0
        grad_stack = torch.stack(grads)
        total_norm = torch.norm(grad_stack)
        
        if total_norm > clip_target:
            scale_factor = clip_target / (total_norm + 1e-6)
            for p in params:
                if p.grad is not None:
                    p.grad.detach().mul_(scale_factor)
        return total_norm.item()

def simulate_accumulation(accum_steps=16, bug_enabled=True):
    # Dummy parameter [100]
    p = nn.Parameter(torch.zeros(100))
    # We want to accumulate 16 steps
    
    # Store "ideal" accumulated gradient (no clipping during accumulation)
    ideal_grad = torch.zeros(100)
    
    # Actual accumulation
    p.grad = torch.zeros(100)
    
    for i in range(accum_steps):
        # Generate random gradient for this batch (e.g. N(0, 0.2))
        batch_grad = torch.randn(100) * 0.2
        
        # Accumulate
        p.grad.add_(batch_grad)
        ideal_grad.add_(batch_grad)
        
        # [THE BUG] In-place clipping every batch during accumulation
        if bug_enabled:
            sanitize_gradients_mock([p], clip_target=1.0)
            
    # Final Hard Clip (applied at the end of accumulation in wrapper_generalist.py:1592)
    final_clip = 1.0 * (accum_steps ** 0.5) # Patch J: sqrt scaling
    
    # Apply to p.grad
    actual_norm = torch.norm(p.grad)
    if actual_norm > final_clip:
        p.grad.mul_(final_clip / actual_norm)
        
    # Apply to ideal_grad
    ideal_norm = torch.norm(ideal_grad)
    if ideal_norm > final_clip:
        ideal_grad.mul_(final_clip / ideal_norm)
        
    return p.grad, ideal_grad

def run_accumulation_analysis():
    logger.info("\n" + "="*60)
    logger.info("TEST P: DESTRUCTIVE ACCUMULATION CLIPPING ANALYSIS")
    logger.info("Checking if in-place clipping mangles accumulated gradients")
    logger.info("="*60)
    
    accum_steps = 16
    logger.info(f"Using {accum_steps} accumulation steps...")
    
    # 1. Buggy Case
    logger.info("\n[CASE 1] Buggy Implementation (Clip every step)")
    actual_b, ideal_b = simulate_accumulation(accum_steps=accum_steps, bug_enabled=True)
    cos_sim_b = torch.nn.functional.cosine_similarity(actual_b, ideal_b, dim=0).item()
    logger.info(f"  Cosine Similarity: {cos_sim_b:.4f}")
    
    # 2. Patched Case
    logger.info("\n[CASE 2] Patched Implementation (Compute norm only, clip at end)")
    actual_p, ideal_p = simulate_accumulation(accum_steps=accum_steps, bug_enabled=False)
    cos_sim_p = torch.nn.functional.cosine_similarity(actual_p, ideal_p, dim=0).item()
    norm_ratio_p = (torch.norm(actual_p) / torch.norm(ideal_p)).item()
    
    logger.info(f"  Cosine Similarity: {cos_sim_p:.4f}")
    logger.info(f"  Magnitude Ratio: {norm_ratio_p:.4f}")
    
    issues = []
    if cos_sim_p < 0.99:
        issues.append(f"PATCH FAILURE: Direction still biased ({cos_sim_p:.4f})")
    if norm_ratio_p < 0.95:
        issues.append(f"PATCH FAILURE: Energy still lost ({norm_ratio_p:.4f})")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    if not issues:
        logger.info("\u2705 TEST P PASSED: Patch restored 100% signal integrity")
    else:
        logger.warning("\u26a0\ufe0f TEST P FAILED: Patch is insufficient")
        for issue in issues:
            logger.warning(f"   - {issue}")
            
    return not bool(issues)

if __name__ == "__main__":
    run_accumulation_analysis()
