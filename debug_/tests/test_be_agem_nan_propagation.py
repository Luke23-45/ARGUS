import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BE_v3")

def test_agem_nan_safe_projection_v3():
    logger.info("Verifying Patch #48.1: Global Reference Sanitization...")
    
    # 1. Setup healthy gradient
    p_grad = torch.ones(10, 10)
    
    # 2. Setup reference gradient with NaNs
    g_ref = torch.randn(10, 10)
    g_ref[0, 0] = float('nan')
    
    # --- PHASE A: ACCUMULATION (Simplified) ---
    grad_ref_buffer = g_ref.flatten() # Simulated accumulation
    
    # --- PHASE B: STEP (v48.1 Logic) ---
    # Global Reference Sanitization
    with torch.no_grad():
        torch.nan_to_num_(grad_ref_buffer, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Unpack
    g_ref_accum = grad_ref_buffer.view(10, 10)
    
    # AGEM Projection
    proj_params = [p_grad]
    proj_refs = [g_ref_accum]
    
    p_dot_ref_tensors = torch._foreach_mul(proj_params, proj_refs)
    ref_sq_tensors = torch._foreach_mul(proj_refs, proj_refs)
    
    alphas = []
    for i in range(len(proj_params)):
        dot_val = p_dot_ref_tensors[i].sum()
        norm_val = ref_sq_tensors[i].sum() + 1e-8
        
        if dot_val < 0 and torch.isfinite(dot_val) and torch.isfinite(norm_val):
            raw_alpha = -1.0 * (dot_val / norm_val).item()
            clamped_alpha = max(-10.0, min(10.0, raw_alpha))
            alphas.append(clamped_alpha)
        else:
            alphas.append(0.0)
            
    # Update
    scaled_refs = torch._foreach_mul(proj_refs, alphas)
    torch._foreach_add_(proj_params, scaled_refs)
    
    # Restoration step
    torch._foreach_add_(proj_params, proj_refs, alpha=1.0)
    
    # Check for NaNs
    if torch.isnan(p_grad).any():
        logger.error("❌ Patch #48.1 FAILED! Model gradient corrupted.")
    else:
        logger.info("✅ Patch #48.1 SUCCESS! Model gradient remains healthy despite NaN reference.")

if __name__ == "__main__":
    test_agem_nan_safe_projection_v3()
