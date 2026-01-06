
import torch
import torch.nn as nn
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.components.bgsl_loss import BGSLLoss
from icu.models.components.loss_scaler import UncertaintyLossScaler

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrashRepro")

def test_bgsl_extremes():
    logger.info("=== Testing BGSLLoss Extremes ===")
    loss_fn = BGSLLoss()
    
    # Batch size 4, time 24
    B, T, D = 4, 24, 28
    
    # Case 1: All Zeros (Division by Zero check)
    logger.info("Case 1: All Zeros input")
    pred = torch.zeros(B, T, 1, requires_grad=True)
    target = torch.zeros(B, T, 1)
    past = torch.zeros(B, T, D)
    mask = torch.zeros(B, T).bool() # All valid
    
    out = loss_fn(pred, target, past, mask=mask)
    logger.info(f"Zeros Result: {out['loss'].item()}")
    if torch.isnan(out['loss']) or torch.isinf(out['loss']):
        logger.error("FAIL: Zeros caused NaN/Inf")
        return False
        
    out['loss'].backward()
    if torch.isnan(pred.grad).any():
        logger.error("FAIL: Zeros caused NaN Gradients")
        return False
    logger.info("PASS: Zeros handled.")

    # Case 2: Extreme Logits (Sigmoid instability)
    logger.info("Case 2: Extreme Logits (+/- 1000)")
    pred = torch.cat([torch.full((B//2, T, 1), 1000.0), torch.full((B//2, T, 1), -1000.0)], dim=0)
    pred.requires_grad = True
    out = loss_fn(pred, target, past, mask=mask)
    logger.info(f"Extreme Logits Result: {out['loss'].item()}")
    if torch.isnan(out['loss']) or torch.isinf(out['loss']):
        logger.error("FAIL: Extreme values caused NaN/Inf")
        return False
    out['loss'].backward()
    logger.info("PASS: Extreme Logits handled.")

    # Case 3: NaN Inputs (Sanitization check)
    logger.info("Case 3: NaN/Inf Inputs")
    pred = torch.randn(B, T, 1)
    pred[0, 0, 0] = float('nan')
    pred[0, 1, 0] = float('inf')
    pred.requires_grad = True
    
    # We expect the sanitization to handle this inside forward
    try:
        out = loss_fn(pred, target, past, mask=mask)
        logger.info(f"NaN Input Result: {out['loss'].item()}")
        if torch.isnan(out['loss']):
             logger.error("FAIL: NaN persisted through sanitization")
             return False
        out['loss'].backward()
    except Exception as e:
        logger.error(f"FAIL: Crashed on NaN input: {e}")
        return False
    # Case 4: Target = 1.0 (Fixed Binary Mapping)
    logger.info("Case 4: Target = 1.0 (Binary Sepsis)")
    pred = torch.full((B, T, 1), 15.0, requires_grad=True) # Confident
    target = torch.full((B, T, 1), 1.0) # Sepsis
    out = loss_fn(pred, target, past, mask=mask)
    logger.info(f"Target = 1.0 Result: {out['loss'].item()}")
    if out['loss'].item() < -1e-5:
        logger.error(f"FAIL: Target=1.0 caused NEGATIVE loss: {out['loss'].item()}")
        return False
    logger.info("PASS: Target = 1.0 is stable.")
    
    # Case 5: The "Lethal ASL" Gradient (The structural source)
    logger.info("Case 5: Gradient of ASL(target=2.0)")
    # We remove the clamp for this test to find the origin
    raw_pred = torch.full((B, T, 1), 5.0, requires_grad=True)
    raw_target = torch.full((B, T, 1), 2.0)
    
    # Simulate internal ASL logic
    probs = torch.sigmoid(raw_pred)
    p_target = probs * raw_target + (1-probs) * (1 - raw_target)
    # If p_target > 1.0 (because target=2.0), then (1 - p_target) is NEGATIVE.
    base = 1.0 - p_target
    logger.info(f"ASL Base: {base.mean().item()} (Should be negative)")
    
    # Power operation on negative base: undefined gradient or NaN
    try:
        w = torch.pow(base, 4.0)
        logger.info(f"ASL Weight: {w.mean().item()} (Value is finite)")
        grad = torch.autograd.grad(w.sum(), raw_pred)[0]
        logger.info(f"ASL Gradient Mean: {grad.mean().item()}")
        if torch.isnan(grad).any() or torch.isinf(grad).any():
             logger.error("FAIL: ASL Gradient is NaN/Inf with target=2.0")
             return False
    except Exception as e:
        logger.error(f"FAIL: ASL Pow crashed: {e}")
        return False
    
    return True

def test_scaler_collapse():
    logger.info("\n=== Testing UncertaintyLossScaler Collapse ===")
    scaler = UncertaintyLossScaler(num_tasks=6)
    
    # Case 1: Infinite Loss Component
    logger.info("Case 1: Input Loss is -inf")
    losses = {
        'diffusion': torch.tensor(1.0),
        'bgsl': torch.tensor(float('-inf'))
    }
    
    total, metrics = scaler(losses)
    logger.info(f"Scaler Output with -inf: {total.item()}")
    
    # Check gradients of log_vars
    total.backward()
    logger.info(f"LogVars Gradients: {scaler.log_vars.grad}")
    
    if torch.isnan(scaler.log_vars.grad).any() or torch.isinf(scaler.log_vars.grad).any():
        logger.error("FAIL: Scaler log_vars gradient is NaN/Inf with -inf loss")
        return False
        
    logger.info("PASS: Scaler gradients are finite.")
    return True

if __name__ == "__main__":
    try:
        bgsl_ok = test_bgsl_extremes()
        scaler_ok = test_scaler_collapse()
        
        if bgsl_ok and scaler_ok:
            logger.info("\n[SUCCESS] Systems are robust.")
            sys.exit(0)
        else:
            logger.error("\n[FAILURE] Vulnerabilities found.")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
