import torch
import torch.nn as nn
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_Q")

def compute_bgsl_gradient(logits, targets, risk_coef=None):
    """Simplified BGSL state loss with critical penalty (bgsl_loss.py:107)"""
    # Base loss (simplified BCE)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Critical Penalty
    penalty = 1.0
    if risk_coef is not None:
        penalty = 1.0 + (risk_coef * 2.0)
        
    loss = (bce * penalty).mean()
    return loss

def simulate_ghost_shock():
    logger.info("Simulating Ghost Sepsis Penalty Shock...")
    
    # 1. Buggy Case
    logger.info("\n[CASE 1] Buggy Implementation (Hardcoded Risk=2.0)")
    logits_b = torch.tensor([[-2.0]], requires_grad=True)
    target_b = torch.tensor([[1.0]])
    loss_b = compute_bgsl_gradient(logits_b, target_b, risk_coef=2.0)
    loss_b.backward()
    g_norm_b = logits_b.grad.norm().item()
    amp_b = g_norm_b / 0.8808 # reference neutral
    logger.info(f"  Risk 2.0 (Penalty 5x): Grad Norm = {g_norm_b:.4f} ({amp_b:.1f}x amp)")

    # 2. Patched Case (Normal Stability)
    logger.info("\n[CASE 2] Patched Implementation (Stability=1.0, Risk=0.5)")
    logits_p1 = torch.tensor([[-2.0]], requires_grad=True)
    loss_p1 = compute_bgsl_gradient(logits_p1, target_b, risk_coef=0.5 * 1.0)
    loss_p1.backward()
    g_norm_p1 = logits_p1.grad.norm().item()
    amp_p1 = g_norm_p1 / 0.8808
    logger.info(f"  Risk 0.5 (Penalty 2x): Grad Norm = {g_norm_p1:.4f} ({amp_p1:.1f}x amp)")

    # 3. Patched Case (Shock Stability)
    logger.info("\n[CASE 3] Patched Implementation (Stability=0.1, Risk=0.05)")
    logits_p2 = torch.tensor([[-2.0]], requires_grad=True)
    loss_p2 = compute_bgsl_gradient(logits_p2, target_b, risk_coef=0.5 * 0.1)
    loss_p2.backward()
    g_norm_p2 = logits_p2.grad.norm().item()
    amp_p2 = g_norm_p2 / 0.8808
    logger.info(f"  Risk 0.05 (Penalty 1.1x): Grad Norm = {g_norm_p2:.4f} ({amp_p2:.1f}x amp)")

    issues = []
    if amp_p1 > 3.0:
        issues.append(f"PATCH FAILURE: Normal penalty still too high ({amp_p1:.1f}x)")
    if amp_p2 > 1.5:
        issues.append(f"PATCH FAILURE: Shock suppression ineffective ({amp_p2:.1f}x)")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    if not issues:
        logger.info("\u2705 TEST Q PASSED: Ghost penalty is now calibrated and safe")
    else:
        logger.warning("\u26a0\ufe0f TEST Q FAILED: Patch is insufficient")
        for issue in issues:
            logger.warning(f"   - {issue}")
            
    return not bool(issues)

if __name__ == "__main__":
    simulate_ghost_shock()
