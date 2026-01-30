import torch
import torch.nn as nn
from icu.models.components.loss_scaler import BayesianProjectedScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BW_v2")

def test_bayesian_safe_start_fixed():
    logger.info("Verifying Patch #110: Bayesian Safe-Start (Initialization)...")
    
    # Initialize scaler
    scaler = BayesianProjectedScaler(num_tasks=7)
    
    # Check log_vars
    # indices: diff=0, critic=1, aux=2, acl=3, bgsl=4, tcb=5, phys=6
    expected_suppression = {
        4: 0.5, # bgsl
        5: 1.0, # tcb
        6: 2.0  # phys
    }
    
    log_vars = scaler.log_vars.data
    logger.info(f"Initialized log_vars: {log_vars.tolist()}")
    
    # Calculate weights: precision = exp(-log_var)
    # Weights = 0.5 * precision
    precision = torch.exp(-log_vars)
    weights = 0.5 * precision
    
    logger.info(f"Calculated Weights: {weights.tolist()}")
    
    # Primary tasks (0-3) should have log_var=0 => weight=0.5
    for i in range(4):
        if abs(weights[i].item() - 0.5) < 1e-4:
            logger.info(f"✅ Task {i} (Primary) weight is nominal (0.5)")
        else:
            logger.error(f"❌ Task {i} weight mismatch: {weights[i].item()}")

    # Suppressed tasks (4-6)
    for i, target_lv in expected_suppression.items():
        if abs(log_vars[i].item() - target_lv) < 1e-4:
            logger.info(f"✅ Task {i} (Extra) correctly suppressed with log_var={target_lv:.1f} (Weight={weights[i].item():.4f})")
        else:
            logger.error(f"❌ Task {i} suppression mismatch: Found {log_vars[i].item()}, Expected {target_lv}")

    if weights[6] < 0.1:
        logger.info("✅ Patch #110 SUCCESS! Physics task is heavily suppressed at start (Safe-Start Active).")
    else:
        logger.error(f"❌ Verification Failed. Physics suppression too weak (Weight={weights[6].item():.4f})")

if __name__ == "__main__":
    test_bayesian_safe_start_fixed()
