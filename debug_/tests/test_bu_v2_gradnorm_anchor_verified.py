import torch
import torch.nn as nn
from icu.core.gradnorm import GradNormBalancer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_BU_v2")

def test_gradnorm_anchor_verification():
    logger.info("Verifying GradNorm EMA-Anchor (#117)...")
    
    # Mock parameters
    shared_params = [nn.Parameter(torch.randn(1, 1))]
    num_tasks = 2
    gn = GradNormBalancer(num_tasks=num_tasks, shared_params=shared_params)
    
    # Phase 1: Warmup (Steps 0-100)
    # Initial loss = 10.0
    for _ in range(101):
        l1 = shared_params[0].sum() * 0 + 10.0
        l2 = shared_params[0].sum() * 0 + 10.0
        gn.update(torch.stack([l1, l2]))
        
    logger.info(f"Anchor after 100 steps: {gn.initial_losses.tolist()}")
    
    # Phase 2: Model Improvement (Steps 101-200)
    # Loss drops to 1.0
    for _ in range(100):
        l1 = shared_params[0].sum() * 0 + 1.0
        l2 = shared_params[0].sum() * 0 + 1.0
        gn.update(torch.stack([l1, l2]))
        
    anchor = gn.initial_losses
    logger.info(f"Anchor after 200 steps (should have started adapting): {anchor.tolist()}")
    
    # Check rel_rates
    # Current loss = 1.0
    rel_rates = torch.tensor([1.0, 1.0]) / (anchor + 1e-8)
    logger.info(f"Effective Relative Rates: {rel_rates.tolist()}")
    
    # SUCCESS CRITERIA:
    # Anchor should be less than 10.0.
    if anchor[0] < 10.0:
        logger.info(f"✅ Verification SUCCESS! Anchor is dynamic: {anchor[0].item():.4f} < 10.0")
    else:
        logger.error("❌ Verification FAILED! Anchor is still frozen at 10.0.")

if __name__ == "__main__":
    test_gradnorm_anchor_verification()
