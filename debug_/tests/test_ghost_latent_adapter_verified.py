import torch
import logging
from icu.models.components.ghost_bank import SepsisGhostBank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_GHOST_ADAPTER")

def test_ghost_latent_adapter_verification():
    logger.info("Verifying Ghost Latent Adapter (#135)...")
    
    latent_dim = 16
    bank = SepsisGhostBank(capacity=10, latent_dim=latent_dim, latent_adapter_strength=0.1)
    
    # 1. Fill bank with historical anchors (all zeros)
    vitals = torch.zeros(5, 24, 28)
    masks = torch.ones(5, 24, 28)
    labels = torch.ones(5, dtype=torch.long)
    latents = torch.zeros(5, latent_dim) # Historical state: 0
    
    bank.update(vitals, masks, labels, latents)
    
    # 2. Set current manifold prototype (all ones)
    bank.prototype_ema.fill_(1.0)
    
    # 3. Sample from bank
    out = bank.sample(num_ghosts=5, seed=42)
    
    sampled_anchors = out["anchors"]
    
    # 4. Check adaptation
    # Expected: (1 - strength) * 0.0 + strength * 1.0 = strength
    expected_val = 0.1
    actual_val = sampled_anchors[0, 0].item()
    
    logger.info(f"Historical Anchor Val: 0.0")
    logger.info(f"Current Prototype Val: 1.0")
    logger.info(f"Sampled (Adapted) Val: {actual_val:.4f} (Expected: {expected_val:.4f})")
    
    if abs(actual_val - expected_val) < 1e-4:
        logger.info("✅ Verification SUCCESS! Ghost Latent Adapter is functioning correctly.")
    else:
        logger.error(f"❌ Verification FAILED! Adaptation logic error. Got {actual_val}, expected {expected_val}.")

if __name__ == "__main__":
    test_ghost_latent_adapter_verification()
