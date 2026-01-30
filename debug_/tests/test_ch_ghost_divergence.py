import torch
import torch.nn as nn
import logging
from unittest.mock import patch
from icu.models.components.ghost_bank import SepsisGhostBank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CH")

def test_ghost_bank_rank_parity_fix():
    logger.info("Verifying Fix for Smoking Gun #161: Ghost Bank Prototype Rank Parity...")
    
    latent_dim = 128
    bank_r0 = SepsisGhostBank(latent_dim=latent_dim, prototype_ema_decay=0.9)
    bank_r1 = SepsisGhostBank(latent_dim=latent_dim, prototype_ema_decay=0.9)
    
    # Mock distributed
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.all_reduce") as mock_all_reduce:
        
        # Simulated global signal: 
        # Rank 0 has 8 samples. Rank 1 has 0.
        # Global Sum will be Rank 0's sum. Global Count will be 8.
        
        latents_r0 = torch.randn(8, latent_dim)
        r0_sum = latents_r0.sum(dim=0, keepdim=True)
        
        def mock_all_reduce_side_effect(buffer, op=None):
            # Simulation: buffer is [sum, count]
            # Since r0 has signal and r1 has none.
            # In a real run, both ranks would call all_reduce and the result would be the same.
            # Here we just force the result to be r0's stats.
            buffer[:-1] = r0_sum.flatten()
            buffer[-1] = 8.0
            return None
        
        mock_all_reduce.side_effect = mock_all_reduce_side_effect
        
        # Rank 0 call
        bank_r0._update_prototype(latents_r0)
        
        # Rank 1 call (with empty input)
        bank_r1._update_prototype(torch.zeros(0, latent_dim))
        
        # 2. Check for Parity
        diff_ema = torch.abs(bank_r0.prototype_ema - bank_r1.prototype_ema).sum().item()
        logger.info(f"Prototype EMA Difference: {diff_ema:.6e}")
        
        if diff_ema < 1e-6:
            logger.info("✅ Fix for Smoking Gun #161 VERIFIED! Ghost Bank Prototypes matched across ranks.")
        else:
            logger.error(f"❌ Fix FAILED! Divergence detected: {diff_ema:.6e}")

if __name__ == "__main__":
    test_ghost_bank_rank_parity_fix()
