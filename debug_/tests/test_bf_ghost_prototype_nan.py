import torch
from icu.models.components.ghost_bank import SepsisGhostBank
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BF_v2")

def test_ghost_prototype_nan_v2():
    logger.info("Verifying Patch #51: NaN-Resistant Prototype...")
    
    bank = SepsisGhostBank(capacity=256, history_len=24, feature_dim=28, latent_dim=128)
    
    # 1. Healthy Update
    latents_ok = torch.randn(32, 128)
    bank._update_prototype(latents_ok)
    logger.info(f"Prototype Norm after healthy update: {bank.prototype_ema.norm().item():.4f}")
    
    # 2. NaN Batch Update
    latents_nan = torch.randn(32, 128)
    latents_nan[0, 0] = float('nan')
    bank._update_prototype(latents_nan)
    
    # Check for NaNs
    if torch.isnan(bank.prototype_ema).any():
        logger.error("❌ Patch #51 FAILED! Ghost Prototype infected by NaN.")
    else:
        logger.info("✅ Patch #51 SUCCESS! Prototype remained healthy after NaN batch.")
        
    # 3. Subsequent healthy update
    bank._update_prototype(latents_ok)
    if torch.isnan(bank.prototype_ema).any():
        logger.error("❌ Prototype remains NaN. Total Failure.")
    else:
        logger.info("✅ Prototype is still functional and healthy.")

if __name__ == "__main__":
    test_ghost_prototype_nan_v2()
