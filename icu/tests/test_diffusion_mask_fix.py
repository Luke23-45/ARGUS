
import torch
import logging
import sys
from unittest.mock import MagicMock

# [TEST FIX] Mock kagglehub to prevent ImportError in CI/Test
sys.modules["kagglehub"] = MagicMock()

from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestDiffusionMask")

def test_diffusion_mask_handling():
    logger.info("--- Starting Diffusion Mask Verification ---")
    
    # 1. Setup Config with Imputation Masks Enabled
    cfg = ICUConfig(
        input_dim=28,
        static_dim=6,
        history_len=24,
        pred_len=6,
        d_model=32,       # Small model for test speed
        n_heads=4,
        n_layers=2,
        encoder_layers=2,
        use_imputation_masks=True, # [v12.0] ENABLED
        use_self_conditioning=True
    )
    
    model = ICUUnifiedPlanner(cfg)
    model.eval()
    
    # 2. Create Mock Batch
    B = 2
    T_obs = cfg.history_len
    T_pred = cfg.pred_len
    D = cfg.input_dim
    
    batch = {
        "observed_data": torch.randn(B, T_obs, D),   # [B, 24, 28]
        "future_data":   torch.randn(B, T_pred, D),  # [B, 6, 28]
        "static_context": torch.randn(B, 6),         # [B, 6]
        "src_mask":      torch.rand(B, T_obs, D).bernoulli() # [B, 24, 28] IMPUTATION MASK
    }
    
    logger.info(f"Input Shapes: {batch['observed_data'].shape}, Mask: {batch['src_mask'].shape}")
    
    # 3. Test Forward Pass (Training)
    try:
        logger.info("Testing Training Forward Pass...")
        out = model(batch)
        loss = out["loss"]
        logger.info(f"Forward Pass Successful. Loss: {loss.item()}")
    except Exception as e:
        logger.error(f"Forward Pass FAILED: {e}")
        raise e
        
    # 4. Test Sampling (Inference)
    try:
        logger.info("Testing Inference Sampling...")
        with torch.no_grad():
            sample = model.sample(batch)
        logger.info(f"Sampling Successful. Output Shape: {sample.shape}")
        assert sample.shape == (B, T_pred, D)
    except Exception as e:
        logger.error(f"Sampling FAILED: {e}")
        raise e

    # 5. Test Backward Compatibility (Flag Disabled)
    logger.info("Testing Backward Compatibility (use_imputation_masks=False)...")
    cfg_legacy = ICUConfig(
        input_dim=28,
        d_model=32,
        use_imputation_masks=False # DISABLED
    )
    model_legacy = ICUUnifiedPlanner(cfg_legacy)
    try:
        out = model_legacy(batch)
        logger.info("Legacy Forward Pass Successful.")
    except Exception as e:
        logger.error(f"Legacy Forward Pass FAILED: {e}")
        raise e
        
    logger.info("--- Verification Complete: GREEN LIGHT ---")

if __name__ == "__main__":
    test_diffusion_mask_handling()
