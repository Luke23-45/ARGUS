
import torch
import torch.nn as nn
import logging
import sys
import os
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.wrapper_generalist import ICUGeneralistWrapper

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResumptionTest")

def test_elastic_resizing():
    logger.info("=== Testing Elastic Buffer Resizing ===")
    
    # 1. Base Configuration (Capacity 1024 / 256)
    conf = {
        "model": {
            "input_dim": 28,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "encoder_layers": 1,
            "history_len": 24,
            "pred_len": 6,
            "use_auxiliary_head": True,
            "num_phases": 6,
            "dropout": 0.0,
            "timesteps": 10
        },
        "train": {
            "tcb_capacity": 1024,
            "ghost_capacity": 256,
            "balancing_mode": "sota_2025",
            "lr": 1e-4,
            "weight_decay": 1e-4
        }
    }
    cfg = OmegaConf.create(conf)
    
    # 2. Initialize Model
    wrapper = ICUGeneralistWrapper(cfg)
    logger.info(f"Initial TCB Capacity: {wrapper.tcb_buffer.capacity}")
    logger.info(f"Initial Ghost Capacity: {wrapper.ghost_bank.capacity}")
    
    # 3. Simulate Scaled Checkpoint (6021 / 1505)
    # We take the current state_dict and replace specific buffers with scaled versions
    state_dict = wrapper.state_dict()
    
    # TCB Buffer Scale
    scaled_tcb_queue = torch.randn(6021, 32)
    state_dict["tcb_buffer.queue"] = scaled_tcb_queue
    
    # Ghost Bank Scale
    scaled_vitals = torch.randn(1505, 24, 28)
    state_dict["ghost_bank.raw_vitals"] = scaled_vitals
    state_dict["ghost_bank.raw_masks"] = torch.randn(1505, 24, 28)
    state_dict["ghost_bank.raw_labels"] = torch.zeros(1505, dtype=torch.long)
    state_dict["ghost_bank.latent_anchors"] = torch.randn(1505, 32)
    state_dict["ghost_bank.uncertainties"] = torch.randn(1505, 1)
    
    checkpoint = {"state_dict": state_dict}
    
    # 4. Trigger on_load_checkpoint (The Hook)
    logger.info("Triggering on_load_checkpoint...")
    wrapper.on_load_checkpoint(checkpoint)
    
    logger.info(f"Resized TCB Capacity: {wrapper.tcb_buffer.capacity}")
    logger.info(f"Resized Ghost Capacity: {wrapper.ghost_bank.capacity}")
    
    # Verification
    if wrapper.tcb_buffer.capacity != 6021:
        logger.error(f"FAIL: TCB Capacity mismatch. Expected 6021, got {wrapper.tcb_buffer.capacity}")
        return False
        
    if wrapper.ghost_bank.capacity != 1505:
        logger.error(f"FAIL: Ghost Bank Capacity mismatch. Expected 1505, got {wrapper.ghost_bank.capacity}")
        return False

    # 5. Verify load_state_dict (Strict Mode)
    # This is where it used to crash.
    try:
        wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
        logger.info("PASS: load_state_dict successful with strict=True.")
    except Exception as e:
        logger.error(f"FAIL: load_state_dict crashed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_elastic_resizing()
    if success:
        logger.info("\n[SUCCESS] Elastic Resizing Fix Validated.")
        sys.exit(0)
    else:
        logger.error("\n[FAILURE] Elastic Resizing Fix Failed.")
        sys.exit(1)
