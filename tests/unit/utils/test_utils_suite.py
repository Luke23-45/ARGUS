import pytest
import torch
import torch.nn as nn
import os
import shutil
from pathlib import Path
from icu.utils.train_utils import (
    EMA, 
    RotationalSaver, 
    SurgicalCheckpointLoader,
    setup_logger
)

# ==============================================================================
# 1. EMA TESTS
# ==============================================================================

def test_ema_weight_averaging():
    """Verify EMA correctly tracks a moving average of weights."""
    model = nn.Linear(5, 1)
    # Set weights to all zeros
    nn.init.constant_(model.weight, 0.0)
    
    ema = EMA(model, decay=0.9) # 0.9 * shadow + 0.1 * model
    
    # Update model weights to all ones
    with torch.no_grad():
        model.weight.fill_(1.0)
    
    ema.update(model)
    
    # New shadow should be 0.9*0 + 0.1*1 = 0.1
    assert torch.allclose(ema.shadow['weight'], torch.tensor(0.1))

def test_ema_apply_restore():
    """Verify EMA can swap weights in and out of a model."""
    model = nn.Linear(5, 1)
    nn.init.constant_(model.weight, 5.5)
    
    ema = EMA(model)
    ema.shadow['weight'] = torch.ones_like(model.weight) * 1.1
    
    # Apply EMA weights to model
    ema.apply_shadow(model)
    assert torch.allclose(model.weight, torch.tensor(1.1))
    
    # Restore student weights
    ema.restore(model)
    assert torch.allclose(model.weight, torch.tensor(5.5))

# ==============================================================================
# 2. SURGICAL CHECKPOINT LOADER TESTS
# ==============================================================================

def test_surgical_loading_remap(tmp_path):
    """Verify partial weight loading with key remapping."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(5, 1)
    
    model = SimpleModel()
    
    # Create fake checkpoint with different layout (Flat state dict)
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "backbone.weight": torch.ones(1, 5) * 7.7, 
        "backbone.bias": torch.zeros(1)
    }, ckpt_path)
    
    loader = SurgicalCheckpointLoader()
    # Map 'backbone' in checkpoint to 'net' in model
    SurgicalCheckpointLoader.load_model(model, str(ckpt_path), key_mapping={"backbone": "net"}, strict=False)
    
    assert torch.allclose(model.net.weight, torch.tensor(7.7))

# ==============================================================================
# 3. ROTATIONAL SAVER TESTS
# ==============================================================================

def test_rotational_saver_cleanup(tmp_path):
    """Verify saver maintains rolling window and deletes old files."""
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    
    # Keep only 2 most recent checkpoints
    saver = RotationalSaver(str(save_dir), keep_last_n=2)
    
    dummy_data = {"val": 1}
    
    # Save 3 epochs
    saver.save(dummy_data, epoch=1)
    saver.save(dummy_data, epoch=2)
    saver.save(dummy_data, epoch=3)
    
    # Check that epoch 1 was deleted, leaving 2 and 3
    files = list(save_dir.glob("checkpoint_epoch_*.pt"))
    epochs = sorted([int(f.stem.split("_")[-1]) for f in files])
    
    assert epochs == [2, 3]
    assert not (save_dir / "checkpoint_epoch_1.pt").exists()

def test_rotational_saver_best(tmp_path):
    """Verify best model is tracked and copied."""
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    saver = RotationalSaver(str(save_dir))
    
    saver.save({"acc": 0.5}, epoch=1, is_best=True)
    assert (save_dir / "best_model.pt").exists()
    
    # Verify content (simple check)
    best = torch.load(save_dir / "best_model.pt")
    assert best["acc"] == 0.5

# ==============================================================================
# 4. LOGGING TESTS
# ==============================================================================

def test_setup_logger(tmp_path):
    """Verify logger creation and file output."""
    log_dir = tmp_path / "logs"
    logger = setup_logger("test_run", str(log_dir))
    
    logger.info("Test Message")
    
    log_file = log_dir / "test_run.log"
    assert log_file.exists()
    assert "Test Message" in log_file.read_text()
