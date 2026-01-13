"""
working/unified_ema.py
----------------------
SOTA Unified EMA (Exponential Moving Average) management for PyTorch Lightning.
Prevents 'Teacher Drift' in Multi-Task Planning models.
"""

import torch
from pytorch_lightning import Callback
from icu.utils.train_utils import TieredEMA

class UnifiedEMACallback(Callback):
    """
    Manages a single authoritative EMA 'Teacher' instance.
    CPU-offloaded shadow weights via TieredEMA.
    """
    def __init__(self, decay: float = 0.9999, cpu_offload: bool = True):
        super().__init__()
        self.decay = decay
        self.cpu_offload = cpu_offload
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        """Unified initialization: attaches TieredEMA to the scientific model."""
        if self.ema is None:
            self.ema = TieredEMA(
                pl_module.model, 
                decay=self.decay, 
                cpu_offload=self.cpu_offload
            )
            # Authority check: attach to pl_module for wrapper access
            pl_module.ema = self.ema

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update shadow weights synchronously with optimizer steps."""
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            self.ema.update(pl_module.model)

    def on_validation_start(self, trainer, pl_module):
        """Swap to shadow weights to evaluate the 'Teacher'."""
        if self.ema:
            self.ema.swap(pl_module.model)

    def on_validation_end(self, trainer, pl_module):
        """Restore 'Student' weights for further training."""
        if self.ema:
            self.ema.swap(pl_module.model)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save shadow weights into the main checkpoint."""
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore shadow weights from checkpoint if present."""
        if "ema_state_dict" in checkpoint:
            if self.ema is None:
                # Early init if needed
                self.ema = TieredEMA(pl_module.model, decay=self.decay)
                pl_module.ema = self.ema
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
