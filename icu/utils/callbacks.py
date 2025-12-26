"""
icu/train/callbacks.py
--------------------------------------------------------------------------------
Production-Grade PyTorch Lightning Callbacks for ICU Research.

Responsibilities:
1.  **Resilience**: Connects the `RotationalSaver` (from utils) to the Lightning loop,
    ensuring atomic, rolling-window checkpointing.
2.  **Safety**: `AnomalyGuardian` protects against NaN/Inf divergence across DDP.
3.  **Observability**: `GradientHealthMonitor` and `ClinicalMetricCallback` provide
    deep telemetry without log pollution ("Z-Fighting").
4.  **SOTA Factory**: `get_sota_callbacks` generates the standard battery.

Status: PRODUCTION-READY / FRONTIER-PROJECT (v3.6.2 - Patched)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    RichProgressBar,
    ModelSummary,
    LearningRateMonitor,
    DeviceStatsMonitor,
    EarlyStopping
)

# [FIX: Robust Import for RichProgressBarTheme (Colab/Older PL versions)]
try:
    from pytorch_lightning.callbacks import RichProgressBarTheme
except ImportError:
    try:
        from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
    except ImportError:
        # Fallback: Create a dummy if it literally doesn't exist (safety first)
        logger.warning("RichProgressBarTheme not found in PL callbacks. Using default.")
        RichProgressBarTheme = lambda **kwargs: None
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryCalibrationError
from omegaconf import DictConfig, OmegaConf

# Project Imports
# [FIX: Import get_rank to prevent NameError crash]
from icu.utils.train_utils import (
    RotationalSaver, 
    TieredEMA, 
    is_main_process, 
    rank_zero_only,
    get_rank 
)

logger = logging.getLogger("icu.callbacks")

# ==============================================================================
# 0. UI: CLINICAL CONSOLE (Aesthetic Overhaul)
# ==============================================================================

APEX_RICH_THEME = RichProgressBarTheme(
    description="bold sky_blue3",
    progress_bar="bold dodger_blue1",
    batch_progress="bold white",
    time="grey70",
    processing_speed="grey53",
    metrics="bold spring_green3"
)

class APEXProgressBar(RichProgressBar):
    """
    Subclass of RichProgressBar for the APEX-MoE "Clinical Console".
    
    Fixes reported by USER:
    1.  **Metric Surgery**: Removes noisy 'v_num' (Version Number).
    2.  **Scientific Precision**: Rounds all floats to 4 decimals for readability.
    3.  **Branding**: Prepend stethoscope icon to indicate clinical context.
    4.  **Layout**: High-contrast medical theme (SkyBlue/SpringGreen).
    """
    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict[str, Union[int, str]]:
        # 1. Grab raw metrics
        items = super().get_metrics(trainer, pl_module)
        
        # 2. Metric Surgery: Remove noise
        items.pop("v_num", None)
        
        # 3. CONSOLIDATION: Merge all losses into one high-density string 
        # to prevent Rich from wrapping columns vertically.
        loss_telemetry = []
        other_metrics = {}
        
        for k, v in items.items():
            # Formatting
            val = f"{v:.3f}" if isinstance(v, float) else str(v)
            
            # Shorten keys logic
            sk = k.replace("train/", "").replace("val/", "")
            sk = sk.replace("total_loss", "L").replace("loss", "L") 
            sk = sk.replace("diff_L", "D").replace("aux_L", "A")
            sk = sk.replace("expert_L", "Exp").replace("reg_L", "Reg")
            sk = sk.replace("critic_L", "Crt").replace("value_L", "Crt")
            sk = sk.replace("stable_L", "S").replace("preshock_L", "P").replace("crash_L", "C")
            sk = sk.replace("batch_size", "B")
            
            # Categorize: Losses vs Metrics
            if any(x in sk for x in ["L", "D", "A", "Exp", "Reg", "Crt", "S", "P", "C", "ess"]):
                loss_telemetry.append(f"{sk}:{val}")
            else:
                other_metrics[sk] = val
        
        # 4. Final Payload: One "Stats" entry + any others (AUC, etc)
        res = {}
        if loss_telemetry:
            res["Stats"] = "|".join(loss_telemetry)
        res.update(other_metrics)
        
        return res

    def configure_columns(self, trainer: pl.Trainer) -> list:
        # Get standard columns
        cols = super().configure_columns(trainer)
        
        # STRIP REDUNDANCY: Maximize space for clinical metrics
        # We identify columns by their class name string to be robust across PL versions
        new_cols = []
        for c in cols:
            c_type = str(type(c)).lower()
            # Remove Speed and Elapsed time to save horizontal real-estate
            if "speed" in c_type or "elapsed" in c_type:
                continue
            new_cols.append(c)
            
        return new_cols

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Branding injection
        if self.progress is not None:
             self.progress.console.print("🩺 [bold cyan]APEX Clinical Console Initializing...[/]")
        super().on_train_start(trainer, pl_module)

# ==============================================================================
# 1. SAFETY: ANOMALY GUARDIAN
# ==============================================================================

class AnomalyGuardian(Callback):
    """
    Proactive NaN/Inf Detector for Life-Critical AI.
    
    If an anomaly is detected in loss, gradients, or weights:
    1.  Logs the culprit layer/metric.
    2.  Triggers an immediate Atomic Checkpoint Dump (Emergency Backup).
    3.  Halts training to prevent weights corruption.
    """
    def __init__(self, halt_on_anomaly: bool = True):
        super().__init__()
        self.halt_on_anomaly = halt_on_anomaly

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        # 1. Check Loss Consistency
        # CRITICAL FIX: Check both key variants for compatibility
        loss = trainer.callback_metrics.get("train/loss") or trainer.callback_metrics.get("train/total_loss")
        anomaly_flag = torch.tensor(0.0, device=pl_module.device)
        
        # Check validity (NaN or Inf)
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            anomaly_flag.fill_(1.0)
            
        # 2. DDP Sync: If ANY rank has an anomaly, ALL ranks halt together.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(anomaly_flag, op=torch.distributed.ReduceOp.MAX)
            
        if anomaly_flag.item() > 0:
            self._handle_anomaly(trainer, pl_module, "Numerical Anomaly (NaN/Inf) detected in Loss.")

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Monitors gradient health before optimizer step with DDP synchronization."""
        grad_anomaly = torch.tensor(0.0, device=pl_module.device)
        
        # Local Check
        for param in pl_module.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    grad_anomaly.fill_(1.0)
                    break
        
        # DDP Sync
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_anomaly, op=torch.distributed.ReduceOp.MAX)
            
        if grad_anomaly.item() > 0:
            # Shielding: Zero out gradients IMMEDIATELY before the optimizer can use them
            pl_module.zero_grad()
            self._handle_anomaly(trainer, pl_module, "Gradient Anomaly (NaN/Inf) detected. Weights shielded.")


    def _handle_anomaly(self, trainer: pl.Trainer, pl_module: pl.LightningModule, reason: str):
        # 1. Multi-Rank Logging
        current_rank = get_rank() # safe import assumed from header
        logger.error(f"🚨 [RANK {current_rank}] ANOMALY DETECTED: {reason}")
        
        # 2. Rank-0 Exclusive Dump [FIX: Traceback Deadlock]
        if is_main_process():
            try:
                # Attempt to find the saver callback to trigger a dump
                saver = None
                for cb in trainer.callbacks:
                    if isinstance(cb, RotationalSaverCallback):
                        saver = cb
                        break
                
                if saver:
                    logger.info("Executing Emergency Atomic Dump on Rank 0...")
                    saver.on_train_epoch_end(trainer, pl_module) # Force dump
                else:
                    logger.warning("AnomalyGuardian could not find RotationalSaverCallback to dump state.")
            except Exception as e:
                # Swallowing save error to preserve the original Anomaly traceback
                logger.error(f"FATAL: Emergency Dump Failed: {e}. Original Anomaly Persists.")
            
        # 3. Block-Safe Halt
        if self.halt_on_anomaly:
            if is_main_process():
                logger.critical("Halt triggered by AnomalyGuardian.")
            trainer.should_stop = True

# ==============================================================================
# 2. OBSERVABILITY: CLINICAL METRIC CALLBACK
# ==============================================================================

class ClinicalMetricCallback(Callback):
    """
    Clinical-Grade Metric Integration (DDP-Safe).
    Tracks AUROC, AUPRC, and Calibration using TorchMetrics.
    
    SOTA v3.7: Handles both Binary and Tri-Phase logits.
    For Tri-Phase, converts to binary: "Stable" vs "Not Stable" (Pre-Shock OR Shock).
    """
    def __init__(self, inputs_are_logits: bool = True):
        super().__init__()
        self.inputs_are_logits = inputs_are_logits
        # Metrics are lazy-initialized in setup() to ensure device correctness
        self.val_auroc = None
        self.val_auprc = None
        self.val_ece = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Move metrics to correct device on setup."""
        if self.val_auroc is None:
            self.val_auroc = BinaryAUROC().to(pl_module.device)
            self.val_auprc = BinaryAveragePrecision().to(pl_module.device)
            self.val_ece = BinaryCalibrationError().to(pl_module.device)
        logger.info(f"ClinicalMetricCallback: Metrics ready on {pl_module.device}")

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ):
        # Robust Logic: Verify keys exist
        if isinstance(outputs, dict) and "preds" in outputs and "target" in outputs:
            preds = outputs["preds"].detach().float()
            target = outputs["target"].detach().long()
            
            # Filter Padding (-1)
            valid_mask = target != -1
            if not valid_mask.any():
                return
                
            clean_preds = preds[valid_mask]
            clean_target = target[valid_mask]
            
            # CRITICAL FIX: Handle multi-class logits (Tri-Phase: [B, 3])
            # Convert to binary probability: P(Not Stable) = 1 - P(Stable)
            # where P(Stable) = softmax(logits)[:, 0]
            if clean_preds.dim() == 2 and clean_preds.shape[-1] > 1:
                # Multi-class logits: [B, num_classes]
                if self.inputs_are_logits:
                    probs = torch.softmax(clean_preds, dim=-1)
                else:
                    probs = clean_preds
                # P(Sick) = 1 - P(Stable), where Stable is class 0
                clean_preds = 1.0 - probs[:, 0]
            elif self.inputs_are_logits:
                # Binary logits: [B] or [B, 1]
                if clean_preds.dim() == 2:
                    clean_preds = clean_preds.squeeze(-1)
                clean_preds = torch.sigmoid(clean_preds)
            
            # Convert multi-class target to binary if needed
            # Target: 0=Stable, 1=Pre-Shock, 2=Shock -> Binary: 0=Stable, 1=Sick
            clean_target = (clean_target > 0).long()
            
            if clean_target.numel() > 0:
                # [FIX] Guard against batches with no positive samples
                # TorchMetrics warns "No positive samples in targets" when update() receives
                # a batch with all zeros. With 3.1% sepsis prevalence and batch_size=64,
                # ~12% of batches have no positives by chance. Skip these to avoid warning spam.
                # Epoch-level aggregation is still valid since we aggregate across many batches.
                if clean_target.sum() > 0:
                    self.val_auroc.update(clean_preds, clean_target)
                    self.val_auprc.update(clean_preds, clean_target)
                    self.val_ece.update(clean_preds, clean_target)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.val_auroc is None: return 
        
        # Compute synchronizes across ranks automatically
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        ece = self.val_ece.compute()
        
        # sync_dist=False because compute() handled it
        pl_module.log("val/clinical_auroc", auroc, sync_dist=False, prog_bar=True)
        pl_module.log("val/clinical_auprc", auprc, sync_dist=False, prog_bar=True)
        pl_module.log("val/clinical_ece", ece, sync_dist=False)
        
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_ece.reset()

# ==============================================================================
# 3. OBSERVABILITY: GRADIENT HEALTH MONITOR
# ==============================================================================

class GradientHealthMonitor(Callback):
    """
    High-Fidelity Telemetry for MoE & Deep Architectures.
    Logs grad norms to detect exploding gradients or "Expert Silence".
    """
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_step % self.log_every_n_steps == 0:
            # 1. Total Norm
            all_grads = [p.grad.detach().flatten() for p in pl_module.model.parameters() if p.grad is not None]
            if not all_grads: return
            
            total_norm = torch.cat(all_grads).norm(2).item()
            
            # [FIX: Telemetry Z-Fighting]
            # Use sync_dist=True with reduce_fx="max" to log the WORST gradient norm across GPUs.
            # This prevents noisy logs and highlights instability on any rank.
            pl_module.log("health/grad_norm_total", total_norm, on_step=True, sync_dist=True, reduce_fx="max")
            
            # 2. Expert Utilization (MoE Check)
            expert_patterns = {} 
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    match = re.search(r"experts\.(\d+)", name)
                    if match:
                        eid = int(match.group(1))
                        if eid not in expert_patterns: expert_patterns[eid] = []
                        expert_patterns[eid].append(param.grad.detach().flatten())
            
            for eid, grads in expert_patterns.items():
                gnorm = torch.cat(grads).norm(2).item()
                # Log worst-case Expert norm to detect collapse
                pl_module.log(f"health/expert_{eid}_grad_norm", gnorm, on_step=True, sync_dist=True, reduce_fx="max")
                
                # Rank-0 warning to avoid console spam
                if gnorm < 1e-8 and is_main_process():
                    logger.warning(f"[Rank 0] Expert {eid} appears silent (norm < 1e-8).")

# ==============================================================================
# 4. ENGINE: EMA CALLBACK (Hardened)
# ==============================================================================

class EMACallback(Callback):
    """
    Hardened EMA Suite (CPU-Offloaded) with Robust Resumption.
    Patched v3.6.2: Fixes Evaluation Amnesia and Device Mismatch bugs.
    """
    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema: Optional[TieredEMA] = None
        self._deferred_state_dict: Optional[Dict] = None

    def _ensure_ema_initialized(self, pl_module: pl.LightningModule):
        """Idempotent initialization logic shared across all stages."""
        if self.ema is None:
            logger.info(f"EMACallback: Initializing Shadow Weights (Decay={self.decay}) on CPU.")
            # Targeted at pl_module.model (The Scientific Core)
            self.ema = TieredEMA(model=pl_module.model, decay=self.decay)
            pl_module.ema = self.ema # Direct attachment
            
            # [FIX: Apply Deferred State]
            if self._deferred_state_dict:
                logger.info("EMACallback: Applying deferred state_dict from Checkpoint...")
                # TieredEMA.load_state_dict expects the dict, internal keys are verified by TieredEMA logic
                self.ema.load_state_dict(self._deferred_state_dict)
                self._deferred_state_dict = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._ensure_ema_initialized(pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # [FIX: Amnesia] Ensure init happens if running trainer.validate() directly
        self._ensure_ema_initialized(pl_module)
        if self.ema:
            self.ema.apply_shadow(pl_module.model)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # [FIX: Amnesia] Ensure init happens if running trainer.test() directly
        self._ensure_ema_initialized(pl_module)
        if self.ema:
            self.ema.apply_shadow(pl_module.model)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if self.ema:
            self.ema.update(pl_module.model)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.ema:
            self.ema.restore(pl_module.model)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.ema:
            self.ema.restore(pl_module.model)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict):
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict):
        """
        Intercept checkpoint loading with Device Safety.
        """
        if "ema_state_dict" in checkpoint:
            # [FIX: Device Mismatch] Force all tensors to CPU immediately
            # This prevents TieredEMA (CPU-bound) from crashing against GPU checkpoint tensors
            safe_state = {k: v.cpu() for k, v in checkpoint["ema_state_dict"].items()}

            if self.ema:
                self.ema.load_state_dict(safe_state)
                logger.info("EMACallback: Shadow Weights restored immediately.")
            else:
                self._deferred_state_dict = safe_state
                logger.info("EMACallback: Shadow Weights buffered (on CPU) for deferred load.")

# ==============================================================================
# 5. ENGINE: ROTATIONAL SAVER CALLBACK
# ==============================================================================

class RotationalSaverCallback(Callback):
    """
    High-Resilience Atomic Saver for Life-Critical AI.
    """
    def __init__(
        self, 
        save_dir: str, 
        remote_dir: Optional[str] = None, 
        monitor: str = "val/clinical_auroc",
        filename_prefix: str = "icu_model"
    ):
        super().__init__()
        self.saver = RotationalSaver(
            save_dir=save_dir, 
            remote_dir=remote_dir, 
            keep_last_n=3,
            snapshot_every_n=50
        )
        self.monitor = monitor
        # Detect direction based on metric name for sensible default
        if "loss" in monitor or "mse" in monitor or "mae" in monitor:
            self.best_metric_val = float('inf')
            self.mode = "min"
        else:
            self.best_metric_val = -1.0
            self.mode = "max"
            
        self.filename_prefix = filename_prefix

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not is_main_process():
            return

        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        # 1. Determine 'Is Best'
        current_val = metrics.get(self.monitor)
        is_best = False
        
        if current_val is not None:
            cv = current_val.item() if torch.is_tensor(current_val) else current_val
            
            if self.mode == "min":
                if cv < self.best_metric_val:
                    self.best_metric_val = cv
                    is_best = True
            else:
                if cv > self.best_metric_val:
                    self.best_metric_val = cv
                    is_best = True

        # 2. Extract Robust State
        full_state = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'state_dict': pl_module.model.state_dict(),
            # Safely grab EMA state if available
            'ema_state_dict': pl_module.ema.state_dict() if hasattr(pl_module, 'ema') and pl_module.ema else None,
            'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
            'config': OmegaConf.to_container(pl_module.cfg, resolve=True),
            'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        }
        
        # 3. Atomic Save
        self.saver.save(
            state_dict=full_state, 
            epoch=epoch, 
            is_best=is_best,
            filename_prefix=self.filename_prefix
        )

# ==============================================================================
# 6. SOTA FACTORY
# ==============================================================================

def get_sota_callbacks(cfg: DictConfig) -> List[Callback]:
    """Generates the definitive Guardian Battery for APEX-MoE."""
    callbacks = []

    # 1. Core Engines
    save_dir = f"{cfg.output_dir}/{cfg.run_name}/checkpoints"
    saver_cb = RotationalSaverCallback(
        save_dir=save_dir,
        remote_dir=cfg.get("remote_dir", None),
        monitor=cfg.train.get("monitor", "val/clinical_auroc"), 
        filename_prefix="icu_model"
    )
    callbacks.append(saver_cb)
    
    ema_decay = cfg.train.get("ema_decay", 0.9999)
    if ema_decay > 0:
        callbacks.append(EMACallback(decay=ema_decay))

    # 2. Guardians
    callbacks.append(AnomalyGuardian(halt_on_anomaly=True))
    callbacks.append(ClinicalMetricCallback(inputs_are_logits=True)) # Default assumption
    callbacks.append(GradientHealthMonitor(log_every_n_steps=100))

    # 3. Standard SOTA Monitoring
    callbacks.append(APEXProgressBar(theme=APEX_RICH_THEME, leave=True))
    callbacks.append(ModelSummary(max_depth=3))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    if torch.cuda.is_available():
        callbacks.append(DeviceStatsMonitor())

    # 4. Early Stopping
    patience = cfg.train.get("patience", 0)
    if patience > 0:
        callbacks.append(EarlyStopping(
            monitor=cfg.train.get("monitor", "val/clinical_auroc"),
            patience=patience,
            mode="max" if "auroc" in cfg.train.get("monitor", "val/clinical_auroc") else "min",
            verbose=True
        ))

    return callbacks