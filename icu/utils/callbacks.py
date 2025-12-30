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
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import (
    BinaryAUROC, 
    BinaryAveragePrecision, 
    BinaryCalibrationError
)
from pytorch_lightning.callbacks import (
    Callback,
    RichProgressBar,
    ModelSummary,
    LearningRateMonitor,
    DeviceStatsMonitor,
    EarlyStopping,
    TQDMProgressBar
)

from icu.utils.train_utils import (
    get_rank, 
    is_main_process, 
    get_world_size,
    RotationalSaver,
    TieredEMA
)

# [FIX: Robust Import for RichProgressBarTheme (Colab/Older PL versions)]
logger = logging.getLogger("icu.callbacks")

try:
    from rich.text import Text
    from rich.progress import TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Column
except ImportError:
    # Fallback for environments without Rich
    logger.warning("Rich library not found. Progress bars will be degraded.")
    class Text:
        def __init__(self, s, **kwargs): self.s = s
        def __str__(self): return self.s
    
    class TextColumn:
        def __init__(self, *args, **kwargs): pass
    
    class BarColumn:
        pass
        
    class TimeRemainingColumn:
        pass
        
    class Column:
        pass

class ProcessingSpeedColumn(TextColumn):
    def __init__(self, style="grey53"):
        if 'TextColumn' in globals() and hasattr(TextColumn, '__init__'):
            super().__init__("", style=style)

    def render(self, task) -> Text:
        if not hasattr(task, 'speed') or task.speed is None:
            return Text("", style=getattr(self, 'style', '')) 
        return Text(f"{task.speed:.2f} it/s", style=getattr(self, 'style', ''))

class APEXMetricsColumn(TextColumn):
    def __init__(self):
        if 'TextColumn' in globals() and hasattr(TextColumn, '__init__'):
            super().__init__("") 

    def render(self, task) -> Text:
        m_str = task.fields.get("metrics_str", "")
        return Text(m_str, style="spring_green3", no_wrap=True, overflow="ellipsis")



def format_metric_sota(v: Any) -> str:
    """
    SOTA Metric Formatter.
    
    Logic:
    - Decimal (.3f) for values >= 0.001
    - Scientific (.2e) for values < 0.001 (and non-zero)
    - Clean '0.000' for zero
    - Standard string for non-numerics
    """
    if not isinstance(v, (float, int)):
        return str(v)
    
    abs_v = abs(float(v))
    if abs_v == 0:
        return "0.000"
    elif abs_v < 0.001:
        return f"{v:.2e}"
    else:
        return f"{v:.3f}"


class APEXProgressBar(RichProgressBar):
    """
    Standard PL RichProgressBar with 'SOTA' metric formatting.
    Removes dangerous custom column overrides to ensure stability.
    """
    def get_metrics(self, trainer, pl_module) -> Dict[str, str]:
        # 1. Let Parent calculate standard metrics (Loss, v_num, etc.)
        items = super().get_metrics(trainer, pl_module)
        
        # 2. Remove internal keys we don't want to see
        items.pop("v_num", None)
        
        # 3. Format the remaining metrics tightly
        clean_metrics = {}
        for k, v in items.items():
            # [v16.6] Adaptive Scientific Formatting
            val = format_metric_sota(v)
            
            # Create Short Keys (SOTA Style)
            # Example: val/clinical_auroc -> A

            sk = k.replace("train/", "").replace("val/", "").replace("health/", "")
            sk = sk.replace("total_loss", "L").replace("loss", "L")
            sk = sk.replace("diff_L", "D").replace("aux_L", "A").replace("value_L", "V")
            sk = sk.replace("clinical_", "").replace("auroc", "AUC").replace("auprc", "PRC")
            sk = sk.replace("generative_mse", "GMSE").replace("generative_mae", "GMAE")
            sk = sk.replace("policy_entropy", "E").replace("explained_variance", "EV")
            sk = sk.replace("preshock_L", "PS").replace("stable_L", "S").replace("crash_L", "C")
            sk = sk.replace("ood_rate", "OOD").replace("grad_norm_total", "GN").replace("max_weight", "MW")
            
            clean_metrics[sk] = val
            
        return clean_metrics



class APEXTQDMProgressBar(TQDMProgressBar):
    """
    Stabilized TQDM Bar for ICU Research.
    Guarantees 'In-Place' updates (No Newline Spam) while keeping SOTA metric formatting.
    """
    def __init__(self, refresh_rate: int = 20):
        # Update every 20 steps by default to reduce IO overhead
        super().__init__(refresh_rate=refresh_rate)

    def init_train_tqdm(self) -> tqdm:
        """Standardizes the Training TQDM bar with APEX branding."""
        bar = super().init_train_tqdm()
        bar.set_description("🏥 APEX Training")
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """Standardizes the Validation TQDM bar."""
        bar = super().init_validation_tqdm()
        bar.set_description("🩺 APEX Validation")
        return bar

    def get_metrics(self, trainer, pl_module) -> Dict[str, Union[int, str]]:
        # 1. Get standard metrics from Lightning
        items = super().get_metrics(trainer, pl_module)
        
        # 2. Remove internal/noisy keys
        items.pop("v_num", None)
        
        # 3. Reformat Keys for 'SOTA' Compactness
        # Output will look like: L:0.342 | AUC:0.891 | E:0.012
        clean_metrics = {}
        for k, v in items.items():
            # [v16.6] Adaptive Scientific Formatting
            val = format_metric_sota(v)
            
            # Shorten Keys
            # train/loss -> L, val/clinical_auroc -> A, diff_L -> D
            sk = k.replace("train/", "").replace("val/", "").replace("health/", "")
            sk = sk.replace("total_loss", "L").replace("loss", "L")
            sk = sk.replace("diff_L", "D").replace("aux_L", "A").replace("value_L", "V")
            sk = sk.replace("clinical_", "").replace("auroc", "AUC").replace("auprc", "PRC")
            sk = sk.replace("generative_mse", "GMSE").replace("generative_mae", "GMAE")
            sk = sk.replace("policy_entropy", "E").replace("explained_variance", "EV")
            sk = sk.replace("preshock_L", "PS").replace("stable_L", "S").replace("crash_L", "C")
            sk = sk.replace("router_ce_L", "RCE").replace("load_balance_L", "LB")
            sk = sk.replace("ood_rate", "OOD").replace("grad_norm_total", "GN").replace("max_weight", "MW")
            
            clean_metrics[sk] = val
            
        return clean_metrics

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
                    # [FIX: Alignment] Use the new unified save method
                    saver.trigger_emergency_save(trainer, pl_module)
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
            # v8.1: Check if model provided explicit 'sepsis_prob' (APEX-MoE Multi-Expert)
            if "sepsis_prob" in outputs and outputs["sepsis_prob"] is not None:
                sepsis_prob = outputs["sepsis_prob"].detach().float()
                # Ensure binary format
                if sepsis_prob.dim() == 2:
                    sepsis_prob = sepsis_prob.squeeze(-1)
                # [FIX v14.0] Apply the same valid_mask to ensure shape alignment
                clean_preds = sepsis_prob[valid_mask]

            
            # Legacy/Fallback Logic
            elif clean_preds.dim() == 2 and clean_preds.shape[-1] > 1:
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
                # [PATCH] CRITICAL: Do NOT check "clean_target.sum() > 0".
                # Updates must occur even for all-negative batches to correctly 
                # accumulate True Negatives and False Positives for global AUROC.
                
                # Ensure targets are binary integers for AUROC
                clean_target = clean_target.long() 
                
                # Update metrics globally
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
            
            # Use sync_dist=True with reduce_fx="max" to log the WORST gradient norm across GPUs.
            # This prevents noisy logs and highlights instability on any rank.
            pl_module.log("health/grad_norm_total", total_norm, on_step=True, sync_dist=True, reduce_fx="max", prog_bar=True)
            
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
        """
        [v3.7 FIX] Accumulation-Aware Updates.
        Only update EMA when the optimizer actually steps.
        """
        if self.ema:
            # Check for Gradient Accumulation
            # Logic: (batch_idx + 1) % accumulate_grad_batches == 0
            # OR it's the last batch of the epoch.
            accum = trainer.accumulate_grad_batches
            batch_idx = kwargs.get("batch_idx", 0) # robust get
            
            # Robust total batches retrieval
            try:
                total_batches = trainer.num_training_batches
            except:
                try:
                    total_batches = len(trainer.train_dataloader)
                except:
                    total_batches = float('inf') # Fallback

            is_accum_step = (batch_idx + 1) % accum == 0
            is_last_batch = (batch_idx + 1) == total_batches
            
            if is_accum_step or is_last_batch:
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
    def __init__(self, save_dir: str, remote_dir: Optional[str] = None, monitor: str = "val/clinical_auroc", filename_prefix: str = "icu_model"):
        super().__init__()
        self.saver = RotationalSaver(save_dir=save_dir, remote_dir=remote_dir, keep_last_n=3, snapshot_every_n=50)
        self.monitor = monitor
        self.filename_prefix = filename_prefix
        
        # Determine mode
        if any(x in monitor for x in ["loss", "mse", "mae", "error"]):
            self.mode = "min"
            self.best_metric_val = float('inf')
        else:
            self.mode = "max"
            self.best_metric_val = -float('inf') 
            
    def trigger_emergency_save(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Standardized entry point for AnomalyGuardian dumps."""
        return self._save_internal(trainer, pl_module, is_emergency=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not is_main_process(): return
        if trainer.sanity_checking: return 
        
        # Standard validation save
        return self._save_internal(trainer, pl_module, is_emergency=False)

    def _save_internal(self, trainer: pl.Trainer, pl_module: pl.LightningModule, is_emergency: bool = False):
        """Unified saving kernel for validation and emergency dumps."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        current_val = metrics.get(self.monitor)
        
        is_best = False
        if not is_emergency and current_val is not None:
            cv = current_val.item() if torch.is_tensor(current_val) else current_val
            if self.mode == "min":
                if cv < self.best_metric_val:
                    self.best_metric_val = cv
                    is_best = True
            else:
                if cv > self.best_metric_val:
                    self.best_metric_val = cv
                    is_best = True

        # Prepare State
        full_state = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'is_emergency': is_emergency,
            'state_dict': pl_module.model.state_dict(),
            # Handle potential EMA attachment
            'ema_state_dict': (
                pl_module.ema.state_dict() if hasattr(pl_module, 'ema') and pl_module.ema 
                else getattr(pl_module, '_ema_state_dict', None)
            ),
            'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
            'config': OmegaConf.to_container(pl_module.cfg, resolve=True) if hasattr(pl_module, 'cfg') else {},
            'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        }
        
        suffix = "_emergency" if is_emergency else ""
        self.saver.save(
            state_dict=full_state, 
            epoch=epoch, 
            is_best=is_best, 
            filename_prefix=f"{self.filename_prefix}{suffix}"
        )

    # [FIX 6] Add load_state_dict to fix Amnesia properly
    def load_state_dict(self, state_dict):
        self.best_metric_val = state_dict.get("best_metric_val", self.best_metric_val)

    def state_dict(self):
        return {"best_metric_val": self.best_metric_val}

# ==============================================================================
# 6. SOTA FACTORY
# ==============================================================================

def get_sota_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks = []

    # 1. Core Engines (Saver, EMA) - Keep as is
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

    # 2. Guardians (Anomaly, Metric, Health) - Keep as is
    callbacks.append(AnomalyGuardian(halt_on_anomaly=True))
    callbacks.append(ClinicalMetricCallback(inputs_are_logits=True))
    callbacks.append(GradientHealthMonitor(log_every_n_steps=100))

    # 3. Standard SOTA Monitoring (TQDM Standardized)
    # [FIX] Primacy given to TQDM for terminal stability. 
    # Rich is preserved above as 'APEXProgressBar' for legacy use.
    refresh_rate = cfg.train.get("refresh_rate", 20)
    callbacks.append(APEXTQDMProgressBar(refresh_rate=refresh_rate))
    
    callbacks.append(ModelSummary(max_depth=3))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    if torch.cuda.is_available():
        callbacks.append(DeviceStatsMonitor())

    # 4. Early Stopping - Keep as is
    patience = cfg.train.get("patience", 0)
    if patience > 0:
        callbacks.append(EarlyStopping(
            monitor=cfg.train.get("monitor", "val/clinical_auroc"),
            patience=patience,
            mode="max" if "auroc" in cfg.train.get("monitor", "val/clinical_auroc") else "min",
            verbose=True
        ))

    return callbacks