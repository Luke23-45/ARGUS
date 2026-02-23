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
    TQDMProgressBar,
    ModelCheckpoint
)

from icu.utils.train_utils import (
    get_rank, 
    is_main_process, 
    get_world_size,
    RotationalSaver,
    TieredEMA,
    ScalingSteward
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
    # [v16.7 FIX] Tensor Unpacking & Granular Scientific Format
    if hasattr(v, 'item'):
        try: v = float(v.item())
        except: return str(v)
    
    if not isinstance(v, (float, int)):
        return str(v)
    
    abs_v = abs(float(v))
    if abs_v == 0:
        return "0.000"
    elif abs_v < 0.01:
        return f"{v:.3e}"
    elif abs_v < 1.0:
        return f"{v:.4f}"
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
            sk = sk.replace("diff_L", "D").replace("diff_loss", "D")
            sk = sk.replace("critic_L", "V").replace("critic_loss", "V").replace("value_L", "V")
            sk = sk.replace("phys_L", "P").replace("phys_loss", "P")
            sk = sk.replace("aux_L", "A").replace("aux_loss", "A")
            sk = sk.replace("awr_ess", "ESS")
            sk = sk.replace("explained_var", "EV").replace("explained_variance", "EV")
            sk = sk.replace("curr_phys_weight", "PW").replace("w_aux", "WA")
            
            # Validation Specifics
            sk = sk.replace("mse_global", "GMSE").replace("sepsis_auroc", "AUC").replace("sepsis_AUC", "AUC")
            sk = sk.replace("sepsis_acc", "ACC").replace("ood_rate_avg", "OOD").replace("ood_rate", "OOD")
            sk = sk.replace("safe_trajectories_avg", "SAFE").replace("phys_violation_rate", "PV")
            
            sk = sk.replace("clinical_", "").replace("auroc", "AUC").replace("auprc", "PRC")
            sk = sk.replace("grad_norm_total", "GN")
            
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
            sk = sk.replace("diff_L", "D").replace("diff_loss", "D")
            sk = sk.replace("critic_L", "V").replace("critic_loss", "V").replace("value_L", "V")
            sk = sk.replace("phys_L", "P").replace("phys_loss", "P")
            sk = sk.replace("aux_L", "A").replace("aux_loss", "A")
            sk = sk.replace("awr_ess", "ESS")
            sk = sk.replace("explained_var", "EV").replace("explained_variance", "EV")
            sk = sk.replace("curr_phys_weight", "PW").replace("w_aux", "WA")
            
            # Validation Specifics
            sk = sk.replace("mse_global", "GMSE").replace("sepsis_auroc", "AUC").replace("sepsis_AUC", "AUC")
            sk = sk.replace("sepsis_acc", "ACC").replace("ood_rate_avg", "OOD").replace("ood_rate", "OOD")
            sk = sk.replace("safe_trajectories_avg", "SAFE").replace("phys_violation_rate", "PV")
            
            sk = sk.replace("clinical_", "").replace("auroc", "AUC").replace("auprc", "PRC")
            sk = sk.replace("grad_norm_total", "GN")
            
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
    def __init__(self, halt_on_anomaly: bool = True, check_interval: int = 50):
        super().__init__()
        self.halt_on_anomaly = halt_on_anomaly
        self.check_interval = check_interval
        self._batch_counter = 0

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        self._batch_counter += 1
        if self._batch_counter % self.check_interval != 0:
            return

        # 1. Check Loss Consistency (Periodic Sync)
        loss = trainer.callback_metrics.get("train/loss") or trainer.callback_metrics.get("train/total_loss")
        anomaly_flag = torch.tensor(0.0, device=pl_module.device)
        
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            anomaly_flag.fill_(1.0)
            
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(anomaly_flag, op=torch.distributed.ReduceOp.MAX)
            
        if anomaly_flag.item() > 0:
            self._handle_anomaly(trainer, pl_module, f"Numerical Anomaly detected in Loss (Checked every {self.check_interval} steps).")

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Monitors gradient health periodically to avoid hot-path stalls."""
        if self._batch_counter % self.check_interval != 0:
            return

        grad_anomaly = torch.tensor(0.0, device=pl_module.device)
        
        # Local Check (Expensive but periodic)
        for param in pl_module.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    grad_anomaly.fill_(1.0)
                    break
        
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_anomaly, op=torch.distributed.ReduceOp.MAX)
            
        if grad_anomaly.item() > 0:
            # Shielding: Zero out gradients if anomaly detected to prevent poisoning the weights
            # before the next optimizer step (though optimizer usually runs immediately after).
            pl_module.zero_grad()
            self._handle_anomaly(trainer, pl_module, "Gradient Anomaly (NaN/Inf) detected.")
            # [v2026 Phase 19 FIX] Accumulation Cycle Reset (Audit Finding F-3)
            # Rationale: Zeroing gradients mid-accumulation without resetting the 
            # accumulation counter causes should_step to fire with incomplete gradient
            # history in the next cycle. Reset to 0 for a clean restart.
            if hasattr(pl_module, "_shadow_grad_accum_idx"):
                pl_module._shadow_grad_accum_idx = 0
            self._handle_anomaly(trainer, pl_module, "Gradient Anomaly (NaN/Inf) detected. Weights shielded. Accumulation cycle reset.")


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
                    logger.warning("AnomalyGuardian: RotationalSaverCallback not found. Attempting standard emergency dump...")
                    # Fallback to standard Trainer save
                    dump_path = os.path.join(trainer.default_root_dir, "emergency_dump_anomaly.ckpt")
                    trainer.save_checkpoint(dump_path)
                    logger.info(f"Emergency Checkpoint saved to: {dump_path}")
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
                # [SOTA FIX] ASL Alignment: Independent Sigmoids, NOT Softmax
                if self.inputs_are_logits:
                    probs = torch.sigmoid(clean_preds)
                else:
                    probs = clean_preds
                
                # Metric: P(Any Sepsis) = 1.0 - P(All Healthy)
                # Assuming Class 0 is 'Stable', Classes 1+ are 'Risk'
                # Probabilistic Union: 1 - Product(1 - P_risk)
                sick_probs = probs[:, 1:]
                clean_preds = 1.0 - (1.0 - sick_probs).prod(dim=1)
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
            # 1. Total Norm (SOTA Fused Implementation)
            # [SOTA FIX] Avoid torch.cat() VRAM spike via _foreach_norm
            grads = [p.grad for p in pl_module.model.parameters() if p.grad is not None]
            if not grads: return
            
            # Math: Global L2 = sqrt(sum(local_L2^2))
            if hasattr(torch, "_foreach_norm"):
                local_norms = torch._foreach_norm(grads, 2)
                total_norm = torch.linalg.vector_norm(torch.stack(local_norms), 2).item()
            else:
                total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2).item()
            
            # Use sync_dist=True with reduce_fx="max" to log the WORST gradient norm across GPUs.
            pl_module.log("health/grad_norm_total", total_norm, on_step=True, sync_dist=True, reduce_fx="max", prog_bar=True)
            
            # 2. Expert Utilization (MoE Check)
            expert_patterns = {} 
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    match = re.search(r"experts\.(\d+)", name)
                    if match:
                        eid = int(match.group(1))
                        if eid not in expert_patterns: expert_patterns[eid] = []
                        expert_patterns[eid].append(param.grad)
            
            for eid, grads in expert_patterns.items():
                # [SOTA FIX] Fused Expert Norm
                if hasattr(torch, "_foreach_norm"):
                    local_norms = torch._foreach_norm(grads, 2)
                    gnorm = torch.linalg.vector_norm(torch.stack(local_norms), 2).item()
                else:
                    gnorm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2).item()
                
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
    SOTA Unified EMA Callback (2025).
    Maintains a single authoritative 'Teacher' instance via TieredEMA.
    
    Why this is SOTA:
    1. Swapping: Instantly replaces student weights with teacher weights for Val/Test.
    2. Zero-Copy: Uses pointer swapping to avoid memory overhead.
    3. Manual Opt Aware: Syncs update steps with custom optimization loops.
    """
    def __init__(self, decay: float = 0.9999, cpu_offload: bool = True, update_every: int = 1):
        super().__init__()
        self.decay = decay
        self.cpu_offload = cpu_offload
        self.update_every = update_every
        self.ema: Optional[TieredEMA] = None
        self._deferred_ema_state: Optional[Dict] = None # For checkpoint loading

    def _init_ema(self, pl_module: pl.LightningModule):
        if self.ema is None:
            logger.info(f"EMA: Initializing Teacher (Decay={self.decay})")
            self.ema = TieredEMA(
                pl_module.model, 
                decay=self.decay
            )
            pl_module.ema = self.ema # Authoritative attachment for training_step

            # Apply deferred state if available
            if self._deferred_ema_state:
                # [v112.0 SOTA] Forensic restoration logging
                stats = ""
                if "weight" in self._deferred_ema_state:
                    w = self._deferred_ema_state["weight"]
                    stats = f" (W: mean={w.mean().item():.4f}, std={w.std().item():.4f})"
                
                logger.info(f"EMA: Applying deferred state_dict from Checkpoint.{stats}")
                self.ema.load_state_dict(self._deferred_ema_state)
                self._deferred_ema_state = None # Clear after applying

    def on_fit_start(self, trainer, pl_module):
        self._init_ema(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # [SOTA FIX] Manual Optimization Guard
        # If the model handles optimization manually (wrapper_generalist), it MUST manage EMA stepping.
        # Otherwise, we get double-updates (Decay Squared) and race conditions.
        if not getattr(pl_module, "automatic_optimization", True):
            return

        # Legacy/Automatic Optimization Path
        if getattr(pl_module.cfg.train, 'manual_ema_update', False):
            return  # Wrapper handles update via its own ema.update() call
        
        if self.ema and (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            self.ema.update(
                pl_module.model, 
                global_step=trainer.global_step, 
                update_every=self.update_every
            )


    def on_validation_start(self, trainer, pl_module):
        self._init_ema(pl_module)
        if self.ema:
            self.ema.apply_shadow(pl_module.model)

    def on_validation_end(self, trainer, pl_module):
        if self.ema:
            self.ema.restore(pl_module.model) # Restore student weights for next training epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        [v110.0 SOTA] Teacher Consensus Protocol.
        Rationale: Synchronizes EMA shadow weights across all ranks at the 
        end of the epoch to prevent teacher divergence in DDP.
        """
        if self.ema:
            self.ema.synchronize()

    def on_test_start(self, trainer, pl_module):
        self._init_ema(pl_module)
        if self.ema:
            self.ema.apply_shadow(pl_module.model)

    def on_test_end(self, trainer, pl_module):
        if self.ema:
            self.ema.restore(pl_module.model) # Restore student weights post-testing

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            # We don't have pl_module.model yet in some PL versions' load_checkpoint
            # but usually it's passed. We ensure init happens on_fit_start or here.
            if self.ema is None:
                # Placeholder for deferred load if model not ready
                self._deferred_ema_state = checkpoint["ema_state_dict"]
            else:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])

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
        # [v2026 FIX] Full Module State Capture (Resumption Trauma Fix)
        # Rationale: Using pl_module.state_dict() instead of model.state_dict() ensures
        # we capture wrapper-attached modules like GhostBank and LossScaler.
        full_state = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'is_emergency': is_emergency,
            'state_dict': pl_module.state_dict(), 
            # Handle potential EMA attachment
            'ema_state_dict': (
                pl_module.ema.state_dict() if hasattr(pl_module, 'ema') and pl_module.ema 
                else getattr(pl_module, '_ema_state_dict', None)
            ),
            'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
            'config': OmegaConf.to_container(pl_module.cfg, resolve=True) if hasattr(pl_module, 'cfg') else {},
            'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()},
            'pytorch-lightning_version': pl.__version__
        }

        # [v2026 CRITICAL FIX] Invoke Manual Save Hook
        # Rationale: wrapper_generalist.on_save_checkpoint contains critical logic for
        # AWR state, GradNorm, and Accumulation Index. We MUST invoke it manually.
        if hasattr(pl_module, "on_save_checkpoint"):
             pl_module.on_save_checkpoint(full_state)
        
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
# 5.5. ENGINE: SAMPLER STEWARD
# ==============================================================================

class SamplerSteward(Callback):
    """
    [v4.2] Ensures Sampler state is saved/loaded with checkpoint.
    This prevents 'Resumption Trauma' where the sampler resets to the
    start of the epoch, potentially repeating seen data.
    """
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.train_dataloader is not None:
            # Handle potential Multiple DataLoaders
            try:
                # Access the underlying dataloader(s)
                dls = trainer.train_dataloader
                if not isinstance(dls, list): dls = [dls]
                
                sampler_states = []
                for dl in dls:
                    # Check for our custom EpisodeAwareSampler or similar
                    if hasattr(dl, "sampler") and hasattr(dl.sampler, "state_dict"):
                        sampler_states.append(dl.sampler.state_dict())
                    else:
                        sampler_states.append(None)
                checkpoint["sampler_states"] = sampler_states
            except Exception as e:
                logger.warning(f"[SamplerSteward] Failed to save sampler state: {e}")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Note: DataLoaders might not be initialized yet.
        # We store the state in the module temporarily.
        if "sampler_states" in checkpoint:
            pl_module.pending_sampler_states = checkpoint["sampler_states"]
            logger.info("[SamplerSteward] Found Sampler states in checkpoint. Queued for restoration.")

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, "pending_sampler_states") and pl_module.pending_sampler_states:
            try:
                dls = trainer.train_dataloader
                if dls is None: return
                if not isinstance(dls, list): dls = [dls]
                
                for i, state in enumerate(pl_module.pending_sampler_states):
                    if state is not None and i < len(dls):
                        # Check if the new sampler supports loading
                        if hasattr(dls[i], "sampler") and hasattr(dls[i].sampler, "load_state_dict"):
                            dls[i].sampler.load_state_dict(state)
                            logger.info(f"[SamplerSteward] Restored state for Sampler {i}.")
                
                # Clear to prevent re-application
                pl_module.pending_sampler_states = None
            except Exception as e:
                logger.warning(f"[SamplerSteward] Failed to restore sampler state: {e}")


# ==============================================================================
# 5.7  DEBUG: SURGICAL SNAPSHOTS
# ==============================================================================

class DebugSnapshotCallback(Callback):
    """
    Saves 'Surgical Snapshots' at specific epochs defined in config.
    Standardized on 0-based indexing (Epoch 0 is the first completed epoch).
    """
    def __init__(self, epochs: List[int], save_dir: str):
        super().__init__()
        self.epochs = set(epochs) 
        self.save_dir = save_dir
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 1. Skip if sanity checking
        if trainer.sanity_checking:
            return
            
        # 2. Skip if current epoch is not in list
        current = trainer.current_epoch
        if current not in self.epochs:
            return
            
        # 3. Guard for DDP (Only Rank 0 saves)
        if is_main_process():
            filename = f"debug_snapshot_epoch_{current}.ckpt"
            path = os.path.join(self.save_dir, filename)
            
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                # Standard PL save (includes optimizer state for full resume)
                trainer.save_checkpoint(path)
                logger.info(f"🚨 [DEBUG] Captured Surgical Snapshot: {path}")
            except Exception as e:
                logger.error(f"Failed to save debug snapshot: {e}")


# ==============================================================================
# 5.8  PERSISTENCE: LATEST SHADOW MIRROR (RAM OPTIMIZATION)
# ==============================================================================

class SOTAUnifiedPersistence(Callback):
    """
    [v2026 AXE-SHARPENED v2] Single-Write Persistence Manager.
    Rationale: Prevents the redundant 3.2GB write that occurs when an epoch is 
    both 'best' and 'last'. Uses hard-links to manage all aliases.
    """
    def __init__(self, run_name: str, dirpath: str):
        super().__init__()
        self.run_name = run_name
        self.dirpath = Path(dirpath)
        
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        [PATCH SG-CKPT] Checkpoint Mirror — fires AFTER ModelCheckpoint.
        Rationale: PL's ModelCheckpoint (save_last=True) saves the bridge during
        on_validation_end. The old on_train_epoch_end hook fired BEFORE validation,
        so the bridge file never existed when the mirror ran.
        Moving to on_validation_end ensures correct ordering.
        Zero serialization = Zero RAM spike.
        """
        if not is_main_process(): return
        if trainer.sanity_checking: return
        
        # The bridge file is created by ModelCheckpoint (save_last=True,
        # CHECKPOINT_NAME_LAST="resumption_bridge"). We just mirror it.
        resumption_path = self.dirpath / "resumption_bridge.ckpt"
        
        try:
            if resumption_path.exists():
                self._mirror_to_latest(trainer, resumption_path)
                logger.info(f"💾 [SOTA-SAVE] Epoch {trainer.current_epoch}: Bridge exists, mirrored to latest/.")
            else:
                logger.warning(f"⚠️ [SOTA-SAVE] Epoch {trainer.current_epoch}: Bridge not found at {resumption_path}. "
                               f"ModelCheckpoint may not have saved yet.")
        except Exception as e:
            import traceback
            logger.error(f"❌ [SOTA-SAVE] Mirror Failure: {e}\n{traceback.format_exc()}")

    def _mirror_to_latest(self, trainer: pl.Trainer, bridge_path: Path):
        """
        [v2026 AXE-SHARPENED v4] Descriptive Subfolder Mirroring.
        Rationale: Organizes the single most recent model into a 'latest/' 
        subfolder with human-readable epoch/metric tracking.
        """
        if not bridge_path.exists(): return

        # 1. Metric Extraction (Dynamic)
        monitor_val = 0.0
        monitor_key = "loss"
        checkpoint_cb = trainer.checkpoint_callback
        if checkpoint_cb and checkpoint_cb.monitor:
            monitor_key = checkpoint_cb.monitor
            monitor_val = trainer.callback_metrics.get(monitor_key, 0.0)
            if hasattr(monitor_val, "item"): monitor_val = monitor_val.item()
        
        # Clean key for filename (e.g. 'val/clinical_auroc' -> 'auroc')
        clean_key = monitor_key.split("/")[-1].replace("clinical_", "").replace("val_", "")

        # 2. Setup Subfolder Singleton
        latest_dir = self.dirpath / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        
        latest_name = f"latest-epoch={trainer.current_epoch:02d}-{clean_key}={monitor_val:.3f}.ckpt"
        latest_path = latest_dir / latest_name

        try:
            if not latest_path.exists():
                import os
                import time
                # Small retry loop for Windows race conditions
                for attempt in range(3):
                    try:
                        os.link(str(bridge_path), str(latest_path))
                        logger.info(f"💾 [MIRROR] Hard-Link Created: latest/{latest_name}")
                        break
                    except (OSError, AttributeError) as e:
                        if "32" in str(e) and attempt < 2:
                            time.sleep(1)
                            continue
                        import shutil
                        shutil.copy2(bridge_path, latest_path)
                        logger.info(f"✨ [MIRROR] Shadow Created (Fallback): {latest_name}")
                        break
                
                # singleton Purge: Remove all other files in latest/ EXCEPT the current one
                for old_file in latest_dir.glob("latest-epoch=*.ckpt"):
                    if old_file.name != latest_name:
                        try:
                            old_file.unlink()
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"⚠️ [MIRROR] Minor Mirror Failure: {e}")


class HighMemoryGuardian(Callback):
    """
    [v2026] Critical OOM Protection for 12GB Environments.
    Forces garbage collection and cache clearing at epoch boundaries.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🛡️ [GUARDIAN] RAM Harvested at Epoch End.")

    def on_validation_epoch_end(self, trainer, pl_module):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🛡️ [GUARDIAN] VRAM Buffers Cleared post-validation.")


# ==============================================================================
# 6. SOTA FACTORY
# ==============================================================================

def get_sota_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks = []

    # 1. Core Engines (Saver, EMA) - Keep as is
    # 1. Core Engines (Saver, EMA)
    # [v2026 AXE-SHARPENED v5] Unified Path Resolution
    # Rationale: If checkpoint_dir is provided, we MUST use it as the root base
    # AND append run_name to ensure experiment isolation. 
    if cfg.get("checkpoint_dir"):
        # Use absolute path to prevent ambiguity
        base_root = os.path.abspath(cfg.checkpoint_dir)
        save_dir = os.path.join(base_root, cfg.run_name, "checkpoints")
        logger.info(f"[PATH] Unified Performance Boot: {save_dir}")
    else:
        save_dir = os.path.join(cfg.output_dir, cfg.run_name, "checkpoints")
        logger.info(f"[PATH] Unified Project Boot: {save_dir}")

    # [v2026] OOM Guardian (PROMOTED TO INDEX 0)
    # Rationale: Must harvest RAM BEFORE ModelCheckpoint triggers.
    callbacks.append(HighMemoryGuardian())

    # [FIX] Use Standard PL ModelCheckpoint for Full Resume Compatibility
    monitor = cfg.train.get("monitor", "val/clinical_auroc")
    mode = "max" if "auroc" in monitor or "acc" in monitor else "min"
    
    saver_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="icu_model-{epoch:02d}-{val/clinical_auroc:.3f}" if "auroc" in monitor else "icu_model-{epoch:02d}-{val_loss:.3f}",
        monitor=monitor,
        mode=mode,
        save_top_k=cfg.train.get("save_top_k", 3),
        save_last=True,  # [v7 FIX] PL handles ALL saves: best epochs get free link, non-best get one save
        save_weights_only=False, # We need optimizer state for resume
        every_n_epochs=1
    )
    # [v7] Override the default 'last.ckpt' filename to 'resumption_bridge.ckpt'
    # This ensures PL creates 'resumption_bridge.ckpt' directly — no renaming needed.
    saver_cb.CHECKPOINT_NAME_LAST = "resumption_bridge"
    callbacks.append(saver_cb)

    # [v2026 AXE-SHARPENED v2] Unified Persistence Manager
    # Rationale: Replaced the simple mirror with a smart saver that eliminates
    # redundant 3GB writes by checking if ModelCheckpoint already saved the best model.
    # Pass the SAME save_dir to ensure synchronization.
    callbacks.append(SOTAUnifiedPersistence(run_name=cfg.run_name, dirpath=save_dir))
    
    # [BACKUP] Optional Remote Mirroring (Simple Copy)
    remote_dir = cfg.get("remote_dir", None)
    if remote_dir:
        class RemoteMirror(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                if is_main_process() and trainer.checkpoint_callback.best_model_path:
                    try:
                        import shutil
                        src = trainer.checkpoint_callback.best_model_path
                        dst = os.path.join(remote_dir, os.path.basename(src))
                        os.makedirs(remote_dir, exist_ok=True)
                        shutil.copy2(src, dst)
                    except Exception as e:
                        logger.warning(f"[BACKUP] Failed to mirror checkpoint: {e}")
        callbacks.append(RemoteMirror())
    
    ema_decay = cfg.train.get("ema_decay", 0.9999)
    ema_update_every = cfg.train.get("ema_update_every", 1)
    
    # [SOTA] Only create EMACallback if use_teacher is enabled
    if cfg.model.get("use_teacher", False) and ema_decay > 0:
        # [SOTA FIX - DYNAMIC BUDGET] Removed broken static initialization scaling.
        # Dynamic scaling based on batches should be executed in on_train_start if desired.
        callbacks.append(EMACallback(
            decay=ema_decay, 
            update_every=ema_update_every
        ))


    # [v14.5] Debug Snapshots (Epoch-Specific)
    debug_epochs = cfg.get("debug_save_epochs", None)
    if debug_epochs:
        # Normalize to list
        if isinstance(debug_epochs, int):
            epochs = [debug_epochs]
        else:
            epochs = list(debug_epochs)
            
        callbacks.append(DebugSnapshotCallback(
            epochs=epochs,
            save_dir=save_dir 
        ))

    # 2. Guardians (Anomaly, Metric, Health) - Keep as is
    callbacks.append(AnomalyGuardian(halt_on_anomaly=True))
    callbacks.append(ClinicalMetricCallback(inputs_are_logits=True))
    callbacks.append(GradientHealthMonitor(log_every_n_steps=100))
    
    # [v4.2.1 SOTA] Sampler Stewardship enabled for StatefulWeightedSampler support.
    callbacks.append(SamplerSteward())

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