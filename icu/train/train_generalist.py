"""
icu/train/train_generalist.py
--------------------------------------------------------------------------------
APEX-MoE: Phase 1 Generalist Training Engine (Ultimate v8.0).

Status: PRODUCTION-READY / SAFETY-CRITICAL

This script orchestrates the training of the Generalist "Donor" model.
It establishes the shared latent space and pre-trains the Sepsis Router,
serving as the foundation for the Phase 2 Specialist system.

"The Generalist is the foundation upon which clinical intelligence is built.
Every carefully tuned weight becomes the seed for life-saving predictions."

Architectural Integration:
1.  **Wrapper-Based Training**: Uses `ICUGeneralistWrapper` for clean
    separation of model logic from training infrastructure.
2.  **Hardware v8.0**: Leverages `get_hardware_context` for auto-TF32/BF16.
3.  **Data v2.0**: Deploys `ICUGeneralistDataModule` with Tiered Acquisition.
4.  **Observability**: Integrated WandB + CSV logging with System Telemetry.
5.  **Safety**: Graceful shutdown, checkpoint recovery, DDP-safe calibration.

Training Features:
1.  **Full Observability**: Logs Generative (MSE/MAE) and Clinical (AUROC/ECE).
2.  **Critic Priming**: Pre-trains the Value Head on outcome signals.
3.  **DDP-Safe Calibration**: Computes physiological bounds without race conditions.
4.  **Hardware Hygiene**: Uses TieredEMA (CPU offload) and Atomic Checkpointing.
5.  **EMA Shadow Sync**: Ensures calibrated normalizer is propagated to EMA.
6.  **OneCycle LR**: Robust learning rate scheduling for training from scratch.

Usage:
    python icu/train/train_generalist.py experiment=phase1_frontier

Authors: APEX Research Team
Version: 8.0 (Ultimate - Production Grade)
"""

from __future__ import annotations

import sys
import os
import logging
import traceback
import collections
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    RichProgressBar,
    DeviceStatsMonitor,
    EarlyStopping
)

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
logger = logging.getLogger("APEX_Phase1_Engine")
# [FIX] PyTorch 2.6 Security: Bypass strict mode for trusted checkpoints
# monkey-patch torch.load to always use weights_only=False
_original_load = torch.load
def strict_mode_bypass_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_mode_bypass_load
logger.warning("[SECURITY] PyTorch 2.6+ strict mode disabled for checkpoint loading (Monkey-Patch active).")

# Add project root to path for local imports
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Project Imports ---
from icu.models.wrapper_generalist import ICUGeneralistWrapper
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn, ensure_data_ready
from icu.utils.callbacks import get_sota_callbacks, EMACallback
from icu.utils.train_utils import (
    set_seed,
    get_hardware_context,
    print_apex_branding,
    get_rank,
    is_main_process,
    count_parameters,
    format_parameters,
    SurgicalCheckpointLoader
)

# Initialize Script-Level Logger



# =============================================================================
# 1. ROBUST CHECKPOINT LOADER
# =============================================================================

def load_checkpoint_robust(
    system: pl.LightningModule,
    ckpt_path: str,
    trainer: pl.Trainer
) -> Optional[str]:
    """
    Surgically inspects and loads a checkpoint, bypassing PL's faulty migration
    if the 'pytorch-lightning_version' key is missing.
    """
    if not ckpt_path:
        return None
        
    logger.info(f"[RESUME] Inspecting checkpoint: {ckpt_path}")
    
    try:
        # Load metadata only to check keys
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Check for PyTorch Lightning metadata
        if "pytorch-lightning_version" not in checkpoint:
            logger.warning("[RESUME] Detected Incomplete/SOTA Checkpoint. Transitioning to Manual Restoration Suite...")
            
            # 1. Restore Model Weights
            # SurgicalCheckpointLoader handles prefix stripping and shape validation
            SurgicalCheckpointLoader.load_model(system.model, ckpt_path)
            logger.info("[RESUME] Model weights restored manually.")
            
            # 2. Restore EMA state if callback exists
            ema_cb = None
            for cb in trainer.callbacks:
                if isinstance(cb, EMACallback):
                    ema_cb = cb
                    break
            
            if ema_cb and "ema_state_dict" in checkpoint:
                # Force-load EMA state dict
                ema_cb.on_load_checkpoint(trainer, system, checkpoint)
                logger.info("[RESUME] EMA weights restored manually.")
            elif hasattr(system, 'ema') and system.ema is not None:
                # [CRITICAL FIX] "Random Teacher" Prevention
                # If we loaded the model but have no EMA state, the Teacher is still random.
                # We must force-sync it to the Student to start with valid targets.
                logger.warning("[RESUME] EMA state missing from checkpoint. Force-syncing Teacher (EMA) to Student (Model)...")
                system.ema._register(system.model)
                logger.info("[RESUME] EMA shadow weights re-initialized from loaded model.")
                
            # 3. Return None to trainer.fit to prevent it from trying to migrate
            # The weights are already in 'system', so a "fresh" PL run will use them.
            return None
            
        return ckpt_path
        
    except Exception as e:
        logger.error(f"[RESUME] Checkpoint inspection failed: {e}. Falling back to default loader.")
        return ckpt_path


# =============================================================================
# 2. ROBUST DATAMODULE
# =============================================================================

class ICUGeneralistDataModule(pl.LightningDataModule):
    """
    DDP-Safe Data Orchestrator.
    
    Decouples data setup from model logic for clean re-instantiation.
    Supports both single-GPU and distributed training.
    
    Features:
    - Tiered Acquisition: Downloads from HuggingFace if local data missing
    - Rank-0 Guarded: Only main process handles downloads
    - Prefetch Optimization: Configurable prefetching for speed
    - Schema Validation: Ensures data integrity before training
    
    Attributes:
        cfg: Hydra configuration
        pin_memory: Whether to pin memory for GPU transfer
        train_ds: Training dataset (set in setup())
        val_ds: Validation dataset (set in setup())
    """
    
    def __init__(self, cfg: DictConfig, pin_memory: bool = True):
        """
        Initialize the DataModule.
        
        Args:
            cfg: Hydra configuration with dataset and model settings
            pin_memory: Enable memory pinning for faster GPU transfers
        """
        super().__init__()
        self.cfg = cfg
        self.pin_memory = pin_memory
        self.train_ds: Optional[ICUSotaDataset] = None
        self.val_ds: Optional[ICUSotaDataset] = None

    def prepare_data(self):
        """
        Called only on Rank 0 (Download/Unzip).
        Ensures the 'Tiered Acquisition' pipeline has run.
        """
        if is_main_process():
            ensure_data_ready(
                dataset_dir=self.cfg.dataset.dataset_dir,
                hf_repo_id=self.cfg.dataset.get("hf_repo", None),
                force_download=self.cfg.dataset.get("force_download", False)
            )

    def setup(self, stage: str = None):
        """
        Called on every GPU (Instantiate Datasets).
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Training Set (with Augmentation)
            self.train_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="train",
                history_len=self.cfg.model.history_len,
                pred_len=self.cfg.model.pred_len,
                augment_noise=self.cfg.dataset.get("augment_noise", 0.005),
                augment_mask_prob=self.cfg.dataset.get("augment_mask_prob", 0.0),
                validate_schema=True
            )
            
            # Validation Set (Clean - No Augmentation)
            self.val_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="val",
                history_len=self.cfg.model.history_len,
                pred_len=self.cfg.model.pred_len,
                augment_noise=0.0,
                augment_mask_prob=0.0,
                validate_schema=True
            )
            
            logger.info(
                f"[Rank {get_rank()}] Datasets Ready: "
                f"Train={len(self.train_ds)}, Val={len(self.val_ds)}"
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        from icu.utils.samplers import EpisodeAwareSampler

        num_workers = self.cfg.train.num_workers
        
        # [SOTA] Use EpisodeAwareSampler to prevent LRU Cache Thrashing
        # This keeps 'shuffle' behavior (random episodes) but sequential frames.
        sampler = EpisodeAwareSampler(
            self.train_ds, 
            shuffle=True, 
            seed=self.cfg.seed,
            drop_last=True
        )
        
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            sampler=sampler,
            shuffle=False, # Sampler takes control
            num_workers=num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            # [PERF] Prefetch only if workers exist to avoid PyTorch warning
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False # Sampler handles dropping logic if needed, but usually redundant with sampler
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        num_workers = self.cfg.train.num_workers
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )


# =============================================================================
# 2. CALLBACK FACTORY
# =============================================================================

# =============================================================================
# 2. CALLBACK FACTORY
# =============================================================================

# =============================================================================
# 2. CALLBACK FACTORY (Legacy removed - Using icu.utils.callbacks)
# =============================================================================
# See main() where get_sota_callbacks is used.


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

@hydra.main(version_base=None, config_path="../../conf", config_name="generalist")
def main(cfg: DictConfig):
    """
    Main entry point for Phase 1 Generalist Training.
    
    Execution Flow:
    1. Boot Sequence (Branding, Hardware Detection, Seed)
    2. Data Pipeline (DataModule)
    3. System Architecture (Wrapper)
    4. Telemetry (Loggers)
    5. Trainer Configuration
    6. Launch Training
    
    Args:
        cfg: Hydra configuration
    """
    # =========================================================================
    # 0. BOOT SEQUENCE
    # =========================================================================
    print_apex_branding()
    
    # Resolve relative paths for Hydra
    if not os.path.isabs(cfg.dataset.dataset_dir):
        cfg.dataset.dataset_dir = os.path.abspath(os.path.join(
            hydra.utils.get_original_cwd(), cfg.dataset.dataset_dir
        ))
    
    # Hardware detection (TF32/BF16/FlashAttn checks)
    hw_ctx = get_hardware_context()
    logger.info(f"Hardware Context: {hw_ctx}")

    # --- Path Resolution & Creation ---
    # Ensure all distinct directories exist
    if cfg.get("checkpoint_dir"):
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    if cfg.get("log_dir"):
        os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.get("backup_dir"):
        os.makedirs(cfg.backup_dir, exist_ok=True)


    # Determinism
    set_seed(cfg.seed)
    
    # =========================================================================
    # 1. DATA PIPELINE
    # =========================================================================
    datamodule = ICUGeneralistDataModule(cfg, pin_memory=hw_ctx["pin_memory"])
    
    # =========================================================================
    # 2. SYSTEM ARCHITECTURE
    # =========================================================================
    # Instantiates the wrapper which contains:
    # - ICUUnifiedPlanner (DiT Backbone)
    # - ICUAdvantageCalculator (AWR)
    # - ClinicalNormalizer (Physics-Aware)
    # - EMA Teacher (Stabilization)
    system = ICUGeneralistWrapper(cfg)
    
    # Log model statistics
    if is_main_process():
        trainable, total = count_parameters(system)
        logger.info(
            f"[MODEL] Parameters: Trainable={format_parameters(trainable)}, "
            f"Total={format_parameters(total)}"
        )
    
    # =========================================================================
    # 3. TELEMETRY (LOGGERS)
    # =========================================================================
    # [FIX] Respect configured log_dir instead of generic output_dir
    log_save_dir = cfg.get("log_dir", cfg.output_dir)
    loggers = [CSVLogger(save_dir=log_save_dir, name="csv_logs")]
    
    if cfg.logging.use_wandb:
        # Handle offline mode override
        if cfg.logging.get("wandb_mode") == "offline":
            os.environ["WANDB_MODE"] = "offline"
            
        loggers.append(WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=log_save_dir,
            log_model=False  # We use our own checkpointing
        ))
        
    # =========================================================================
    # 4. TRAINER CONFIGURATION
    # =========================================================================
    # CRITICAL: We disable 'sanity_val_steps' to allow on_fit_start() to run
    # FIRST. This ensures the Normalizer is calibrated before any validation
    # batches are processed, preventing "Uncalibrated Normalizer" warnings.
    
    strategy = hw_ctx["strategy"]
    if strategy == "ddp":
        # [ROBUSTNESS] find_unused_parameters=True required for AuxHead handling
        # independent of loss weighting. Prevents DDP crash on unused subgraphs.
        strategy = DDPStrategy(find_unused_parameters=True)
    
    # Configure Callbacks (SOTA)
    # Configure Callbacks (SOTA)
    # [FIX] Filter out EMACallback because ICUGeneralistWrapper manages EMA manually
    # for Teacher-Student distillation logic. Double-update avoidance.
    # [FIX] Inject explicit ModelCheckpoint path
    raw_callbacks = get_sota_callbacks(cfg)
    
    # Filter callbacks if necessary (but KEEP TieredEMACallback if use_teacher is enabled)
    callbacks = []
    for cb in raw_callbacks:
        # If use_teacher is enabled, we expect TieredEMACallback to be present and we should keep it.
        # Otherwise, if it's a generic EMACallback and use_teacher is NOT enabled, we filter it out
        # to avoid double-updating if the wrapper handles EMA manually.
        if isinstance(cb, EMACallback) and not cfg.model.get("use_teacher", False):
            logger.info(f"[CALLBACKS] Filtering out {type(cb).__name__} as use_teacher is False or wrapper handles EMA.")
            continue
        
        # If it's a TieredEMACallback and use_teacher is enabled, we keep it.
        # If it's any other callback, we keep it.
        callbacks.append(cb)

    # Ensure ModelCheckpoint callback uses the correct dirpath if present
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            # If checkpoint_dir is set in config, override likely defaults from get_sota_callbacks
            if cfg.get("checkpoint_dir"):
                cb.dirpath = cfg.checkpoint_dir
                logger.info(f"[CONFIG] Checkpoint Dir overridden to: {cb.dirpath}")

    class EMARestoration(pl.Callback):
        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            if "ema_state_dict" in checkpoint and hasattr(pl_module, 'ema'):
                logger.info(f"[RESUME] Found EMA state in checkpoint. Restoring to {pl_module.ema.decay} decay...")
                # Force CPU load to ensure TieredEMA doesn't spike VRAM
                safe_state = {k: v.cpu() for k, v in checkpoint["ema_state_dict"].items()}
                pl_module.ema.load_state_dict(safe_state)
                
    callbacks.append(EMARestoration())
            
    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        max_epochs=cfg.train.epochs,
        accelerator=hw_ctx["accelerator"],
        devices=hw_ctx["devices"],
        strategy=strategy,
        precision=hw_ctx["precision"],
        callbacks=callbacks,
        logger=loggers,
        # [SOTA PERFORMANCE] Disabled automatic clipping to allow 'fused' AdamW
        # Handled manually in Wrapper.on_before_optimizer_step()
        gradient_clip_val=0,
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 10),
        enable_checkpointing=True,
        num_sanity_val_steps=0,  # [FIX] Allow calibration first
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        val_check_interval=cfg.train.get("val_check_interval", 1.0),
        # [PERF] Limit validation batches for speed during experimentation
        limit_val_batches=cfg.get("debug_limit_val", 20) if cfg.get("debug", False) else cfg.train.get("limit_val_batches", 1.0),
        # [DEBUG] Limit training batches for rapid smoke test
        limit_train_batches=cfg.get("debug_limit_batches", 200) if cfg.get("debug", False) else 1.0,
        # [ROBUSTNESS] Enable model summary
        enable_model_summary=True,
        # [DDP] Sync batch norm for distributed training
        sync_batchnorm=True if hw_ctx["devices"] > 1 else False
    )
    
    # =========================================================================
    # 5. LAUNCH TRAINING
    # =========================================================================
    logger.info("="*80)
    logger.info(f"[LAUNCH] Starting Phase 1 Generalist Training")
    logger.info(f"[CONFIG] Output Dir: {cfg.output_dir}")
    logger.info(f"[CONFIG] Epochs: {cfg.train.epochs}")
    logger.info(f"[CONFIG] Batch Size: {cfg.train.batch_size}")
    logger.info(f"[CONFIG] Learning Rate: {cfg.train.lr}")
    logger.info("="*80)
    
    try:
        # Check for resume checkpoint
        ckpt_path = cfg.get("resume_from", None)
        if ckpt_path and not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint {ckpt_path} not found. Starting fresh.")
            ckpt_path = None
        elif ckpt_path:
            # [FIX: Robust Resumption] Use surgical loader for SOTA/Legacy checkpoints
            ckpt_path = load_checkpoint_robust(system, ckpt_path, trainer)
            if ckpt_path:
                logger.info(f"[RESUME] Resuming from checkpoint via Lightning: {ckpt_path}")
            else:
                logger.info("[RESUME] Resuming via Manual Restoration Suite (Weights Only).")
            
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
        
        # Post-training summary
        if is_main_process():
            logger.info("="*80)
            logger.info("[SUCCESS] Phase 1 Training Complete!")
            logger.info(f"[BEST MODEL] {trainer.checkpoint_callback.best_model_path}")
            logger.info(f"[BEST SCORE] val/sepsis_auroc = {trainer.checkpoint_callback.best_model_score:.4f}")
            
            # [BACKUP] Copy best model to backup_dir if configured
            if cfg.get("backup_dir") and trainer.checkpoint_callback.best_model_path:
                import shutil
                best_path = Path(trainer.checkpoint_callback.best_model_path)
                backup_path = Path(cfg.backup_dir) / best_path.name
                try:
                    shutil.copy2(best_path, backup_path)
                    logger.info(f"[BACKUP] Successfully backed up best model to: {backup_path}")
                except Exception as e:
                    logger.error(f"[BACKUP] Failed to backup model: {e}")

            logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("[STOP] Training interrupted by user. Saving state...")
        # PyTorch Lightning handles graceful shutdown automatically
        
    except Exception as e:
        logger.critical(f"[CRITICAL] Training crashed: {e}")
        logger.critical(traceback.format_exc())
        raise e
        
    finally:
        logger.info("[DONE] Phase 1 Execution Finished.")


# =============================================================================
# 4. STANDALONE HELPER: PHYSICS CALIBRATION
# =============================================================================

def calibrate_physics_standalone(cfg: DictConfig, model: nn.Module):
    """
    Computes global stats on Rank 0 and calibrates the normalizer.
    
    This is a standalone function for use outside of Lightning training.
    During normal training, calibration is handled by on_fit_start() in the wrapper.
    
    Args:
        cfg: Hydra configuration
        model: Model with normalizer to calibrate
    """
    if not is_main_process():
        return

    logger.info("[PHYSICS] Calibrating Normalizer on Rank 0...")
    
    # Init temporary dataset just for stats
    ds = ICUSotaDataset(
        dataset_dir=cfg.dataset.dataset_dir,
        split="train",
        augment_noise=0.0
    )
    
    # Get channel count from first sample
    ts_channels = ds[0]["observed_data"].shape[-1]
    g_min = torch.full((ts_channels,), float('inf'))
    g_max = torch.full((ts_channels,), float('-inf'))
    
    # Use DataLoader for speed
    loader = DataLoader(ds, batch_size=2048, num_workers=4, collate_fn=robust_collate_fn)
    
    for batch in tqdm(loader, desc="Calibration Scan"):
        vitals = torch.cat([batch["observed_data"], batch["future_data"]], dim=1)
        flat = vitals.view(-1, ts_channels)
        
        g_min = torch.minimum(g_min, flat.min(dim=0).values)
        g_max = torch.maximum(g_max, flat.max(dim=0).values)
        
    logger.info(f"Calibration Complete. Range: [{g_min.min().item():.2f}, {g_max.max().item():.2f}]")
    
    # Inject into model buffers
    if hasattr(model, 'normalizer'):
        model.normalizer.ts_min.copy_(g_min)
        model.normalizer.ts_max.copy_(g_max)
        model.normalizer.is_calibrated.fill_(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()