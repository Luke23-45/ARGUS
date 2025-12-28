"""
icu/train/train_specialist.py
--------------------------------------------------------------------------------
APEX-MoE: Phase 2 Specialist Training Engine (Ultimate v8.0).

Status: PRODUCTION-READY / SAFETY-CRITICAL

This script orchestrates the "Specialization" phase, transforming the Generalist
into a Phase-Locked Mixture-of-Experts (MoE) via Surgical Bootstrapping.

"The Specialist is where clinical intelligence crystalizes into life-saving
predictions. Each expert becomes a guardian specialized in detecting a
specific phase of patient deterioration."

Architectural Pillars:
1.  **Surgical Bootstrapping**: Automatically loads Phase 1 weights via the 
    `ICUSpecialistWrapper`, freezing Perception and cloning Experts.
2.  **Gradient Surgery**: The LightningModule (Wrapper) handles the routing 
    of gradients to specific experts based on ground-truth phase labels.
3.  **DDP-Safe MoE**: Configures DistributedDataParallel to handle frozen 
    sub-modules (Perception/Router) without deadlocking.
4.  **AWR Integration**: Prepares the environment for Advantage-Weighted 
    Regression calculations synced across ranks.

Training Features:
1.  **Expert Specialization**: Each expert focuses on Stable/PreShock/Shock phases.
2.  **Cross-Expert Regularization**: Prevents mode collapse between experts.
3.  **Physics-Guided Sampling**: Uses physiological constraints during generation.
4.  **EMA Teacher Network**: Stabilizes training with exponential moving average.
5.  **Comprehensive Telemetry**: Tracks expert utilization, routing quality, etc.

Usage:
    python icu/train/train_specialist.py experiment=phase2_frontier

Authors: APEX Research Team
Version: 8.0 (Ultimate - Production Grade)
"""

from __future__ import annotations

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

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

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Project Imports ---
from icu.models.wrapper_apex import ICUSpecialistWrapper
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn, ensure_data_ready
from icu.utils.callbacks import get_sota_callbacks
from icu.utils.train_utils import (
    set_seed, 
    get_hardware_context,
    print_apex_branding,
    get_rank,
    is_main_process,
    count_parameters,
    format_parameters
)

# Initialize Script-Level Logger
logger = logging.getLogger("APEX_Phase2_Specialist")


# =============================================================================
# 1. LIGHTNING DATAMODULE (DDP-Safe)
# =============================================================================

class ICUSpecialistDataModule(pl.LightningDataModule):
    """
    Robust DataModule ensuring DDP-safe setup and referencing.
    
    Required for the 'on_fit_start' hooks in the SpecialistWrapper to find 
    the dataset for AWR statistics calculation.
    
    Features:
    - Tiered Acquisition: Downloads from HuggingFace if local data missing
    - Rank-0 Guarded: Only main process handles downloads
    - Schema Validation: Ensures data integrity for Phase 2
    - History/Pred Length: Configurable sequence lengths
    
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
            cfg: Hydra configuration
            pin_memory: Enable memory pinning for GPU transfers
        """
        super().__init__()
        self.cfg = cfg
        self.pin_memory = pin_memory
        self.train_ds: Optional[ICUSotaDataset] = None
        self.val_ds: Optional[ICUSotaDataset] = None

    def prepare_data(self):
        """
        Called only on Rank 0. Download/Cache data here.
        """
        if is_main_process():
            ensure_data_ready(
                dataset_dir=self.cfg.dataset.dataset_dir,
                hf_repo_id=self.cfg.dataset.get("hf_repo", None),
                force_download=self.cfg.dataset.get("force_download", False)
            )

    def setup(self, stage: str = None):
        """
        Called on every GPU. Instantiate datasets here.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Phase 2 requires strict schema validation
            self.train_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="train",
                history_len=self.cfg.model.history_len,
                pred_len=self.cfg.model.pred_len,
                augment_noise=self.cfg.dataset.get("augment_noise", 0.005),
                augment_mask_prob=self.cfg.dataset.get("augment_mask_prob", 0.0),
                validate_schema=True
            )
            
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
                f"[Rank {get_rank()}] Specialist Data Setup: "
                f"Train={len(self.train_ds)}, Val={len(self.val_ds)}"
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        num_workers = self.cfg.train.num_workers
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True  # Ensures consistent batch sizes for MoE routing
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

@hydra.main(version_base=None, config_path="../../conf", config_name="specialist")
def main(cfg: DictConfig):
    """
    Main entry point for Phase 2 Specialist Training.
    
    Execution Flow:
    1. Boot Sequence (Branding, Hardware Detection, Seed)
    2. Pretrained Checkpoint Validation
    3. System Architecture (Specialist Wrapper)
    4. Data Pipeline (DataModule)
    5. Telemetry (Loggers)
    6. Trainer Configuration
    7. Launch Training
    
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
    
    # Hardware detection
    hw_ctx = get_hardware_context()
    logger.info(f"Specialist Hardware Context: {hw_ctx}")

    # Determinism
    set_seed(cfg.seed)
    
    # =========================================================================
    # 1. PRETRAINED CHECKPOINT VALIDATION
    # =========================================================================
    logger.info("="*80)
    logger.info("[SYSTEM] Initializing APEX-MoE Specialist System...")
    
    pretrained_path = cfg.train.get("pretrained_path")
    if not pretrained_path:
        raise ValueError(
            "CRITICAL: Phase 2 config requires 'train.pretrained_path' "
            "pointing to a Phase 1 checkpoint."
        )
    
    # Resolve relative pretrained path
    if not os.path.isabs(pretrained_path):
        pretrained_path = os.path.abspath(os.path.join(
            hydra.utils.get_original_cwd(), pretrained_path
        ))
        
    if not Path(pretrained_path).exists():
        raise FileNotFoundError(
            f"CRITICAL: Phase 1 Checkpoint not found at: {pretrained_path}"
        )
    
    logger.info(f"[BOOTSTRAP] Phase 1 Checkpoint: {pretrained_path}")
    
    # =========================================================================
    # 2. SYSTEM ARCHITECTURE
    # =========================================================================
    system = ICUSpecialistWrapper(cfg)
    
    # Log model statistics
    if is_main_process():
        trainable, total = count_parameters(system)
        logger.info(
            f"[MODEL] Parameters: Trainable={format_parameters(trainable)}, "
            f"Total={format_parameters(total)}"
        )
        
        # Log frozen vs trainable experts
        if hasattr(system, 'model') and hasattr(system.model, 'experts'):
            num_experts = len(system.model.experts)
            logger.info(f"[MODEL] MoE Configuration: {num_experts} Experts")
    
    # =========================================================================
    # 3. DATA PIPELINE
    # =========================================================================
    datamodule = ICUSpecialistDataModule(cfg, pin_memory=hw_ctx["pin_memory"])
    
    # =========================================================================
    # 4. TELEMETRY (LOGGERS)
    # =========================================================================
    loggers = [CSVLogger(save_dir=cfg.output_dir, name="specialist_logs")]
    
    if cfg.logging.use_wandb:
        # Handle offline mode override
        if cfg.logging.get("wandb_mode") == "offline":
            os.environ["WANDB_MODE"] = "offline"
            
        loggers.append(WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=cfg.output_dir,
            group="Phase2_Specialist",
            log_model=False  # We use our own checkpointing
        ))
        
    # =========================================================================
    # 5. TRAINER CONFIGURATION
    # =========================================================================
    # DDP Strategy Configuration
    # IMPORTANT: find_unused_parameters=True is required for MoE because
    # not all experts receive gradients every batch (routing-dependent).
    strategy = hw_ctx["strategy"]
    if strategy == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=True,
            # Static graph optimization (if using same expert routing pattern)
            static_graph=False  # Set to True if routing is deterministic
        )

    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        max_epochs=cfg.train.epochs,
        accelerator=hw_ctx["accelerator"],
        devices=hw_ctx["devices"],
        strategy=strategy,
        precision=hw_ctx["precision"],
        callbacks=get_sota_callbacks(cfg),
        logger=loggers,
        # [SOTA PERFORMANCE] Disabled automatic clipping to allow 'fused' AdamW
        # Handled manually in ICUSpecialistWrapper.on_before_optimizer_step()
        gradient_clip_val=0,
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 10),
        enable_checkpointing=True,
        num_sanity_val_steps=0,  # [FIX] Allow calibration first
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        val_check_interval=cfg.train.get("val_check_interval", 1.0),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
        enable_model_summary=True,
        # [DDP] Sync batch norm for distributed MoE
        sync_batchnorm=True if hw_ctx["devices"] > 1 else False
    )
    
    # =========================================================================
    # 6. LAUNCH TRAINING
    # =========================================================================
    logger.info("="*80)
    logger.info("[LAUNCH] Starting Phase 2 Specialist Training")
    logger.info(f"[CONFIG] Output Dir: {cfg.output_dir}")
    logger.info(f"[CONFIG] Epochs: {cfg.train.epochs}")
    logger.info(f"[CONFIG] Batch Size: {cfg.train.batch_size}")
    logger.info(f"[CONFIG] Learning Rate: {cfg.train.lr}")
    logger.info(f"[CONFIG] Bootstrapping from: {pretrained_path}")
    logger.info(f"[CONFIG] Crash Weight: {cfg.train.get('crash_weight', 'N/A')}")
    logger.info(f"[CONFIG] Reg Weight: {cfg.train.get('lambda_reg', 'N/A')}")
    logger.info("="*80)
    
    try:
        # Check for Resume (Mid-Phase 2 resumption)
        # This is different from "pretrained_path" (Phase 1 source).
        # This checks if Phase 2 crashed and needs to continue.
        ckpt_path = cfg.train.get("resume_from_checkpoint", None)
        if ckpt_path:
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.abspath(os.path.join(
                    hydra.utils.get_original_cwd(), ckpt_path
                ))
            if Path(ckpt_path).exists():
                logger.info(f"[RESUME] Resuming Specialist Training from: {ckpt_path}")
            else:
                logger.warning(
                    f"[RESUME] Checkpoint {ckpt_path} not found. Starting Phase 2 fresh."
                )
                ckpt_path = None
        
        # fit() triggers the Wrapper's on_fit_start()
        # This handles AWR stats sync and Normalizer calibration across ranks.
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
        
        # Post-training summary
        if is_main_process():
            logger.info("="*80)
            logger.info("[SUCCESS] Phase 2 Specialist Training Complete!")
            if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                logger.info(f"[BEST MODEL] {trainer.checkpoint_callback.best_model_path}")
                if trainer.checkpoint_callback.best_model_score is not None:
                    logger.info(
                        f"[BEST SCORE] val/loss = {trainer.checkpoint_callback.best_model_score:.4f}"
                    )
            logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("[STOP] Training interrupted by user. Saving state...")
        # PyTorch Lightning handles graceful shutdown automatically
        
    except Exception as e:
        logger.critical(f"[CRITICAL] Training crashed: {e}")
        logger.critical(traceback.format_exc())
        raise e
        
    finally:
        logger.info("[DONE] Phase 2 Execution Finished.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
