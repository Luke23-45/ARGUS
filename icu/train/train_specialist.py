"""
scripts/train_specialist.py
--------------------------------------------------------------------------------
Phase 2: Specialist (APEX-MoE) Training Engine (SOTA v4.0).

Status: PRODUCTION-READY / SAFETY-CRITICAL

This script orchestrates the "Specialization" phase, transforming the Generalist
into a Phase-Locked Mixture-of-Experts (MoE) via Surgical Bootstrapping.

Architectural Pillars:
1.  **Surgical Bootstrapping**: Automatically loads Phase 1 weights via the 
    `ICUSpecialistWrapper`, freezing Perception and cloning Experts.
2.  **Gradient Surgery**: The LightningModule (Wrapper) handles the routing 
    of gradients to specific experts based on ground-truth phase labels.
3.  **DDP-Safe MOE**: Configures DistributedDataParallel to handle frozen 
    sub-modules (Perception/Router) without deadlocking.
4.  **AWR Integration**: Prepares the environment for Advantage-Weighted 
    Regression calculations synced across ranks.

Usage:
    python scripts/train_specialist.py experiment=phase2_frontier
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Project Imports ---
from icu.models.wrapper_apex import ICUSpecialistWrapper
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn, ensure_data_ready
from icu.utils.train_utils import set_seed, rank_zero_only, get_rank
from icu.utils.callbacks import get_sota_callbacks

logger = logging.getLogger("Phase2_Specialist")

# ==============================================================================
# 1. LIGHTNING DATAMODULE (DDP-Safe)
# ==============================================================================

class ICUSpecialistDataModule(pl.LightningDataModule):
    """
    Robust DataModule ensuring DDP-safe setup and referencing.
    Required for the 'on_fit_start' hooks in the SpecialistWrapper to find 
    the dataset for AWR statistics calculation.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[ICUSotaDataset] = None
        self.val_ds: Optional[ICUSotaDataset] = None

    def prepare_data(self):
        """Called only on Rank 0. Download/Cache data here."""
        ensure_data_ready(
            dataset_dir=self.cfg.dataset.dataset_dir,
            hf_repo_id=self.cfg.dataset.hf_repo,
            force_download=self.cfg.dataset.get("force_download", False)
        )

    def setup(self, stage: str = None):
        """Called on every GPU. Instantiate datasets here."""
        if stage == 'fit' or stage is None:
            # Phase 2 requires strict schema validation
            self.train_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="train",
                augment_noise=self.cfg.dataset.augment_noise,
                validate_schema=True
            )
            self.val_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="val",
                augment_noise=0.0
            )
            logger.info(f"[Rank {get_rank()}] Specialist Data Setup: Train={len(self.train_ds)}, Val={len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=True,
            persistent_workers=True if self.cfg.train.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=True
        )

# ==============================================================================
# 2. HYDRA ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="../../conf", config_name="specialist")
def main(cfg: DictConfig):
    # 1. Determinism (SOTA Replicability)
    set_seed(cfg.seed)
    
    # 2. System Init
    # The ICUSpecialistWrapper handles the "Surgical Bootstrapping" internally.
    logger.info("⚡ [SYSTEM] Initializing APEX-MoE Specialist System...")
    
    # Critical Check: Phase 1 Checkpoint must exist
    pretrained_path = cfg.train.get("pretrained_path")
    if not pretrained_path:
        raise ValueError("CRITICAL: Phase 2 config requires 'train.pretrained_path'.")
    if not Path(pretrained_path).exists():
        raise FileNotFoundError(f"CRITICAL: Phase 1 Checkpoint not found at: {pretrained_path}")

    system = ICUSpecialistWrapper(cfg)
    
    # 3. Data Init
    datamodule = ICUSpecialistDataModule(cfg)
    
    # 4. Logger Init
    loggers = []
    # CSV Logger (Local Backup)
    loggers.append(CSVLogger(save_dir=cfg.output_dir, name="specialist_logs"))
    
    # WandB Logger (Cloud Telemetry)
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=cfg.output_dir,
            group="Phase2_Specialist"
        )
        loggers.append(wandb_logger)
        
    # 5. Callback Init (AtomicSaver, RichProgressBar, AnomalyGuardian)
    callbacks = get_sota_callbacks(cfg)
    
    # 6. DDP Strategy Configuration
    # SOTA v3.0: Phase 2 freezes large parts of the model (Encoder/Router).
    # 'find_unused_parameters=True' is MANDATORY to prevent DDP deadlock 
    # when gradients are not generated for frozen parameters.
    ddp_strategy = "auto"
    if torch.cuda.device_count() > 1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)

    # 7. Trainer Init
    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy=ddp_strategy,
        precision=cfg.train.get("precision", "16-mixed"),
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=10,
        enable_checkpointing=False, # Handled exclusively by RotationalSaverCallback
        num_sanity_val_steps=2 
    )
    
    # 8. Execution
    logger.info("🚀 [LAUNCH] Phase 2 Specialist Training")
    logger.info(f"   Bootstrapping from: {pretrained_path}")
    logger.info(f"   Crash Weight: {cfg.train.get('crash_weight', 'N/A')}")
    logger.info(f"   Reg Weight:   {cfg.train.get('lambda_reg', 'N/A')}")
    
    # Check for Resume (Mid-Phase 2 resumption)
    # This is different from "pretrained_path" (Phase 1). 
    # This checks if Phase 2 crashed and needs to continue.
    ckpt_path = cfg.train.get("resume_from_checkpoint", None)
    if ckpt_path:
        if Path(ckpt_path).exists():
            logger.info(f"   Resuming Specialist Training from: {ckpt_path}")
        else:
            logger.warning(f"   Resume checkpoint {ckpt_path} not found. Starting Phase 2 fresh.")
            ckpt_path = None
    
    # fit() triggers the Wrapper's on_fit_start()
    # This handles AWR stats sync and Normalizer calibration across ranks.
    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    
    logger.info("✅ [DONE] Phase 2 Training Complete.")

if __name__ == "__main__":
    main()