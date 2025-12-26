"""
scripts/train_generalist.py
--------------------------------------------------------------------------------
Phase 1: Generalist Training Engine (SOTA v4.0).

Status: PRODUCTION-READY / SAFETY-CRITICAL

This script establishes the "Shared Latent Space" and pre-trains the Sepsis Router.
It drives the 'ICUUnifiedPlanner' backbone using the Guardian Suite for reliability.

Features:
1.  **Full Observability**: Logs both Generative (MSE/MAE) and Clinical (AUROC/ECE)
    metrics during validation using EMA weights.
2.  **Critic Priming**: Pre-trains the Value Head on outcome signals, ensuring
    the Phase 2 Critic starts with a valid representation of patient severity.
3.  **DDP-Safe Calibration**: Computes physiological bounds without race conditions.
4.  **Hardware Hygiene**: Uses TieredEMA (CPU offload) and Atomic Checkpointing.

Usage:
    python scripts/train_generalist.py experiment=phase1_frontier
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Project Imports ---
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn, ensure_data_ready
from icu.utils.train_utils import (
    configure_robust_optimizer,
    rank_zero_only, 
    set_seed,
    is_main_process,
    get_rank,
    get_hardware_context
)
from icu.utils.callbacks import get_sota_callbacks

logger = logging.getLogger("Phase1_Generalist")

# ==============================================================================
# 1. LIGHTNING DATAMODULE
# ==============================================================================

class ICUGeneralistDataModule(pl.LightningDataModule):
    """
    DDP-Safe Data Orchestrator. 
    Decouples data setup from model logic for clean re-instantiation.
    """
    def __init__(self, cfg: DictConfig, pin_memory: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pin_memory = pin_memory
        self.train_ds: Optional[ICUSotaDataset] = None
        self.val_ds: Optional[ICUSotaDataset] = None

    def prepare_data(self):
        """Called only on Rank 0 (Download/Unzip)."""
        ensure_data_ready(
            dataset_dir=self.cfg.dataset.dataset_dir,
            hf_repo_id=self.cfg.dataset.hf_repo,
            force_download=self.cfg.dataset.get("force_download", False)
        )

    def setup(self, stage: str = None):
        """Called on every GPU (Instantiate Datasets)."""
        if stage == 'fit' or stage is None:
            self.train_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="train",
                augment_noise=self.cfg.dataset.augment_noise,
                validate_schema=True
            )
            self.val_ds = ICUSotaDataset(
                dataset_dir=self.cfg.dataset.dataset_dir,
                split="val",
                augment_noise=0.0 # No noise in validation
            )
            logger.info(f"[Rank {get_rank()}] Dataset Setup: Train={len(self.train_ds)}, Val={len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.cfg.train.num_workers > 0 else False,
            prefetch_factor=2 if self.cfg.train.num_workers > 0 else None  # [FIX] Prefetch for speed
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=robust_collate_fn,
            pin_memory=self.pin_memory,
            prefetch_factor=2 if self.cfg.train.num_workers > 0 else None  # [FIX] Prefetch for speed
        )

# ==============================================================================
# 2. LIGHTNING MODULE (The Brain)
# ==============================================================================

class ICUGeneralistModule(pl.LightningModule):
    """
    Phase 1 Engine.
    
    Training Objectives:
    1.  **Diffusion Loss**: MSE(Noise) for future trajectory generation.
    2.  **Auxiliary Loss**: CrossEntropy for Sepsis Classification (Pre-training the Router).
    3.  **Value Loss**: MSE for Outcome Prediction (Pre-training the Critic for Phase 2).
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # 1. Model Initialization
        model_cfg = ICUConfig(**cfg.model)
        self.model = ICUUnifiedPlanner(model_cfg)
        
        self.ema = None 
        self.saver_callback = None

        # 3. Aux Head Weighting (for 3.1% imbalance)
        # Weight Vector: [Stable, Pre-Shock, Shock]
        pos_weight = self.cfg.train.get("pos_weight", 1.0)
        self.register_buffer("aux_weight", torch.tensor([1.0, pos_weight, pos_weight], dtype=torch.float32))

    def on_fit_start(self):
        """
        CRITICAL FIX: DDP-Safe Normalizer Calibration + EMA Sync.
        
        This runs AFTER DDP initialization, so all ranks execute this code.
        We use calibrate_from_stats() which reads from the dataset's index file,
        ensuring deterministic calibration across all ranks.
        
        PATCH v4.1: Force-sync EMA shadow after calibration to prevent
        validation from using uncalibrated normalizer buffers.
        """
        if not hasattr(self.trainer, "datamodule") or not self.trainer.datamodule:
            logger.warning("No DataModule. Normalizer uncalibrated.")
            return
            
        try:
            dataset = self.trainer.datamodule.train_dataloader().dataset
            index_path = getattr(dataset, "index_path", None)
            metadata = getattr(dataset, "metadata", {})
            ts_cols = metadata.get("ts_columns", [])
            
            if index_path and ts_cols:
                logger.info(f"[Rank {get_rank()}] Calibrating normalizer from {index_path}")
                self.model.normalizer.calibrate_from_stats(index_path, ts_cols)
                
                # [CRITICAL FIX] EMA Shadow Sync
                # The EMA callback initializes shadow weights BEFORE this calibration runs.
                # We must force-update the shadow's normalizer buffers to match the calibrated state.
                if hasattr(self, 'ema') and self.ema is not None:
                    logger.info(f"[Rank {get_rank()}] Syncing EMA shadow with calibrated normalizer...")
                    # Update only the normalizer buffers in the shadow
                    for name, buffer in self.model.normalizer.named_buffers():
                        full_name = f"normalizer.{name}"
                        if full_name in self.ema.shadow:
                            self.ema.shadow[full_name] = buffer.data.detach().cpu().clone()
                    logger.info(f"[Rank {get_rank()}] EMA shadow sync complete.")
            else:
                logger.warning(f"[Rank {get_rank()}] Missing index_path or ts_columns. Normalizer uncalibrated.")
        except Exception as e:
            logger.error(f"[Rank {get_rank()}] Calibration failed: {e}. Proceeding uncalibrated.")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Multitask Training Step.
        Returns combined loss for backprop.
        """
        # 1. Forward Pass (Diffusion + Aux Head inside model)
        # Model returns {'loss', 'diffusion_loss', 'aux_loss', 'pred_value'}
        # Inject the class weights for the auxiliary head
        batch["aux_weight"] = self.aux_weight
        out = self.model(batch)
        
        diff_loss = out["diffusion_loss"]
        aux_loss = out["aux_loss"]
        
        # 2. Value Function Priming (Pre-training for AWR)
        # We define a proxy reward for Phase 1: The ground truth binary label.
        # This helps the Critic learn "Being sick is high value/low value" logic early.
        value_loss = torch.tensor(0.0, device=self.device)
        
        if "outcome_label" in batch:
            pred_value = out.get("pred_value") # Shape [B, T_pred] or [B, 1] depending on head
            target_value = batch["outcome_label"].float()
            
            # Broadcast scalar outcome to trajectory if needed
            if pred_value is not None:
                if pred_value.dim() > target_value.dim(): # [B, T] vs [B]
                    target_seq = target_value.unsqueeze(1).expand_as(pred_value)
                    value_loss = F.mse_loss(pred_value, target_seq)
                else:
                    value_loss = F.mse_loss(pred_value, target_value)

        # 3. Aggregation
        # Weights: Diffusion (1.0) + Router (0.1) + Critic (0.1)
        # Critic is kept low to not interfere with representation learning
        total_loss = diff_loss + 0.1 * aux_loss + 0.1 * value_loss
        
        # 4. Telemetry (batch_size explicit to silence PL warning)
        # prog_bar=True for essential metrics to be visible during training
        B = batch["observed_data"].shape[0]
        self.log("train/loss", total_loss, prog_bar=True, batch_size=B)
        self.log("train/diff_loss", diff_loss, prog_bar=True, batch_size=B)  # [FIX] Show in progress bar
        self.log("train/aux_loss", aux_loss, prog_bar=True, batch_size=B)    # [FIX] Show in progress bar
        self.log("train/value_loss", value_loss, batch_size=B)
        
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Dual-Validation: Clinical Metrics + Generative Quality.
        Executed under EMA context (via Callback).
        """
        # A. Clinical Validation (The Router)
        # We manually run the Encoder + AuxHead path to get logits
        # This fixes "Metric Blindness" by feeding the ClinicalMetricCallback
        past_norm, static_norm = self.model.normalize(batch["observed_data"], batch["static_context"])
        _, global_ctx, _ = self.model.encoder(past_norm, static_norm, batch.get("src_mask"))
        logits = self.model.aux_head(global_ctx) # Sepsis logits
        
        # B. Generative Validation (The Vitals)
        if batch_idx < 10: # Only sample a few batches to save time
            self._run_generative_sampling(batch)

        # C. Return Payload for Callbacks
        return {
            "preds": logits,
            "target": batch.get("outcome_label", torch.tensor([])) 
        }

    def _run_generative_sampling(self, batch: Dict[str, torch.Tensor]):
        """Helper for autoregressive vitals generation."""
        sample_steps = self.cfg.train.get("val_sample_steps", 50)
        pred_future = self.model.sample(batch, num_steps=sample_steps)
        gt_future = batch["future_data"]
        
        mse = F.mse_loss(pred_future, gt_future)
        mae = F.l1_loss(pred_future, gt_future)
        
        B = batch["observed_data"].shape[0]
        self.log("val/generative_mse", mse, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/generative_mae", mae, on_epoch=True, sync_dist=True, batch_size=B)

    def configure_optimizers(self):
        """SOTA Optimizer Configuration."""
        optimizer = configure_robust_optimizer(
            self.model,
            learning_rate=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            use_fused=True
        )
        
        # OneCycleLR is robust for training from scratch
        # Estimate steps
        if self.trainer.estimated_stepping_batches:
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            # Safe fallback approximation
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches
            total_steps = steps_per_epoch * self.cfg.train.epochs

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.train.lr,
            total_steps=total_steps,
            pct_start=self.cfg.train.get("warmup_ratio", 0.05),
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# ==============================================================================
# 3. HELPER: PHYSICS CALIBRATION
# ==============================================================================

def calibrate_physics(cfg: DictConfig, model: nn.Module):
    """
    Computes global stats on Rank 0 and saves them. 
    However, for DDP safety with Lightning, we let each process load 
    pre-computed stats if they exist, or rely on the Datamodule.
    
    Current strategy: Iterating the dataset is fast. 
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
    
    # 28 Channels (Frontier Spec)
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
        
    logger.info(f"Calibration Complete. Max Val: {g_max.max().item():.2f}")
    
    # Inject into model buffers
    # Note: In DDP, this happens on Rank 0 BEFORE .fit(), 
    # but the model is copied to other ranks. 
    # To be absolutely safe, we should save these to a file and load on all ranks,
    # OR we rely on Lightning's broadcast if using 'spawn'.
    # Given modern 'fork'/'spawn' nuance, we inject directly here.
    model.normalizer.ts_min.copy_(g_min)
    model.normalizer.ts_max.copy_(g_max)
    model.normalizer.is_calibrated.fill_(1)

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="../../conf", config_name="generalist")
def main(cfg: DictConfig):
    # 0. Hardware Detection & Optimization
    hw_ctx = get_hardware_context()
    logger.info(f"Hardware Context: {hw_ctx}")

    # 1. Determinism
    set_seed(cfg.seed)
    
    # 2. DataModule
    datamodule = ICUGeneralistDataModule(cfg, pin_memory=hw_ctx["pin_memory"])
    
    # 3. System
    system = ICUGeneralistModule(cfg)
    
    # 4. Physics (Pre-Flight)
    # NOTE: Calibration is now handled in ICUGeneralistModule.on_fit_start()
    # which runs AFTER DDP init, ensuring all ranks are synchronized.
    # The old calibrate_physics() function is kept for reference but not called.
        
    # 5. Callbacks & Loggers
    callbacks = get_sota_callbacks(cfg)
    
    loggers = [CSVLogger(save_dir=cfg.output_dir, name="csv_logs")]
    if cfg.logging.use_wandb:

        if cfg.logging.wandb_mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        loggers.append(WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=cfg.output_dir
        ))
        
    # 6. Trainer
    # NOTE: num_sanity_val_steps=0 is CRITICAL!
    # The sanity check runs BEFORE on_fit_start(), which means the Normalizer 
    # would not be calibrated yet, causing "Normalizer used without calibration" warnings.
    
    strategy = hw_ctx["strategy"]
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)
        
    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        max_epochs=cfg.train.epochs,
        accelerator=hw_ctx["accelerator"],
        devices=hw_ctx["devices"],
        strategy=strategy,
        precision=hw_ctx["precision"],
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=10,
        enable_checkpointing=False, # Handled by RotationalSaver
        num_sanity_val_steps=0, # [FIX] Disable sanity check to allow on_fit_start calibration first
    )
    
    # 7. Launch
    logger.info("[LAUNCH] Phase 1 Generalist Training")
    
    ckpt_path = cfg.train.get("resume_from_checkpoint", None)
    if ckpt_path and not Path(ckpt_path).exists():
        logger.warning(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
        ckpt_path = None
        
    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    
    logger.info("[DONE] Phase 1 Complete.")

if __name__ == "__main__":
    main()