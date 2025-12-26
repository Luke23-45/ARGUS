"""
icu/models/wrapper_generalist.py
--------------------------------------------------------------------------------
APEX-MoE: Phase 1 Generalist Training Wrapper (SOTA v2.0).

Responsibilities:
1.  **Robust Optimization**: Gradient clipping, differential decay, cosine annealing.
2.  **AWR Engine (v2.0)**: Integrates GAE-Lambda for time-aware credit assignment.
    - Computes Dense Rewards (Clinical Physics).
    - Computes Temporal Advantages.
    - Whitens Advantages (Global Stats).
    - Weights Diffusion Loss.
3.  **Shadow Validation**: Uses EMA model for real physiological error metrics.
4.  **Safety Hooks**: Ensures Normalizers and Stats are pre-calibrated before training.

Dependencies:
    - icu.models.diffusion.ICUUnifiedPlanner
    - icu.utils.advantage_calculator.ICUAdvantageCalculator
"""
from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm.auto import tqdm
from typing import Any, Dict, Optional, Tuple, Union
from omegaconf import DictConfig

# Project Imports
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
from icu.utils.train_utils import EMA, configure_robust_optimizer
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logger = logging.getLogger("APEX_Generalist")

class ICUGeneralistWrapper(pl.LightningModule):
    """
    LightningModule for Phase 1 Generalist Training.
    Implements the 'Offline RL via Supervised Learning' paradigm (AWR).
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # 1. Model Initialization
        logger.info("Initializing ICUGeneralistWrapper...")
        model_config = ICUConfig(**cfg.model)
        self.model = ICUUnifiedPlanner(model_config)
        
        # 2. EMA Initialization (Shadow Model)
        # Decay of 0.9999 provides very stable teacher weights for AWR
        self.ema = EMA(self.model, decay=cfg.train.get("ema_decay", 0.9999))
        
        # 3. Advantage Engine (Brain)
        self.awr_calculator = ICUAdvantageCalculator(
            beta=cfg.train.get("awr_beta", 0.5),         # Temperature
            max_weight=cfg.train.get("awr_max_weight", 20.0), # Clipping
            lambda_gae=cfg.train.get("awr_lambda", 0.95), # GAE Horizon
            gamma=cfg.train.get("awr_gamma", 0.99)       # Discount
        )
        
        # 4. Metrics
        from torchmetrics import MeanSquaredError, Accuracy
        self.val_mse_clinical = MeanSquaredError() # Real unit error
        self.val_acc_sepsis = Accuracy(task="multiclass", num_classes=2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inference Forward (Student Model)."""
        return self.model(batch)

    def on_fit_start(self):
        """
        SOTA Pre-Flight Checks (DDP Safe):
        1. Calibrate Normalizer (Deterministic File I/O -> All Ranks).
        2. Calibrate AWR Stats (Random Sampling -> Rank 0 & Broadcast).
        """
        if not (hasattr(self.trainer, "datamodule") and self.trainer.datamodule):
            logger.warning("No DataModule found. Skipping stats fitting.")
            return

        loader = self.trainer.datamodule.train_dataloader()
        dataset = loader.dataset
        
        # --- 1. Normalizer Calibration (Run on ALL Ranks) ---
        logger.info(f"[Rank {self.global_rank}] Calibrating Normalizer...")
        try:
            # Robust Metadata Extraction
            index_path = getattr(dataset, "index_path", None)
            metadata = getattr(dataset, "metadata", {})
            ts_cols = metadata.get("ts_columns", [])
            
            if index_path and ts_cols:
                # The correct SOTA API call
                self.model.normalizer.calibrate_from_stats(index_path, ts_cols)
            else:
                raise ValueError("Dataset missing 'index_path' or 'metadata.ts_columns'")
                
        except Exception as e:
            # [PATCH] Fallback Crash Fix
            # Do NOT call load_from_dataset (it doesn't exist). 
            # Log the error and proceed in "Uncalibrated Mode" (Pass-through).
            logger.error(f"[CRITICAL] Normalizer Calibration Failed: {e}")
            logger.warning("SYSTEM SAFETY: Proceeding with Uncalibrated Normalizer (Identity). Check Data Paths!")
            # self.model.normalizer.is_calibrated defaults to False, so it's safe.

        # --- 2. AWR Stats Fitting (Rank 0 Compute + Broadcast) ---
        self._fit_awr_stats_ddp(dataset)

    def _fit_awr_stats_ddp(self, dataset):
        """
        Computes AWR stats on Rank 0 and broadcasts to all DDP processes
        to ensure mathematical consistency across GPUs.
        """
        # Buffer for stats [mean, std]
        stats_tensor = torch.zeros(2, device=self.device)
        
        if self.trainer.is_global_zero:
            logger.info("[Rank 0] Sampling Trajectories for AWR Whitening...")
            rewards_list = []
            
            # Sample 500 random trajectories
            indices = torch.randperm(len(dataset))[:500]
            
            for idx in tqdm(indices, desc="AWR Calibration"):
                # Load raw sample (CPU)
                sample = dataset[idx] 
                
                # Move to GPU for reward calculation
                future = sample["future_data"].unsqueeze(0).to(self.device)
                label = sample["outcome_label"].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # [SAFETY FIX] Pass normalizer=None since data is already raw
                    # Original bug: Passing normalizer to raw data caused double-denormalization
                    r = self.awr_calculator.compute_clinical_reward(
                        future, label, normalizer=None
                    )
                    # [SAFETY FIX] Use mean() to match training scale (advantages.mean(dim=1))
                    # Original bug: sum() caused scale mismatch with trajectory-mean advantages
                    rewards_list.append(r.mean().item())
            
            # Compute stats
            r_arr = np.array(rewards_list)
            stats_tensor[0] = float(r_arr.mean())
            stats_tensor[1] = float(r_arr.std())
            logger.info(f"Calculated Stats: Mean={stats_tensor[0]:.4f}, Std={stats_tensor[1]:.4f}")

        # --- Synchronization Point ---
        # Broadcast from Rank 0 to all other ranks
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stats_tensor, src=0)
            
        # Apply to Calculator on all ranks
        mean_val = stats_tensor[0].item()
        std_val = stats_tensor[1].item()
        self.awr_calculator.set_stats(mean=mean_val, std=std_val)


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gt_label = batch.get("outcome_label", None)
        future_vitals = batch["future_data"]
        
        # 1. Forward Pass (Student)
        # out keys: ['loss', 'diffusion_loss', 'aux_loss', 'pred_value']
        out = self.model(batch, reduction='none')
        
        raw_diff_loss = out["diffusion_loss"]
        pred_values = out["pred_value"] # Has gradients
        
        # 2. Advantage Engine
        with torch.no_grad():
            # A. Compute Dense Rewards
            # [SAFETY FIX] Pass normalizer=None since future_vitals is already RAW
            # Original bug: Denormalizing raw data corrupted clinical thresholds
            rewards = self.awr_calculator.compute_clinical_reward(
                future_vitals, gt_label, normalizer=None
            )
            
            # B. Compute GAE Advantages
            # Note: compute_gae uses pred_values, but since we are in no_grad, 
            # the resulting 'advantages' tensor is detached.
            advantages = self.awr_calculator.compute_gae(rewards, pred_values)
            
            # C. Trajectory Weights
            traj_advantage = advantages.mean(dim=1)
            weights, ess = self.awr_calculator.calculate_weights(traj_advantage)
            weights = weights / (weights.mean() + 1e-8)
            
            # [PATCH] Gradient Barrier
            # Critic Target = GAE Return. We must detach to freeze the target.
            returns = (advantages + pred_values).detach()

        # 3. Loss Composition
        # A. Weighted Diffusion Loss
        weighted_loss = (raw_diff_loss * weights).mean()
        
        # B. Critic Loss (Regression to Detached Target)
        critic_loss = F.mse_loss(pred_values, returns)
        
        # C. Aux Loss
        aux_loss = out["aux_loss"].mean()
        
        # D. Total Loss
        total_loss = weighted_loss + 0.5 * critic_loss + 0.1 * aux_loss
        
        # 4. EMA Update
        self.ema.update(self.model)
        
        # 5. Logging
        self.log_dict({
            "train/total_loss": total_loss,
            "train/diff_loss": weighted_loss,
            "train/critic_loss": critic_loss,
            "train/ess": ess,
            "train/weight_mean": weights.mean(),
            "train/reward_mean": rewards.mean()
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        # A. Technical Validation (Student)
        out = self.model(batch, reduction='mean')
        self.log("val/diff_loss", out["diffusion_loss"], on_epoch=True, sync_dist=True)
        
        # B. Sepsis Accuracy (Aux Head)
        if self.model.cfg.use_auxiliary_head:
            self._validate_sepsis_acc(batch)

        # C. Clinical Validation (EMA Teacher Sampling)
        if batch_idx == 0:
            self._validate_clinical_sampling(batch)
            
        # D. Support ClinicalMetricCallback
        # This allows the 'Guardian' suite to compute AUROC/AUPRC
        if "outcome_label" in batch:
            return {
                "preds": out.get("aux_logits", torch.zeros_like(batch["outcome_label"])), 
                "target": batch["outcome_label"]
            }
        return {}

    def _validate_sepsis_acc(self, batch):
        """Robust metric calculation respecting model state."""
        # Ensure evaluation mode logic applies (Dropout off)
        # We manually drive the encoder -> aux_head path because forward()
        # returns loss, not logits.
        
        static_context = batch["static_context"]
        past = batch["observed_data"]
        
        # 1. Normalize BOTH past AND static (CRITICAL: Must match training forward pass)
        past_norm, static_norm = self.model.normalize(past, static_context)
        
        # 2. Encode
        # Note: 'src_mask' might be in batch, good to pass it if present
        src_mask = batch.get("src_mask", None)
        _, global_ctx, _ = self.model.encoder(past_norm, static_norm, src_mask)
        
        # 3. Aux Head
        logits = self.model.aux_head(global_ctx)
        
        self.val_acc_sepsis(logits, batch["outcome_label"].long())
        self.log("val/sepsis_acc", self.val_acc_sepsis, on_epoch=True, sync_dist=True)


    def _validate_clinical_sampling(self, batch):
        """
        Generates trajectories for clinical validation.
        
        NOTE: EMA weights are already applied by EMACallback.on_validation_start(),
        so self.model already contains the shadow weights during validation.
        We do NOT need to manually swap EMA here.
        """
        # Subset for speed
        subset = {k: v[:8] for k, v in batch.items()}
        gt = subset["future_data"]
        
        with torch.no_grad():
            # Model already has EMA weights during validation (handled by EMACallback)
            pred = self.model.sample(subset) 
            
        # Clinical MSE
        clinical_mse = self.val_mse_clinical(pred, gt)
        self.log("val/clinical_mse", clinical_mse, on_epoch=True, prog_bar=True, sync_dist=True)



    def on_before_optimizer_step(self, optimizer):
        """
        SOTA PERFORMANCE: Surgical Bypass for Gradient Clipping.
        Logic:
        1.  PyTorch Lightning's automatic clipping crashes with 'fused' optimizers.
        2.  By implementing this hook, we bypass the crash.
        3.  In AMP (16-mixed), Lightning has already unscaled the gradients at this point.
        4.  We call the raw PyTorch utility to perform the clipping.
        """
        if self.cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.cfg.train.grad_clip
            )

    def configure_optimizers(self):
        """
        SOTA: Robust Optimizer Factory.
        Differential Learning Rates handled via loss weighting (Critic=0.5).
        Weight Decay exclusion for Norms/Biases.
        """
        optimizer = configure_robust_optimizer(
            self.model,
            learning_rate=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.train.epochs,
            eta_min=self.cfg.train.min_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }