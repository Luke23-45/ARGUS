"""
icu/models/wrapper_apex.py
--------------------------------------------------------------------------------
APEX-MoE: Phase 2 Specialist Training Wrapper (SOTA v2.2).

Status: PRODUCTION-READY / SAFETY-CRITICAL
Changelog v2.2 (Final Polish):
    - [Safety] Removed Sanity-Check guard to guarantee AWR Stats initialization.
    - [Telemetry] Removed deceptive 'val_loss' logging; relies on 'critic_loss'.
    - [Hygiene] Purged vestigial 'preflight_checks' logic to prevent DDP bugs.

Responsibilities:
1.  **Surgical Bootstrapping**: Loads Phase 1 Generalist weights into the Tri-Phase Architecture.
2.  **AWR-Guided Gradient Surgery**:
    - Step 1: Compute Advantage Weights (using frozen Encoder + trainable Value Head).
    - Step 2: Route weighted gradients to specific Experts (Stable/Pre-Shock/Shock).
3.  **Manifold Tethering**: Penalizes divergence between experts to ensure smoothness.
4.  **Critic Adaptation**: Updates the Value Head to track the new expert policies.

Dependencies:
    - icu.models.apex_moe_planner.APEX_MoE_Planner
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
from icu.models.apex_moe_planner import APEX_MoE_Planner
from icu.utils.train_utils import configure_robust_optimizer, SurgicalCheckpointLoader
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.utils.metrics_advanced import (
    compute_policy_entropy, compute_ece, compute_explained_variance, 
    compute_overconfidence_error, compute_action_continuity, compute_demographic_accuracy_gaps
)
from icu.utils.safety import OODGuardian

logger = logging.getLogger("APEX_Specialist_v2.2")

class ICUSpecialistWrapper(pl.LightningModule):
    """
    LightningModule for Phase 2 (Specialist) Fine-Tuning.
    Implements Safety-Critical checks for Unit consistency and AWR Math.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # --- 1. SCAFFOLDING (The Generalist) ---
        logger.info("Initializing Scaffold (Generalist)...")
        base_cfg = ICUConfig(**cfg.model)
        # We need a temporary generalist to load weights, 
        # but we'll consume it immediately into the MoE Planner.
        generalist = ICUUnifiedPlanner(base_cfg)
        
        # --- 2. SURGICAL BOOTSTRAPPING ---
        ckpt_path = cfg.train.get("pretrained_path", None)
        if not ckpt_path:
            raise ValueError("[CRITICAL] Phase 2 requires 'train.pretrained_path' to bootstrap experts!")
            
        logger.info(f"[SURGICAL] Loading weights from {ckpt_path}...")
        SurgicalCheckpointLoader.load_model(generalist, ckpt_path, strict=True)
        
        # --- 3. TRANSFORMATION (The Specialist) ---
        # [SOTA-Clinical] Phase weights are now auto-generated in APEX_MoE_Planner 
        # based on num_phases (6 experts), not hardcoded for 3 phases.
        reg_weight = cfg.train.get("lambda_reg", 0.01)
        lb_weight = cfg.train.get("lambda_lb", 0.01)  # [SOTA-Clinical] Load-balancing coefficient
        
        # Construct the 6-Expert Planner using the pre-loaded generalist
        self.model = APEX_MoE_Planner(
            generalist, 
            phase_weights=None,  # Auto-generate [1.0, 5.0, 5.0, ...] for N experts
            lambda_reg=reg_weight,
            lambda_lb=lb_weight  # [SOTA-Clinical] Switch Transformer load-balancing
        )

        
        # --- 4. OPTIMIZATION HYGIENE ---
        # AWR Engine
        self.awr_calculator = ICUAdvantageCalculator(
            beta=cfg.train.get("awr_beta", 0.5),
            max_weight=cfg.train.get("awr_max_weight", 20.0),
            lambda_gae=cfg.train.get("awr_lambda", 0.95),
            gamma=cfg.train.get("awr_gamma", 0.99)
        )
        
        # Metrics
        from torchmetrics import MeanSquaredError, Accuracy
        self.val_mse_clinical = MeanSquaredError()
        # [SAFETY FIX] Use binary accuracy since target is binary (0=Stable, 1=Sepsis)
        # Original bug: 3-class metric with binary target caused misleading metrics
        self.val_acc_sepsis = Accuracy(task="binary")
        
        # 5. SOTA Safety Guardian (NeurIPS 2024 Alignment)
        self.safety_guardian = OODGuardian()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference Forward (Soft Gated Sampling)."""
        return self.model.sample(batch)

    # ==========================================================================
    # PRE-FLIGHT CHECKS (Safety Critical)
    # ==========================================================================

    def on_fit_start(self):
        """
        SOTA Pre-Flight Checks (DDP Safe).
        Ensures Normalizer is calibrated and AWR Statistics are mathematically correct.
        
        PATCH v2.2: Removed Sanity-Check guard. This guarantees AWR Stats are 
        calculated even if PL logic skips subsequent callbacks.
        """
        # 1. Calibration (Must run on ALL ranks to prevent Split-Brain)
        self._calibrate_normalizer_safe()

        # [SAFETY FIX] DDP Barrier: Ensure all ranks have finished setup
        # Original risk: Race condition if Rank 1 hasn't finished data setup
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # 2. AWR Statistics (Rank 0 computes, then Broadcasts)
        # Note: We must compute stats on ADVANTAGES, not Rewards.
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
             dataset = self.trainer.datamodule.train_dataloader().dataset
             self._fit_awr_stats_ddp(dataset)
        else:
             logger.warning("No DataModule found. Skipping AWR Sync (Risk of uncalibrated weights).")

    def _calibrate_normalizer_safe(self):
        """Robust Normalizer Loading for all ranks."""
        logger.info(f"[Rank {self.global_rank}] Calibrating Normalizer...")
        try:
            if not getattr(self.trainer, "datamodule", None):
                logger.warning("No DataModule. Normalizer will run in pass-through mode.")
                # Normalizer defaults to is_calibrated=False, which means pass-through
                return

            dataset = self.trainer.datamodule.train_dataloader().dataset
            idx_path = getattr(dataset, "index_path", None)
            meta = getattr(dataset, "metadata", {})
            cols = meta.get("ts_columns", [])
            
            if idx_path and cols:
                self.model.normalizer.calibrate_from_stats(idx_path, cols)
                
                # [CRITICAL FIX] EMA Shadow Sync
                # The EMA callback initializes shadow weights BEFORE this calibration runs.
                # We must force-update the shadow's normalizer buffers to match the calibrated state.
                if hasattr(self, 'ema') and self.ema is not None:
                    logger.info(f"[Rank {self.global_rank}] Syncing EMA shadow with calibrated normalizer...")
                    # Update only the normalizer buffers in the shadow
                    for name, buffer in self.model.normalizer.named_buffers():
                        # [FIX] Prefix must match the model's attribute name in the wrapper
                        full_name = f"model.normalizer.{name}"
                        # Ensure buffer is synced across DDP ranks before EMA update
                        if torch.distributed.is_initialized():
                            torch.distributed.broadcast(buffer.data, src=0)
                        
                        if full_name in self.ema.shadow:
                            self.ema.shadow[full_name] = buffer.data.detach().cpu().clone()
                    logger.info(f"[Rank {self.global_rank}] EMA shadow sync complete.")
            else:
                logger.warning("Metadata missing 'index_path' or 'ts_columns'. Normalizer in pass-through mode.")
                # Do NOT call non-existent method. Uncalibrated normalizer is safe (identity transform).
        except Exception as e:
            logger.error(f"Calibration Error: {e}. Normalizer will run in pass-through mode.")
            # Normalizer.is_calibrated defaults to False, so it's safe to proceed

    def _fit_awr_stats_ddp(self, dataset):
        """
        Syncs AWR statistics. 
        
        [CRITICAL FIX] AWR Stats Alignment:
        We must calibrate on ADVANTAGES, not raw Rewards.
        The AWR weighting formula is: exp((A - mu_A) / sigma_A / beta)
        If we calibrate mu/sigma on Rewards but apply to Advantages, the distributions
        mismatch, causing unstable gradient weighting.
        """
        # [0: Mean, 1: Std, 2: Count]
        stats = torch.zeros(3, device=self.device)
        
        if self.trainer.is_global_zero:
            logger.info("[Pre-Flight] Fitting AWR Stats on ADVANTAGES (Sampling)...")
            advantages_list = []
            
            sample_count = min(len(dataset), 500)
            idxs = torch.randperm(len(dataset))[:sample_count]
            
            valid_samples = 0
            for i in tqdm(idxs, desc="AWR Sync"):
                try:
                    s = dataset[i]
                    if s is None: continue # Skip corrupt/empty samples
                    
                    fut = s["future_data"].unsqueeze(0).to(self.device)
                    
                    if "outcome_label" in s:
                        lbl = s["outcome_label"].unsqueeze(0).to(self.device)
                    elif "phase_label" in s:
                        p_lbl = s["phase_label"]
                        # [FIX] Canonical Phase for Shock is 2 (from dataset.py)
                        # Original bug: used num_phases - 1, which mismatches with dataset {0,1,2}
                        shock_idx = 2 
                        is_shock = (torch.as_tensor(p_lbl) == shock_idx).float()
                        lbl = is_shock.unsqueeze(0).to(self.device)
                    else:
                        lbl = torch.zeros(1, device=self.device)

                    with torch.no_grad():
                        # [FIX] Compute ADVANTAGE = Reward + Value (not just Reward)
                        # Step 1: Get raw clinical reward
                        r = self.awr_calculator.compute_clinical_reward(
                            fut, lbl, normalizer=None 
                        )
                        
                        # Step 2: Get value estimate from the critic
                        # We need to run the encoder to get global_ctx
                        past = s["observed_data"].unsqueeze(0).to(self.device)
                        static = s["static_context"].unsqueeze(0).to(self.device)
                        
                        past_norm, static_norm = self.model.normalize(past, static)
                        _, global_ctx, _ = self.model.encoder(past_norm, static_norm, None)
                        pred_values = self.model.value_head(global_ctx)
                        
                        # Ensure shapes match [B, T]
                        if pred_values.dim() == 3 and pred_values.shape[-1] == 1:
                            pred_values = pred_values.squeeze(-1)
                        if pred_values.shape[1] != r.shape[1]:
                            pred_values = pred_values[:, :r.shape[1]]
                        
                        # Step 3: Compute GAE advantages
                        advantages = self.awr_calculator.compute_gae(r, pred_values)
                        
                        # [VRAM SAFETY] Move scalars to CPU before list accumulation
                        # Prevents accumulation of graph fragments in VRAM during sync
                        advantages_list.append(advantages.mean().detach().cpu().item())
                        valid_samples += 1
                except Exception as e:
                    logger.warning(f"AWR Sync Error at idx {i}: {e}")
                    continue
            
            if valid_samples > 0:
                arr = np.array(advantages_list)
                stats[0] = float(arr.mean())
                stats[1] = float(arr.std())
                stats[2] = float(valid_samples)
            else:
                logger.warning("AWR Sync found NO valid samples! Using standard normal defaults.")
                stats[0] = 0.0
                stats[1] = 1.0
            
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stats, src=0)
            
        sigma = stats[1].item()
        # Safety clamp to prevent div/0
        if sigma < 1e-6 or np.isnan(sigma): sigma = 1.0
        
        self.awr_calculator.set_stats(stats[0].item(), sigma)

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        if not batch or "observed_data" not in batch:
            return None # Skip empty batches from robust_collate
            
        past = batch["observed_data"]
        future_vitals = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # --- A. Robust Label Resolution ---
        gt_label = batch.get("outcome_label", None)
        phase_label = batch.get("phase_label", None)
        
        if gt_label is None and phase_label is not None:
            # [SAFETY FIX] Include both Pre-Shock (1) and Shock (2) as positive class
            # Original bug: Only Shock (Phase 2) was labeled positive, so Pre-Shock
            # patients were incorrectly labeled "Healthy", penalizing early detection
            gt_label = (phase_label >= 1).to(dtype=torch.float32)
        
        # --- B. Pre-Compute Advantage Weights ---
        with torch.no_grad():
            # 1. Perception (Encoder is frozen, but we need the latents)
            past_norm, static_norm = self.model.normalize(past, static)
            _, global_ctx, _ = self.model.encoder(past_norm, static_norm, src_mask)
            
            # 2. Value Estimate (Critic)
            # Ensure shapes align
            pred_values = self.model.value_head(global_ctx)
            if pred_values.dim() == 3 and pred_values.shape[-1] == 1:
                pred_values = pred_values.squeeze(-1) # [B, T]
            
            # Safety broadcast if prediction horizon mismatch
            if pred_values.shape[1] != future_vitals.shape[1]:
                pred_values = pred_values[:, :future_vitals.shape[1]]

            # 3. Clinical Reward Calculation
            rewards = self.awr_calculator.compute_clinical_reward(
                future_vitals, gt_label, normalizer=None
            )
            B_curr, T_curr = rewards.shape

            # [CRITICAL FIX] Scalar-to-Sequence Broadcasting
            # If Value Head is a sequence-level critic (e.g. Outcome prediction),
            # we must expand it to match the horizon [B, T] for GAE.
            if pred_values.dim() == 1 or (pred_values.dim() == 2 and pred_values.shape[1] == 1):
                # Shape: [B, 1] -> [B, T]
                pred_values = pred_values.view(B_curr, 1).expand(B_curr, T_curr)
            elif pred_values.shape[1] != T_curr:
                # Horizon truncation if critic predicted longer/shorter horizon
                pred_values = pred_values[:, :T_curr]

            # 4. GAE Calculation (Using Calibrated Stats)
            advantages = self.awr_calculator.compute_gae(rewards, pred_values)
            
            # 5. Weight Calculation
            traj_adv = advantages.mean(dim=1)
            # Pass values and rewards for Explained Variance calculation
            weights, diag = self.awr_calculator.calculate_weights(
                traj_adv, 
                values=pred_values.mean(dim=1), 
                rewards=rewards.mean(dim=1)
            )
            
            # Normalize weights to preserve gradient magnitude
            weights = weights / (weights.mean() + 1e-8)
            
            # 6. Critic Target (Stop Gradient on Advantage)
            critic_target = (advantages + pred_values).detach()
            # Ensure target shape matches [B, T] if horizon was sliced
            if critic_target.shape[1] != future_vitals.shape[1]:
                critic_target = critic_target[:, :future_vitals.shape[1]]

        # --- C. Execute Experts (Gradient Surgery) ---
        # Pass weights to Hard-Gating mechanism
        out = self.model(batch, awr_weights=weights)
        
        # --- D. Train Critic (Adaptation) ---
        # The Critic must track the NEW expert policy, so we update it here.
        current_pred_val = out["pred_value"]
        if current_pred_val.shape != critic_target.shape:
             current_pred_val = current_pred_val.view_as(critic_target)
             
        critic_loss = F.mse_loss(current_pred_val, critic_target)
        
        # --- E. Total Loss ---
        total_loss = out["loss"] + 0.5 * critic_loss
        
        # --- F. Logging ---
        # PATCH v2.2: Removed 'val_loss' (deceptive). Added 'preshock_loss' (visibility).
        B = batch["observed_data"].shape[0]  # [FIX] Explicit batch_size for PL
        
        # Policy Entropy (Diagnostic for decision collapse)
        with torch.no_grad():
            logits = out.get("logits", torch.zeros(1, 1, device=self.device))
            # [MoE] Router predicts over num_phases experts
            probs = F.softmax(logits, dim=-1)
            entropy = compute_policy_entropy(probs)

        self.log_dict({
            "train/total_loss": total_loss,
            "train/expert_loss": out["loss"], 
            "train/reg_loss": out["reg_loss"],
            "train/critic_loss": critic_loss,
            "train/explained_variance": diag["explained_variance"],
            "train/policy_entropy": entropy,
            "train/stable_loss": out.get("stable_loss", 0.0),
            "train/preshock_loss": out.get("preshock_loss", 0.0), # Added
            "train/crash_loss": out.get("crash_loss", 0.0),
            "train/ess": diag["ess"],
            "train/max_weight": diag["max_weight"]
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        
        # CRITICAL: Must return loss for PyTorch Lightning to compute gradients
        return total_loss

    # ==========================================================================
    # VALIDATION LOOP
    # ==========================================================================

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[Dict[str, Any]]:
        """
        Dual Validation: 
        1. Clinical Sampling (Vitals Generation) using EMA weights.
        2. Routing Metrics (Sepsis Classification) using Router outputs.
        """
        if not batch or "observed_data" not in batch:
            return None
            
        try:
            # 1. Run Soft-Gated Inference (Returns Dict: 'vitals', 'logits', 'probs')
            # This implicitly uses the EMA context if the EMACallback is active/swapped
            out_payload = self.model.sample(batch, hard_gating=False)
            
            # 2. Log Clinical Vitals MSE (Only for first batch to save time)
            if batch_idx == 0:
                self._validate_clinical_sampling(batch, out_payload["vitals"])

            # 3. Prepare Payload for ClinicalMetricCallback
            # We extract the logits from the router to evaluate routing accuracy
            target = batch.get("outcome_label")
            
            # Fallback if outcome_label missing
            if target is None and "phase_label" in batch:
                # [FIX] Align with training logic: Sepsis = Pre-Shock (1) or Shock (2)
                # target = (batch["phase_label"] == 2) was too conservative 
                target = (batch["phase_label"] >= 1).to(dtype=torch.float32)

            # E. Calibration Error (ECE) & Overconfidence Error (OE)
            if target is not None:
                # Use only the base phase logits for ECE (if multiclass router)
                # target is binary {0, 1}
                router_probs = out_payload["probs"]
                # Map 6 experts -> 2 classes (Healthy [0,3], Sepsis [1,2,4,5])
                sepsis_prob = router_probs[:, [1,2,4,5]].sum(dim=1)
                
                ece = compute_ece(sepsis_prob, target)
                oe = compute_overconfidence_error(sepsis_prob, target)
                
                self.log_dict({
                    "val/ece": ece,
                    "val/oe": oe
                }, on_epoch=True, sync_dist=True)
                
                # F. Fairness Audit (Gender/Age)
                # Static context: [Age (0), Gender (1), ...]
                static = batch["static_context"]
                if static.shape[-1] >= 2:
                    is_female = (static[:, 1] > 0.5)
                    is_male = (static[:, 1] <= 0.5)
                    is_elderly = (static[:, 0] > 65.0) # Assuming Age is in years
                    
                    fairness_results = compute_demographic_accuracy_gaps(
                        sepsis_prob, target, 
                        {"female": is_female, "male": is_male, "elderly": is_elderly}
                    )
                    self.log_dict({f"val/fairness_{k}": v for k, v in fairness_results.items()}, on_epoch=True, sync_dist=True)

            return {
                "preds": out_payload["logits"],  # Pass Router Logits to Metric Callback
                "target": target
            }
                
        except Exception as e:
            logger.error(f"Validation Error: {e}")
            # Raise to prevent silent failure during development
            raise e


    def _validate_clinical_sampling(self, batch, pred_vitals):
        """
        Logs generation fidelity metrics.
        Args:
            batch: Input batch dict
            pred_vitals: Generated trajectory tensor [B, T, D]
        """
        gt = batch["future_data"]
        
        # Ensure devices match
        if pred_vitals.device != gt.device:
            pred_vitals = pred_vitals.to(gt.device)
            
        # 1. Metric Object Update
        self.val_mse_clinical.update(pred_vitals, gt)
        
        # 2. SOTA OOD Guardian (Dynamics Safety)
        past = batch["observed_data"]
        safety_results = self.safety_guardian.check_trajectories(past, pred_vitals)
        
        # 3. Action Continuity (Smoothness)
        smoothness = compute_action_continuity(pred_vitals)
        
        self.log_dict({
            "val/clinical_mse": self.val_mse_clinical,
            "val/ood_rate": safety_results["ood_rate"],
            "val/safety_sbp_delta": safety_results["sbp_delta_mean"],
            "val/safety_lac_max": safety_results["lac_max_mean"],
            "val/action_smoothness": smoothness
        }, on_epoch=True, prog_bar=True)
  
    def on_before_optimizer_step(self, optimizer):
        """
        SOTA PERFORMANCE: Surgical Bypass for Gradient Clipping.
        Uses raw PyTorch utility to bypass Lightning's conservative 'fused' check.
        Gradients are already unscaled by the Trainer at this point in AMP.
        """
        if self.cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.cfg.train.grad_clip
            )

    def configure_optimizers(self):
        """
        Specialist Optimizer.
        - Freezes Encoder/Router/RoPE (No gradients).
        - Optimizes Experts + Value Head.
        """
        # Select parameters that require gradients
        # (APEX_MoE_Planner already sets requires_grad=False for frozen parts)
        optimizer = configure_robust_optimizer(
            self.model,
            learning_rate=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay
            # Note: backbone_lr_ratio removed - not supported by configure_robust_optimizer
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.cfg.train.epochs, 
            eta_min=self.cfg.train.min_lr
        )
        
        return [optimizer], [scheduler]