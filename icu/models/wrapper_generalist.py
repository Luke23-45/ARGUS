"""
icu/models/wrapper_generalist.py
--------------------------------------------------------------------------------
APEX-MoE: Phase 1 Generalist Training Wrapper (Ultimate Edition v12.0).

"The last line of defense. The first hope for survival."

This script represents the culmination of our safety-critical optimization pipeline.
It is designed not just to train a model, but to cultivate a clinical agent 
capable of reasoning about life-support trajectories with physiological fidelity.

This module implements a training wrapper with SOTA techniques from:
- Offline Reinforcement Learning (AWR, IQL concepts, TD3BC regularization)
- Diffusion Model Training (Two-Pass Self-Conditioning - "Analog Bits")
- Target Network Stabilization (EMA Teacher for value bootstrapping)
- Dynamic Curriculum Learning (Progressive physiological constraint hardening)
- Safety-Critical Clinical AI (Physiological bounds, OOD detection)

Architectural Pillars (Ultimate v12.0):
1.  **Two-Pass Self-Conditioning**: 50% of training steps use a preliminary
    x0 estimate as conditioning, teaching the model to "fix its own mistakes".
2.  **True Target Network**: EMA weights are used as a frozen Teacher for 
    value estimation, preventing the "Dead Critic" problem in offline RL.
3.  **Dynamic Physiological Curriculum**: Physics penalties start low (0.01)
    and ramp up over 50% of training, allowing exploration then enforcing safety.
4.  **Robust Optimizer Groups**: Explicit separation of weight-decay eligible
    parameters (kernels) from exclusion groups (biases/norms/embeddings).
5.  **Holistic Safety Aggregation**: Validation aggregates 'Safe Trajectory'
    statistics across the entire validation corpus for deployment confidence.
6.  **Granular Clinical Telemetry**: Decomposes error metrics into Hemodynamic,
    Metabolic, Respiratory, and Neurological components for targeted diagnosis.
7.  **Gradient Accumulation Aware**: Properly handles weight updates with
    configurable gradient accumulation steps.
8.  **Mixed Precision Safe**: All operations are designed to be FP16-safe
    with proper gradient scaling awareness.
9.  **DDP Synchronized**: AWR statistics are computed on Rank 0 and broadcast
    to ensure mathematical consistency across distributed training.
10. **Warmup + Cosine Annealing**: Learning rate schedule with linear warmup
    for stable early training followed by cosine decay.

References:
    - Peng et al., "Advantage-Weighted Regression" (AWR)
    - Kostrikov et al., "Implicit Q-Learning" (IQL)
    - Chen et al., "Analog Bits: Generating Discrete Data" (Self-Conditioning)
    - He et al., "Momentum Contrast" (EMA Teacher Networks)
    - Bengio et al., "Curriculum Learning" (Progressive Hardening)
    - Sepsis-3 Consensus (2016) - Clinical threshold definitions

Dependencies:
    - icu.models.diffusion.ICUUnifiedPlanner
    - icu.utils.advantage_calculator.ICUAdvantageCalculator
"""
from __future__ import annotations

import logging
import contextlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm.auto import tqdm
from typing import Any, Dict, Optional, Tuple, List, Union
from omegaconf import DictConfig

# Project Imports
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig, PhysiologicalConsistencyLoss
from icu.utils.train_utils import EMA
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.utils.metrics_advanced import (
    compute_policy_entropy, 
    compute_ece, 
    compute_explained_variance, 
    compute_overconfidence_error
)
from icu.utils.safety import OODGuardian

# Specialized Metric Collection
from torchmetrics import MeanSquaredError, Accuracy, MeanMetric, AUROC

logger = logging.getLogger("APEX_Generalist_v12")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.
    
    This is the SOTA choice for transformer training, providing:
    1. Stable early training via linear warmup
    2. Smooth convergence via cosine annealing
    3. Prevention of learning rate cliff at the end
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of steps for linear warmup
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as a ratio of initial LR (default 0.01 = 1%)
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# MAIN WRAPPER CLASS
# =============================================================================

class ICUGeneralistWrapper(pl.LightningModule):
    """
    LightningModule for Phase 1 Generalist Training.
    
    Implements SOTA Offline RL with:
    - Target Network Stabilization (EMA Teacher)
    - Two-Pass Self-Conditioning (Analog Bits)
    - Dynamic Physiological Curricula
    - Advantage-Weighted Regression (AWR)
    - Holistic Safety Monitoring
    
    This wrapper is designed for safety-critical clinical AI applications
    where model stability, physiological plausibility, and robust training
    are paramount.
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # =====================================================================
        # 1. CORE ARCHITECTURE
        # =====================================================================
        logger.info("Initializing ICUGeneralistWrapper (Ultimate Edition v12.0)...")
        model_config = ICUConfig(**cfg.model)
        # [v12.0] Inject training config into model config for validation parity
        model_config.aux_loss_scale = cfg.train.get("aux_loss_scale", 0.1)
        self.model = ICUUnifiedPlanner(model_config)
        
        # =====================================================================
        # 2. STABILIZATION & DISTILLATION (EMA Teacher Network)
        # =====================================================================
        # Decay 0.9999 provides extreme stability for the Value Target.
        # This acts as the 'Teacher' in our distillation process.
        # With 1000 steps, this means the teacher is ~90% influenced by
        # weights from 10 steps ago, providing a stable target.
        self.ema = EMA(self.model, decay=cfg.train.get("ema_decay", 0.9999))
        
        # =====================================================================
        # 3. ADVANTAGE ENGINE (AWR with GAE-Lambda)
        # =====================================================================
        # Uses Advantage-Weighted Regression to prioritize high-survival trajectories.
        # Key parameters:
        # - beta: Temperature (0.5 = focused, 1.0 = exploration)
        # - max_weight: Clipping to prevent gradient explosions
        # - lambda_gae: Bias-variance tradeoff (0.95 = low bias)
        # - gamma: Discount factor (0.99 = ~100 step horizon)
        self.awr_calculator = ICUAdvantageCalculator(
            beta=cfg.train.get("awr_beta", 0.5),
            max_weight=cfg.train.get("awr_max_weight", 20.0),
            lambda_gae=cfg.train.get("awr_lambda", 0.95),
            gamma=cfg.train.get("awr_gamma", 0.99)
        )
        
        # =====================================================================
        # 4. SAFETY & BIOLOGICAL CONSTRAINTS
        # =====================================================================
        # Base weight for physics loss; dynamically scaled during training
        # via curriculum learning (_get_curr_physics_weight)
        self.base_phys_weight = cfg.train.get("phys_loss_weight", 0.1)
        self.phys_loss = PhysiologicalConsistencyLoss(weight=1.0)  # Scaled dynamically
        self.safety_guardian = OODGuardian()
        
        # =====================================================================
        # 5. TRAINING TELEMETRY (Accumulated Metrics)
        # =====================================================================
        # These are accumulated over batches and logged at epoch end
        self.train_loss_total = MeanMetric()
        self.train_loss_diff = MeanMetric()
        self.train_loss_critic = MeanMetric()
        self.train_loss_phys = MeanMetric()
        self.train_loss_aux = MeanMetric()
        self.train_awr_ess = MeanMetric()  # Effective Sample Size
        self.train_explained_var = MeanMetric()
        
        # =====================================================================
        # 6. VALIDATION TELEMETRY (Global Aggregation)
        # =====================================================================
        # Global MSE
        self.val_mse_global = MeanSquaredError()
        
        # Granular MSE by clinical category (for targeted debugging)
        # Indices based on Frontier 28 schema:
        # - Hemodynamic: indices 0-6 (HR, MAP, Temp, SpO2, SBP, DBP, RespRate)
        # - Labs: indices 7-17 (WBC, Lactate, Creatinine, BUN, Glucose, HCT, Hgb, Platelets, Bilirubin, INR, PTT)
        # - Electrolytes: indices 18-21 (Na, K, Ca, Mg)
        # - Static: indices 22-27 (Age, Gender, Unit1, Unit2, AdmTime, LOS)
        self.val_mse_hemo = MeanSquaredError()      # Hemodynamic (0-6)
        self.val_mse_labs = MeanSquaredError()      # Labs (7-17)
        self.val_mse_electrolytes = MeanSquaredError()  # Electrolytes (18-21)
        
        # Classification Metrics
        self.val_acc_sepsis = Accuracy(task="multiclass", num_classes=cfg.model.get("num_phases", 3))
        self.val_auroc_sepsis = AUROC(task="binary")
        
        # Safety Accumulators
        self.val_ood_rate = MeanMetric()
        self.val_safe_traj_count = MeanMetric()
        self.val_phys_violation_rate = MeanMetric()
        
        # Calibration
        self.val_ece = MeanMetric()
        self.val_oe = MeanMetric()
        
        # =====================================================================
        # 7. STATE FLAGS
        # =====================================================================
        self._awr_stats_initialized = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass (used for simple inference/debugging).
        For training, we use the custom logic in training_step.
        """
        return self.model(batch)

    # =========================================================================
    # SOTA TRAINING LOGIC (The "Heart")
    # =========================================================================

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        The Ultimate Life-Critical Training Step.
        
        Orchestrates:
        1. Context encoding (shared between passes)
        2. Two-Pass Self-Conditioning (Analog Bits)
        3. Advantage-Weighted diffusion loss
        4. Teacher-Student critic distillation
        5. Auxiliary sepsis classification
        6. Dynamic physiological curriculum
        
        Returns:
            Scalar loss tensor for backpropagation
        """
        gt_label = batch.get("outcome_label", None)
        
        # --- 1. Unpack & Normalize (Shared) ---
        past = batch["observed_data"]
        fut = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # Normalize inputs for the Diffusion backbone
        past_norm, static_norm = self.model.normalize(past, static)
        fut_norm, _ = self.model.normalize(fut, None)
        
        B = past.shape[0]
        device = self.device
        
        # [v12.1] Robust Padding Detection:
        # A timestep is PADDING if ALL channels in src_mask are missing (0.0).
        # [v16.0] FIX: Disable unsafe padding inference. Fixed window dataset has no padding.
        padding_mask = None
        
        ctx_seq, global_ctx, ctx_mask = self.model.encoder(
            past_norm, static_norm, 
            imputation_mask=src_mask,
            padding_mask=padding_mask  # [FIX] Ignore ghost history tokens
        )
        
        # --- 3. Diffusion Training (Student) ---
        t = torch.randint(0, self.model.cfg.timesteps, (B,), device=device)
        noisy_fut, noise_eps = self.model.scheduler.add_noise(fut_norm, t)
        
        # --- 4. Two-Pass Self-Conditioning ("Analog Bits") ---
        # 50% of the time, we force "Cold Start" (zeros) to learn initial generation.
        # 50% of the time, we make a preliminary guess, detach it, and condition on it.
        # This teaches the model to "read its own thoughts" and refine predictions.
        
        use_self_cond = self.model.cfg.use_self_conditioning and (torch.rand(1).item() < 0.5)
        self_cond_tensor = None
        
        if self.model.cfg.use_self_conditioning:
            # Initialize with cold start (zeros)
            self_cond_tensor = torch.zeros_like(noisy_fut)
            
            if use_self_cond:
                with torch.no_grad():
                    # Pass 1: "Guess" x0 using the Student (current weights)
                    guess_eps = self.model.backbone(
                        noisy_fut, t, ctx_seq, global_ctx, ctx_mask, self_cond=self_cond_tensor
                    )
                    
                    # Reconstruct x0 approximation (DDIM equation)
                    alpha_bar = self.model.scheduler.alphas_cumprod[t][:, None, None]
                    guess_x0 = (noisy_fut - torch.sqrt(1 - alpha_bar) * guess_eps) / torch.sqrt(alpha_bar)
                    
                    # Update conditioning tensor (CRITICAL: Must be detached!)
                    self_cond_tensor = guess_x0.detach()

        # Pass 2 (Final): The actual gradient step with conditioning
        pred_noise = self.model.backbone(
            noisy_fut, t, ctx_seq, global_ctx, ctx_mask, self_cond=self_cond_tensor
        )
        
        # --- 5. Loss Computation ---
        
        # A. Raw Diffusion Loss (Per sample, for AWR weighting)
        raw_diff_loss = F.mse_loss(pred_noise, noise_eps, reduction='none').mean(dim=[1, 2])
        
        # B. Advantage Calculation (Teacher-Student paradigm)
        with torch.no_grad():
            # Calculate Dense Rewards (Survival + Clinical Stability)
            # We use raw `fut` (unnormalized) for reward calculation
            # as clinical thresholds are defined in physical units
            rewards = self.awr_calculator.compute_clinical_reward(
                fut, gt_label, normalizer=None
            )
            
            # [SOTA] TRUE TARGET NETWORK (Bootstrapping with EMA Teacher)
            # We must use the EMA weights (Teacher) to estimate Value.
            # This is critical for Offline RL stability - prevents "Dead Critic"
            # where the student chases its own changing predictions.
            with self.ema_teacher_context():
                # Re-encode using Teacher weights
                _, teacher_global_ctx, _ = self.model.encoder(past_norm, static_norm, src_mask)
                target_values = self.model.value_head(teacher_global_ctx)
            
            # Compute GAE (Generalized Advantage Estimation)
            # Shape: [B, T_pred]
            advantages = self.awr_calculator.compute_gae(rewards, target_values)
            
            # Compute AWR Weights
            # We aggregate to trajectory level for sample weighting
            traj_adv = advantages.mean(dim=1)  # [B]
            weights, diag = self.awr_calculator.calculate_weights(
                traj_adv, 
                values=target_values.mean(dim=1), 
                rewards=rewards.mean(dim=1)
            )
            
            # Normalize weights for gradient stability
            # This ensures weights have mean 1.0, preserving gradient scale
            weights = weights / (weights.mean() + 1e-8)
            
            # Bootstrapped Returns for Critic Update
            # The student tries to predict these stable targets
            returns = (advantages + target_values).detach()

        # C. Weighted Diffusion Loss (AWR Core)
        # We focus learning on trajectories that actually lead to survival
        weighted_diff_loss = (raw_diff_loss * weights).mean()
        
        # D. Critic Loss (Student → Teacher Distillation)
        # The student critic learns to match the Teacher's stable value estimates
        pred_values_student = self.model.value_head(global_ctx)
        critic_loss = F.mse_loss(pred_values_student, returns)
        
        # E. Auxiliary Sepsis Loss (If enabled)
        # Multi-task learning: Predict clinical phase alongside diffusion
        aux_loss = torch.tensor(0.0, device=device)
        if self.model.cfg.use_auxiliary_head and "phase_label" in batch:
            aux_logits = self.model.aux_head(global_ctx)
            
            # Handle potential class mismatch (e.g., 6 experts vs 3 clinical phases)
            w = batch.get("aux_weight", None)
            effective_logits = aux_logits
            if w is not None and aux_logits.shape[-1] > w.shape[0]:
                effective_logits = aux_logits[:, :w.shape[0]]
            
            aux_loss = F.cross_entropy(
                effective_logits, 
                batch["phase_label"].long(),
                weight=w
            )

        # F. Dynamic Physiological Penalty (Curriculum Learning)
        # We reconstruct the x0 approximation and penalize biological violations.
        # The penalty weight increases as training progresses, allowing the model
        # to learn the data distribution first, then refine biological realism.
        alpha_bar_t = self.model.scheduler.alphas_cumprod[t][:, None, None]
        x0_approx = (noisy_fut - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        
        curr_phys_weight = self._get_curr_physics_weight()
        phys_penalty = self.phys_loss(x0_approx) * curr_phys_weight

        # G. Total Loss Composition
        # Loss weighting follows established practices:
        # - Diffusion: 1.0 (Primary task)
        # - Critic: 0.5 (Value learning, lower to not dominate)
        # - Aux: Configured (0.182 optimal)
        # - Physics: Dynamic (0.01 → base_weight over curriculum)
        aux_scale = self.cfg.train.get("aux_loss_scale", 0.1)
        total_loss = weighted_diff_loss + 0.5 * critic_loss + aux_scale * aux_loss + phys_penalty
        
        # --- 6. EMA Update (Student -> Teacher) ---
        # [v12.1] Accumulation-Aware Update:
        # Only update EMA when the weights are actually updated by the optimizer.
        # This prevents the Teacher from drifting too fast during accumulation steps.
        accum = self.trainer.accumulate_grad_batches
        if (batch_idx + 1) % accum == 0:
            self.ema.update(self.model)
        
        # --- 7. Policy Entropy Diagnostic ---
        # Monitors mode collapse in the router (if aux head is used)
        entropy = 0.0
        if self.model.cfg.use_auxiliary_head:
            with torch.no_grad():
                aux_logits_for_entropy = self.model.aux_head(global_ctx)
                router_probs = F.softmax(aux_logits_for_entropy, dim=-1)
                entropy = compute_policy_entropy(router_probs)
        
        # --- 8. Telemetry Accumulation ---
        self.train_loss_total.update(total_loss)
        self.train_loss_diff.update(weighted_diff_loss)
        self.train_loss_critic.update(critic_loss)
        self.train_loss_phys.update(phys_penalty)
        self.train_loss_aux.update(aux_loss)
        self.train_awr_ess.update(diag["ess"])
        self.train_explained_var.update(diag["explained_variance"])
        
        # Step-level logging (for live monitoring)
        self.log_dict({
            "train/loss_step": total_loss,
            "train/diff_loss": weighted_diff_loss,
            "train/critic_loss": critic_loss,
            "train/awr_ess": diag["ess"],              # Effective Sample Size (RL health)
            "train/awr_max_w": diag["max_weight"],     # Check for exploding weights
            "train/explained_var": diag["explained_variance"],
            "train/policy_entropy": entropy,
            "train/reward_mean": rewards.mean(),
            "train/curr_phys_weight": curr_phys_weight,  # Monitor curriculum
            "train/lr": self.optimizers().param_groups[0]["lr"]  # Learning rate
        }, on_step=True, on_epoch=False, prog_bar=True)
        
        return total_loss

    def on_train_epoch_end(self):
        """Log accumulated metrics for the epoch and reset."""
        self.log_dict({
            "train/epoch_loss_total": self.train_loss_total.compute(),
            "train/epoch_loss_diff": self.train_loss_diff.compute(),
            "train/epoch_loss_critic": self.train_loss_critic.compute(),
            "train/epoch_loss_phys": self.train_loss_phys.compute(),
            "train/epoch_loss_aux": self.train_loss_aux.compute(),
            "train/epoch_awr_ess": self.train_awr_ess.compute(),
            "train/epoch_explained_var": self.train_explained_var.compute(),
        }, sync_dist=True)
        
        # Reset for next epoch
        self.train_loss_total.reset()
        self.train_loss_diff.reset()
        self.train_loss_critic.reset()
        self.train_loss_phys.reset()
        self.train_loss_aux.reset()
        self.train_awr_ess.reset()
        self.train_explained_var.reset()

    # =========================================================================
    # VALIDATION & SAFETY CHECKS (Holistic)
    # =========================================================================

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Validation: Checks metrics and runs 'Safety Guardian' on predictions.
        
        Performs:
        1. Standard diffusion loss monitoring
        2. Risk prediction calibration (ECE, Overconfidence Error)
        3. Clinical trajectory sampling (with granular error analysis)
        4. Safety checks (OOD detection, physiological bounds)
        
        Returns predictions and targets for external callbacks.
        """
        # 1. Standard Loss Monitoring
        out = self.model(batch, reduction='mean')
        self.log("val/diff_loss", out["diffusion_loss"], on_epoch=True, sync_dist=True)
        
        # 2. Risk Prediction Calibration
        if "outcome_label" in batch and self.model.cfg.use_auxiliary_head:
            logits = out.get("aux_logits", None)
            if logits is not None:
                probs = F.softmax(logits, dim=-1)
                
                # For binary AUROC: Sum sepsis-related probabilities
                # Assuming index 0 = Stable, indices 1+ = Sepsis stages
                if probs.shape[-1] > 1:
                    risk_prob = probs[:, 1:].sum(dim=1)
                else:
                    risk_prob = torch.sigmoid(logits.squeeze())
                
                # Binary label for AUROC (0 = Stable, 1 = Sepsis/Shock)
                binary_label = (batch["outcome_label"] > 0).long()
                
                # ECE and Overconfidence Error
                ece = compute_ece(risk_prob, binary_label)
                oe = compute_overconfidence_error(risk_prob, binary_label)
                
                self.val_ece.update(ece)
                self.val_oe.update(oe)
                self.val_acc_sepsis.update(logits, batch["outcome_label"].long())
                self.val_auroc_sepsis.update(risk_prob, binary_label)

        # 3. Clinical Trajectory Sampling (Every 5th batch to save compute)
        # This ensures we cover diverse samples while being efficient
        if batch_idx % 5 == 0:
            self._validate_clinical_sampling(batch)

        # Return for external callbacks (e.g., ClinicalMetricCallback)
        result = {}
        if "outcome_label" in batch:
            result["preds"] = out.get("aux_logits", torch.zeros_like(batch["outcome_label"]))
            result["target"] = batch["outcome_label"]
        
        return result

    def _validate_clinical_sampling(self, batch: Dict[str, torch.Tensor]):
        """
        Generates full trajectories and validates them against clinical reality.
        
        This is the most clinically meaningful validation:
        1. Generates future vitals using the model
        2. Compares against ground truth (granular MSE)
        3. Checks physiological plausibility (OOD Guardian)
        4. Measures safety constraint violations
        """
        # Take a subset to save compute (16 samples per batch)
        subset_size = min(16, batch["observed_data"].shape[0])
        subset = {k: v[:subset_size] for k, v in batch.items()}
        gt = subset["future_data"]
        
        # Sample using the *Teacher* (EMA) for best generation quality
        with self.ema_teacher_context():
            with torch.no_grad():
                pred = self.model.sample(subset)
        
        # A. Global MSE (Overall prediction quality)
        self.val_mse_global.update(pred, gt)
        
        # B. Granular MSE by Clinical Category
        # This helps identify which subsystem the model struggles with
        
        # Hemodynamic (indices 0-6): HR, MAP, Temp, SpO2, SBP, DBP, RespRate
        if pred.shape[-1] > 6:
            self.val_mse_hemo.update(pred[..., :7], gt[..., :7])
        
        # Labs (indices 7-17): Metabolic panel
        if pred.shape[-1] > 17:
            self.val_mse_labs.update(pred[..., 7:18], gt[..., 7:18])
        
        # Electrolytes (indices 18-21): Na, K, Ca, Mg
        if pred.shape[-1] > 21:
            self.val_mse_electrolytes.update(pred[..., 18:22], gt[..., 18:22])
        
        # C. Safety Check (The "Hard Deck")
        # OOD Guardian checks if generated trajectories are biologically plausible
        safety_results = self.safety_guardian.check_trajectories(subset["observed_data"], pred)
        
        self.val_ood_rate.update(safety_results["ood_rate"])
        self.val_safe_traj_count.update(safety_results["safe_count"])
        
        # D. Physiological Constraint Violations
        # Check how often predictions exceed the 2.5σ bounds
        with torch.no_grad():
            pred_norm, _ = self.model.normalize(pred, None)
            violations = (torch.abs(pred_norm) > 2.5).float().mean()
            self.val_phys_violation_rate.update(violations)

    def on_validation_epoch_end(self):
        """
        Aggregates safety stats across the entire validation set.
        This provides a holistic view of model readiness for deployment.
        """
        self.log_dict({
            # Prediction Quality
            "val/mse_global": self.val_mse_global.compute(),
            "val/mse_hemo": self.val_mse_hemo.compute(),
            "val/mse_labs": self.val_mse_labs.compute(),
            "val/mse_electrolytes": self.val_mse_electrolytes.compute(),
            
            # Classification Metrics
            "val/sepsis_acc": self.val_acc_sepsis.compute(),
            "val/sepsis_auroc": self.val_auroc_sepsis.compute(),
            
            # Calibration
            "val/ece": self.val_ece.compute(),
            "val/oe": self.val_oe.compute(),
            
            # Safety Metrics (Critical for deployment decisions)
            "val/ood_rate_avg": self.val_ood_rate.compute(),
            "val/safe_trajectories_avg": self.val_safe_traj_count.compute(),
            "val/phys_violation_rate": self.val_phys_violation_rate.compute(),
        }, prog_bar=True, sync_dist=True)
        
        # Reset all metrics
        self.val_mse_global.reset()
        self.val_mse_hemo.reset()
        self.val_mse_labs.reset()
        self.val_mse_electrolytes.reset()
        self.val_acc_sepsis.reset()
        self.val_auroc_sepsis.reset()
        self.val_ece.reset()
        self.val_oe.reset()
        self.val_ood_rate.reset()
        self.val_safe_traj_count.reset()
        self.val_phys_violation_rate.reset()

    # =========================================================================
    # UTILITIES & SETUP
    # =========================================================================

    def _get_curr_physics_weight(self) -> float:
        """
        Curriculum Learning for Physiological Constraints.
        
        Ramps up the physics loss weight from 0.01 to base_weight over 50% of epochs.
        This allows the model to:
        1. Learn the data distribution first (exploration phase)
        2. Then refine biological realism (exploitation phase)
        
        This is inspired by curriculum learning principles from Bengio et al.
        """
        if self.trainer.max_epochs is None:
            return self.base_phys_weight
            
        current_epoch = self.current_epoch
        warmup_epochs = self.trainer.max_epochs * 0.5
        
        if current_epoch < warmup_epochs:
            # Linear ramp: 0.01 → base_weight
            progress = current_epoch / warmup_epochs
            return 0.01 + (self.base_phys_weight - 0.01) * progress
        else:
            return self.base_phys_weight

    def ema_teacher_context(self):
        """
        Context manager that temporarily swaps Student weights with 
        Teacher (EMA) weights for stable inference.
        
        This is essential for:
        1. Value estimation in training (prevents "Dead Critic")
        2. Generation during validation (best quality outputs)
        
        The swap is safe for DDP as it operates on the local model only.
        """
        # [v15.3] SOTA Fix: Delegate to TieredEMA's robust context manager
        # TieredEMA handles CPU offloading, pinning, and restoration automatically.
        if hasattr(self, 'ema') and self.ema is not None:
            return self.ema.swap()
        return contextlib.nullcontext()

    def on_fit_start(self):
        """
        Pre-flight checks (DDP Safe):
        1. Calibrate Normalizer (Deterministic file I/O → All Ranks).
        2. Whitening AWR Stats (Random Sampling → Rank 0 & Broadcast).
        3. Sync EMA shadow with calibrated normalizer.
        """
        if not (hasattr(self.trainer, "datamodule") and self.trainer.datamodule):
            logger.warning("No DataModule found. Skipping stats fitting.")
            return

        loader = self.trainer.datamodule.train_dataloader()
        dataset = loader.dataset
        
        # --- 1. Normalizer Calibration (Run on ALL Ranks) ---
        # Check if already calibrated (e.g. from checkpoint) to avoid jitter
        if self.model.normalizer.is_calibrated > 0:
            logger.info(f"[Rank {self.global_rank}] Normalizer already calibrated. Skipping.")
            return

        logger.info(f"[Rank {self.global_rank}] Calibrating Normalizer...")
        try:
            index_path = getattr(dataset, "index_path", None)
            metadata = getattr(dataset, "metadata", {})
            ts_cols = metadata.get("ts_columns", [])
            
            if index_path and ts_cols:
                self.model.normalizer.calibrate_from_stats(index_path, ts_cols)
                
                # [CRITICAL] EMA Shadow Sync
                # The EMA was initialized BEFORE normalizer calibration.
                # We must force-update the shadow's normalizer buffers.
                if hasattr(self, 'ema') and self.ema is not None:
                    logger.info(f"[Rank {self.global_rank}] Syncing EMA shadow with calibrated normalizer...")
                    for name, buffer in self.model.normalizer.named_buffers():
                        full_name = f"normalizer.{name}"
                        if full_name in self.ema.shadow:
                            self.ema.shadow[full_name] = buffer.data.detach().cpu().clone()
                    logger.info(f"[Rank {self.global_rank}] EMA shadow sync complete.")
            else:
                logger.warning("Dataset missing 'index_path' or 'metadata.ts_columns'. Using identity normalization.")
                
        except Exception as e:
            logger.error(f"[CRITICAL] Normalizer Calibration Failed: {e}")
            logger.warning("SYSTEM SAFETY: Proceeding with Uncalibrated Normalizer. Check data paths!")

        # --- 2. AWR Stats Fitting (Rank 0 Compute + Broadcast) ---
        self._fit_awr_stats_ddp(dataset)

    def _fit_awr_stats_ddp(self, dataset):
        """
        Computes Mean/Std of rewards for advantage whitening.
        
        Computed on Rank 0 and broadcasted to ensure all GPUs use same stats.
        This is critical for DDP consistency - different random samples on
        different ranks would cause divergence.
        """
        stats_tensor = torch.zeros(2, device=self.device)
        
        if self.trainer.is_global_zero:
            logger.info("[Rank 0] Sampling Trajectories for AWR Whitening...")
            rewards_list = []
            
            # [v15.3] SOTA Upgrade: Population-Level Statistics
            # We scan the ENTIRE dataset to compute exact mu/sigma logic.
            # This is "Utmost Quality" for medical safety (prevents batch jitter).
            num_samples = len(dataset)
            indices = range(num_samples)
            
            logger.info(f"[Rank 0] Starting Population Scan (N={num_samples})... This may take a few minutes.")
            indices = torch.arange(num_samples)
            
            for idx in tqdm(indices, desc="AWR Calibration", disable=not self.trainer.is_global_zero):
                sample = dataset[int(idx)]
                
                # Skip invalid samples
                if sample is None:
                    continue
                    
                future = sample["future_data"].unsqueeze(0).to(self.device)
                label = sample["outcome_label"].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Compute reward (using mean to match training scale)
                    r = self.awr_calculator.compute_clinical_reward(future, label, normalizer=None)
                    rewards_list.append(r.mean().item())
            
            if len(rewards_list) > 0:
                r_arr = np.array(rewards_list)
                stats_tensor[0] = float(r_arr.mean())
                stats_tensor[1] = float(r_arr.std())
                logger.info(f"AWR Stats: Mean={stats_tensor[0]:.4f}, Std={stats_tensor[1]:.4f}")
            else:
                logger.warning("No valid samples found for AWR calibration. Using defaults.")
                stats_tensor[0] = 0.0
                stats_tensor[1] = 1.0

        # --- DDP Synchronization ---
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stats_tensor, src=0)
            
        self.awr_calculator.set_stats(mean=stats_tensor[0].item(), std=stats_tensor[1].item())
        self._awr_stats_initialized = True

    def on_before_optimizer_step(self, optimizer):
        """
        SOTA: Surgical Bypass for Gradient Clipping.
        
        This hook runs AFTER gradients are computed but BEFORE optimizer.step().
        In mixed precision training, Lightning has already unscaled gradients.
        
        We manually clip here because:
        1. PyTorch Lightning's auto-clipping can crash with 'fused' optimizers
        2. We have fine-grained control over clipping threshold
        """
        if self.cfg.train.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.cfg.train.grad_clip
            )

    def configure_optimizers(self):
        """
        Robust Optimizer Configuration (SOTA v8.0).
        Uses factory from train_utils for explicit parameter hygiene.
        """
        from icu.utils.train_utils import configure_robust_optimizer, get_cosine_schedule_with_warmup

        # 1. Configure Robust AdamW (Fused + Parameter Hygiene)
        optimizer = configure_robust_optimizer(
            model=self.model,
            learning_rate=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=(0.9, 0.999),
            use_fused=True  # [FIX] Enable Fused Kernels for H100/A100 speedup
        )
        
        # 2. Learning Rate Scheduler
        # [FIX] Use estimated_stepping_batches for accurate total count
        # (Handles accumulation, limit_batches, and DDP sharding correctly)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.cfg.train.get("warmup_ratio", 0.05))
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
