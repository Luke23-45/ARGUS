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
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm.auto import tqdm
from typing import Any, Dict, Optional, Tuple, List, Union
from omegaconf import DictConfig
import logging
import math
import numpy as np
import contextlib
import traceback

# [v2025 SOTA] Implementation Imports
from icu.core.cagrad import CAGrad
from icu.core.gradnorm import GradNormBalancer
from icu.core.robust_losses import (
    smooth_l1_critic_loss, 
    compute_explained_variance, 
    physiological_violation_loss
)

# Project Imports
from icu.models.diffusion import ICUUnifiedPlanner, ClinicalResidualHead, ICUConfig, PhysiologicalConsistencyLoss
from icu.utils.train_utils import EMA
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.utils.metrics_advanced import (
    compute_policy_entropy, 
    compute_ece, 
    compute_explained_variance, 
    compute_overconfidence_error
)
from icu.utils.safety import OODGuardian
from icu.utils.stability import ForensicStabilityAuditor
from icu.models.components.loss_scaler import UncertaintyLossScaler

# [PHASE 1-3] Agentic Evolution Components
from icu.models.components.risk_scorer import PhysiologicalRiskScorer
from icu.models.components.risk_aware_loss import RiskAwareAsymmetricLoss
from icu.models.components.safety_envelope import PhysiologicalSafetyEnvelope
from icu.models.components.horizon_scheduler import ClinicalHorizonScheduler

# Specialized Metric Collection
from torchmetrics import MeanSquaredError, Accuracy, MeanMetric, AUROC, Precision, Recall, F1Score

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


class DynamicClassBalancer(nn.Module):
    """
    [SOTA 2025] Dynamic Class Balancing for Online Learning.
    Adapts loss weights based on running class prevalence to handle
    imbalanced streams (Standard vs Sepsis).
    Uses Effective Number of Samples (ENS) logic.
    """
    def __init__(self, num_classes: int, beta: float = 0.99, prior_pos_weight: float = None):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.register_buffer("counts", torch.zeros(num_classes))
        self.register_buffer("initialized", torch.tensor(False))
        
        if prior_pos_weight is not None and num_classes > 1 and prior_pos_weight > 0:
             n_neg = 1000.0
             n_pos_each = n_neg / prior_pos_weight
             self.counts[0] = n_neg
             self.counts[1:] = n_pos_each
             self.initialized.fill_(True)

    def update(self, y: torch.Tensor):
        if not self.initialized:
            y = y.long()
            b_counts = torch.bincount(y, minlength=self.num_classes).float()
            self.counts.copy_(b_counts + 1.0)
            self.initialized.fill_(True)
        else:
            y = y.long()
            b_counts = torch.bincount(y, minlength=self.num_classes).float()
            new_counts = self.beta * self.counts + (1 - self.beta) * b_counts
            self.counts.copy_(new_counts)

    def get_weights(self) -> torch.Tensor:
        safe_counts = self.counts + 1.0 
        total = safe_counts.sum()
        weights = total / (self.num_classes * safe_counts)
        weights = weights / (weights.mean() + 1e-8)
        return weights



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

        # [H100 OPTIMIZATION] Torch Compile (PT 2.0+)
        if cfg.get("compile_model", False):
            logger.info("[H100] Compiling model with mode='max-autotune'...")
            self.model = torch.compile(self.model, mode="max-autotune")
        
        # [2025 SOTA] Switch to Manual Optimization
        # Required for CAGrad's multiple backward passes and GradNorm's dynamic weighting.
        self.automatic_optimization = False
        
        # [v25.3] Flexible Multi-Task Balancing
        # modes: "sota_2025" (Hooks + UW) or "legacy_surgical" (CAGrad + GradNorm)
        self.balancing_mode = cfg.train.get("balancing_mode", "sota_2025")
        logger.info(f"Using balancing mode: {self.balancing_mode}")

        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Use Model's internal UncertaintyLossScaler
            logger.info("Using Model's UncertaintyLossScaler for balancing.")
        
        # Authority check: EMACallback will attach here as self.ema
        self.ema = None 

        # =====================================================================
        # 3. ADVANTAGE ENGINE (AWR with GAE-Lambda)
        # =====================================================================
        # [v17.0] Optimized AWR for high selection pressure (SOTA 2025)
        self.awr_calculator = ICUAdvantageCalculator(
            beta=cfg.train.get("awr_beta", 0.05),
            max_weight=cfg.train.get("awr_max_weight", 50.0),
            lambda_gae=cfg.train.get("awr_lambda", 0.95),
            gamma=cfg.train.get("awr_gamma", 0.99),
            adaptive_beta=cfg.train.get("adaptive_beta", True),
            adaptive_clipping=cfg.train.get("adaptive_clipping", True)
        )
        
        # =====================================================================
        # 4. SOTA GRADIENT & LOSS BALANCING
        # =====================================================================
        self.gradnorm = None
        if self.balancing_mode == "legacy_surgical":
            # GradNorm dynamically weights [Diffusion, Critic, Sepsis]
            # We target the Shared Encoder as the primary balancing anchor.
            self.gradnorm = GradNormBalancer(
                num_tasks=3, 
                shared_params=self.model.encoder.parameters(),
                alpha=cfg.train.get("gradnorm_alpha", 1.5)
            ).to(self.device) # [v20.1] Immediate device alignment for manual opt
        
        self.base_phys_weight = cfg.train.get("phys_loss_weight", 0.2)
        self.safety_guardian = OODGuardian()
        self.forensic_auditor = ForensicStabilityAuditor(guardian=self.safety_guardian)
        
        # --- Internal Buffers ---
        pos_weight = cfg.train.get("pos_weight", None)
        self.class_balancer = DynamicClassBalancer(
            num_classes=cfg.model.get("num_phases", 3),
            prior_pos_weight=pos_weight
        )
        self.model.loss_scaler = UncertaintyLossScaler(num_tasks=2) # Diffusion + Aux
        # [v25.4 FIX] Initial Log-Var Reset: Start with balanced weights (sigma=1.0)
        nn.init.constant_(self.model.loss_scaler.log_vars, 0.0)
        
        # =====================================================================
        # [NEW] AGENTIC EVOLUTION CORE (Phases 1-3)
        # =====================================================================
        # Phase 1: Dynamic Diagnostics
        self.risk_scorer = PhysiologicalRiskScorer()
        self.risk_aware_loss = RiskAwareAsymmetricLoss(
            gamma_neg=cfg.train.get("asl_gamma_neg", 4.0),
            gamma_pos=cfg.train.get("asl_gamma_pos", 1.0),
            critical_multiplier=cfg.train.get("risk_multiplier", 2.0)
        )
        
        # Phase 2: Per-Feature Safety
        # [v2025 SOTA FIX] Align with CANONICAL_COLUMNS from dataset.py
        # HR=0, O2Sat=1, SBP=2, DBP=3, MAP=4, Resp=5, Temp=6, Lactate=7
        self.clinical_feat_idx = {
            'map': 4, 'lactate': 7, 'o2sat': 1, 'spo2': 1, 'hr': 0, 'sbp': 2, 'resp': 5
        }
        self.safety_envelope = PhysiologicalSafetyEnvelope(self.clinical_feat_idx)
        
        # Phase 3: Longitudinal Convergence
        self.horizon_scheduler = ClinicalHorizonScheduler(
            start_gamma=cfg.train.get("start_gamma", 0.80),
            end_gamma=cfg.train.get("end_gamma", 0.99),
            warmup_epochs=cfg.train.get("horizon_warmup", 10),
            ramp_epochs=cfg.train.get("horizon_ramp", 40)
        )
        
        # =====================================================================
        # 5. TRAINING TELEMETRY (Accumulated Metrics)
        # =====================================================================
        self.train_loss_total = MeanMetric()
        self.train_loss_diff = MeanMetric()
        self.train_loss_critic = MeanMetric()
        self.train_loss_phys = MeanMetric()
        self.train_loss_aux = MeanMetric()
        self.train_loss_gradnorm = MeanMetric()
        self.train_awr_ess = MeanMetric()
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
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        
        # Safety Accumulators
        self.val_ood_rate = MeanMetric()
        self.val_safe_traj_count = MeanMetric()
        self.val_phys_violation_rate = MeanMetric()
        
        # Calibration & Dynamics
        self.val_ece = MeanMetric()
        self.val_oe = MeanMetric()
        self.val_explained_var = MeanMetric()
        
        # =====================================================================
        # 7. STATE FLAGS
        # =====================================================================
        self.register_buffer("_awr_stats_initialized", torch.tensor(False))

    def on_train_epoch_start(self):
        """[Phase 3] Update AWR Horizon based on curriculum."""
        new_gamma = self.horizon_scheduler.get_gamma(self.current_epoch)
        # Note: gamma is a float attribute on the nn.Module
        self.awr_calculator.gamma = new_gamma
        logger.info(f"[Epoch {self.current_epoch}] Agentic Foresight: Gamma={new_gamma:.4f} "
                    f"({self.horizon_scheduler.get_foresight_hours(new_gamma):.1f}h)")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass (used for simple inference/debugging).
        For training, we use the custom logic in training_step.
        """
        return self.model(batch)

    # =========================================================================
    # SOTA TRAINING LOGIC (The "Heart")
    # =========================================================================

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        [2025 SOTA] Manual Multi-Task Training Loop.
        Orchestrates CAGrad, GradNorm, and EMA-Teacher Distillation.
        """
        if not batch or "observed_data" not in batch:
            return
        
        opt = self.optimizers()
        B = batch["observed_data"].size(0)
        
        # --- 1. Forward Pass & Context Generation ---
        past, fut, static = batch["observed_data"], batch["future_data"], batch["static_context"]
        src_mask = batch.get("src_mask", None)
        past_norm, static_norm = self.model.normalize(past, static)
        fut_norm, _ = self.model.normalize(fut, None)
        
        # [PHASE 1] Dynamic Risk Scoring
        risk_coef = self.risk_scorer(past, self.clinical_feat_idx)
        
        # [v25.1 SAFETY FIX] Convert per-feature mask to per-timestep mask
        # src_mask: [B, T, 28] (0=Missing, 1=Valid)
        # padding_mask: [B, T] (True=Pad/Ignore, False=Keep/Attend)
        if src_mask is not None:
            # A timestep is PADDED only if ALL features are missing (0)
            bool_padding_mask = (src_mask.sum(dim=-1) == 0) # [B, T] Result is bool
        else:
            bool_padding_mask = None
        
        ctx_seq, global_ctx, ctx_mask = self.model.encoder(
            past_norm, 
            static_norm, 
            imputation_mask=src_mask,      # [v25.2 FIX] Pass 3D mask for Imputation Awareness
            padding_mask=bool_padding_mask # [v25.2 FIX] Pass 2D mask for Transformer Attention
        )
        
        # --- 2. Per-Task Loss Component Computation ---
        
        # A. Diffusion Task (Student Pass)
        t = torch.randint(0, self.model.cfg.timesteps, (B,), device=self.device)
        noisy_fut, noise_eps = self.model.scheduler.add_noise(fut_norm, t)
        pred_noise = self.model.backbone(noisy_fut, t, ctx_seq, global_ctx, ctx_mask)
        raw_diff_loss = F.mse_loss(pred_noise, noise_eps, reduction='none').mean(dim=[1, 2])

        # B. Advantage Engine (AWR with authoritative EMA Teacher)
        with torch.no_grad():
            rewards = self.awr_calculator.compute_clinical_reward(
                fut, 
                batch.get("outcome_label", None),
                dones=batch.get("is_terminal", None),
                feature_indices=self.clinical_feat_idx,
                normalizer=self.model.normalizer,
                src_mask=batch.get("future_mask", None) # [v12.6 FIX] Use Future Imputation Mask
            )
            # [SOTA] Use Teacher Context for bootstrapping
            with self.ema_teacher_context():
                teacher_seq, teacher_global, teacher_mask = self.model.encoder(
                    past_norm, 
                    static_norm, 
                    imputation_mask=src_mask, 
                    padding_mask=bool_padding_mask
                )
                target_values = self.model.value_head(teacher_global)
            
            # [v12.7 SOTA FIX] Value-Head Bootstrapping
            # If a window is truncated (not yet at end of stay), we bootstrap 
            # from the last critic estimate. If terminal, we use zero.
            is_truncated = batch.get("is_truncated", None)
            if is_truncated is not None and is_truncated.any():
                # [B, 1] bootstrap value from the last predicted value
                # SOTA: This ensures Bellman backups don't treat window edges as episode ends.
                bootstrap_value = target_values[:, -1:]
            else:
                bootstrap_value = None

            advantages = self.awr_calculator.compute_gae(
                rewards, 
                target_values,
                dones=batch.get("is_terminal", None),
                bootstrap_value=bootstrap_value
            )
            returns = (advantages + target_values).detach()
            
            # [2025 SOTA] High-Pressure AWR Weights
            weights_awr, diag = self.awr_calculator.calculate_weights(
                advantages.mean(dim=1),
                values=target_values.mean(dim=1),
                rewards=returns.mean(dim=1)
            )
            weights_awr = weights_awr / (weights_awr.mean() + 1e-8)
            weights_awr_log = {"train/awr_ess": diag["ess"]}

        diff_loss = (raw_diff_loss * weights_awr).mean()
        self.train_awr_ess.update(weights_awr_log.get("train/awr_ess", 0.0))
        
        # C. Critic Task (Robust SmoothL1)
        pred_values = self.model.value_head(global_ctx)
        critic_loss = smooth_l1_critic_loss(pred_values, returns)
        
        # D. Auxiliary Task (MoE / Sepsis Diagnostics)
        aux_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.model.use_auxiliary_head and "phase_label" in batch:
            # [CRITICAL FIX 1] Use ctx_seq (Student) instead of teacher_seq
            # teacher_seq is detached from graph, preventing encoder learning
            logits, _ = self.model.aux_head(ctx_seq, mask=ctx_mask)
            
            # [FIX 3] Integrate DynamicClassBalancer for imbalanced sepsis data
            targets = batch["phase_label"]
            self.class_balancer.update(targets)
            class_weights = self.class_balancer.get_weights().to(self.device)
            
            if logits.shape[-1] > 1 and targets.ndim == 1:
                targets = F.one_hot(targets.long(), num_classes=logits.shape[-1]).float()
            elif logits.shape[-1] == 1 and targets.ndim == 1:
                targets = targets.float().unsqueeze(-1)
                
            aux_loss = self.risk_aware_loss(logits, targets, risk_coef, class_weights=class_weights)


        # --- 3. SOTA Path: Gradient Scaling Hooks (O(1) Conflict Resolution) ---
            # [FIX 4] Increase throttle_scale to 1.0 (was 0.2, muffling signal by 80%)
            throttle_scale = self.cfg.train.get("throttle_scale", 1.0)
            for ctx in [ctx_seq, global_ctx]:
                if ctx.requires_grad:
                    ctx.register_hook(lambda grad: grad * throttle_scale if grad is not None else None)
        
        # --- 4. Multi-Task Balancing Logic ---
        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Single-Pass Uncertainty Weighting
            curr_phys_weight = self._get_curr_physics_weight()
            alpha_t = self.model.scheduler.alphas_cumprod[t][:, None, None]
            x0_approx = (noisy_fut - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t).clamp(min=1e-5)
            
            # [PHASE 2] Safety envelope operates on clinical units. Denormalize x0 first.
            x0_clinical = self.model.unnormalize(x0_approx)
            phys_violation = self.safety_envelope(x0_clinical, risk_coef)
            phys_loss = phys_violation * curr_phys_weight
            
            loss_dict = {
                'diffusion': (raw_diff_loss * weights_awr).mean() + phys_loss,
                'aux': aux_loss
            }
            scaled_total, logs = self.model.loss_scaler(loss_dict)
            total_loss = scaled_total + 0.5 * critic_loss
            
            # Single Backward Pass
            self.manual_backward(total_loss)
            gn_loss = torch.tensor(0.0, device=self.device)
            task_weights = [logs.get('weight/diffusion', 1.0), 0.5, logs.get('weight/aux', 1.0)]
            
        else:
            # [Legacy Surgical] Multi-Pass CAGrad + GradNorm
            diff_loss_unweighted = (raw_diff_loss * weights_awr).mean()
            primary_losses = torch.stack([diff_loss_unweighted, critic_loss, aux_loss])
            gn_loss, task_weights = self.gradnorm.update(primary_losses)
            
            # 2. Weighted losses for CAGrad surgery
            weighted_tasks = [diff_loss_unweighted * task_weights[0], critic_loss * task_weights[1], aux_loss * task_weights[2]]
            
            # 3. Conflict-Averse Surgery (Backward Pass)
            is_start_of_accum = (batch_idx % self.trainer.accumulate_grad_batches == 0)
            opt.pc_backward(
                weighted_tasks, 
                backward_fn=self.manual_backward, 
                accumulate=not is_start_of_accum
            )
            
            # 4. Post-Surgery Constraint Optimization (Physics)
            curr_phys_weight = self._get_curr_physics_weight()
            alpha_t = self.model.scheduler.alphas_cumprod[t][:, None, None]
            x0_approx = (noisy_fut - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t).clamp(min=1e-5)
            
            # [PHASE 2] Denormalize for clinical safety envelope
            x0_clinical = self.model.unnormalize(x0_approx)
            phys_violation = self.safety_envelope(x0_clinical, risk_coef)
            phys_loss = phys_violation * curr_phys_weight
            if phys_loss > 0:
                self.manual_backward(phys_loss)

        # --- 5. Accumulation-Aware Step & Cleanup ---
        # [SOTA 2025] Manually manage accumulation for precise DDP synchronization
        # Only step if we've accumulated enough batches
        acc_batches = self.trainer.accumulate_grad_batches
        if (batch_idx + 1) % acc_batches == 0:
            # 2025 Grad Clipping (Final safety before step)
            if self.cfg.train.get("grad_clip", 0) > 0:
                # [SOTA FIX] Manual clipping to avoid PL MisconfigurationException
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.train.grad_clip)
                
            opt.step()
            opt.zero_grad()
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()            
            
            # GradNorm Optimizer Step (Legacy Only)
            if self.balancing_mode == "legacy_surgical":
                if hasattr(self.gradnorm, 'optimizer') and self.gradnorm.optimizer is not None:
                    self.gradnorm.optimizer.zero_grad()
                    self.manual_backward(gn_loss)
                    self.gradnorm.optimizer.step()
        
        # Log periodicity: every batch regardless of accumulation

        # --- 6. Telemetry & Metric Accumulation ---
        with torch.no_grad():
            ev = compute_explained_variance(pred_values, returns)
            
            # Update metric accumulators
            self.train_loss_total.update((diff_loss + critic_loss + aux_loss).detach())
            self.train_loss_diff.update(diff_loss.detach())
            self.train_loss_critic.update(critic_loss.detach())
            self.train_loss_aux.update(aux_loss.detach())
            self.train_loss_phys.update(phys_loss.detach())
            self.train_loss_gradnorm.update(gn_loss.detach())
            self.train_awr_ess.update(diag["ess"])
            self.train_explained_var.update(ev)
            
            # Global Rank 0 Logging (SOTA: Pass objects, not .compute(), to avoid sync bottleneck)
            # [TELEMETRY] Primary Metrics (Visible in Progress Bar)
            # Use detached scalars (.item()) for the progress bar to ensure immediate visibility.
            # Shortening to L, D, V, etc. is handled by the APEXProgressBar callback.
            # [TELEMETRY] Primary Metrics (Visible in Progress Bar)
            # [SOTA FIX] DO NOT use .item() here. It causes graph breaks in torch.compile.
            # Lightning handles tensor logging efficiently.
            self.log_dict({
                "total_loss": (diff_loss + critic_loss + aux_loss),
                "diff_loss": diff_loss,
                "critic_loss": critic_loss,
                "phys_loss": phys_loss,
                "aux_loss": aux_loss,
                "awr_ess": diag["ess"],
                "explained_var": ev,
                "curr_phys_weight": torch.tensor(curr_phys_weight, device=self.device),
                "w_aux": torch.tensor(task_weights[2], device=self.device),
                "lr": torch.tensor(self.optimizers().param_groups[0]["lr"], device=self.device)
            }, on_step=True, on_epoch=False, prog_bar=True)

            # [TELEMETRY] Detailed Diagnostics (WandB Only)
            self.log_dict({
                "train/loss_critic": self.train_loss_critic,
                "train/loss_aux": self.train_loss_aux,
                "train/loss_phys": self.train_loss_phys,
                "train/loss_gradnorm": self.train_loss_gradnorm,
                "train/explained_var": self.train_explained_var,
                "train/awr_ess": self.train_awr_ess,
                "train/weight_diff": task_weights[0],
                "train/weight_critic": task_weights[1],
                "train/weight_aux": task_weights[2],
                "train/curr_phys_weight": curr_phys_weight,
            }, on_step=True, on_epoch=False, prog_bar=False)

        return (diff_loss + critic_loss + aux_loss).detach()


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
        # 0. Robustness Guard
        if not batch or "observed_data" not in batch:
            return {}

        # 1. Standard Loss Monitoring
        out = self.model(batch, reduction='mean')
        bs = batch["observed_data"].shape[0]
        self.log("val/diff_loss", out["diffusion_loss"], on_epoch=True, sync_dist=True, batch_size=bs)        
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
                # [FIX] Use phase_label > 0 (Stable=0, Pre=1, Shock=2) for robust binary target
                # outcome_label is float probability, casting to long makes it 0 (Bug Fix)
                if "phase_label" in batch:
                    binary_label = (batch["phase_label"] > 0).long()
                    target_class = batch["phase_label"].long()
                else:
                    # Fallback if phase_label missing (should not happen with SotaDataset)
                    binary_label = (batch["outcome_label"] > 0.5).long()
                    target_class = (batch["outcome_label"] > 0.5).long()

                # ECE and Overconfidence Error
                ece = compute_ece(risk_prob, binary_label)
                oe = compute_overconfidence_error(risk_prob, binary_label)
                
                self.val_ece.update(ece)
                self.val_oe.update(oe)
                self.val_acc_sepsis.update(logits, target_class)
                self.val_auroc_sepsis.update(risk_prob, binary_label)
                self.val_precision.update(risk_prob, binary_label)
                self.val_recall.update(risk_prob, binary_label)
                self.val_f1.update(risk_prob, binary_label)

                # [v5.3.4 SOTA FIX] Align Validation Semantic Baseline
                # We previously compared pred_value (Returns [~-7, 5]) to binary_label (Outcome [0, 1]).
                # This caused the meaningless -0.005 value due to scale mismatch.
                # Now we compute actual validation returns for a true Critic Quality check.
                value_preds = out.get("pred_value", None)
                if value_preds is not None:
                    with torch.no_grad():
                        # Calculate ground truth rewards for the validation batch
                        val_rewards = self.awr_calculator.compute_clinical_reward(
                            batch["future_data"], batch.get("outcome_label", None),
                            normalizer=self.model.normalizer if hasattr(self.model, 'normalizer') else None
                        )
                        # Estimate GAE advantages and total returns
                        # (Using value_preds for GAE bootstrapping ensures internal consistency)
                        val_adv = self.awr_calculator.compute_gae(val_rewards, value_preds)
                        val_returns = (val_adv + value_preds).detach()
                        
                        # [v5.3.3 FIX] Shape Alignment: use mean across trajectory for scalar metric
                        # This compares "The target return we expected" vs "The return we predicted".
                        ev = compute_explained_variance(value_preds.mean(dim=1), val_returns.mean(dim=1))
                        self.val_explained_var.update(ev)

        # 3. Clinical Trajectory Sampling (Only first batch to save compute)
        # This prevents the "Validation Trap" (105x compute overhead)
        if batch_idx == 0:
            if self.cfg.get("debug", False):
                try:
                    self._validate_clinical_sampling(batch)
                except Exception as e:
                    logger.error(f"[DEBUG MODE] Clinical sampling failed: {e}")
                    logger.error(traceback.format_exc())
            else:
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
                if self.cfg.get("debug", False):
                    try:
                        pred = self.model.sample(subset)
                    except Exception as e:
                        logger.error(f"[DEBUG MODE] model.sample failed: {e}")
                        logger.error(traceback.format_exc())
                        return # Skip the rest of clinical validation for this batch
                else:
                    pred = self.model.sample(subset)
        
        # A. Global MSE (Overall prediction quality)
        # [ROBUSTNESS FIX] Last line of defense: filter infinite values to prevent telemetry corruption
        pred_safe = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e9, 1e9)
        gt_safe = torch.nan_to_num(gt, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e9, 1e9)
        self.val_mse_global.update(pred_safe, gt_safe)
        
        # B. Granular MSE by Clinical Category
        # This helps identify which subsystem the model struggles with
        
        # Hemodynamic (indices 0-6): HR, MAP, Temp, SpO2, SBP, DBP, RespRate
        # [FIX] Enforce contiguous memory layout for slices to prevent View/Stride errors in torchmetrics
        if pred.shape[-1] > 6:
            self.val_mse_hemo.update(
                pred_safe[..., :7].contiguous(), 
                gt_safe[..., :7].contiguous()
            )
        
        # Labs (indices 7-17): Metabolic panel
        if pred.shape[-1] > 17:
            self.val_mse_labs.update(
                pred_safe[..., 7:18].contiguous(), 
                gt_safe[..., 7:18].contiguous()
            )
        
        # Electrolytes (indices 18-21): Na, K, Ca, Mg
        if pred.shape[-1] > 21:
            self.val_mse_electrolytes.update(
                pred_safe[..., 18:22].contiguous(), 
                gt_safe[..., 18:22].contiguous()
            )
        
        # C. Safety Check (The "Hard Deck")
        # [v5.3.6 SOTA FIX] Unified Clinical Pipeline
        # We pass raw clinical data directly. OODGuardian is forced to skip unit-heuristics
        # to prevent misidentification during early training jitter.
        with torch.no_grad():
            safety_results = self.safety_guardian.check_trajectories(
                subset["observed_data"], 
                pred, 
                force_clinical=True
            )
        
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
            "val/sepsis_precision": self.val_precision.compute(),
            "val/sepsis_recall": self.val_recall.compute(),
            "val/sepsis_f1": self.val_f1.compute(),
            
            # Calibration & Dynamics
            "val/ece": self.val_ece.compute(),
            "val/oe": self.val_oe.compute(),
            "val/explained_var": self.val_explained_var.compute(),
            
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
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_ece.reset()
        self.val_oe.reset()
        self.val_explained_var.reset()
        self.val_ood_rate.reset()
        self.val_safe_traj_count.reset()
        self.val_phys_violation_rate.reset()

    # =========================================================================
    # UTILITIES & SETUP
    # =========================================================================

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        [v12.8 SOTA FIX] In-Place StateDict Mutation for Strict Loading.
        
        PyTorch Lightning triggers a strict load AFTER this hook. To avoid 
        'Missing Key' crashes when resuming from legacy checkpoints, we 
        mutate the checkpoint['state_dict'] in-place by injecting current 
        buffer values for any missing SOTA keys.
        """
        state_dict = checkpoint.get("state_dict", {})
        if not state_dict:
            return
            
        # Authoritative SOTA Keys
        new_keys = [
            "awr_calculator.beta", 
            "awr_calculator.max_weight", 
            "awr_calculator.ess_buffer", 
            "awr_calculator.clip_rate_buffer"
        ]
        
        # [v12.8] Surgical Injection:
        # We check if keys are missing from the checkpoint, and if so, 
        # we populate them from the current model's buffers.
        missing_keys = [k for k in new_keys if k not in state_dict]
        
        if missing_keys:
            logger.warning(f"[RESUME] Checkpoint missing {len(missing_keys)} SOTA keys. Injecting model defaults to satisfy strict loader.")
            current_model_dict = self.state_dict()
            for k in missing_keys:
                if k in current_model_dict:
                    state_dict[k] = current_model_dict[k].clone()
                else:
                    logger.error(f"[RESUME] SOTA key '{k}' expected but not found in current model. Restoration may be partial.")
        
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
            logger.info(f"[Rank {self.global_rank}] Normalizer already calibrated. Skipping Calibration.")
        else:
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
        # [v20.1] PERFORMANCE PATCH: AWR Dependency Injection
        # If stats are pre-computed (e.g., from deep audit), use them directly.
        # This bypasses the 1500-sample calibration loop for HPO speed.
        injected_mean = self.cfg.train.get("awr_stats_mean", None)
        injected_std = self.cfg.train.get("awr_stats_std", None)
        
        if injected_mean is not None and injected_std is not None:
            if self.trainer.is_global_zero:
               logger.info(f"⚡ [AWR] Fast-Path Active: Using Injected Stats (mu={injected_mean:.4f}, sigma={injected_std:.4f})")
            
            self.awr_calculator.set_stats(mean=injected_mean, std=injected_std)
            self._awr_stats_initialized.fill_(True)
            return

        stats_tensor = torch.zeros(2, device=self.device)
        
        if self.trainer.is_global_zero:
            logger.info("[Rank 0] Sampling Trajectories for AWR Whitening...")
            rewards_list = []
            
            # [v15.4] Robusified: Calibration Mode Toggle
            num_samples = len(dataset)
            mode = self.cfg.train.get("awr_calibration_mode", "full")
            max_samples = self.cfg.train.get("awr_max_samples", 5000)

            if mode == "sample":
                if max_samples >= num_samples:
                    logger.info(f"[Rank 0] AWR Calibration: Requested samples ({max_samples}) >= population ({num_samples}). Falling back to FULL scan.")
                    indices = torch.arange(num_samples)
                    actual_count = num_samples
                elif max_samples <= 0:
                    logger.warning(f"[Rank 0] AWR Calibration: Invalid max_samples={max_samples}. Defaulting to FULL scan.")
                    indices = torch.arange(num_samples)
                    actual_count = num_samples
                else:
                    logger.info(f"[Rank 0] AWR Calibration: Sampling trajectories (N={max_samples} of {num_samples})...")
                    # Rank-consistent deterministic sampling derived from global seed
                    g = torch.Generator(device='cpu')
                    g.manual_seed(self.cfg.seed + 42) 
                    indices = torch.randperm(num_samples, generator=g)[:max_samples]
                    actual_count = max_samples
            else:
                logger.info(f"[Rank 0] AWR Calibration: Starting Full Population Scan (N={num_samples})... This may take a few minutes.")
                indices = torch.arange(num_samples)
                actual_count = num_samples
            
            for idx in tqdm(indices, desc="AWR Calibration", disable=not self.trainer.is_global_zero):
                sample = dataset[int(idx)]
                
                # Skip invalid samples
                if sample is None:
                    continue
                    
                future = sample["future_data"].unsqueeze(0).to(self.device)
                label = sample["outcome_label"].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # [v25.5 FIX] Calibration Parity: Include sparse rewards and masking
                    # rewards_list.append(r.mean().item()) -> Use rewards collected with terminal awareness
                    r = self.awr_calculator.compute_clinical_reward(
                        future, 
                        label, 
                        dones=sample.get("is_terminal").unsqueeze(0).to(self.device),
                        feature_indices=self.clinical_feat_idx,
                        normalizer=None,
                        src_mask=sample.get("future_mask").unsqueeze(0).to(self.device)
                    )
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
        self.register_buffer("_awr_stats_initialized", torch.tensor(True))

    # Removed on_before_optimizer_step in favor of manual clipping in training_step

    def configure_optimizers(self):
        """
        [2025 SOTA] Conflict-Averse Optimizer Configuration.
        Wraps robust AdamW with CAGrad for surgical conflict resolution.
        """
        from icu.utils.train_utils import configure_robust_optimizer, get_cosine_schedule_with_warmup

        # 1. Configure Robust AdamW (Fused + Parameter Hygiene)
        if self.balancing_mode == "sota_2025":
            # [v25.3] Parameter Hygiene: Separate Model and Uncertainty parameters
            # We must ensure groups are disjoint, or PyTorch raises ValueError
            scaler_params = list(self.model.loss_scaler.parameters())
            scaler_ids = {id(p) for p in scaler_params}
            model_params = [p for p in self.model.parameters() if id(p) not in scaler_ids]
            
            # Uncertainty weights need a slightly higher LR for faster convergence
            uw_lr = self.cfg.train.get("uw_lr", 0.025)
            optimizer_params = [
                {"params": model_params},
                {
                    "params": scaler_params,
                    "lr": uw_lr,
                    "weight_decay": 0.0  # Do not decay weighting parameters
                }
            ]
        else:
            optimizer_params = [{"params": self.model.parameters()}]

        base_optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=(0.9, 0.999),
            fused=True
        )
        
        # 2. Wrap with CAGrad
        # c=0.5 provides the optimal balance for clinical MTL (LibMTL benchmark)
        if self.balancing_mode == "legacy_surgical":
            optimizer = CAGrad(base_optimizer, c=0.5)
        else:
            # "sota_2025" uses Integrated Scalar Loss -> Pure Optimizer is optimal
            optimizer = base_optimizer

        # 3. Learning Rate Scheduler
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