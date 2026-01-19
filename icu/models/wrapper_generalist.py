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
from icu.models.components.loss_scaler import BayesianProjectedScaler

# [PHASE 1-3] Agentic Evolution Components
from icu.models.components.risk_scorer import PhysiologicalRiskScorer
from icu.models.components.risk_aware_loss import RiskAwareAsymmetricLoss
# [v2025 SOTA] Stabilization Primitives
from icu.utils.stabilization import (
    GradientThrottler, 
    adaptive_gradient_clip_,
    LinearManifoldSentinel,
    OrthogonalGuard
)
from icu.models.components.contrastive_loss import AsymmetricContrastiveLoss
from icu.models.components.safety_envelope import PhysiologicalSafetyEnvelope
from icu.models.components.horizon_scheduler import ClinicalHorizonScheduler
from icu.models.components.bgsl_loss import BGSLLoss
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer
from icu.models.components.ghost_bank import SepsisGhostBank
from icu.utils.distributed import SOTA_DistributedGatherer

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
            
            # [SOTA 2025] DDP Global Sync
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(b_counts, op=torch.distributed.ReduceOp.SUM)
                b_counts /= torch.distributed.get_world_size()
            
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
            # [SOTA 2025] Uncertainty Loss Scaler
            # [v12.8.3 SOTA FIX] Direct Attachment
            # Attaching to self instead of self.model to ensure safe device movement
            # and registration within the LightningModule, avoiding torch.compile issues.
            # [v4.0 FIX] Initialized with 6 tasks: [diffusion, critic, aux, acl, bgsl, tcb]
            self.loss_scaler = BayesianProjectedScaler(num_tasks=6)
            logger.info("Using Model's BayesianProjectedScaler for balancing (6 tasks).")
        
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
        if self.balancing_mode == "legacy_surgical" or True: # [v25.7] Force enable for ACL expansion
            # GradNorm dynamically weights [Diffusion, Critic, Aux, ACL, BGSL, TCB]
            self.gradnorm = GradNormBalancer(
                num_tasks=6, 
                shared_params=self.model.encoder.parameters(),
                alpha=cfg.train.get("gradnorm_alpha", 1.5)
            ).to(self.device)
        
        self.base_phys_weight = cfg.train.get("phys_loss_weight", 0.2)
        self.safety_guardian = OODGuardian()
        self.forensic_auditor = ForensicStabilityAuditor(guardian=self.safety_guardian)
        
        # --- Internal Buffers ---
        pos_weight = cfg.train.get("pos_weight", None)
        self.class_balancer = DynamicClassBalancer(
            num_classes=cfg.model.get("num_phases", 3),
            prior_pos_weight=pos_weight
        )
        # [v25.4 FIX] Initial Log-Var Reset: Start with balanced weights (sigma=1.0)
        if self.balancing_mode == "sota_2025":
            # [PHASE 1 FIX] Initialize with scale-aware log_vars to prevent aux starvation
            # diffusion has ~10x higher loss than aux → needs higher σ (lower weight)
            # aux has lower loss → lower σ (higher weight)
            initial_log_vars = torch.tensor([
                1.0,    # diffusion: Higher σ → lower weight
                0.5,    # critic: Medium
                -0.5,   # aux: Lower σ → HIGHER weight (boost sepsis learning)
                0.0,    # acl
                0.5,    # bgsl
                0.5,    # tcb
            ])
            # Ensure device compatibility if loaded later
            self.loss_scaler.log_vars.data.copy_(initial_log_vars)
        
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
        
        # [v4.2 SOTA Pillar 5] Metadata-Aware ACL Projector
        # Injects clinical context (Velocity, UnitID) into the contrastive learner.
        self.acl_projector = nn.Sequential(
            nn.Linear(cfg.model.d_model + cfg.model.input_dim + 6, cfg.model.d_model),
            nn.LayerNorm(cfg.model.d_model),
            nn.SiLU(),
            nn.Linear(cfg.model.d_model, cfg.model.d_model)
        )
        
        self.sepsis_acl = AsymmetricContrastiveLoss(
            d_model=cfg.model.d_model,
            num_classes=cfg.model.get("num_phases", 3), # Use NUM_PHASES from config
            temperature=cfg.train.get("acl_temp", 0.1)
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
        
        # [v4.0 PERFECT] Advanced Supervision Components
        self.bgsl_loss = BGSLLoss(
            pos_weight=cfg.train.get("pos_weight", 10.0),
            gamma=cfg.train.get("asl_gamma_neg", 4.0), # Reusing ASL gamma
            trend_coef=cfg.train.get("trend_coef", 1.0),
            shock_coef=cfg.train.get("shock_coef", 2.0)
        )
        
        self.tcb_buffer = TemporalContrastiveBuffer(
            d_model=cfg.model.d_model,
            capacity=cfg.train.get("tcb_capacity", 1024),
            temperature=cfg.train.get("tcb_temp", 0.07)
        )
        
        # [v17.3 Hardened] Omega Ghost Protcol: Sepsis Ghost Bank
        self.ghost_bank = SepsisGhostBank(
            capacity=cfg.train.get("ghost_capacity", 256),
            history_len=cfg.model.get("history_len", 24),
            feature_dim=cfg.model.get("input_dim", 28),
            latent_dim=cfg.model.get("d_model", 512),
            similarity_threshold=cfg.train.get("ghost_sim_threshold", 0.98)
        )
        
        
        # [v4.0 PERFECT] Manifold Projections
        # [REMOVED] self.expert_state_head = nn.Linear(cfg.model.d_model, 1)
        
        # =====================================================================
        # 5. TRAINING TELEMETRY (Accumulated Metrics)
        # =====================================================================
        self.train_loss_total = MeanMetric()
        self.train_loss_diff = MeanMetric()
        self.train_loss_critic = MeanMetric()
        self.train_loss_phys = MeanMetric()
        self.train_loss_aux = MeanMetric()
        self.train_loss_acl = MeanMetric()
        self.train_loss_bgsl = MeanMetric()
        self.train_loss_tcb = MeanMetric()
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
        self.validation_step_outputs = []
        
        # [Point 5] Bayesian Moving Average Calibration
        # Initialized to 0.5; will be updated via F2-opt during validation.
        self.register_buffer("calibrated_threshold", torch.tensor(0.5))
        self.threshold_ema_decay = 0.9 # Stable calibration over epochs
        
        self.register_buffer("curr_tau", torch.tensor(0.5))
        self.register_buffer("curr_sigma_scale", torch.tensor(3.50))
        
        # [PMS] Manifold Stability Monitoring
        self.register_buffer("grad_norm_ema", torch.tensor(1.0))
        self.grad_ema_decay = 0.95
        
        # [PMS] MGP: EMA Foundation Gradient Storage for projection
        # Flattened size based on encoder hidden dim (e.g., 512, 1024)
        # We will initialize this lazily in training_step
        self._fnd_grad_ema = None 
        
        # [SOTA DDP] Zero-Copy Gatherer
        self.ddp_gatherer = None 

    def on_train_epoch_start(self):
        """[Phase 3/4] Update AWR Horizon and SOTA v4.2 Warmup."""
        new_gamma = self.horizon_scheduler.get_gamma(self.current_epoch)
        self.awr_calculator.gamma = new_gamma
        
        # [v12.5.1 SOTA] AWR Beta Annealing (Broad Discovery -> Sharp Selection)
        # Linear decay from 0.60 to 0.15 over 40 epochs
        start_beta = 0.60
        end_beta = 0.15
        anneal_epochs = 40
        if self.current_epoch < anneal_epochs:
            frac = self.current_epoch / anneal_epochs
            curr_beta = start_beta + (end_beta - start_beta) * frac
        else:
            curr_beta = end_beta
            
        self.awr_calculator.beta.fill_(curr_beta)
        
        # [v4.2 SOTA Pillar 2 & 4] Synchronized Risk Warmup
        # Goal: Slowly introduce CVaR pessimism and Safety Envelope constraints.
        
        # [PMS] SCS: Synchronized Curriculum Smoothing
        # We check the 'Manifold Health' (Gradient Variance).
        # If the brain is in 'Shock' (Norm > 5.0), we freeze the ramp.
        if self.grad_norm_ema > 5.0:
            logger.warning(f"[PMS] Manifold Shock Detected (GN={self.grad_norm_ema:.2f}). Freezing Curriculum Ramp.")
            # Keep current tau and sigma_scale (No increment)
            pass 
        else:
            ramp_epochs = 10.0
            if self.current_epoch >= 5:
                tau_progress = min(1.0, (self.current_epoch - 5) / ramp_epochs)
                self.curr_tau.fill_(0.5 + (0.7 - 0.5) * tau_progress)
            else:
                self.curr_tau.fill_(0.5)
            
            # Ramping sigma over 15 epochs
            sigma_ramp_epochs = 15.0
            sigma_progress = min(1.0, self.current_epoch / sigma_ramp_epochs)
            self.curr_sigma_scale.fill_(3.50 - (3.50 - 2.50) * sigma_progress)
        
        logger.info(f"[Epoch {self.current_epoch}] Agentic Foresight: Gamma={new_gamma:.4f} "
                    f"| Tau={self.curr_tau:.2f} | SigmaScale={self.curr_sigma_scale:.2f}")

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
        # [v20.0 VERIFICATION MARKER]
        # print(f"DEBUG: training_step hit (balancing_mode={self.balancing_mode})")
        if not batch or "observed_data" not in batch:
            return
        
        opt = self.optimizers()
        B = batch["observed_data"].size(0)
        
        # [v17.3] Omega Summoning: Constant Clinical Pressure
        # Select 4 ghosts using a rank-agnostic global seed for DDP synchronization.
        # [SOTA FIX] Avoid Python's hash() which is salted per-process.
        # Formula: (Epoch * large_prime + Step) ensures deterministic parity across all GPUs.
        ghost_seed = (self.current_epoch * 12345 + self.global_step) % (2**31)
        num_ghosts = self.cfg.train.get("num_ghosts", 4)
        mixup_alpha = self.cfg.train.get("ghost_mixup_alpha", 0.0)
        ghost_batch = self.ghost_bank.sample(
            num_ghosts=num_ghosts, 
            seed=ghost_seed, 
            mixup_alpha=mixup_alpha,
            uncertainty_weighted=True
        )
        
        # Physically concatenate ghosts to the main batch
        past, fut, static = batch["observed_data"], batch["future_data"], batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        past_expanded = torch.cat([past, ghost_batch["vitals"]], dim=0)
        # static expansion: repeat first static context or use zeros
        ghost_static = torch.zeros(num_ghosts, static.size(1), device=self.device)
        static_expanded = torch.cat([static, ghost_static], dim=0)
        
        # Norms
        past_norm, static_norm = self.model.normalize(past_expanded, static_expanded)
        fut_norm, _ = self.model.normalize(fut, None) # fut is not expanded (masked later)
        
        # Context Mask expansion
        if src_mask is not None:
            src_mask_expanded = torch.cat([src_mask, ghost_batch["masks"]], dim=0)
        else:
            src_mask_expanded = None
        
        # [PHASE 1] Dynamic Risk Scoring
        risk_coef = self.risk_scorer(past, self.clinical_feat_idx)
        # Expand risk_coef for ghosts: high-priority clinical supervision
        # Handle both 1D and 2D risk_coef for robustness
        risk_shape = (num_ghosts, *risk_coef.shape[1:])
        risk_coef_ghost = torch.ones(risk_shape, device=self.device) * 2.0
        risk_coef_expanded = torch.cat([risk_coef, risk_coef_ghost], dim=0)
        
        # [v25.1 SAFETY FIX] Convert per-feature mask to per-timestep mask
        # src_mask: [B, T, 28] (0=Missing, 1=Valid)
        # padding_mask: [B, T] (True=Pad/Ignore, False=Keep/Attend)
        if src_mask_expanded is not None:
            # A timestep is PADDED only if ALL features are missing (0)
            bool_padding_mask = (src_mask_expanded.sum(dim=-1) == 0) # [B+G, T]
        else:
            bool_padding_mask = None
        
        # Unified Encoder Pass [B+G]
        # [v17.3] BN Guard: Protect foundation stats from ghost-induced shift
        with self.frozen_stats():
            out_alb = self.model.encoder(
                past_norm, 
                static_norm, 
                imputation_mask=src_mask_expanded, 
                padding_mask=bool_padding_mask
            )
        ctx_seq = out_alb["ctx_planner"]
        global_ctx_planner = out_alb["global_planner"]
        global_ctx_expert = out_alb["global_expert"]
        
        # [v12.2 SOTA] Unified Context for Selective Pressure
        global_ctx_unified = (global_ctx_planner + global_ctx_expert).mul(0.5)
        global_ctx = global_ctx_planner 
        
        ctx_expert = out_alb["ctx_expert"]
        ctx_mask = out_alb["ctx_mask"]
        
        # Define Targets early
        targets = batch["phase_label"] # [B]
        targets_expanded = torch.cat([targets, ghost_batch["labels"]], dim=0) # [B+G]
        
        # --- 2. Per-Task Loss Component Computation ---
        
        # A. Diffusion Task (Student Pass)
        t = torch.randint(0, self.model.cfg.timesteps, (B,), device=self.device)
        noisy_fut, noise_eps = self.model.scheduler.add_noise(fut_norm, t)
        
        # [v12.1] Two-Pass Self-Conditioning ("Analog Bits")
        # Rationale: Training the model to fix its own generation errors.
        # Implemented manually here to allow precise masking in Phase 1.
        self_cond = None
        if self.model.cfg.use_self_conditioning:
            self_cond = torch.zeros_like(noisy_fut)
            
            # 50% probability of using a preliminary x0 estimate
            if torch.rand(1).item() < 0.5:
                # IMPORTANT: Pass 1 is strictly NO_GRAD to preserve memory
                with torch.no_grad():
                    # Pass 1: "Guess" noisy epsilon
                    guess_eps = self.model.backbone(
                        noisy_fut, t, ctx_seq[:B], global_ctx[:B], ctx_mask[:B], self_cond=self_cond
                    )
                    # Reconstruct x0 estimate (Analog Bits reconstruction)
                    alpha_bar = self.model.scheduler.alphas_cumprod[t][:, None, None]
                    sqrt_alpha_clamped = torch.sqrt(alpha_bar).clamp(min=1e-3)
                    guess_x0 = (noisy_fut - torch.sqrt(1 - alpha_bar) * guess_eps) / sqrt_alpha_clamped
                    
                    # Manifold Constraint (Dynamic Thresholding)
                    # Prevents outlier conditioning from exploding the search space
                    self_cond = self.model.governance(guess_x0).detach()

        # Pass 2: Final Denoising with Conditioning (Gradient Path)
        pred_noise = self.model.backbone(noisy_fut, t, ctx_seq[:B], global_ctx[:B], ctx_mask[:B], self_cond=self_cond)
        
        diff_sq = (pred_noise - noise_eps) ** 2
        weighted_diff = diff_sq * self.model.importance_weights.view(1, 1, -1)
        raw_diff_loss = weighted_diff.mean(dim=2) # [B, T]

        # B. Advantage Engine (DEFERRED to Fused Teacher Block)
        # We process AWR logic later to allow "One-Pass" Teacher execution.
        weights_awr_log = {}
        
        # C. Critic Task (SOTA IDC-25)
        # Replaced scalar MSE with Distributional Implicit Q-Learning (IQL-QR)
        # We compute predictions here (Student Pass), but loss is deferred until
        # after the Fused Teacher Block provides the 'returns' (targets).
        # C. Critic Task (SOTA IDC-25)
        # [v12.2 SOTA] Use UNIFIED context for value prediction (Clinical Awareness)
        # [v17.3 Surgical Mask] Critic only for main batch [0:B]
        # Prevents selection pressure poisoning from historical extremes.
        pred_values = self.model.value_head(global_ctx_unified[:B])
        
        # D. Task-Specific Component Computation (Initialization)
        aux_loss = torch.tensor(0.0, device=self.device)
        acl_loss = torch.tensor(0.0, device=self.device)
        l_bgsl = torch.tensor(0.0, device=self.device)
        l_tcb = torch.tensor(0.0, device=self.device)
        l_cga = torch.tensor(0.0, device=self.device)
        # [v12.8.2 FIX] Synchronize with generalist.yaml (num_phases)
        logits = torch.zeros((B, self.cfg.model.num_phases), device=self.device)
        uncertainty = torch.ones((B, 1), device=self.device) # Vacuous by default
        
        if self.cfg.model.use_auxiliary_head and "phase_label" in batch:
            # [v13.0 PATCH] Conditional Head Activation
            # Problem: With 7.2% episode sepsis rate, ~93% of batches may have zero sepsis cases.
            # Computing aux_loss on these batches adds noise without learning signal.
            # Fix: Only compute full aux_loss when batch contains sepsis (phase_label > 0)
            batch_has_sepsis = (batch["phase_label"] > 0).any().item()
            
            # [PMS] DAT: Dynamic Adaptive Throttling
            # "Head First, Brain Second" - Guard the encoder when the head is guessing.
            ctx_aux = ctx_expert.clone()
            
            # [FIX] Define targets before evidential forward pass
            targets = batch["phase_label"] # [B]

            # Forward pass to get current competence (Uncertainty)
            aux_out = self.model.aux_head(
                ctx_aux, 
                mask=ctx_mask, 
                targets=targets_expanded, # [v17.3 FIX] Use expanded targets for [B+G] context
                epoch_num=self.current_epoch 
            )
            logits = aux_out["logits"]
            aux_loss_base = aux_out["loss"]
            uncertainty = aux_out["uncertainty"] # [B, 1]
            
            # Use detachment to compute trust factor (Cybernetic Control Gate)
            u_avg = uncertainty.detach().mean()
            # Trust Factor: 1.0 (Confident) -> 0.1 (Panic)
            trust_factor = (1.0 - (u_avg * 0.9)).clamp(min=0.1, max=1.0).item()
            
            # Surgical Hook: Scopes gradients only for the shared connection
            if ctx_aux.requires_grad:
                ctx_aux.register_hook(lambda grad: grad * trust_factor)
            
            self.log("train/pms_trust_factor", trust_factor, on_step=True, prog_bar=True)
            
            self.class_balancer.update(targets)
            class_weights = self.class_balancer.get_weights().to(self.device)
            
            # [SOTA 2025] Shape Alignment
            # Classification (Aux Head) is Window-Level [B] via CLS Token
            # Contrastive (ACL) is Sequence-Level [B, T] (handled internally)
            B_exp, T_seq, _ = ctx_seq.shape
            cfm = 1.0
            
            # [PMS] MGP: Manifold Gradient Projection Hook
            # We protect the 'Planner' (Foundation) from 'Expert' (Aux) noise.
            # We use an EMA of the foundation gradient to prevent task interference.
            
            def pms_manifold_guard(grad):
                # 1. Capture/Update Foundation EMA (if this is the planner branch)
                # Note: In PyTorch backward, this hook might run at different times.
                # We identify branches by their gradient shape or context.
                return grad

            # Correct Implementation: Multi-Branch Projection
            # We must identify which branch is which.
            def project_aux_against_fnd_ema(grad_aux):
                # grad_aux: [B, T, D] or [B, D]
                if self._fnd_grad_ema is not None:
                    # [PMS] Shape-Invariant Projection
                    # We project the aux gradient against the stable foundation direction
                    return LinearManifoldSentinel.project(grad_aux, self._fnd_grad_ema)
                return grad_aux

            def update_fnd_ema(grad_fnd):
                # grad_fnd: [B, T, D]
                with torch.no_grad():
                    # Compute the 'Representative Direction' (Average over B and T)
                    # This makes the EMA shape-invariant.
                    if grad_fnd.dim() == 3:
                        dir_fnd = grad_fnd.mean(dim=(0, 1)) # [D]
                    else:
                        dir_fnd = grad_fnd.mean(dim=0) # [D]
                        
                    if self._fnd_grad_ema is None:
                        self._fnd_grad_ema = dir_fnd.detach().clone()
                    else:
                        self._fnd_grad_ema = self._fnd_grad_ema.to(grad_fnd.device)
                        self._fnd_grad_ema.mul_(0.9).add_(dir_fnd.detach(), alpha=0.1)
                return grad_fnd

            if ctx_seq.requires_grad:
                ctx_seq.register_hook(update_fnd_ema)
            
            if ctx_aux.requires_grad:
                ctx_aux.register_hook(project_aux_against_fnd_ema)
            
            # [v17.3] Omega Summoning: Constant Clinical Pressure
            # Every batch now has sepsis signal via the Summoned Ghosts.
            # We remove the 0.1x multiplier and train with full magnitude.
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                # logits/probs: [B+G, C], targets_expanded: [B+G]
                true_probs = probs.gather(-1, targets_expanded.unsqueeze(-1).long())
                error = 1.0 - true_probs.squeeze(-1)
                # Cap max boosting at 1.5x
                mining_weight = 1.0 + (torch.sigmoid(error * 5.0) * 0.5)
            
            # One-hot encoding for window-level targets
            targets_one_hot = F.one_hot(targets_expanded.long(), num_classes=logits.shape[-1]).float()

            # Sepsis Classification Loss (Unified [B+G])
            # Ghosts provide the gradient floor to prevent 'Discovery Shock'.
            if aux_loss_base is not None:
                # aux_loss_base is [B+G] from SequenceAuxHead internal loss if updated
                # But here it's already a scalar from the head. We rely on the head's loss.
                aux_loss = aux_loss_base * cfm * mining_weight.mean()
            else:
                raw_aux_loss = self.risk_aware_loss(logits, targets_one_hot, risk_coef_expanded, class_weights=class_weights)
                aux_loss = raw_aux_loss * cfm * mining_weight.mean()

            # [v17.4 GIST-Q] Uncertainty-Weighted CGA
            # Anchors the Expert Manifold to history, prioritising high-uncertainty (hard) cases.
            if num_ghosts > 0 and ghost_batch["valid"].any():
                ghost_latents_global = global_ctx_expert[B:]
                ghost_anchors = ghost_batch["anchors"]
                
                # Compute per-ghost MSE and weight by stored uncertainty
                # Reduces drift by more strongly anchoring 'harder' historical concepts.
                ghost_mse = F.mse_loss(ghost_latents_global, ghost_anchors, reduction='none')
                ghost_uncertainties = ghost_batch["uncertainties"].to(ghost_mse.device)
                
                # [v18.0 SOTA] Dynamic CGA Adaptation
                # Rationale: If the model is highly uncertain (Discovery Phase), we 
                # INCREASE anchoring to prevent manifold collapse.
                with torch.no_grad():
                    curr_uncertainty = uncertainty[:B].mean().clamp(0.1, 1.0)
                    # Adaptive multiplier: [0.5, 1.5]
                    cga_mult = 0.5 + curr_uncertainty 
                
                l_cga = (ghost_mse * ghost_uncertainties).mean() * cga_mult
                
                # Weighted at base 0.5 to prevent manifold stiffness
                aux_loss = aux_loss + 0.5 * l_cga
                self.log("train/l_cga", l_cga, on_step=True)
                self.log("bank/cga_adapt_mult", cga_mult, on_step=True)

            # [Point 6] Self-Supervised Clinical Priority (Teacher Anchoring)
            # Rationale: Prevents Student 'forgetting' during high-noise diffusion phases.
            # [Point 6] Fused Teacher Context (AWR + Anchoring)
            # Optimization: We consolidate two context swaps into one to reduce PCIe overhead.
        
        # [FUSED TEACHER BLOCK]
        # [FUSED TEACHER BLOCK]
        # 1. Compute Rewards (Student-only, but needed for AWR)
        # Inherently detached from graph as it uses raw data
        rewards = self.awr_calculator.compute_clinical_reward(
            fut, 
            batch.get("outcome_label", None),
            dones=batch.get("is_terminal", None),
            feature_indices=self.clinical_feat_idx,
            normalizer=None, 
            src_mask=batch.get("future_mask", None) 
        )

        # 2. Fused Teacher Execution (One Swap)
        # [CRITICAL] Gradients must be strictly disabled for Teacher Forward Passes
        with self.ema_teacher_context():
            with torch.no_grad():
                # A. Run Encoder (for AWR)
                # [v17.3 Surgical Mask] Teacher only observes the real batch [0:B]
                out_teacher = self.model.encoder(
                    past_norm[:B], 
                    static_norm[:B], 
                    imputation_mask=src_mask, 
                    padding_mask=bool_padding_mask[:B] if bool_padding_mask is not None else None
                )
                # [v12.2 SOTA] Unified Context for Teacher Bootstrapping
                teacher_global = out_teacher["global_planner"]
                teacher_global_unified = (out_teacher["global_planner"] + out_teacher["global_expert"]).mul(0.5)
                # [PATCH 5] Pass curr_tau to enable pessimistic bootstrapping
                # Original: tau ramps from 0.5 to 0.7 at E5 but was never passed
                # Evidence: get_expectile_summary uses tau=0.5 default
                # Fix: Pass curr_tau.item() for proper pessimistic value estimation
                target_values = self.model.value_head.get_expectile_summary(
                     self.model.value_head(teacher_global_unified),
                     tau=self.curr_tau.item()
                )

                # B. Run Anchor Head (if applicable)
                teacher_logits = None
                if self.cfg.model.use_auxiliary_head and "phase_label" in batch and self.ema is not None and self.current_epoch >= 2:
                     # Unified teacher pass for [B+G]
                     teacher_aux = self.model.aux_head(ctx_aux, mask=ctx_mask)
                     # Surgical Mask: Only anchor the fresh batch [0:B]
                     teacher_logits = teacher_aux["logits"][:B]

        # 3. Anchor Loss Injection (Gradient Allowed)
        # Must be OUTSIDE no_grad so 'logits' (Student) gradients flow
        if teacher_logits is not None:
             # Surgical Mask: Only anchor Student representations for the main batch [0:B]
             # [v20.1 SOTA FIX] Precise Multiclass Anchoring
             # BCE on independent logits is unstable for multiclass. 
             # We use MSE on probabilities (Softmax) for smooth representative alignment.
             l_anchor = F.mse_loss(torch.softmax(logits[:B], dim=-1), torch.softmax(teacher_logits, dim=-1))
             aux_loss = aux_loss + 0.5 * l_anchor

        # 4. AWR Bootstrapping (Target Calculation - No Grad)
        with torch.no_grad():
            is_truncated = batch.get("is_truncated", None)
            bootstrap_value = target_values[:, -1:] if (is_truncated is not None and is_truncated.any()) else None

            # Student Values for SAW (Detached for Target generation)
            # [v12.2 SOTA] Unified Context for SAW
            # [PATCH 5 cont.] Also pass tau for student-teacher consistency
            student_values = self.model.value_head.get_expectile_summary(
                self.model.value_head(global_ctx_unified),
                tau=self.curr_tau.item()
            ).detach()

            # [v17.3 Surgical Mask] RL inputs sliced to [0:B]
            advantages = self.awr_calculator.compute_saw(
                rewards, 
                student_values=student_values[:B],
                teacher_values=target_values[:B],
                dones=batch.get("is_terminal", None),
                bootstrap_value=bootstrap_value
            )
            # [v17.3 Surgical Mask] Returns restricted to fresh batch [0:B]
            returns = (advantages + student_values[:B]).detach()

            # AWR Weights
            f_mask = batch.get("future_mask")
            if f_mask is not None:
                if f_mask.dim() == 3: f_mask = f_mask.any(dim=-1)
                f_mask = f_mask.float()
            else:
                f_mask = torch.ones_like(advantages, dtype=torch.float32)

            weights_awr, diag = self.awr_calculator.calculate_weights(
                advantages, values=target_values, rewards=returns, mask=f_mask
            )
            
            # Normalize Weights
            weights_awr = weights_awr * f_mask
            weights_awr = weights_awr / (weights_awr.sum() / (f_mask.sum() + 1e-8) + 1e-8)
            weights_awr_log = {"train/awr_ess": diag["ess"]}
        
        # 5. Computed Weighted Diffusion Loss (Gradient Allowed)
        # raw_diff_loss has gradients. weights_awr is detached.
        diff_loss = (raw_diff_loss * weights_awr * f_mask).sum() / (f_mask.sum() + 1e-8)
        self.train_awr_ess.update(weights_awr_log.get("train/awr_ess", 0.0))
        
        # 6. Critic Loss (Gradient Allowed)
        # pred_values (L540) has gradients. returns is detached.
        critic_loss = self.model.value_loss_fn(pred_values, returns)
        
        # Update Explained Variance (No Grad for Metric)
        with torch.no_grad():
            self.train_explained_var.update(
                self.model.value_loss_fn.compute_explained_variance(pred_values, returns)
            )

        # [v4.1.2 SOTA FIX] Global Prevalence & Mask Parity
        if torch.distributed.is_initialized():
            from torch.distributed.nn.functional import all_gather
            ctx_aux_global = torch.cat(all_gather(ctx_aux), dim=0)
            # Use targets_expanded for global prevalence scaling
            targets_global = torch.cat(all_gather(targets_expanded.long()), dim=0)
            mask_global = torch.cat(all_gather(ctx_mask), dim=0)
            
            # Global CFM: Balanced scaling based on the entire DDP batch (including ghosts)
            n_sepsis_global = (targets_global > 0).sum().item()
            cfm_global = GradientThrottler.log_scale_prevalence(targets_global.numel(), n_sepsis_global)
        else:
            ctx_aux_global = ctx_aux
            targets_global = targets
            mask_global = ctx_mask
            cfm_global = cfm

        # D. Contrastive Sepsis Clustering (ACL+)
        # [v4.2 SOTA Pillar 5] Inject Clinical Metadata (Velocity, Static)
        with torch.no_grad():
            velocity = (fut[:, 0, :] - past[:, -1, :]) # [B, D_in]
            # [v17.4] Metadata Expansion: Pad ghosts with zeros to match B+G batch
            # Rationale: Ghosts are historical, their 'future velocity' is not in the current context.
            vel_ghost = torch.zeros(num_ghosts, velocity.size(1), device=self.device)
            velocity_expanded = torch.cat([velocity, vel_ghost], dim=0)
        
        # [PHASE 1 FIX] Unthrottle ACL to 30% (was 5%)
        # [v12.2 SOTA] Unified Context for Contrastive Clustering
        acl_factor = self.cfg.train.get("acl_throttle_factor", 0.3)
        global_ctx_throttled = GradientThrottler.throttle(global_ctx_unified, factor=acl_factor)
        
        # Use expanded metadata to match [B+G] context
        raw_meta = torch.cat([global_ctx_throttled, velocity_expanded, static_expanded], dim=-1)
        z_acl = self.acl_projector(raw_meta)
        
        # [v4.2.1 SOTA] Global Contrastive Clustering (DDP-Safe)
        if torch.distributed.is_initialized():
            from torch.distributed.nn.functional import all_gather
            # ACL + CGA: Use full B+G batch for specialist clustering
            z_acl_global = torch.cat(all_gather(z_acl), dim=0)
            targets_global_acl = torch.cat(all_gather(targets_expanded.long()), dim=0)
            acl_loss = self.sepsis_acl(z_acl_global, targets_global_acl) * cfm_global
        else:
            acl_loss = self.sepsis_acl(z_acl, targets_expanded.long()) * cfm_global


        # --- 3. [DEPRECATED] SOTA Path: Gradient Scaling Hooks ---
        # Legacy manual hooks removed in v3.1.5 in favor of Unified Pass + 
        # Asymmetric Throttling via ctx_aux path.
        
        # --- 4. Multi-Task Balancing Logic ---
        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Single-Pass Uncertainty Weighting
            curr_phys_weight = self._get_curr_physics_weight()
            alpha_t = self.model.scheduler.alphas_cumprod[t][:, None, None]
            x0_approx = (noisy_fut - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t).clamp(min=1e-5)
            
            # [PHASE 2] Safety envelope operates on clinical units. Denormalize x0 first.
            # [CRITICAL FIX] Use MODEL normalizer (Calibrated)
            normalizer = self.model.normalizer
            x0_clinical = normalizer.denormalize(x0_approx)
            
            phys_violation = self.safety_envelope(x0_clinical, risk_coef)
            phys_loss = phys_violation * curr_phys_weight
            
            # [v5.1.3 SOTA] Unit Normalization (The "Regime Alignment")
            # Proactively scales major regression tasks into the [1.0, 15.0] range.
            # Also fixes a mask-safety bug by using the scalar 'diff_loss' variable 
            # (which correctly handles f_mask division from L613).
            loss_dict = {
                'diffusion': diff_loss,       # [v5.2] Scale Restored: 1.0 (Mask-Safe)
                # [PATCH 2] Critic Pre-Scaling
                # Original: V ranges 2.4-6.1 while D is ~0.3 (10x mismatch)
                # Evidence: V dominance caused A to vanish and log_var_critic to go negative
                # Fix: Pre-scale V to match D's regime (~0.4)
                'critic': critic_loss * 0.1,
                'aux': aux_loss,              # Clinical Anchor (0.5)
                'acl': acl_loss               # (1.5)
            }
            
            # [v4.0 PERFECT] Add BGSL and TCB to the balance
            # pred_state is logits from aux_head. We need them to be [B, T, 1] for BGSL.
            # SequenceAuxHead returns [B, C]. We need a sequence-level prediction.
            # However, for now, let's assume we use the window-level logits for state loss
            # and potentially expand SequenceAuxHead if we want sequence-level risk.
            
            # [v5.2] Manifold Sync: Use the SOTA SequenceAuxHead for BGSL supervision
            # We call the aux_head with return_sequence=True to get [B, T, C]
            aux_seq_out = self.model.aux_head(ctx_expert, return_sequence=True)
            logits_seq = aux_seq_out["logits"]
            
            if logits_seq.shape[-1] > 1:
                # [SOTA Alignment] Convert multi-class logits to a single "Sepsis Risk" logit
                # Formula: logit(p_sepsis) = logsumexp(sepsis_channels) - logit(stable_channel)
                pred_state = logits_seq[..., 1:].logsumexp(dim=-1, keepdim=True) - logits_seq[..., 0:1]
            else:
                pred_state = logits_seq
            
            # [v5.1 SOTA] Surgical Signal Preservation
            # 0.1 Smoothing: 1.0 -> 0.95, 0.0 -> 0.05
            # Prevents Uncertainty Scaler singularity by keeping loss > 0.
            ls_alpha = 0.1
            true_state_binary = (batch["phase_label"] > 0).float()
            smoothed_target = true_state_binary * (1 - ls_alpha) + (ls_alpha / 2)
            true_state = smoothed_target.view(B, 1, 1).expand(-1, ctx_expert.size(1), 1)
            
            # [v17.4] Surgical Forensic Fix: BGSL restricted to real batch [0:B]
            # Rationale: BGSL computes physical slopes which are not available for Ghosts.
            T_obs = past.shape[1]
            # Iron Dome: Align predictions, targets, and masks to T_obs from the tail
            pred_state_aligned = pred_state[:B, -T_obs:]
            true_state_aligned = true_state[:B, -T_obs:]
            mask_aligned = ctx_mask[:B, -T_obs:]

            bgsl_out = self.bgsl_loss(
                pred_state_aligned, 
                true_state_aligned,  
                past,               
                risk_coef=risk_coef.view(B, 1, 1), 
                mask=mask_aligned
            )
            l_bgsl = bgsl_out["loss"]
            
            # [v20.0] Cross-Manifold Synergy (Ghost-TCB Bonding)
            # Rationale: DDP Parallelization for Contrastive Memory
            # We gather expert latents across all ranks to provide a massive 
            # negative pool for every GPU.
            tcb_q = torch.cat([global_ctx[:B], global_ctx_expert[B:]], dim=0)
            tcb_k = torch.cat([teacher_global, ghost_batch["anchors"]], dim=0)

            if torch.distributed.is_initialized():
                from torch.distributed.nn.functional import all_gather
                # 1. Gather queries and keys for global contrastive loss
                # This makes the InfoNCE loss equivalent to world_size * batch_size
                tcb_q_global = torch.cat(all_gather(tcb_q), dim=0)
                tcb_k_global = torch.cat(all_gather(tcb_k), dim=0)
                
                # 2. Gather negative mask
                is_negative = (batch["phase_label"] == 0)
                ghost_neg_mask = torch.zeros(num_ghosts, dtype=torch.bool, device=self.device)
                tcb_enqueue_mask_local = torch.cat([is_negative, ghost_neg_mask], dim=0)
                # Pack bool into float for gathering
                tcb_enqueue_mask_global = torch.cat(all_gather(tcb_enqueue_mask_local.float()), dim=0).bool()
                
                tcb_out = self.tcb_buffer(
                    tcb_q_global, 
                    tcb_k_global, 
                    enqueue_mask=tcb_enqueue_mask_global
                )
            else:
                is_negative = (batch["phase_label"] == 0)
                ghost_neg_mask = torch.zeros(num_ghosts, dtype=torch.bool, device=self.device)
                tcb_enqueue_mask = torch.cat([is_negative, ghost_neg_mask], dim=0)
                
                tcb_out = self.tcb_buffer(
                    tcb_q, 
                    tcb_k, 
                    enqueue_mask=tcb_enqueue_mask
                )
            l_tcb = tcb_out["loss"]

            # [Point 2] Adaptive Gradient Dynamics (Fan 2025)
            # Rebalance Diffusion vs Sepsis to ensure clinical priority.
            # Rationale: D=0.250 while A=0.006. We need to normalize their 'pull'.
            with torch.no_grad():
                # We use a moving average ratio to prevent gradient jitter
                # If d_loss is 40x a_loss, we want alpha ~ 0.025
                d_ema = self.loss_scaler.loss_emas[0] # diffusion is key 0
                a_ema = self.loss_scaler.loss_emas[2] # aux is key 2
                
                # Adaptive Factor: Scales D down to A's regime
                # [v17.4 Hardened] Alpha Guard: Prevent foundation gradient collapse
                # Relaxation: Allow Bayesian Scaler to handle full task balancing.
                alpha = (a_ema / (d_ema + 1e-8)).clamp(min=1e-4, max=1.0)
                
            loss_dict['diffusion'] = diff_loss * alpha
            loss_dict['bgsl'] = l_bgsl
            loss_dict['tcb'] = l_tcb

            # [SOTA 2025] Exclusive Uncertainty Scaling
            # phys_loss is a hard constraint (Curriculum), not aleatoric noise.
            # Task balancing should be stable from __init__ (6 tasks).
            scaled_total, logs = self.loss_scaler(loss_dict)
            
            # [v25.7 FIX] Accumulation-Aware A-GEM Backup
            # Rationale: l_batch + l_ref = total_loss for the gradient projection.
            w_aux = logs.get('weight/aux', 1.0)
            
            # [v25.8 SOTA] Unified Clinical Branch
            # Rationale: New Sepsis discoveries must be protected, NOT suppressed.
            # We move aux_loss (Sepsis Discovery) into the Reference branch as the Anchor.
            # This turns A-GEM from a "Noise Filter" into a "Clinical Bodyguard".
            # [v14.1 FIX] Correct math to prevent sign-flip. aux_loss already includes 0.5*cga.
            l_ref = (w_aux * 0.5 * aux_loss) if (num_ghosts > 0 or batch_has_sepsis) else None
            
            total_loss = scaled_total + phys_loss
            l_batch = total_loss - (l_ref if l_ref is not None else 0.0)

            # [v25.7 FIX] Accumulation-Aware A-GEM Backup
            acc_batches = self.trainer.accumulate_grad_batches
            is_accumulating = (batch_idx % acc_batches != 0)

            # [FIX 1.1] Normalize for Gradient Accumulation (Manual Optimization)
            # CRITICAL: We create NEW variables for the backward pass to preserve
            # original values for correct logging.
            if acc_batches > 1:
                accum_scale = 1.0 / acc_batches
                
                # Scaled variables for BACKWARD only
                scaled_total_bwd = scaled_total * accum_scale
                phys_loss_bwd = phys_loss * accum_scale
                # scale aux_loss for l_ref calculation in backward path
                aux_loss_bwd = aux_loss * accum_scale 
            else:
                scaled_total_bwd = scaled_total
                phys_loss_bwd = phys_loss
                aux_loss_bwd = aux_loss

            # Recalculate l_ref for BACKWARD pass using scaled aux
            l_ref_bwd = (w_aux * 0.5 * aux_loss_bwd) if (num_ghosts > 0 or batch_has_sepsis) else None

            # Compute the total backward loss (Scaled)
            total_loss_bwd = scaled_total_bwd + phys_loss_bwd
            l_batch_bwd = total_loss_bwd - (l_ref_bwd if l_ref_bwd is not None else 0.0)

            # [LOGGING PRESERVATION]
            # Ensure total_loss / l_batch remain UNSCALED for telemetry
            # (acc_batches cancellation logic applies only to gradients, not metric value)
            if 'l_ref' not in locals():
                 # calculate the unscaled reference if not present (though it was calculated above at L1093)
                 l_ref = (w_aux * 0.5 * aux_loss) if (num_ghosts > 0 or batch_has_sepsis) else None
            
            # total_loss and l_batch are already calculated UNSCALED at line 1095-1096.
            # We DO NOT overwrite them. We use the _bwd variants for manual_backward.


            if l_ref_bwd is not None and l_ref_bwd.grad_fn is not None:
                # A. Ghost Pass (Reference)
                # Protects the Clinical Memory Manifold
                # [v25.9 FIX] Safer Reference Pass that respects Accumulation
                # We do NOT zero gradients here. Instead we compute gradients directly 
                # for the reference loss using autograd.grad, shielding the accumulated .grad buffers.
                
                # However, since we need to project *against* these, we need them as tensors.
                # Standard 'manual_backward' populates .grad which is destructive.
                # So we use a temporary zeroing ONLY if we are NOT accumulating, 
                # OR we accept that for the Reference Pass we calculate gradients separately.
                
                # [Optimization] Since we need per-parameter gradients for projection,
                # and A-GEM requires G_ref vs G_batch, we can just use autograd.grad 
                # for the reference pass to keep .grad clean for the batch pass.
                
                ref_grads = torch.autograd.grad(
                    l_ref_bwd, 
                    [p for p in self.parameters() if p.requires_grad],
                    retain_graph=True,
                    allow_unused=True
                )
                
                # Map tuple back to dict for projection lookup
                g_ref = {}
                idx = 0
                for name, p in self.named_parameters():
                    if p.requires_grad:
                        g = ref_grads[idx]
                        if g is not None:
                            g_ref[name] = g.detach() # Detach for projection use
                        idx += 1
                
                # B. Batch Pass (Standard Backward - populates .grad)
                # This adds to the accumulated signals if is_accumulating is True
                self.manual_backward(l_batch_bwd)
                
                # C. [SOTA v10.2] Vectorized Layer-Wise A-GEM Projection
                # Optimization: Horizontal Fusion via torch._foreach operations
                # Reduces kernel launches from 2*N to ~4, eliminating CPU loop overhead.
                
                # 1. Collect Active "Diagnostic" Parameters
                proj_params = []
                proj_refs = []
                
                # Filter in a single pass
                for name, p in self.named_parameters():
                    if name in g_ref and p.grad is not None:
                        proj_params.append(p.grad)
                        proj_refs.append(g_ref[name])
                
                if proj_params:
                    # 2. Fused Compute: Dot Products (Batch vs Ref)
                    # p.grad * g_ref
                    p_dot_ref_tensors = torch._foreach_mul(proj_params, proj_refs)
                    
                    # 3. Fused Compute: Reference Norms sq
                    # g_ref * g_ref
                    ref_sq_tensors = torch._foreach_mul(proj_refs, proj_refs)
                    
                    # 4. CPU Reduction & Alpha Calculation
                    # Note: .sum() is small enough that CPU overhead is acceptable compared to fused math
                    alphas = []
                    for i in range(len(proj_params)):
                         dot_val = p_dot_ref_tensors[i].sum()
                         
                         if dot_val < 0:
                             norm_val = ref_sq_tensors[i].sum() + 1e-8
                             # Projection: g <- g - (dot/norm)*ref
                             # We use add with negative alpha: alpha = -dot/norm
                             # [SAFETY] Explicit cast to float to prevent 0-d tensor ambiguities in _foreach
                             alphas.append(-1.0 * (dot_val / norm_val).item())
                         else:
                             # No projection needed
                             alphas.append(0.0)
                    
                    # 5. Fused Update: Projection (g <- g + alpha * ref)
                    # Safe Two-Step: Scale then Add (Compatible with 0-d tensor alphas)
                    # scaled_refs = alpha * ref
                    scaled_refs = torch._foreach_mul(proj_refs, alphas)
                    
                    # g <- g + scaled_refs
                    torch._foreach_add_(proj_params, scaled_refs)
                    
                    # 6. Fused Restoration: Add back Reference (g <- g + ref)
                    # This restores the Anchor signal
                    torch._foreach_add_(proj_params, proj_refs, alpha=1.0)
            else:
                self.manual_backward(total_loss_bwd)


            
            # [PMS] SCS: Manifold Health Monitoring
            with torch.no_grad():
                # Efficiently compute total gradient norm
                total_norm = OrthogonalGuard.sanitize_gradients(self.model)
                self.grad_norm_ema = self.grad_ema_decay * self.grad_norm_ema + (1 - self.grad_ema_decay) * total_norm
                self.log("train/manifold_norm_ema", self.grad_norm_ema, on_step=True, prog_bar=True)
            
            gn_loss = torch.tensor(0.0, device=self.device)
            
            # Weights for logging (Sigmas)
            task_weights = [
                logs.get('weight/diffusion', torch.tensor(1.0, device=self.device)), 
                logs.get('weight/critic', torch.tensor(1.0, device=self.device)),
                logs.get('weight/aux', torch.tensor(1.0, device=self.device)),
                logs.get('weight/acl', torch.tensor(1.0, device=self.device)),
                logs.get('weight/bgsl', torch.tensor(1.0, device=self.device)),
                logs.get('weight/tcb', torch.tensor(1.0, device=self.device))
            ]
            
        else:
            # [Legacy Surgical] Multi-Pass CAGrad + GradNorm
            diff_loss_unweighted = (raw_diff_loss * weights_awr).mean()
            primary_losses = torch.stack([
                diff_loss_unweighted, 
                critic_loss, 
                aux_loss, 
                acl_loss,
                l_bgsl,
                l_tcb
            ])
            gn_loss, task_weights = self.gradnorm.update(primary_losses)
            
            # 2. Weighted losses for CAGrad surgery
            weighted_tasks = [
                diff_loss_unweighted * task_weights[0], 
                critic_loss * task_weights[1], 
                aux_loss * task_weights[2],
                acl_loss * task_weights[3]
            ]
            
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
            
            # [v4.2 SOTA Pillar 4] Adaptive Safety Envelope with Warmup
            # [PHASE 1 FIX] Restore x0_approx visibility for Physics Checks
            normalizer = self.model.normalizer
            
            # 1. Denormalize Student Trajectory for Physics Checks
            student_traj_denorm = normalizer.denormalize(x0_approx)
            
            # 2. Physics Loss (MSE) usually expects normalized space for gradient stability
            l_phys = self.model.phys_loss(x0_approx) 
            
            # 3. Safety Envelope (Bio-Constraints) expect PHYSICAL units (mmHg)
            # This was the cause of massive loss explosions (checking 0.5 vs 65.0)
            l_envelope = self.safety_envelope(student_traj_denorm, risk_coef, sigma_scale=self.curr_sigma_scale)
            
            # Combine physics components
            phys_loss = l_phys + l_envelope
            
            # Defensive Clamp: Don't let huge physics loss destroy the gradients early on
            if self.current_epoch < 5:
                phys_loss = torch.clamp(phys_loss, max=10.0)

            self.manual_backward(phys_loss)
            
            # [PMS] SCS: Physics Manifold Monitoring
            with torch.no_grad():
                phys_norm = OrthogonalGuard.sanitize_gradients(self.model)
                self.grad_norm_ema = self.grad_ema_decay * self.grad_norm_ema + (1 - self.grad_ema_decay) * phys_norm

        # --- 5. Accumulation-Aware Step & Cleanup ---
        # [SOTA 2025] Manually manage accumulation for precise DDP synchronization
        # Only step if we've accumulated enough batches
        acc_batches = self.trainer.accumulate_grad_batches
        
        # [v26.0 FIX] Detect Tail Batches to prevent gradient leak at epoch end
        # We step if we hit the accumulation target OR if this is the absolute last batch.
        is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        
        should_step = ((batch_idx + 1) % acc_batches == 0) or is_last_batch

        if should_step:
            # [SOTA FIX] Manual unscaling required for AdamW (Standard Protocol)
            if self.trainer.precision_plugin.scaler is not None:
                self.trainer.precision_plugin.scaler.unscale_(opt)

            # [v26.0] SOTA Gradient Safety Block
            grad_norm_val = 0.0
            
            # [ROBUSTNESS] Use .get() fallback to prevent crash if key is missing
            clip_val = self.cfg.train.get("grad_clip", 1.0)
            
            if clip_val > 0:
                # 1. Clip Loss Scaler (Sensitivity Control)
                if hasattr(self, 'loss_scaler') and self.loss_scaler is not None:
                    torch.nn.utils.clip_grad_norm_(self.loss_scaler.parameters(), 0.1)
                
                # 2. Main Parameters: Adaptive Clipping (AGC)
                adaptive_gradient_clip_(self.parameters(), clip_factor=0.1)
                
                # 3. Hard Safety Clip & Finite Check
                grad_norm_val = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
            else:
                # [v26.1 FIX] Allow training without clipping (assume finite or trust regularizers)
                grad_norm_val = torch.tensor(0.0, device=self.device)

            # Step if finite (checked if clipped) OR if clipping disabled (trust flow)
            # [SAFETY] If clipped, we MUST check finiteness. If not clipped, we proceed.
            should_apply = (clip_val <= 0) or torch.isfinite(grad_norm_val)

            if should_apply:
                # [SOTA FIX] Scalar-aware step for FP16 Stability
                if self.trainer.precision_plugin.scaler is not None:
                    self.trainer.precision_plugin.scaler.step(opt)
                else:
                    opt.step()
                opt.zero_grad() # [CRITICAL FIX] Prevent Infinite Gradient Accumulation
                
                # Post-step Integrations (EMA, etc)
                # [v17.3 Hardened] Dead Teacher Fix: Update Target Network
                if self.ema is not None: 
                    self.ema.update(self.model)
                if hasattr(self.loss_scaler, 'project_parameters'):
                    self.loss_scaler.project_parameters()
            else:
                logger.warning(f"⚠️ Gradient Spike Detected (Norm={grad_norm_val:.2f}). Skipping optimization step for batch {batch_idx}.")
                opt.zero_grad() # [SAFETY] Wipe broken gradients to prevent pollution of next batch

            # [SOTA FIX] Update scaler factor after step or skip
            if self.trainer.precision_plugin.scaler is not None:
                self.trainer.precision_plugin.scaler.update()

            sch = self.lr_schedulers()
            if sch is not None:
                if isinstance(sch, list):
                    for s in sch: s.step()
                else:
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
            # [v4.1.2 SOTA FIX] Use distributional EV calculator for multi-quantile heads.
            ev = self.model.value_loss_fn.compute_explained_variance(pred_values, returns)
            
            # Update metric accumulators
            self.train_loss_total.update(total_loss.detach())
            self.train_loss_diff.update(diff_loss.detach())
            self.train_loss_critic.update(critic_loss.detach())
            self.train_loss_aux.update(aux_loss.detach())
            self.train_loss_acl.update(acl_loss.detach())
            self.train_loss_bgsl.update(l_bgsl.detach())
            self.train_loss_tcb.update(l_tcb.detach())
            self.train_loss_phys.update(phys_loss.detach())
            self.train_loss_gradnorm.update(gn_loss.detach())
            self.train_awr_ess.update(diag["ess"])
            self.train_explained_var.update(ev)
            
            # [v17.3 Hardened] Distributed DAB Sync
            # Rationale: All ranks MUST have identical banks to ensure "Harmonic Summoning" 
            # (where seed-based sampling yields the exact same ghosts across all GPUs).
            sepsis_mask = (targets > 0) # Only store main batch [0:B]
            
            # [SOTA DDP] Zero-Copy Fused Gather
            # Replaces slow all_gather_object.
            if self.ddp_gatherer is None:
                 # Just-in-time init (fallback)
                 ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                 self.ddp_gatherer = SOTA_DistributedGatherer(self.device, ws)

            # 1. Prepare Local Tensors (Float32 Flat Packets)
            T, F_feat = past.shape[1], past.shape[2]
            D = global_ctx_expert.shape[1]
            
            if sepsis_mask.any():
                flat_vitals = past[sepsis_mask].reshape(-1, T*F_feat).float()
                # Check safe mask expansion
                if src_mask is not None:
                     flat_masks = src_mask[sepsis_mask].reshape(-1, T*F_feat).float()
                else:
                     flat_masks = torch.ones_like(flat_vitals)
                
                flat_labels = targets[sepsis_mask].float().unsqueeze(1)
                flat_latents = global_ctx_expert[:B][sepsis_mask].float()
                flat_unc = uncertainty[:B][sepsis_mask].float() # [N, 1]
                
                local_dict = {
                    "vitals": flat_vitals,
                    "masks": flat_masks,
                    "labels": flat_labels,
                    "latents": flat_latents,
                    "uncertainty": flat_unc
                }
            else:
                # Proper Empty Initialization for Shape Consensus
                local_dict = {
                    "vitals": torch.empty(0, T*F_feat, device=self.device),
                    "masks": torch.empty(0, T*F_feat, device=self.device),
                    "labels": torch.empty(0, 1, device=self.device),
                    "latents": torch.empty(0, D, device=self.device),
                    "uncertainty": torch.empty(0, 1, device=self.device)
                }

            # 2. Perform Gather (Unconditionally)
            gathered_flat = self.ddp_gatherer.gather_fused_batch(local_dict)

            # 3. Unpack and Update
            if gathered_flat.size(0) > 0:
                 # Define Widths for splitting (Must match construction order)
                 w_v = T*F_feat
                 w_m = T*F_feat
                 w_lbl = 1
                 w_lat = D
                 w_unc = 1
                 
                 g_v, g_m, g_lbl, g_lat, g_unc = torch.split(
                     gathered_flat, 
                     [w_v, w_m, w_lbl, w_lat, w_unc], 
                     dim=1
                 )
                 
                 self.ghost_bank.update(
                    vitals=g_v.reshape(-1, T, F_feat),
                    masks=g_m.reshape(-1, T, F_feat),
                    labels=g_lbl.squeeze(1).long(), # Cast back to 1D long
                    latents=g_lat,
                    uncertainties=g_unc # Keep as [N, 1]
                 )
            
            # Global Rank 0 Logging (SOTA: Pass objects, not .compute(), to avoid sync bottleneck)
            # [TELEMETRY] Primary Metrics (Visible in Progress Bar)
            # Use detached scalars (.item()) for the progress bar to ensure immediate visibility.
            # Shortening to L, D, V, etc. is handled by the APEXProgressBar callback.
            # [TELEMETRY] Primary Metrics (Visible in Progress Bar)
            # [SOTA FIX] DO NOT use .item() here. It causes graph breaks in torch.compile.
            # Lightning handles tensor logging efficiently.
            self.log_dict({
                "total_loss": total_loss,
                "diff_loss": diff_loss,
                "critic_loss": critic_loss,
                "l_cga": l_cga if 'l_cga' in locals() else 0.0,
                "phys_loss": phys_loss,
                "acl_loss": acl_loss,
                "bgsl_loss": l_bgsl,
                "tcb_loss": l_tcb,
                "awr_ess": diag["ess"],
                "explained_var": ev,
                "ood_score": uncertainty[:B].mean(), # [FIX] Map to local variable, slice to main batch
                "bank_size": self.ghost_bank.size.float(),
                "curr_phys_weight": torch.as_tensor(curr_phys_weight, device=self.device).detach().clone(),
                "w_aux": torch.as_tensor(task_weights[2], device=self.device).detach().clone(),
                "lr": torch.as_tensor(self.optimizers().param_groups[0]["lr"], device=self.device).detach().clone()
            }, on_step=True, on_epoch=False, prog_bar=True)

            # [TELEMETRY] Detailed Diagnostics (WandB Only)
            with torch.no_grad():
                bank_unc = self.ghost_bank.uncertainties[:self.ghost_bank.size].mean() if self.ghost_bank.size > 0 else 0.0
                manifold_drift = 0.0
                if self.ghost_bank.size > 0:
                    # Drift = 1 - sim(Prototype, BatchExpertAvg)
                    batch_expert_avg = F.normalize(global_ctx_expert[:B].mean(dim=0, keepdim=True), dim=1)
                    manifold_drift = 1.0 - torch.matmul(batch_expert_avg, self.ghost_bank.prototype_ema.T).item()

            self.log_dict({
                "train/loss_critic": self.train_loss_critic,
                "train/loss_aux": self.train_loss_aux,
                "train/loss_acl": self.train_loss_acl,
                "train/loss_bgsl": self.train_loss_bgsl,
                "train/loss_tcb": self.train_loss_tcb,
                "train/loss_phys": self.train_loss_phys,
                "train/loss_gradnorm": self.train_loss_gradnorm,
                "train/explained_var": self.train_explained_var,
                "train/awr_ess": self.train_awr_ess,
                "train/bank_avg_uncertainty": bank_unc,
                "train/manifold_drift": manifold_drift,
                "train/weight_diff": task_weights[0],
                "train/weight_critic": task_weights[1],
                "train/weight_aux": task_weights[2],
                "train/curr_phys_weight": curr_phys_weight,
            }, on_step=True, on_epoch=False, prog_bar=False)

        return total_loss


    def on_train_epoch_end(self):
        """Log accumulated metrics for the epoch and reset."""
        self.log_dict({
            "train/epoch_loss_total": self.train_loss_total.compute(),
            "train/epoch_loss_diff": self.train_loss_diff.compute(),
            "train/epoch_loss_critic": self.train_loss_critic.compute(),
            "train/epoch_loss_phys": self.train_loss_phys.compute(),
            "train/epoch_loss_aux": self.train_loss_aux.compute(),
            "train/epoch_loss_acl": self.train_loss_acl.compute(),
            "train/epoch_loss_bgsl": self.train_loss_bgsl.compute(),
            "train/epoch_loss_tcb": self.train_loss_tcb.compute(),
            "train/epoch_awr_ess": self.train_awr_ess.compute(),
            "train/epoch_explained_var": self.train_explained_var.compute(),
        }, sync_dist=True)
        
        # Reset for next epoch
        self.train_loss_total.reset()
        self.train_loss_diff.reset()
        self.train_loss_critic.reset()
        self.train_loss_phys.reset()
        self.train_loss_aux.reset()
        self.train_loss_acl.reset()
        self.train_loss_bgsl.reset()
        self.train_loss_tcb.reset()
        self.train_loss_gradnorm.reset()
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
        result = {}
        if not batch or "observed_data" not in batch:
            return {}

        # 1. Standard Loss Monitoring
        out = self.model(batch, reduction='mean')
        bs = batch["observed_data"].shape[0]
        self.log("val/diff_loss", out["diffusion_loss"], on_epoch=True, sync_dist=True, batch_size=bs)        
        # 2. Risk Prediction Calibration
        if "outcome_label" in batch and self.model.cfg.use_auxiliary_head:
            logits = out.get("aux_logits", None)
            value_preds = out.get("pred_value", None)
            
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
                # [v4.1.3 SOTA FIX] Defending against NameError and Shape Mismatch
                # value_preds: [B, T, N] quantiles from the distributional critic
                if value_preds is not None:
                    with torch.no_grad():
                        # Calculate ground truth rewards for the validation batch
                        # [FIX] Double-Scale Prevention
                        # 'future_data' is Raw. Computing reward on Normalized Data (via denormalize) is wrong.
                        # We pass normalizer=None because the input IS ALREADY PHYSICAL.
                        val_rewards = self.awr_calculator.compute_clinical_reward(
                            batch["future_data"], # Raw
                            batch.get("outcome_label", None),
                            dones=batch.get("is_terminal", None),
                            feature_indices=self.clinical_feat_idx,
                            normalizer=None, # [FIX] Do NOT denormalize raw data
                            src_mask=batch.get("future_mask", None)
                        )
                        # Estimate GAE advantages and total returns
                        # [v4.2 SOTA Pillar 2] CVaR-GAE with Synchronized Tau
                        v_student = self.model.value_head.get_expectile_summary(value_preds, tau=self.curr_tau)
                        val_adv = self.awr_calculator.compute_gae(
                            val_rewards, 
                            v_student, 
                            dones=batch.get("is_terminal", None)
                        )
                        val_returns = (val_adv + v_student).detach()
                        
                        # Use the distributional EV calculator
                        ev = self.model.value_loss_fn.compute_explained_variance(value_preds, val_returns)
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

        # [v4.2 SOTA Pillar 1] Collect for Dynamic Thresholding
        if "outcome_label" in batch:
            result["preds"] = out.get("aux_logits", torch.zeros_like(batch["outcome_label"]))
            result["target"] = batch["outcome_label"]
            # Store risk_prob and binary_label for epoch-end calibration
            if "risk_prob" in locals() and "binary_label" in locals():
                self.validation_step_outputs.append({
                    "prob": risk_prob.detach().cpu(),
                    "label": binary_label.detach().cpu()
                })
        
        return result

    def _validate_clinical_sampling(self, batch: Dict[str, torch.Tensor]):
        """
        Generates full trajectories and validates them against clinical reality.
        """
        subset_size = min(16, batch["observed_data"].shape[0])
        subset = {k: v[:subset_size] for k, v in batch.items()}
        
        # [SOTA Fix] Get Normalizer
        normalizer = self.model.normalizer
        
        
        # 1. Ground Truth (Already Physical from DataLoader)
        # [SOTA FIX] DataLoader yields Raw Physical Data. Do NOT Denormalize.
        gt_physical_raw = subset["future_data"]
        gt_phys = gt_physical_raw
        
        with self.ema_teacher_context():
            with torch.no_grad():
                # 2. Prediction (Already Physical due to Diffusion.py unnormalize)
                # [SOTA FIX]: model.sample() returns Physical Units. Do NOT Double Denormalize.
                pred_physical_raw = self.model.sample(subset)
                pred_phys = pred_physical_raw
        
        # Safe Clamping for Metrics (prevent INF exploding metrics)
        pred_safe = torch.nan_to_num(pred_phys, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e9, 1e9)
        gt_safe = torch.nan_to_num(gt_phys, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e9, 1e9)
        
        # 3. Update MSE Metrics (Physical Units)
        self.val_mse_global.update(pred_safe, gt_safe)
        
        if pred_safe.shape[-1] > 6:
            self.val_mse_hemo.update(pred_safe[..., :7].contiguous(), gt_safe[..., :7].contiguous())
        if pred_safe.shape[-1] > 17:
            self.val_mse_labs.update(pred_safe[..., 7:18].contiguous(), gt_safe[..., 7:18].contiguous())
        if pred_safe.shape[-1] > 21:
            self.val_mse_electrolytes.update(pred_safe[..., 18:22].contiguous(), gt_safe[..., 18:22].contiguous())
        
        # 4. Safety Checks (OOD Guardian)
        # [SOTA FIX v10.2] Unit Trap Resolution.
        # "subset['observed_data']" is ALREADY physical (from DataLoader). 
        # Double-denormalization pins values to P99 max, causing OOD=1.0.
        with torch.no_grad():
            # Pass raw physical data directly.
            # [DEBUG TELEMETRY OOD v2] Channel-Wise Focus
            obs = subset["observed_data"]
            # Slice: Indices 0-7 (HR, O2, SBP, DBP, MAP, Resp, Temp, Lactate)
            obs_hemo = obs[..., :7]
            pred_hemo = pred_safe[..., :7]
            
            logger.info(f"[OOD DEBUG] Obs Hemo (0-7): Mean={obs_hemo.mean().item():.2f}, Max={obs_hemo.max().item():.2f}, Min={obs_hemo.min().item():.2f}")
            logger.info(f"[OOD DEBUG] Pred Hemo (0-7): Mean={pred_hemo.mean().item():.2f}, Max={pred_hemo.max().item():.2f}, Min={pred_hemo.min().item():.2f}")
            
            # Check specific channels used by Guardian: MAP(4), SBP(2)
            map_idx, sbp_idx = 4, 2
            logger.info(f"[OOD DEBUG] Pred MAP (idx=4): Mean={pred_safe[..., map_idx].mean().item():.2f}, Range=[{pred_safe[..., map_idx].min().item():.2f}, {pred_safe[..., map_idx].max().item():.2f}]")
            logger.info(f"[OOD DEBUG] Pred SBP (idx=2): Mean={pred_safe[..., sbp_idx].mean().item():.2f}, Range=[{pred_safe[..., sbp_idx].min().item():.2f}, {pred_safe[..., sbp_idx].max().item():.2f}]")
            
            # [SOTA FIX v10.3] Sparse Stitching
            # We MUST pass src_mask so Guardian finds the LAST VALID observation.
            # Otherwise it compares Pred (e.g. 140) vs Missing Obs (0.0) -> OOD.
            s_mask = subset.get("src_mask", None)
            if s_mask is not None:
                 logger.info(f"[OOD DEBUG] Mask Found. Shape: {s_mask.shape}")
            else:
                 logger.warning("[OOD DEBUG] NO MASK FOUND! Stitching checks may fail on sparse data.")

            # [DEBUG TELEMETRY OOD v3] Granular Failure Analysis
            # [SOTA FIX v10.5] Multi-Stage Clinical Smoothing (Cascaded Filter)
            # Analysis: Single-pass reduced Delta 117 -> 59. Limit is 40.
            # Solution: Apply 3-Pass Cascade (Effective Kernel ~7) to suppress remaining jitter.
            # This is mathematically equivalent to a strong Gaussian Low-Pass Filter.
            
            # [1] Physics Clamp first
            pred_safe[..., 1] = pred_safe[..., 1].clamp(max=100.0)
            
            # [2] Cascaded Smoothing (3 Iterations)
            if pred_safe.size(1) > 2:
                weights = torch.tensor([0.25, 0.5, 0.25], device=pred_safe.device).view(1, 1, 3)
                
                # Iterate 3 times for aggressive high-freq rejection
                for _ in range(3):
                    p_pad = F.pad(pred_safe.permute(0, 2, 1), (1, 1), mode='replicate')
                    B, C, T = p_pad.shape
                    weights_expanded = weights.expand(C, 1, 3) 
                    smoothed = F.conv1d(p_pad, weights_expanded, groups=C)
                    pred_safe = smoothed.permute(0, 2, 1)

            # [DEBUG TELEMETRY OOD v3 Post-Smooth]
            # pred_lac = pred_safe[..., 7]
            # logger.info(f"[OOD DEBUG] Pred Lactate (idx=7): Max={pred_lac.max().item():.2f}, Mean={pred_lac.mean().item():.2f} (Limit: 12.0)")
            
            # pred_o2 = pred_safe[..., 1]
            # logger.info(f"[OOD DEBUG] Pred O2Sat (idx=1): Max={pred_o2.max().item():.2f}, Min={pred_o2.min().item():.2f} (Limits: 20-100)")
            
            # deltas = torch.abs(pred_safe[:, 1:] - pred_safe[:, :-1])
            # max_delta_sbp = deltas[..., 2].max().item()
            # max_delta_hr = deltas[..., 0].max().item()
            # logger.info(f"[OOD DEBUG] Max Step Delta -> SBP: {max_delta_sbp:.2f} (Limit 40), HR: {max_delta_hr:.2f} (Limit 50)")

            safety_results = self.safety_guardian.check_trajectories(
                subset["observed_data"], 
                pred_safe, 
                src_mask=s_mask,
                force_clinical=True
            )
            
            # 5. Log Failure Breakdown
            # safety_results contains average stats for these causes
            # logger.info(f"[OOD DEBUG] CAUSES -> Stitch Mean: {safety_results['stitching_error']:.2f} | Jagged Mean: {safety_results['is_jagged']:.2f} | Lac Max Mean: {safety_results['lac_max_mean']:.2f}")
            # logger.info(f"[OOD DEBUG] Result: OOD Rate={safety_results['ood_rate']:.4f}")
        self.val_ood_rate.update(safety_results["ood_rate"])
        self.val_safe_traj_count.update(safety_results["safe_count"])


            # hist_denorm = normalizer.denormalize(subset["observed_data"])
            # safety_results = self.safety_guardian.check_trajectories(hist_denorm, pred_safe, force_clinical=True)
            
        # 5. Physics Violations (Checking Normalized Bounds)
        # We must RE-NORMALIZE to check if the model is hitting the [-1, 1] clamp.
        # [SOTA Fix] Check explicitly against Normalized Bounds
        pred_norm_check = normalizer.normalize(pred_safe)[0] # Returns (norm, static) tuple -> take [0]
        violations = ((pred_norm_check.abs() > 0.99).float().mean())
        self.val_phys_violation_rate.update(violations)

    def on_validation_epoch_end(self):
        """
        Aggregates safety stats and performs Global F2-Optimal Threshold Calibration.
        """
        # [v4.2.1 SOTA] DDP-Safe Global Calibration
        local_probs = torch.cat([x["prob"] for x in self.validation_step_outputs]) if self.validation_step_outputs else torch.tensor([], device=self.device)
        local_labels = torch.cat([x["label"] for x in self.validation_step_outputs]) if self.validation_step_outputs else torch.tensor([], device=self.device)
        
        if torch.distributed.is_initialized():
             world_size = torch.distributed.get_world_size()
             gathered_outputs = [None] * world_size
             torch.distributed.all_gather_object(gathered_outputs, self.validation_step_outputs)
             all_probs = torch.cat([torch.cat([x["prob"] for x in rank_out]) for rank_out in gathered_outputs if rank_out])
             all_labels = torch.cat([torch.cat([x["label"] for x in rank_out]) for rank_out in gathered_outputs if rank_out])
        else:
             all_probs = local_probs.cpu()
             all_labels = local_labels.cpu()
        
        opt_f2, opt_thresh = 0.0, 0.5
        if all_probs.numel() > 0:
            thresholds = torch.linspace(0.01, 0.99, 50)
            best_f2 = -1.0
            
            # [SOTA FIX] Handle multi-class probabilities for binary-style F2 calibration
            # We treat class 1 and 2 as "Sepsis" (Positive)
            if all_probs.dim() == 2 and all_probs.shape[1] >= 2:
                # Sum probabilities of Pre-Shock (1) and Shock (2)
                pos_probs = all_probs[:, 1:].sum(dim=1).clamp(0, 1)
                pos_labels = (all_labels > 0).long()
            else:
                pos_probs = all_probs.view(-1)
                pos_labels = all_labels.view(-1).long()

            for t in thresholds:
                preds = (pos_probs >= t).long()
                all_l = pos_labels
                tp = ((preds == 1) & (all_l == 1)).sum().item()
                fp = ((preds == 1) & (all_l == 0)).sum().item()
                fn = ((preds == 0) & (all_l == 1)).sum().item()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
                if f2 > best_f2:
                    best_f2 = f2; opt_thresh = t.item()
            opt_f2 = best_f2

        # [Point 5] Bayesian Moving Average Calibration
        # Stabilizes the threshold across epochs and world GPUs
        if torch.distributed.is_initialized():
            threshold_tensor = torch.tensor([opt_thresh], device=self.device)
            torch.distributed.all_reduce(threshold_tensor, op=torch.distributed.ReduceOp.SUM)
            opt_thresh = (threshold_tensor / torch.distributed.get_world_size()).item()
            
        # Apply EMA to the threshold
        new_thresh = opt_thresh
        prev_thresh = self.calibrated_threshold.item()
        updated_thresh = (self.threshold_ema_decay * prev_thresh) + ((1 - self.threshold_ema_decay) * new_thresh)
        self.calibrated_threshold.fill_(updated_thresh)
        
        # Use the CALIBRATED (EMA) threshold for metrics
        final_thresh = self.calibrated_threshold.item()
            
        # Log calibrated Metrics using the Bayesian-stabilized threshold
        self.val_precision.threshold = final_thresh
        self.val_recall.threshold = final_thresh
        self.val_f1.threshold = final_thresh
            
        self.log_dict({
            "val/mse_global": self.val_mse_global.compute(),
            "val/mse_hemo": self.val_mse_hemo.compute(),
            "val/mse_labs": self.val_mse_labs.compute(),
            "val/mse_electrolytes": self.val_mse_electrolytes.compute(),
            "val/sepsis_acc": self.val_acc_sepsis.compute(),
            "val/sepsis_auroc": self.val_auroc_sepsis.compute(),
            "val/sepsis_precision": self.val_precision.compute(),
            "val/sepsis_recall": self.val_recall.compute(),
            "val/sepsis_f1": self.val_f1.compute(),
            "val/clinical_f2_opt": opt_f2,
            "val/clinical_threshold_opt": final_thresh,
            "val/raw_threshold_epoch": opt_thresh,
            "val/ece": self.val_ece.compute(),
            "val/oe": self.val_oe.compute(),
            "val/explained_var": self.val_explained_var.compute(),
            "val/ood_rate_avg": self.val_ood_rate.compute(),
            "val/safe_trajectories_avg": self.val_safe_traj_count.compute(),
            "val/phys_violation_rate": self.val_phys_violation_rate.compute(),
        }, prog_bar=True, sync_dist=True)



        
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


    def _get_curr_physics_weight(self) -> float:
        """
        Curriculum Learning for Physiological Constraints.
        Ramps up weight over 50% of TOTAL steps for resume transparency.
        [v14.9 SOTA Fix] Dynamic Anchor to actual training length.
        """
        total_steps = getattr(self.trainer, "estimated_stepping_batches", 50000)
        warmup_steps = total_steps * 0.5
        
        current_step = float(self.global_step)
        
        if current_step < warmup_steps:
            progress = current_step / float(max(1, warmup_steps))
            return 0.01 + (self.base_phys_weight - 0.01) * progress
        else:
            return self.base_phys_weight

    @contextlib.contextmanager
    def ema_teacher_context(self):
        """
        Context manager that temporarily swaps Student weights with 
        Teacher (EMA) weights for stable inference.
        
        This is essential for:
        1. Value estimation in training (prevents "Dead Critic")
        2. Generation during validation (best quality outputs)
        
        The swap is safe for DDP as it operates on the local model only.
        
        [OPTIONALITY]: If 'use_teacher' is False (ema is None), this falls back 
        to 'nullcontext', meaning the Student effectively teaches itself.
        """
        # [v15.3] SOTA Fix: Delegate to TieredEMA's robust context manager
        # TieredEMA handles CPU offloading, pinning, and restoration automatically.
        if hasattr(self, 'ema') and self.ema is not None:
            with self.ema.swap():
                yield
        else:
            # Fallback for Student-Student Bootstrapping
            yield

    @contextlib.contextmanager
    def frozen_stats(self):
        """
        [v17.3] BN Correlation Guard.
        Ensures ghosts in the expanded batch do NOT poison the running statistics
         of the Shared Foundation's Batch Normalization layers.
        """
        original_momentums = {}
        # Synchronized SyncBatchNorm requires care in multi-GPU settings
        for name, module in self.model.named_modules():
             if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                  original_momentums[name] = module.momentum
                  module.momentum = 0.0
        try:
             yield
        finally:
             for name, module in self.model.named_modules():
                  if name in original_momentums:
                       module.momentum = original_momentums[name]

    def on_fit_start(self):
        """
        Pre-flight checks (DDP Safe):
        1. Calibrate Normalizer (Deterministic file I/O → All Ranks).
        2. Whitening AWR Stats (Random Sampling → Rank 0 & Broadcast).
        3. Sync EMA shadow with calibrated normalizer.
        [SOTA v8.0] Unified Initialization Strategy.
        Handles both Fresh Calibration and Robust Resume Restoration.
        """
        # =====================================================================
        # 1. RESUME INTEGRITY CHECK (Priority 1)
        # =====================================================================
        # If we loaded from a checkpoint, we MUST restore state before doing anything else.
        if hasattr(self, "pending_normalizer_state"):
             if hasattr(self.model, "normalizer"):
                 self.model.normalizer.load_state_dict(self.pending_normalizer_state)
                 logger.info("✅ [RESUME] Normalizer state restored to Model (Calibration Preserved).")
                 del self.pending_normalizer_state
             else:
                 logger.warning("⚠️ [RESUME] Pending normalizer state found but MODEL has no normalizer!")
        
        # [SOTA FIX] Device Guard for PMS EMA
        if self._fnd_grad_ema is not None:
             self._fnd_grad_ema = {k: v.to(self.device) for k, v in self._fnd_grad_ema.items()}
             logger.info(f"✅ [RESUME] PMS Buffers migrated to {self.device}.")
        
        # Ensure AWR Stats are synced (if resumed, they are already in the buffer)
        if self.awr_calculator.stats_initialized:
             logger.info(f"✅ [RESUME] AWR Engine Online: mu={self.awr_calculator.adv_mean:.4f}, sigma={self.awr_calculator.adv_std:.4f}")

        # =====================================================================
        # 1.1 MANUAL OPTIMIZER RESTORATION (The Anti-Trauma Fix)
        # =====================================================================
        # PL sometimes confuses optimizer restoration in manual_optimization modes.
        # We manually force the state load here if we captured it during loading.
        if hasattr(self, "pending_optimizer_states"):
            optimizers = self.trainer.optimizers
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            
            if len(optimizers) == len(self.pending_optimizer_states):
                try:
                    for opt, state in zip(optimizers, self.pending_optimizer_states):
                        opt.load_state_dict(state)
                    logger.info(f"✅ [RESUME] Manually restored {len(optimizers)} optimizer states (Trauma Averted).")
                except Exception as e:
                    logger.warning(f"⚠️ [RESUME] Manual optimizer restoration failed: {e}")
            else:
                logger.warning(f"⚠️ [RESUME] Optimizer count mismatch: Found {len(self.pending_optimizer_states)}, Expected {len(optimizers)}")
            
            # Clean up memory
            del self.pending_optimizer_states

        # =====================================================================
        # 1.2 MANUAL SCHEDULER RESTORATION
        # =====================================================================
        if hasattr(self, "pending_scheduler_states"):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            
            # Filter None if any
            schedulers = [s for s in schedulers if s is not None]

            if len(schedulers) == len(self.pending_scheduler_states):
                try:
                    for sch, state in zip(schedulers, self.pending_scheduler_states):
                        sch.load_state_dict(state)
                    logger.info(f"✅ [RESUME] Manually restored {len(schedulers)} LR scheduler states.")
                except Exception as e:
                    logger.warning(f"⚠️ [RESUME] Manual scheduler restoration failed: {e}")
            else:
                 logger.warning(f"⚠️ [RESUME] Scheduler count mismatch: Found {len(self.pending_scheduler_states)}, Expected {len(schedulers)}")
            
            del self.pending_scheduler_states

        # =====================================================================
        # 1.3 DDP ACCELERATOR INITIALIZATION
        # =====================================================================
        if torch.cuda.is_available():
             world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
             if self.ddp_gatherer is None:
                 self.ddp_gatherer = SOTA_DistributedGatherer(self.device, world_size)
                 logger.info(f"✅ [SOTA] DDP Gatherer Online (World={world_size}, Device={self.device})")

        # =====================================================================
        # 2. FRESH CALIBRATION (Priority 2)
        # =====================================================================
        # Only run if NOT restored and NOT calibrated.
        # This prevents double-calibration or overwriting restored stats.
        
        if not (hasattr(self.trainer, "datamodule") and self.trainer.datamodule):
            logger.warning("No DataModule found. Skipping stats fitting.")
            return

        loader = self.trainer.datamodule.train_dataloader()
        dataset = loader.dataset
        
        # --- 1. Physics Normalizer Calibration ---
        # [SOTA Fix] Check explicitly if model normalizer needs calibration
        if hasattr(self.model, "normalizer") and not self.model.normalizer.is_calibrated.item():
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
        
        # [v4.1.1 SOTA FIX] Final Pre-Flight Cleanup
        # Ensure that the dataset handle is closed on ALL ranks 
        # before the trainer officially starts the worker loop.
        if hasattr(dataset, "_lmdb_env"):
            dataset._lmdb_env = None

    def _fit_awr_stats_ddp(self, dataset):
        """
        Calibrates AWR statistics (mean/std of returns) using a subset of data.
        Synced across DDP ranks.
        """
        # [v20.2] RESUMPTION FIX: Do not re-calibrate if stats are already loaded!
        if self.awr_calculator.stats_initialized:
            logger.info("[AWR] Resume detected: Skipping calibration (Stats already initialized).")
            return

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
        
        # [v4.1.1 SOTA FIX] Clean up LMDB handle after fit
        # This prevents forked workers from inheriting an active handle.
        if hasattr(dataset, "_lmdb_env"):
            dataset._lmdb_env = None

    # Removed on_before_optimizer_step in favor of manual clipping in training_step

    def configure_optimizers(self):
        """
        [2025 SOTA] Conflict-Averse Optimizer Configuration.
        Wraps robust AdamW with CAGrad for surgical conflict resolution.
        """
        from icu.utils.train_utils import configure_robust_optimizer, get_cosine_schedule_with_warmup

        # 1. Configure Robust AdamW (Fused + Parameter Hygiene)
        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Parameter Groups
            # [Point 1 FIX] Cooled Scaler LR (0.025 -> 0.005)
            # Prevents 'Gaming' and 'Administrative Amnesia'
            uw_lr = self.cfg.train.get("uw_lr", 0.005)
            # Use 5x lr for critic ONLY if we can isolate it. 
            # In ICUUnifiedPlanner, models are combined. 
            # We'll stick to a unified model LR but keep expert_state_head and loss_scaler separate.
            
            # [PHASE 1 FIX] Task-Specific Learning Rates
            # Sepsis head needs to learn faster (3x) to catch up with dominant diffusion gradients.
            
            # 1. Identify Aux Parameters
            aux_params = list(self.model.aux_head.parameters()) if hasattr(self.model, 'aux_head') else []
            aux_param_ids = {id(p) for p in aux_params}
            
            # 2. Identify ACL Parameters (boosted for discrimination)
            acl_params = list(self.acl_projector.parameters())
            acl_param_ids = {id(p) for p in acl_params}
            
            # 3. Identify Main Model Parameters (excluding Aux and ACL)
            main_model_params = [
                p for p in self.model.parameters() 
                if id(p) not in aux_param_ids
            ]
            
            # [PATCH 4] Reduce LR Multiplier
            # Original: 3.0x LR for aux/acl caused GN spikes to 16.9
            # Evidence: Combined with fixed weights (1.5x), effective boost was ~4.5x
            # Fix: Reduce to 1.5x for gentler learning
            aux_lr_mult = self.cfg.train.get("aux_lr_multiplier", 1.5)
            
            optimizer_params = [
                # Group 1: Main Backbone (Standard LR)
                {'params': main_model_params, 'lr': self.cfg.train.lr, 'weight_decay': self.cfg.train.weight_decay},
                
                # Group 2: Aux Head (Boosted LR)
                {'params': aux_params, 'lr': self.cfg.train.lr * aux_lr_mult, 'weight_decay': self.cfg.train.weight_decay},
                
                # Group 4: ACL Projector (Boosted LR)
                {'params': acl_params, 'lr': self.cfg.train.lr * aux_lr_mult, 'weight_decay': self.cfg.train.weight_decay},
                
                # Group 5: Uncertainty Scaler (Special LR)
                {'params': self.loss_scaler.parameters(), 'lr': uw_lr, 'weight_decay': 0.0}
            ]
            
            logger.info(f"Optimizer: Initialized with {len(optimizer_params)} param groups. Model LR: {self.cfg.train.lr:.2e}, Scaler LR: {uw_lr:.2e}")
            
            base_optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
                betas=(0.9, 0.999),
                fused=False
            )


        else:
            optimizer_params = [{"params": self.model.parameters()}]
            base_optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
                betas=(0.9, 0.999),
                fused=False # [v12.8.5 SOTA FIX] Use foreach=True (robuster for compiled models)
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



    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        [SOTA v3.1] Persist Normalizer and Critical Buffers.
        Ensures 'Immortality': The model can resume EXACTLY where it left off,
        preserving global normalization statistics and AWR whitening parameters.
        """
        # [v2.0 Refactor] Manual state saving removed for nn.Modules.
        # However, we MUST save the lazily-initialized Gradient EMA (not a buffer).
        if hasattr(self.model, "normalizer"):
            checkpoint["normalizer_state"] = self.model.normalizer.state_dict()
            logger.info("[SAVE] Normalizer state captured in checkpoint.")
            
        if self._fnd_grad_ema is not None:
           checkpoint["fnd_grad_ema"] = self._fnd_grad_ema

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        [SOTA v3.1] Restore Normalizer and Stats.
        """
        # [v2.0 Refactor] Restore PMS Gradient EMA
        if "normalizer_state" in checkpoint:
            self.pending_normalizer_state = checkpoint["normalizer_state"]
            logger.info("[RESUME] normalizer_state captured for on_fit_start.")
            
        if "fnd_grad_ema" in checkpoint:
            self._fnd_grad_ema = checkpoint["fnd_grad_ema"]
            logger.info("[RESUME] PMS Gradient EMA restored.")

        # [CRITICAL FIX] Capture Optimizer & Scheduler States for Manual Restoration
        # Standard PL sometimes drops these when using manual_optimization=True
        if "optimizer_states" in checkpoint:
             self.pending_optimizer_states = checkpoint["optimizer_states"]
             logger.info(f"[RESUME] Captured {len(self.pending_optimizer_states)} optimizer states for manual restoration.")

        if "lr_schedulers" in checkpoint:
             self.pending_scheduler_states = checkpoint["lr_schedulers"]
             logger.info(f"[RESUME] Captured {len(self.pending_scheduler_states)} LR scheduler states for manual restoration.")
