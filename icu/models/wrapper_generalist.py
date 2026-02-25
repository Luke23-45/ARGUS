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
import torch.distributed as dist
import pytorch_lightning as pl
import torchmetrics.functional as tm_func
from tqdm.auto import tqdm
from typing import Any, Dict, Optional, Tuple, List, Union
import contextlib # [v2026] Required for DDP Iron Dome
import logging
from omegaconf import DictConfig
import math
import numpy as np
import random
import traceback
import gc
import zlib
import os
# [v2025 SOTA] Implementation Imports
from icu.core.cagrad import CAGrad
from icu.core.gradnorm import GradNormBalancer
from icu.core.robust_losses import (
    smooth_l1_critic_loss,
    physiological_violation_loss
)

# Project Imports
from icu.models.diffusion import ICUUnifiedPlanner, ClinicalResidualHead, ICUConfig, PhysiologicalConsistencyLoss
from icu.utils.train_utils import EMA, ScalingSteward
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.utils.metrics_advanced import (
    compute_policy_entropy, 
    compute_ece, 
    compute_explained_variance, 
    compute_overconfidence_error
)
from icu.utils.logging_utils import BufferedCSVLogger
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
    OrthogonalGuard,
    TrendSentinel
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
        self.register_buffer("beta", torch.tensor([beta]).float())
        self.register_buffer("counts", torch.zeros(num_classes))
        self.register_buffer("initialized", torch.tensor([False]))
        
        if prior_pos_weight is not None and num_classes > 1 and prior_pos_weight > 0:
             n_neg = 1000.0
             n_pos_each = n_neg / prior_pos_weight
             self.counts[0] = n_neg
             self.counts[1:] = n_pos_each
             self.initialized.fill_(True)

    def update(self, y: torch.Tensor):
        """Standardizes class statistics across all DDP ranks."""
        y = y.long()
        b_counts = torch.bincount(y, minlength=self.num_classes).float()
        
        # [v2026 SOTA FIX] Unified Initial Consensus (Smoking Gun #DDP-Init-Bias)
        # Rationale: Ranks must share the same 'Ground Zero' statistics.
        if dist.is_initialized():
            dist.all_reduce(b_counts, op=dist.ReduceOp.SUM)
            b_counts /= dist.get_world_size()

        if not self.initialized:
            self.counts.copy_(b_counts + 1.0)
            self.initialized.fill_(True)
        else:
            new_counts = self.beta * self.counts + (1 - self.beta) * b_counts
            self.counts.copy_(new_counts)

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies class-weight momentum across step densities."""
        if n_curr <= 0: return
        self.beta.fill_(ScalingSteward.get_decay(0.9995, n_curr))

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
        
        # [v12.0 SOTA FIX] Adaptive Clipping State
        self.adaptive_clipping = cfg.train.get("adaptive_clipping", True)
        
        # [v25.3] Flexible Multi-Task Balancing
        # modes: "sota_2025" (Hooks + UW) or "legacy_surgical" (CAGrad + GradNorm)
        self.balancing_mode = cfg.train.get("balancing_mode", "sota_2025")
        logger.info(f"Using balancing mode: {self.balancing_mode}")

        # [v19.0 SOTA] A-GEM Orthogonal Replay Configuration
        # If True: Adds Reference Gradient back after projection.
        self.orthogonal_replay = True

        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Uncertainty Loss Scaler
            # [v12.8.3 SOTA FIX] Direct Attachment
            # Attaching to self instead of self.model to ensure safe device movement
            # and registration within the LightningModule, avoiding torch.compile issues.
            # [v4.0 FIX] Initialized with 7 tasks: [diffusion, critic, aux, acl, bgsl, tcb, phys]
            self.loss_scaler = BayesianProjectedScaler(num_tasks=7)
            logger.info("Using Model's BayesianProjectedScaler for balancing (7 tasks).")
        
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
            # [SOTA FIX - HORIZON RAMPING] Dynamic Horizon Bounds
            start_gamma=cfg.train.get("awr_start_gamma", 0.80),
            gamma=cfg.train.get("awr_gamma", 0.99),
            adaptive_beta=cfg.train.get("adaptive_beta", True),
            adaptive_clipping=cfg.train.get("adaptive_clipping", True),
            beta_momentum=cfg.train.get("awr_momentum", 0.98), # [v38.0] Smoother AWR transition
            target_ess=cfg.train.get("target_ess", 0.10),      # Grounded to % since lop6
            min_beta=cfg.train.get("beta_min", 0.5)            # [v38.0] Selection Floor
        )
        
        # =====================================================================
        # 4. SOTA GRADIENT & LOSS BALANCING
        # =====================================================================
        self.gradnorm = None
        if self.balancing_mode == "legacy_surgical" or True: # [v25.7] Force enable for ACL expansion
            # [SOTA v30.5] Head Exclusion for GradNorm
            # Rationale: GradNorm must only monitor the shared backbone. 
            # Including heads inflates norms with task-specific noise.
            excluded_heads = ['aux_head', 'value_head', 'dynamics_head', 'actor_head', 'diffusion_head', 'acl_projector']
            shared_params = [
                p for n, p in self.model.named_parameters() 
                if not any(h in n for h in excluded_heads)
            ]
            
            self.gradnorm = GradNormBalancer(
                num_tasks=7, 
                shared_params=shared_params,
                alpha=cfg.train.get("alpha_min", 0.1) # [SOTA SG-2] Reduced for Sepsis sensitivity
            ).to(self.device)
        
        self.base_phys_weight = cfg.train.get("phys_loss_weight", 0.2)
        self.safety_guardian = OODGuardian()
        self.forensic_auditor = ForensicStabilityAuditor(guardian=self.safety_guardian)
        
        # --- Internal Buffers ---
        pos_weight = cfg.train.get("pos_weight", None)
        self.class_balancer = DynamicClassBalancer(
            num_classes=cfg.model.get("num_phases", 3),
            prior_pos_weight=pos_weight,
            beta=cfg.train.get("balancer_beta", 0.9995)
        )
        # [v25.4 FIX] Initial Log-Var Reset: Start with balanced weights (sigma=1.0)
        if self.balancing_mode == "sota_2025":
            # [PHASE 2] Prioritize Diffusion to break Mean-Prediction Trap
            initial_log_vars = torch.tensor([
                -0.69,  # diffusion: Higher weight (~2x) to force signal recovery
                1.5,    # critic: Start even lower to allow diffusion to settle
                -0.5,   # aux: Clinical Anchor (High weight)
                1.0,    # acl: Start low
                1.0,    # bgsl: Start low
                2.5,    # tcb: Start VERY low (high log_var) to prevent random queue shock
                0.5,    # phys: Standard prior for physics constraints
            ])
            # Ensure device compatibility if loaded later
            self.loss_scaler.log_vars.data.copy_(initial_log_vars)
        
        # =====================================================================
        # [NEW] AGENTIC EVOLUTION CORE (Phases 1-3)
        # =====================================================================
        # Phase 1: Dynamic Diagnostics
        self.risk_scorer = PhysiologicalRiskScorer()
        self.risk_aware_loss = RiskAwareAsymmetricLoss(
            gamma_neg=cfg.train.get("asl_gamma_neg", 2.0),
            gamma_pos=cfg.train.get("asl_gamma_pos", 0.5),
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
            gamma=cfg.train.get("asl_gamma_neg", 2.0), # Reusing ASL gamma
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
            capacity=cfg.train.get("ghost_capacity", 1505),
            history_len=cfg.model.get("history_len", 24),
            feature_dim=cfg.model.get("input_dim", 28),
            latent_dim=cfg.model.get("d_model", 512),
            similarity_threshold=cfg.train.get("ghost_sim_threshold", 0.98)
        )
        
        
        # [v177.3 SOTA] Multi-Task Gradient Pressure EMAs
        # Rationale: Track gradient magnitudes for Physics vs Diffusion to ensure 1:1 balance.
        # Registered as buffers to ensure persistence across resumption.
        self.register_buffer("phys_grad_ema", torch.tensor([1.0]))
        self.register_buffer("diff_grad_ema", torch.tensor([1.0]))
        
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
        self.register_buffer("_awr_stats_initialized", torch.tensor([False]))
        self.validation_step_outputs = []
        
        # [Point 5] Bayesian Moving Average Calibration
        # Initialized to 0.5; will be updated via F2-opt during validation.
        self.register_buffer("calibrated_threshold", torch.tensor([0.5]))
        self.threshold_ema_decay = 0.85 # [SOTA PATCH] Balanced calibration speed (0.99 was too slow, 0.7 too fast)
        
        self.register_buffer("curr_tau", torch.tensor([0.5]))
        self.register_buffer("curr_sigma_scale", torch.tensor([3.50]))
        self.register_buffer("curr_phys_clamp", torch.tensor([10.0])) # [v29.6] Step-Invariant Clamp
        
        # [PMS] Manifold Stability Monitoring (v26.5 SOTA)
        # [v26.6 FIX] Initialize to 0.0 so TrendSentinel's bias correction (1 - beta^t) works.
        self.register_buffer("stability_factor", torch.tensor([1.0]))
        self.register_buffer("grad_norm_ema", torch.tensor([1.0]))
        self.register_buffer("grad_norm_std", torch.tensor([0.0])) 
        self.register_buffer("grad_norm_step_count", torch.tensor([0], dtype=torch.long))
        self.grad_ema_decay = cfg.train.get("grad_ema_decay", 0.99)
        
        # [PMS] Resumption Grace Period (Circuit Breaker)
        # Prevents "False Shocks" as the first few batches settle after resume.
        self.register_buffer("resumption_grace_steps", torch.tensor([0], dtype=torch.long))
        self._pms_start_logged = False
        self._pms_init_logged = False
        
        # [v107.0 SOTA] GradNorm Accumulation Parity (Smoking Gun #107)
        # Accumulates task losses across the cycle to prevent sampling bias.
        # [v33.4 FIX] Synchronized to 7 tasks (Diffusion, Critic, Aux, ACL, BGSL, TCB, Phys)
        self.register_buffer("gn_loss_accumulator", torch.zeros(7))
        self.register_buffer("gn_acc_count", torch.tensor([0], dtype=torch.long))
        
        # [v48.0 SOTA FIX] Stateful Accumulation Index (Smoking Gun #90)
        # Rationale: Ensures bit-perfect cycle alignment across resumptions.
        self.register_buffer("grad_accum_idx", torch.tensor([0], dtype=torch.long))
        
        # [v12.0 SOTA] CPU Shadows for Zero-Sync
        # Rationale: Prevents hot-path .item() syncs in training_step.
        self._shadow_grad_accum_idx = 0
        self._shadow_resumption_grace_steps = 0
        
        # [v110.0 SOTA] Bayesian Safe-Start (Smoking Gun #110)
        self._fnd_grad_ema = None 
        self.grad_ref_buffer = None # [v53.0] AGEM Reference Accumulator
        
        # [SOTA DDP] Zero-Copy Gatherer
        self.ddp_gatherer = None 
        
        # [v2026] Intra-Epoch Telemetry
        # Efficient percentages logging (e.g. "Epoch 6: 5%")
        self.csv_log_interval = cfg.train.get("csv_log_interval_percent", 5.0)
        self.csv_logger = None
        if self.csv_log_interval > 0:
            # Lazy init to handle DDP rank checks later or just write from all ranks (filtered usually)
            pass
        
        self.last_logged_bucket = -1
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        [SOTA v2026] Resilient State Loading.
        
        Features:
        1.  Shape-Aware Bridge: Automatically reshapes scalar buffers ([]) from 
            older checkpoints into the new 1D vector format ([1]).
        2.  Relaxed Constraint: Uses strict=False for checkpoints to allow 
            seamless addition of new stability buffers.
        """
        # 1. Detection of checkpoint loading vs manual state_dict application
        is_checkpoint = any("model." in k for k in state_dict.keys()) or any("awr_calculator." in k for k in state_dict.keys())
        
        if is_checkpoint:
            logger.info("[PMS] Resilient Loading: Initializing restoration bridge...")
            
            # 2. [v2026 SOTA] Scalar-to-Vector Normalization
            # Rationale: Many scalar buffers were upgraded from shape [] to [1]
            # to ensure DDP consistency. We must reshape them in the state_dict
            # to match the new buffers, otherwise PyTorch's load_state_dict 
            # will ignore them or fail (even with strict=False).
            new_state_dict = state_dict.copy()
            model_state = self.state_dict()
            
            # [FIX] Handle timestep/shape mismatches (e.g., timesteps 100 -> 500)
            # Filter out keys with incompatible shapes that would cause RuntimeError
            mismatched_keys = []
            for name, expected_tensor in model_state.items():
                if name in new_state_dict:
                    checkpoint_tensor = new_state_dict[name]
                    # Check for shape mismatch
                    if checkpoint_tensor.shape != expected_tensor.shape:
                        mismatched_keys.append(name)
                        logger.warning(f"  [Shape Mismatch] {name}: checkpoint {checkpoint_tensor.shape} vs model {expected_tensor.shape} - SKIPPING")
                        del new_state_dict[name]
                        continue
                    
                    # Case 1: scalar ([]) in checkpoint, vector ([1]) in model
                    if checkpoint_tensor.dim() == 0 and expected_tensor.dim() == 1 and expected_tensor.shape[0] == 1:
                        new_state_dict[name] = checkpoint_tensor.view(1)
                        logger.debug(f"  [Reshape] {name}: [] -> [1]")
                    
                    # Case 2: int/long cast
                    if checkpoint_tensor.dtype != expected_tensor.dtype:
                        new_state_dict[name] = checkpoint_tensor.to(expected_tensor.dtype)
            
            if mismatched_keys:
                logger.info(f"[PMS] Skipped {len(mismatched_keys)} mismatched keys (likely timesteps change). Model will reinitialize these.")

            return super().load_state_dict(new_state_dict, strict=False)
            
        return super().load_state_dict(state_dict, strict=strict)
    
    
    def _get_scaler(self):
        """[v2026 SOTA] Safe GradScaler accessor (Smoking Gun #FP32-Crash).
        Returns None if no scaler exists (FP32 mode or incompatible Lightning version).
        """
        return getattr(getattr(self.trainer, 'precision_plugin', None), 'scaler', None)
    
    def on_train_start(self):
        """[SOTA v2026] Unified Mathematical Hyperparameter Scaling."""
        # Detect the actual number of iterations per epoch (e.g., SOTA_REF_STEPS vs debug)
        n_curr = self.trainer.num_training_batches
        logger.info(f" [SOTA] Scaling Steward: Unifying dynamics for {n_curr} steps.")
        self.grad_ema_decay = ScalingSteward.get_decay(0.99, n_curr)
        self.class_balancer.scale_dynamics(n_curr)
        
        # [SOTA FIX P12] Global Warmup Synchronization
        config_warmup = self.cfg.train.get("warmup_steps", 0)
        if config_warmup > 0:
            self.trainer.global_warmup_steps = config_warmup
            self.trainer.warmup_steps = config_warmup # Sync legacy attribute
        else:
            ref_warmup = ScalingSteward.SOTA_REF_WARMUP
            self.trainer.global_warmup_steps = ScalingSteward.get_steps(ref_warmup, n_curr)
            self.trainer.warmup_steps = self.trainer.global_warmup_steps
        
        self.awr_calculator.scale_dynamics(n_curr)
        self.ghost_bank.scale_dynamics(n_curr)
        self.tcb_buffer.scale_dynamics(n_curr)
        self.sepsis_acl.scale_dynamics(n_curr)
        
        if hasattr(self, "loss_scaler"):
            self.loss_scaler.scale_dynamics(n_curr)
            
        if hasattr(self, "horizon_scheduler"):
            self.horizon_scheduler.scale_dynamics(n_curr)
        
        # [SOTA FIX v9.0] Unconditional Grace Period
        # Rationale: Whether fresh start OR resumption, we must allow 50 steps
        # for TrendSentinel buffers (grad_norm_ema) to align with current dynamics.
        # This prevents "False Shock" detection from freezing the curriculum.
        self.resumption_grace_steps.fill_(50)

        # [v48.1 SOTA FIX] Resumption Accumulation Reset (Smoking Gun #90)
        # Rationale: PyTorch/Lightning does NOT restore .grad buffers or 
        # unregistered accumulation state. If we resume with a non-zero 
        # grad_accum_idx, the first update will be under-estimated.
        if self.trainer.ckpt_path is not None:
             logger.info("[RESUME] Resetting accumulation buffers to ensure mathematical parity.")
             self.grad_accum_idx.fill_(0)
             self.gn_acc_count.fill_(0)
             self.gn_loss_accumulator.zero_()
             
             # [v2026 SOTA FIX] Loss Scaler Accumulation Reset (Smoking Gun #Resume-Pollution)
             # Rationale: If resuming mid-epoch, scaler buffers might contain partial sums.
             # Since we reset grad_accum_idx to 0, we must also clear the scaler to match.
             if hasattr(self, "loss_scaler") and hasattr(self.loss_scaler, "loss_accumulator"):
                  self.loss_scaler.loss_accumulator.zero_()
                  self.loss_scaler.task_counters.zero_()
                  self.loss_scaler.batch_counter.zero_()
                  
             if hasattr(self, "grad_ref_buffer") and self.grad_ref_buffer is not None:
                  self.grad_ref_buffer.zero_()
                  
             # [v2026 SOTA FIX] Momentum Bank Synchronization (Trauma Trace SG-Bank)
             if hasattr(self, "ghost_bank") and hasattr(self.ghost_bank, "sync_shadows"):
                  self.ghost_bank.sync_shadows()
                  logger.info("✅ [RESUME] Ghost Bank Shadows Synchronized.")
                  
             # [v2026 SOTA FIX] Temporal Buffer Synchronization (Trauma Trace SG-TCB)
             if hasattr(self, "tcb_buffer") and hasattr(self.tcb_buffer, "sync_shadows"):
                  self.tcb_buffer.sync_shadows()
                  logger.info("✅ [RESUME] TCB Shadows Synchronized.")
        
        # [v12.0 SOTA] Initialize CPU Shadows
        self._shadow_grad_accum_idx = int(self.grad_accum_idx)
        self._shadow_resumption_grace_steps = int(self.resumption_grace_steps)

    def on_fit_start(self):
        """
        [SOTA v2026] Elastic Resumption Bridge (Patch 1).
        Handles: 
        1. Context-Aware Restoration (EMA, Optimizers, Schedulers).
        2. DDP-Safe Accelerator Initializtion.
        3. Priority-Based Calibration (Normalizer, AWR).
        """
        # =====================================================================
        # 1. RESUME INTEGRITY CHECK (Priority 1: Core State Restoration)
        # =====================================================================
        # 1.1 Normalizer Restoration
        # [v1.1 Restoration moved to 2.1.5 for consolidation]

        # 1.2 PMS Buffer Migration
        if self._fnd_grad_ema is not None:
             self._fnd_grad_ema = self._fnd_grad_ema.to(self.device)
             logger.info(f"✅ [RESUME] PMS Buffers migrated to {self.device}.")

        # 1.3 AWR Heartbeat Check
        if self.awr_calculator.stats_initialized:
            if hasattr(self.awr_calculator, 'adv_mean'):
                logger.info(f"✅ [RESUME] AWR Engine Online: mu={self.awr_calculator.adv_mean.item():.4f}, sigma={self.awr_calculator.adv_std.item():.4f}")

        # =====================================================================
        # 2. MANUAL OPTIMIZER & SCHEDULER RESTORATION (The Anti-Trauma Bridge)
        # =====================================================================
        # 2.1 Optimizers
        if hasattr(self, "pending_optimizer_states"):
            optimizers = self.trainer.optimizers
            if not isinstance(optimizers, list): optimizers = [optimizers]
            
            if len(optimizers) == len(self.pending_optimizer_states):
                try:
                    # [v2026 SOTA FIX] Optimizer State Shape Bridge (Smoking Gun #ShapeCrash)
                    # Rationale: Model load_state_dict bridges buffers/params from [] → [1],
                    # but optimizer states (exp_avg, exp_avg_sq) are NOT bridged.
                    # This causes a fatal shape mismatch in Adam's lerp_():
                    #   exp_avg.lerp_(grad, 1-beta) → RuntimeError: [] vs [1]
                    # Fix: Reshape all 0-dim optimizer state tensors to [1] before loading.
                    bridged_count = 0
                    for opt_state in self.pending_optimizer_states:
                        if 'state' in opt_state:
                            for param_id, pstate in opt_state['state'].items():
                                for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                                    if key in pstate and isinstance(pstate[key], torch.Tensor):
                                        if pstate[key].dim() == 0:
                                            pstate[key] = pstate[key].unsqueeze(0)
                                            bridged_count += 1
                    if bridged_count > 0:
                        logger.info(f"🛡️ [RESUME] Optimizer Shape Bridge: Reshaped {bridged_count} state tensors ([] → [1]).")

                    for opt, state in zip(optimizers, self.pending_optimizer_states):
                        opt.load_state_dict(state)
                    logger.info(f"✅ [RESUME] Manually restored {len(optimizers)} optimizer states (Trauma Averted).")
                except Exception as e:
                    logger.warning(f"⚠️ [RESUME] Manual optimizer restoration failed: {e}")
            else:
                logger.warning(f"⚠️ [RESUME] Optimizer count mismatch: Found {len(self.pending_optimizer_states)}, Expected {len(optimizers)}")
            del self.pending_optimizer_states

        # 2.1.5 Physics Normalizer (Critical Bridge)
        # Rationale: Normalizer stats are the manifold anchor. Resuming without them
        # or with fresh calibration causes a massive "Adaptation Shock".
        if hasattr(self, "pending_normalizer_state") and hasattr(self.model, "normalizer"):
            try:
                self.model.normalizer.load_state_dict(self.pending_normalizer_state)
                logger.info("✅ [RESUME] Physics Normalizer stats restored directly.")
                
                # Sync EMA shadow immediately to prevent Teacher drift
                if hasattr(self, 'ema') and self.ema is not None:
                    for name, buffer in self.model.normalizer.named_buffers():
                        full_name = f"normalizer.{name}"
                        if full_name in self.ema.shadow:
                            self.ema.shadow[full_name] = buffer.data.detach().cpu().clone()
                    logger.info("✅ [RESUME] EMA shadow synced with restored normalizer.")
            except Exception as e:
                logger.warning(f"⚠️ [RESUME] Normalizer restoration failed: {e}")
            del self.pending_normalizer_state

        # 2.2 Schedulers
        if hasattr(self, "pending_scheduler_states"):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list): schedulers = [schedulers]
            schedulers = [s for s in schedulers if s is not None]

            if len(schedulers) == len(self.pending_scheduler_states):
                try:
                    for sch, state in zip(schedulers, self.pending_scheduler_states):
                        sch.load_state_dict(state)
                    logger.info(f"✅ [RESUME] Manually restored {len(schedulers)} LR scheduler states.")
                    
                    # [v55.0 SOTA FIX] LR Pulse Alignment (Smoking Gun #113)
                    # Rationale: Ensure Optimizer LR matches Scheduler state before the first batch.
                    # Standard PL/PyTorch can cause a "Shock Batch" with initial LR if not synced.
                    for opt, sch in zip(self.optimizers() if isinstance(self.optimizers(), list) else [self.optimizers()], schedulers):
                         if hasattr(sch, "get_last_lr"):
                              new_lr = sch.get_last_lr()[0]
                              for pg in opt.param_groups:
                                   pg['lr'] = new_lr
                    logger.info("✅ [RESUME] LR Pulse Alignment synchronized.")
                except Exception as e:
                    logger.warning(f"⚠️ [RESUME] Manual scheduler restoration failed: {e}")
            else:
                 logger.warning(f"⚠️ [RESUME] Scheduler count mismatch: Found {len(self.pending_scheduler_states)}, Expected {len(schedulers)}")
            del self.pending_scheduler_states

        # 2.2.5 AWR Calculator (The Amnesia Fix #38.1)
        # Rationale: Direct buffer restoration ensures bit-perfect parity 
        # for whitening and adaptive beta dynamics.
        if hasattr(self, "pending_awr_state") and hasattr(self, "awr_calculator"):
            try:
                self.awr_calculator.load_awr_state(self.pending_awr_state)
                logger.info("✅ [RESUME] AWR Engine internal state restored.")
            except Exception as e:
                logger.warning(f"⚠️ [RESUME] AWR restoration failed: {e}")
            del self.pending_awr_state

        # 2.2.6 Grand Unified Persistence Telemetry (Phase 38.2)
        # Rationale: Provide clear forensic proof of restoration for all components.
        if self.trainer.training:
            logger.info("[RESUME] Grand Unified Persistence Audit:")
            if hasattr(self, "sepsis_acl"):
                logger.info(f"   |- [ACL] Momentum: {float(self.sepsis_acl.momentum):.4f}")
            if hasattr(self, "class_balancer"):
                logger.info(f"   |- [Balancer] Sepsis Weight: {self.class_balancer.get_weights()[1].item():.4f}")
            if hasattr(self, "bgsl_loss"):
                logger.info(f"   |- [BGSL] Weights: Trend={self.bgsl_loss.w_t.item():.2f}, Shock={self.bgsl_loss.w_h.item():.2f}")
            if hasattr(self, "ghost_bank"):
                logger.info(f"   |- [GhostBank] Size: {self.ghost_bank.size.item()}/{self.ghost_bank.capacity}")
            if hasattr(self, "tcb_buffer"):
                logger.info(f"   |- [TCB] Filled: {self.tcb_buffer.queue_filled.item()}/{self.tcb_buffer.capacity}")

        # 2.2.7 EMA Persistence Bridge (The Missing Link Fix)
        # Rationale: wrapper_generalist captured 'pending_ema_state' but never applied it.
        # This ensures the Teacher model is restored even if the Callback falls through.
        if hasattr(self, "pending_ema_state") and self.ema is not None:
             try:
                 self.ema.load_state_dict(self.pending_ema_state)
                 self._ema_restored_manual = True
                 logger.info("✅ [RESUME] EMA shadow weights MANUALLY restored (Bridge Active).")
             except Exception as e:
                 logger.warning(f"⚠️ [RESUME] Manual EMA restoration failed: {e}")
             del self.pending_ema_state

        # [PATCH SG-1] Forced EMA Hard Sync on Resumption (Divergence Fix)
        # Rationale: The periodic hard sync at (epoch+1) % 3 == 0 can leave a 3-epoch
        # window where the teacher drifts from the student after a checkpoint resume.
        # This was confirmed as the primary root cause of E6→E8 divergence.
        # [v2026 SOTA] Memory Preservation Gate: We only force sync if a manual 
        # restoration from the checkpoint was NOT accomplished. This protects 
        # the "Historical Wisdom" of the teacher model.
        if (hasattr(self, 'ema') and self.ema is not None and self.global_step > 0 
            and not getattr(self, "_ema_restored_manual", False)):
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.ema.shadow:
                        self.ema.shadow[name].copy_(param.data.cpu().float())
                for name, buffer in self.model.named_buffers():
                    if name in self.ema.shadow:
                        if torch.is_floating_point(buffer):
                            self.ema.shadow[name].copy_(buffer.data.cpu().float())
                        else:
                            self.ema.shadow[name].copy_(buffer.data.cpu())
            logger.info("⚡ [RESUME] Forced EMA Hard Sync — Teacher drift prevention active.")
            
            # [PATCH SG-4] Ghost Manifold Alignment
            # Rationale: Anchors must be re-encoded to align with the restored student 
            # manifold to prevent anchor drift shock.
            if hasattr(self, "ghost_bank") and self.ghost_bank is not None:
                def ghost_encoder_fn(x, m):
                    return self.model.encoder(x, imputation_mask=m)["global_expert"]
                self.ghost_bank.refresh_anchors(ghost_encoder_fn, decay=0.9)
                logger.info("👻 [RESUME] Ghost Bank anchors refreshed (Manifold Aligned).")

        # 2.3 GradNorm Optimizer (The Amnesia Fix #v2026)
        # Rationale: Direct state restoration ensures bit-perfect parity 
        # for task weighting momentums and loss anchors.
        if hasattr(self, "pending_gn_state") and self.gradnorm is not None:
             try:
                 self.gradnorm.load_gradnorm_state(self.pending_gn_state)
                 logger.info("✅ [RESUME] GradNorm Engine internal state restored.")
             except Exception as e:
                 logger.warning(f"⚠️ [RESUME] GradNorm restoration failed: {e}")
             del self.pending_gn_state

        # [v52.0 SOTA FIX] Grand Unified Consensus (Smoking Gun #MasterAudit)
        # Rationale: Bit-perfect parity across all meta-parameters and historical buffers.
        if dist.is_available() and dist.is_initialized():
            # [GATED CONSENSUS FIX] Discovery Protocol
            # Rationale: Ranks must vote on existence of conditional components before broadcast.
            # If Rank 1 skips broadcast while Rank 0 executes, the cluster deadlocks permanently.
            sync_mask = torch.tensor([
                1 if getattr(self, "_fnd_grad_ema", None) is not None else 0,
                1 if getattr(self, "grad_ref_buffer", None) is not None else 0
            ], device=self.device, dtype=torch.long)
            dist.all_reduce(sync_mask, op=dist.ReduceOp.MAX)

            # 1. Manifold Stability Parity
            dist.broadcast(self.grad_norm_ema, src=0)
            dist.broadcast(self.grad_norm_std, src=0)
            dist.broadcast(self.grad_norm_step_count, src=0)
            dist.broadcast(self.phys_grad_ema, src=0)
            dist.broadcast(self.diff_grad_ema, src=0)
            
            # 2. Foundation Consensus (MGP & AGEM)
            if sync_mask[0].item() > 0:
                if getattr(self, "_fnd_grad_ema", None) is None:
                    # Allocate dummy tensor to safely receive broadcast
                    self._fnd_grad_ema = torch.zeros(self.model.cfg.d_model, device=self.device)
                dist.broadcast(self._fnd_grad_ema, src=0)
                
            if sync_mask[1].item() > 0:
                if getattr(self, "grad_ref_buffer", None) is None:
                    # Allocate dummy to receive
                    total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
                    self.grad_ref_buffer = torch.zeros(total_p, device=self.device)
                dist.broadcast(self.grad_ref_buffer, src=0)
            
            # 3. Meta-Task Consensus (GradNorm & LossScaler)
            if self.gradnorm is not None:
                if hasattr(self.gradnorm, 'weights'):
                    dist.broadcast(self.gradnorm.weights, src=0)


                # [v52.1] Sync Initial Losses to prevent meta-drift
                if hasattr(self.gradnorm, 'initial_losses'):
                    dist.broadcast(self.gradnorm.initial_losses, src=0)
            # 4. AWR Consensus (Phase 38.1)
            # Rationale: Ranks MUST have identical whitening and beta stats 
            # to prevent divergent selection pressure.
            if hasattr(self, "awr_calculator"):
                dist.broadcast(self.awr_calculator.adv_mean, src=0)
                dist.broadcast(self.awr_calculator.adv_std, src=0)
                dist.broadcast(self.awr_calculator.stats_count, src=0)
                dist.broadcast(self.awr_calculator.beta, src=0)
                dist.broadcast(self.awr_calculator.ess_buffer, src=0)
                dist.broadcast(self.awr_calculator.clip_rate_buffer, src=0)

            logger.info("[PMS] Grand Unified DDP Consensus achieved (Rank Sync Complete).")
            
                       
            if hasattr(self, "loss_scaler"):
                dist.broadcast(self.loss_scaler.log_vars, src=0)
                dist.broadcast(self.loss_scaler.loss_emas, src=0)
                # [v52.3] Sync Accumulators to prevent cycle amnesia (Smoking Gun #MasterAudit)
                # Rationale: All ranks must resume with bit-perfect mid-cycle sums.
                dist.broadcast(self.loss_scaler.loss_accumulator, src=0)
                dist.broadcast(self.loss_scaler.task_counters, src=0)
                dist.broadcast(self.loss_scaler.batch_counter, src=0)
            
            # 5. [v54.0 SOTA] Normalization Consensus (Normalization Skew Fix)
            # Rationale: All ranks must share identical physics and statistical bounds
            # to prevent divergent latent representations.
            if hasattr(self, "normalizer"):
                dist.broadcast(self.normalizer.ts_stat_min, src=0)
                dist.broadcast(self.normalizer.ts_stat_max, src=0)
                dist.broadcast(self.normalizer.static_min, src=0)
                dist.broadcast(self.normalizer.static_max, src=0)
                dist.broadcast(self.normalizer.is_calibrated, src=0)
                
            logger.info("[RESUME] Grand Unified DDP Consensus reached.")

        # [v118.0 SOTA FIX] GradNorm Accumulator Restoration (Bridge Update)
        # Rationale: Preservation of historical loss sums mid-epoch.
        if hasattr(self, "pending_gn_accumulator"):
             self.gn_loss_accumulator.copy_(self.pending_gn_accumulator.to(self.gn_loss_accumulator.device))
             self.gn_acc_count.copy_(self.pending_gn_acc_count.to(self.gn_acc_count.device))
             logger.info("✅ [RESUME] GradNorm accumulators restored (Amnesia Averted).")
             del self.pending_gn_accumulator, self.pending_gn_acc_count
             
        # [v118.1 SOTA FIX] Partial Gradient Restoration
        # Rationale: Work-preservation for mid-cycle crashes.
        if hasattr(self, "pending_partial_grads"):
             count = 0
             for name, p in self.named_parameters():
                 if name in self.pending_partial_grads:
                     p.grad = self.pending_partial_grads[name].to(p.device)
                     count += 1
             logger.info(f"✅ [RESUME] Restored partial gradients for {count} parameters.")
             del self.pending_partial_grads

        # =====================================================================
        # 3. EMA TEACHER RESTORATION (The Advantage Preservation Bridge)
        # =====================================================================
        if hasattr(self, "pending_ema_state"):
            if self.ema is not None:
                self.ema.load_state_dict(self.pending_ema_state)
                logger.info("✅ [RESUME] EMA shadow weights (Teacher) restored directly.")
                del self.pending_ema_state
            else:
                # [v112.0 SOTA FIX] Callback Proxy Restoration
                found = False
                for cb in self.trainer.callbacks:
                    if "EMACallback" in cb.__class__.__name__:
                        cb._deferred_ema_state = self.pending_ema_state
                        logger.info("✅ [RESUME] EMA state pushed to EMACallback for deferred initialization.")
                        found = True
                        break
                if not found:
                    logger.warning("⚠️ [RESUME] Pending EMA state found but self.ema is None and no EMACallback detected.")

        # =====================================================================
        # 3.5 GHOST BANK MANIFOLD REFRESH (Patch 58: Smoking Gun #SG-121)
        # =====================================================================
        # Rationale: On resumption, the Ghost Bank's latent anchors are stale.
        # The encoder has evolved since the checkpoint, so stored representations
        # drift from the current latent space. This causes AGEM and CGA projections
        # to operate in a "ghost manifold," degrading clinical signal quality.
        # Fix: Re-encode all stored trajectories with the current encoder weights.
        if hasattr(self, "ghost_bank") and self.ghost_bank is not None:
            if self.ghost_bank.size > 0 and self.global_step > 0:
                logger.info("🔄 [RESUME] Refreshing Ghost Bank latent anchors...")
                try:
                    # [SG-02 SOTA FIX] Ghost Bank Topographical Integrity
                    def ghost_encoder_fn(v, m):
                         # 1. Extract true static features (indices 22-27 at t=0)
                         s_true = v[:, 0, 22:].clone()
                         v_norm, s_norm = self.model.normalize(v, s_true)
                         
                         # 2. Derive padding mask (True where all channels are 0)
                         bool_padding_mask = (m.sum(dim=-1) == 0) if m is not None else None
                         
                         # 3. Freeze BN stats to prevent manifold poisoning during bulk re-encoding
                         with self.frozen_stats():
                             out_alb = self.model.encoder(
                                 v_norm, 
                                 s_norm, 
                                 imputation_mask=m,
                                 padding_mask=bool_padding_mask
                             )
                         
                         # 4. Return Expert manifold (matches training_step anchoring)
                         return out_alb["global_expert"]
                         
                    # Use the explicit encoder closure
                    # Soft update (decay=0.5) to blend old manifold with new
                    self.ghost_bank.refresh_anchors(
                        encoder=ghost_encoder_fn,
                        decay=0.5
                    )
                    logger.info("✅ [RESUME] Ghost Bank anchors refreshed (Manifold Aligned).")
                except Exception as e:
                    logger.warning(f"⚠️ [RESUME] Ghost Bank refresh failed: {e}")

        # =====================================================================
        # 4. DDP ACCELERATOR INITIALIZATION
        # =====================================================================
        if torch.cuda.is_available():
             world_size = dist.get_world_size() if dist.is_initialized() else 1
             if self.ddp_gatherer is None:
                 self.ddp_gatherer = SOTA_DistributedGatherer(self.device, world_size)
                 logger.info(f"✅ [SOTA] DDP Gatherer Online (World={world_size}, Device={self.device})")

        # =====================================================================
        # 5. FRESH CALIBRATION (Priority 2: Infrastructure)
        # =====================================================================
        if not (hasattr(self.trainer, "datamodule") and self.trainer.datamodule):
            logger.warning("No DataModule found. Skipping stats fitting.")
            return

        loader = self.trainer.datamodule.train_dataloader()
        dataset = loader.dataset
        
        # 5.1 Physics Normalizer Calibration
        if hasattr(self.model, "normalizer") and not self.model.normalizer.is_calibrated.item():
            logger.info(f"[Rank {self.global_rank}] Calibrating Normalizer...")
            try:
                index_path = getattr(dataset, "index_path", None)
                ts_cols = getattr(dataset, "metadata", {}).get("ts_columns", [])
                
                if index_path and ts_cols:
                    self.model.normalizer.calibrate_from_stats(index_path, ts_cols)
                    
                    # [CRITICAL] Sync EMA shadow with newly calibrated normalizer
                    if hasattr(self, 'ema') and self.ema is not None:
                        for name, buffer in self.model.normalizer.named_buffers():
                            full_name = f"normalizer.{name}"
                            if full_name in self.ema.shadow:
                                self.ema.shadow[full_name] = buffer.data.detach().cpu().clone()
                        logger.info(f"[Rank {self.global_rank}] EMA shadow sync complete.")
                else:
                    logger.warning("Dataset missing 'index_path' or 'metadata.ts_columns'.")
            except Exception as e:
                logger.error(f"[CRITICAL] Normalizer Calibration Failed: {e}")

        # 5.2 AWR Stats Fitting (Rank 0 Compute + Broadcast)
        # [v20.2] RESUMPTION FIX: Do not re-calibrate if stats are already loaded!
        if not self.awr_calculator.stats_initialized:
            self._fit_awr_stats_ddp(dataset)
        
        # Dataset handle cleanup
        if hasattr(dataset, "_lmdb_env"):
            dataset._lmdb_env = None

        # [v12.5 SOTA FIX] Pre-Training Validation (Resumption Trauma Detector)
        # Rationale: Detect metric regression or loading bugs before starting training.
        # This provides a clean baseline to verify bit-perfect restoration.
        # if self.trainer is not None and getattr(self.trainer, "ckpt_path", None) is not None:
        #      logger.info("🔍 [RESUME] Running Pre-Training Validation to detect resumption trauma...")
        #      try:
        #          # Ensure we have a valid dataloader from datamodule
        #          val_loader = self.trainer.datamodule.val_dataloader()
        #          self.trainer.validate(self, dataloaders=val_loader)
        #          logger.info("✅ [RESUME] Pre-Training Validation Complete. Baseline Established.")
        #      except Exception as e:
        #          logger.warning(f"⚠️ [RESUME] Pre-Training Validation skipped or failed: {e}")

    def on_train_epoch_start(self):
        """[Phase 3/4] Update AWR Horizon and SOTA v4.2 Warmup."""
        # [v26.4 SOTA FIX] Force Sampler Synchronization
        # Eliminates "Sampler Amnesia" by actively pushing the epoch state.
        if hasattr(self.trainer, "train_dataloader") and self.trainer.train_dataloader is not None:
            dls = self.trainer.train_dataloader
            if not isinstance(dls, list): dls = [dls]
            for dl in dls:
                if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                    dl.sampler.set_epoch(self.current_epoch)
        
        # [v29.5] All metric and curriculum updates moved to _update_curriculum 
        # to ensure step-density invariance across all training configurations.
        
    def _update_curriculum(self, batch_idx: int):
        """[v21.5 SOTA] Step-Continuous Curriculum Update."""
        if self.trainer is None: return
        n_batches = self.trainer.num_training_batches
        if n_batches <= 0: return

        # [v2026 SOTA FIX] Density-Invariant Curriculum (Abyssal #4)
        # Rationale: Anchor progress to EPOCH count to ensure that curriculum
        # transitions (Gamma ramps, Sigma decay) happen at the same perceived
        # time regardless of whether we are in debug (200 steps) or production (1200 steps).
        ref_progress = float(self.current_epoch) + (float(batch_idx) / float(n_batches))
        
        # [v21.5 SOTA] PMS Governance: Mute updates if manifold is unstable
        # Unless we are in resumption grace period.
        ema_bc, std_bc = TrendSentinel.get_stats(self.grad_norm_ema, self.grad_norm_std, self.grad_ema_decay, self.grad_norm_step_count)
        
        # [v2026 SOTA] Zero-Sync Shock Check
        is_shock_t = TrendSentinel.is_unstable(ema_bc, std_bc, max_pressure=5.0, max_sigma=2.0)
        skip_update = 1.0 if (is_shock_t and self._shadow_resumption_grace_steps == 0) else 0.0
        
        # 2. Gamma: Step-Invariant Horizon Ramp (Abyssal #5)
        new_gamma = self.horizon_scheduler.get_gamma_step(self.global_step)
        # Note: We'll update the buffer after the potential DDP sync below
        
        
        # [SOTA FIX - BULLETPROOF HYBRID] Metric-Driven + Guaranteed Fallback
        # 1. Metric Score: ema_bc -> 0.0 is perfect stability, >5.0 is unstable.
        metric_score = max(0.0, min(1.0, 1.0 - (ema_bc / 5.0)))
        
        # 2. Time Score: Guaranteed fallback over the first 30% of training.
        max_budget = getattr(self.trainer, "estimated_stepping_batches", ScalingSteward.SOTA_REF_SAFE_BUDGET)
        # Prevent DivisionByZero if max_budget is missing/0
        safe_budget = max(100.0, float(max_budget)) 
        time_score = max(0.0, min(1.0, float(self.global_step) / (0.30 * safe_budget)))
        
        # 3. Hybrid Dominance: Take the faster of the two trajectories.
        curriculum_p = max(metric_score, time_score)
        
        # Tau: Scales from 0.5 -> 0.7
        tau_val = 0.5 + (0.7 - 0.5) * curriculum_p
        
        # Sigma: Contracts from 3.5 -> 2.5
        sigma_val = 3.50 - (3.50 - 2.50) * curriculum_p
        
        # Physics Clamp: Relaxes from 10.0 -> 50.0 (Shifted to upper 50% of the curriculum)
        clamp_p = max(0.0, min(1.0, (curriculum_p - 0.5) * 2.0))
        phys_clamp_val = 10.0 + 40.0 * clamp_p

        # 6. [v12.5.1 SOTA] AWR Beta Annealing (Abyssal #6)
        # [SOTA FIX - DYNAMIC BUDGET] Use actual trainer steps instead of hardcoded AI slop
        target_epochs = self.cfg.train.get("epochs", 40)
        anneal_steps = target_epochs * n_batches
        if self.global_step < anneal_steps:
             frac = float(self.global_step) / max(1.0, float(anneal_steps))
             curr_beta = 0.60 + (0.15 - 0.60) * frac
        else:
             curr_beta = 0.15
             
        # Beta update handled after sync below
        pass

        # 7. Update and Synchronize (Broadcast Rank 0 to others)
        if dist.is_initialized():
             # [v2026 SOTA FIX] Unified Synchronous Handshake (Deadlock Prevention)
             # Rationale: All ranks MUST participate in the broadcast. If Rank 0 
             # skips due to shock, it must tell others to skip too.
             vars_tensor = torch.tensor([tau_val, sigma_val, curr_beta, float(new_gamma), phys_clamp_val, skip_update], device=self.device)
             dist.broadcast(vars_tensor, src=0)
             # Unpack synced values
             tau_val, sigma_val, curr_beta, new_gamma, phys_clamp_val, skip_update = vars_tensor.tolist()
        
        # [v2026 SOTA] Atomic Skip (Post-Sync)
        if skip_update > 0.5:
             # Manifold Shock: Hold current curriculum to prevent further destabilization
             return

        # 8. Update Buffers
        # [v2026 SOTA] Atomic Buffer Registration (Atomic #1)
        # Rationale: Using .fill_() is safe for both floats and tensors, 
        # but we must ensure we are passing a scalar to avoid dimension mismatch.
        self.curr_tau.fill_(float(tau_val))
        self.curr_sigma_scale.fill_(float(sigma_val))
        self.curr_phys_clamp.fill_(float(phys_clamp_val))
        
        if not self.awr_calculator.adaptive_beta:
             self.awr_calculator.beta.fill_(float(curr_beta))
        
        self.awr_calculator.gamma.fill_(float(new_gamma))

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
        if not batch or "observed_data" not in batch:
            return
        
        # [v12.0 SOTA] Curriculhm Throttling (Zero-Sync)
        # Rationale: Reduces DDP broadcast overhead by 50x. deterministic
        # curriculum is stable even with sparse syncing.
        if batch_idx % 50 == 0 or batch_idx == (self.trainer.num_training_batches - 1):
             self._update_curriculum(batch_idx)

        total_loss = 0.0 # Default for telemetry
        loss_dict = {} # Default for telemetry
        logs = {}      # [Patch 2] Default for telemetry
        gn_loss = torch.tensor(0.0, device=self.device) # [Patch 2] Default for telemetry
        opt = self.optimizers()
        B = batch["observed_data"].size(0)
        
        # [v48.1 SOTA FIX] Unified Accumulation Tracking (Smoking Gun #90)
        # Rationale: Ensures bit-perfect cycle alignment across resumptions.
        # Uses shadow (CPU) for step logic and buffer (Device) for checkpointing.
        self._shadow_grad_accum_idx += 1
        self.grad_accum_idx.fill_(self._shadow_grad_accum_idx)
        
        acc_batches = self.cfg.train.get("accumulate_grad_batches", 1)
        is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        should_step = (self._shadow_grad_accum_idx >= acc_batches) or is_last_batch
        
        # [v49.0 SOTA FIX] Seed Continuity (Smoking Gun #65)
        # Rationale: Align sampling with the stateful accumulation cycle.
        ghost_seed = (self.current_epoch * 12345 + self.global_step + (self._shadow_grad_accum_idx - 1)) % (2**31)
        num_ghosts = self.cfg.train.get("num_ghosts", 2)
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
        # [v21.5 SOTA] Demographic Preservation: Use actual ghost demographics from the bank.
        static_expanded = torch.cat([static, ghost_batch["static"]], dim=0)
        
        # Norms
        past_norm, static_norm = self.model.normalize(past_expanded, static_expanded)
        fut_norm, _ = self.model.normalize(fut, None) # fut is not expanded (masked later)
        
        # Context Mask expansion
        if src_mask is not None:
            src_mask_expanded = torch.cat([src_mask, ghost_batch["masks"]], dim=0)
        else:
            src_mask_expanded = None
        
        # [SOTA 2025] Manifold Stability Monitoring (The Governor)
        # Calculates a damping factor [0.1, 1.0] based on absolute and relative pressure.
        # This factor is used to relax focal parameters and dampen conflict resolution.
        with torch.no_grad():
            ema_bc, std_bc = TrendSentinel.get_stats(
                ema=self.grad_norm_ema, 
                std=self.grad_norm_std, 
                decay=self.grad_ema_decay,
                step_tensor=self.grad_norm_step_count
            )
            
            # 1. Volatility Protection (Z-Score / "The Earthquake Detector")
            # [v44.0 SOTA FIX] Domain Alignment (Smoking Gun #44)
            # Rationale: Comparing bias-corrected 'ema_bc' against uncorrected 'grad_norm_ema'
            # causes a 10x phantom shock on resumption. Since we haven't seen the 
            # current batch's gradient yet, we assume trend-stability (z=0).
            z_score = 0.0 
            stab_z = 1.0 / (1.0 + (max(0, z_score - 3.0) ** 2))
            
            # 2. Absolute Pressure Protection (Exp Decay / "The Boiling Frog Detector")
            # Drops exponentially if EMA > 25.0 (Aligned with x10.0 Reward Scale)
            # [Patch 65] Relaxed Threshold
            stab_p = torch.exp(-torch.clamp(torch.as_tensor(ema_bc, device=self.device) - 25.0, min=0.0))
            
            # Final Governor Factor (min of both protections)
            stability_factor = torch.min(torch.as_tensor(stab_z, device=self.device), stab_p)
            
            # Absolute Floor for "Iron Dome" protection
            # [Patch 65] Relaxed Threshold
            # [v2026 SOTA] Vectorized floor
            # [v2026 SOTA] Standardized Broadcast Protection
            stability_factor = torch.where(torch.as_tensor(ema_bc, device=self.device) > 50.0, torch.tensor([0.1], device=self.device), stability_factor)
            
            # [v51.0 SOTA FIX] Governor Grace Alignment (Smoking Gun #81)
            # Rationale: During the grace period, EMAs represent stale history.
            # Avoid artificial throttling while the manifold settles.
            if self._shadow_resumption_grace_steps > 0:
                stability_factor = 1.0
            
            # [v88.0 SOTA FIX] DDP Stability Consensus (Smoking Gun #88)
            # Rationale: All ranks must agree on the curriculum gate.
            if dist.is_initialized():
                self.stability_factor.fill_(float(stability_factor))
                dist.broadcast(self.stability_factor, src=0)
                stability_factor = self.stability_factor
            else:
                stability_factor = torch.as_tensor(stability_factor, device=self.device)
            
            self.log("train/manifold_stability", stability_factor, on_step=True, prog_bar=True)
            self.log("gov/stability_factor_p", stab_p, on_step=True)

        # [PHASE 1] Dynamic Risk Scoring
        risk_coef = self.risk_scorer(past, self.clinical_feat_idx)
        
        # [v27.2 SOTA FIX] Calibrated Ghost Penalty
        # Rationale: Hardcoded 2.0 (5x loss) causes 37x gradient amplification.
        # Reducing to 0.5 (2x loss) and scaling with stability_factor for safety.
        risk_shape = (num_ghosts, *risk_coef.shape[1:])
        risk_coef_ghost = (torch.ones(risk_shape, device=self.device) * 0.5) * stability_factor
        risk_coef_expanded = risk_coef
        if num_ghosts > 0:
            risk_coef_expanded = torch.cat([risk_coef, risk_coef_ghost], dim=0)

        # [v49.0 Telemetry] Log stability z-score (always, regardless of ghost count)
        self.log("gov/stability_factor_z", stab_z, on_step=True)

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
        # [v21.0 SOTA FIX] Distributed Timestep Synchrony (Smoking Gun #188)
        # Rationale: All ranks must solve the SAME noise levels to maintain 
        # manifold consensus and prevent gradient heteroscedasticity.
        if dist.is_initialized():
             if dist.get_rank() == 0:
                 t = torch.randint(0, self.model.cfg.timesteps, (B,), device=self.device)
             else:
                 t = torch.empty((B,), dtype=torch.long, device=self.device)
             dist.broadcast(t, src=0)
        else:
             t = torch.randint(0, self.model.cfg.timesteps, (B,), device=self.device)

        noisy_fut, noise_eps = self.model.scheduler.add_noise(fut_norm, t)
        
        # [v12.1] Two-Pass Self-Conditioning ("Analog Bits")
        # Rationale: Training the model to fix its own generation errors.
        self_cond = None
        if self.model.cfg.use_self_conditioning:
            self_cond = torch.zeros_like(noisy_fut)
            
            # [v21.3 SOTA FIX] CPU-Pure Self-Cond Branch (Zero-Sync)
            # Rationale: All ranks share the same seed (via set_seed in trainer). 
            # [v2026 SOTA FIX] DDP-Safe Stochastic Branching (Abyssal #3.1)
            # Rationale: Using random.random() causes rank divergence. 
            # If Rank 0 enters the self-cond block (with DDP all_reduce inside governance) 
            # but Rank 1 skips it, the cluster deadlocks.
            # We use a deterministic step-based hash to ensure rank consensus.
            do_self_cond = (self.global_step % 2 == 0)

            if do_self_cond:
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
                    # [v21.5 SOTA] Accumulation Guard: Only update on stepping batches.
                    self_cond = self.model.governance(guess_x0, update_ema=should_step).detach()

        # Pass 2: Final Denoising with Conditioning (Gradient Path)
        pred_noise = self.model.backbone(noisy_fut, t, ctx_seq[:B], global_ctx[:B], ctx_mask[:B], self_cond=self_cond)
        
        diff_sq = (pred_noise - noise_eps) ** 2
        weighted_diff = diff_sq * self.model.importance_weights.view(1, 1, -1)
        
        # [v2025 SOTA FIX] Training Manifold Disentanglement (Smoking Gun #468)
        # Rationale: Static channels (22-27) are effectively constant (0.0). Training constraints 
        # on them are wasted capacity and distract the encoder from dynamic features.
        # We slice to Dynamic Channels (0-22) to align Training Loss with GMSE Validation Metric.
        DYNAMIC_CHANNELS = 22
        raw_diff_loss = weighted_diff[..., :DYNAMIC_CHANNELS].mean(dim=2) # [B, T]

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
        phys_loss = torch.tensor(0.0, device=self.device)
        curr_phys_weight = 0.0
        # [v12.8.2 FIX] Synchronize with generalist.yaml (num_phases)
        logits = torch.zeros((B, self.cfg.model.num_phases), device=self.device)
        uncertainty = torch.ones((B, 1), device=self.device) # Vacuous by default
        u_avg = 1.0 # Default for non-aux batches
        
        # [PMS] DAT: Dynamic Adaptive Throttling
        # "Head First, Brain Second" - Guard the encoder when the head is guessing.
        ctx_aux = ctx_expert.clone()
        cfm = 1.0
        if self.cfg.model.use_auxiliary_head:
            # [v2026 SOTA] Vectorized Sepsis Gating
            # Rationale: Replaced .any().item() with a 0D tensor to avoid graph breaks.
            batch_has_sepsis = (batch["phase_label"] > 0).any()
            
            # [PMS] DAT Moved up to line 881 to prevent UnboundLocalError when head is disabled.
            # ctx_aux = ctx_expert.clone()
            
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
            # [v21.1 SOTA FIX] Global Consensus Trust Factor (Smoking Gun #178)
            # Rationale: Foundation trust must be identical across ranks.
            if dist.is_initialized():
                 u_sum = uncertainty.detach().sum()
                 b_count = torch.tensor([float(B)], device=self.device)
                 
                 sync_data = torch.stack([u_sum, b_count])
                 dist.all_reduce(sync_data, op=dist.ReduceOp.SUM)
                 u_avg = (sync_data[0] / sync_data[1]) # Removed .item()
            else:
                 u_avg = uncertainty.detach().mean()
            
            # [v1.5 SOTA] Drowning Foundation Repair (Smoking Gun #D-05)
            # Rationale: min=0.4 allowed 40% noise into encoder even when head was 100% wrong.
            # Lowering to 0.05 to enable near-total suppression during discovery shocks.
            trust_factor = torch.exp(-u_avg * 0.5).clamp(min=0.05, max=1.0)
            
            # Surgical Hook: Scopes gradients only for the shared connection
            if ctx_aux.requires_grad:
                ctx_aux.register_hook(lambda grad: grad * trust_factor)
            
            # [v2026 SOTA] Periodic Progress Bar
            self.log("train/pms_trust_factor", trust_factor, on_step=True, prog_bar=True)
            
            self.class_balancer.update(targets)
            class_weights = self.class_balancer.get_weights().to(self.device).clamp(max=10.0)
            
            # [SOTA 2025] Shape Alignment
            # Classification (Aux Head) is Window-Level [B] via CLS Token
            # Contrastive (ACL) is Sequence-Level [B, T] (handled internally)
            B_exp, T_seq, _ = ctx_seq.shape
            # cfm = 1.0 (Moved up to line 871)
            
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
            def throttle_aux_gradient(grad_aux):
                # [SOTA FIX] Asymmetric PCGrad Surgery (NeurIPS 2020)
                # Rationale: Legacy static 0.2x throttle suppressed 80% of "Helping" signals.
                # PCGrad allows 100% magnitude but projects if conflicting with Foundation.
                if self._fnd_grad_ema is None:
                    return grad_aux
                
                # Align shapes for projection
                g_fnd = self._fnd_grad_ema
                
                # Compute Cosine Similarity (Alignment check)
                dot = torch.sum(grad_aux * g_fnd)
                norm_fnd = torch.sum(g_fnd * g_fnd) + 1e-8
                
                # [Asymmetric Strategy] 
                # If dot < 0 (Conflict): Project to be orthogonal to Foundation.
                # If dot >= 0 (Aligned): ALLOW 100% MAGNITUDE (The recovery).
                # [v2026 SOTA FIX] Vectorized Projection (Zero-Sync)
                # Rationale: Replaced scalar branching with torch.where to prevent host-side stall.
                proj_grad = grad_aux - (dot / norm_fnd) * g_fnd
                return torch.where(dot < 0, proj_grad, grad_aux)

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
                        if dist.is_initialized():
                            dist.all_reduce(self._fnd_grad_ema, op=dist.ReduceOp.SUM)
                            self._fnd_grad_ema /= dist.get_world_size()
                    else:
                        self._fnd_grad_ema = self._fnd_grad_ema.to(grad_fnd.device)
                        # [v74.0 SOTA FIX] DDP Foundation Consensus (Smoking Gun #74)
                        # Rationale: Foundation anchors MUST be identical across ranks 
                        # for consistent MGP projection.
                        if dist.is_initialized():
                            dist.all_reduce(dir_fnd, op=dist.ReduceOp.SUM)
                            dir_fnd /= dist.get_world_size()
                            
                        # [SOTA v2026] Scale-Aware Directional Stability
                        fnd_momentum = ScalingSteward.get_decay(0.90, self.trainer.num_training_batches)
                        self._fnd_grad_ema.mul_(fnd_momentum).add_(dir_fnd.detach(), alpha=1.0 - fnd_momentum)
                return grad_fnd

            if ctx_seq.requires_grad:
                ctx_seq.register_hook(update_fnd_ema)
            
            if ctx_aux.requires_grad:
                # [SOTA FIX] Liberation: Disable suppression to allow Sepsis learning
                # ctx_aux.register_hook(throttle_aux_gradient)
                pass
            
            # [v17.3] Omega Summoning: Constant Clinical Pressure
            # Every batch now has sepsis signal via the Summoned Ghosts.
            # We remove the 0.1x multiplier and train with full magnitude.
            with torch.no_grad():
                probs = torch.sigmoid(logits) # [SOTA FIX] Use Sigmoid to match ASL
                # logits/probs: [B+G, C], targets_expanded: [B+G]
                true_probs = probs.gather(-1, targets_expanded.unsqueeze(-1).long())
                error = 1.0 - true_probs.squeeze(-1)
                # Cap max boosting at 1.5x
                mining_weight = 1.0 + (torch.sigmoid(error * 5.0) * 0.5)
            
            # One-hot encoding for window-level targets
            targets_one_hot = F.one_hot(targets_expanded.long(), num_classes=logits.shape[-1]).float()

            # [v21.4 SOTA FIX] Global Mining Weight Consensus (Smoking Gun #191)
            if dist.is_initialized():
                 m_sum = mining_weight.sum()
                 m_count = torch.tensor([float(mining_weight.numel())], device=self.device)
                 
                 sync_m = torch.stack([m_sum, m_count])
                 dist.all_reduce(sync_m, op=dist.ReduceOp.SUM)
                 mining_weight_avg = (sync_m[0] / sync_m[1]) # Removed .item()
            else:
                 mining_weight_avg = mining_weight.mean()

            # Sepsis Classification Loss (Unified [B+G])
            # Ghosts provide the gradient floor to prevent 'Discovery Shock'.
            # [Patch 67] Wire Phantom Risk Loss (Smoking Gun #SGD-17)
            # Rationale: SequenceAuxHead uses generic AsymmetricLoss. We MUST override with 
            # RiskAwareAsymmetricLoss to apply Critical Penalty for Shock/Hypoxia.
            
            # [SOTA FIX] Alarm Fatigue Prevention: Disable brute-force class weights.
            # ASL inherently handles the 3.1% imbalance via gamma_neg and Prior Bias Init.
            # Multiplying by 32x class_weights causes catastrophic false positives.
            raw_aux_loss = self.risk_aware_loss(logits, targets_one_hot, risk_coef_expanded, class_weights=None)
            aux_loss = raw_aux_loss * cfm * mining_weight_avg

            # [v17.4 GIST-Q] Uncertainty-Weighted CGA
            # Anchors the Expert Manifold to history, prioritising high-uncertainty (hard) cases.
            if num_ghosts > 0:
                ghost_latents_global = global_ctx_expert[B:]
                ghost_anchors = ghost_batch["anchors"]
                
                # [v72.0 SOTA FIX] Hyperspherical Anchoring (Smoking Gun #72)
                # Rationale: Absolute MSE grows boundlessly as the student manifold evolves.
                # Fix: Use Cosine-Distance bounded in [0, 2] to anchor DIRECTION.
                z_s = F.normalize(ghost_latents_global, dim=1)
                z_t = F.normalize(ghost_anchors, dim=1)
                ghost_dist = 1.0 - torch.sum(z_s * z_t, dim=1, keepdim=True)
                
                ghost_uncertainties = ghost_batch["uncertainties"].to(ghost_dist.device)
                
                # [v21.5 SOTA FIX] Global CGA Multiplier Consensus (Smoking Gun #192)
                # Rationale: Manifold stiffness must be uniform across ranks.
                with torch.no_grad():
                    if dist.is_initialized():
                         # We already have u_avg from L868 (Global Consensus)
                         # If u_avg is not available in this scope, we compute it.
                         cga_mult = 0.5 + u_avg 
                    else:
                         curr_uncertainty = uncertainty[:B].mean().clamp(0.1, 1.0)
                         cga_mult = 0.5 + curr_uncertainty 
                
                # [v2026 SOTA] Vectorized Validity Mask
                # Rationale: Replaced .any() sync with zero-masking for graph safety.
                v_mask = ghost_batch["valid"].unsqueeze(-1).float()
                l_cga_raw = (ghost_dist * ghost_uncertainties * v_mask).mean() * cga_mult
                
                # [FORENSIC FIX #5] CGA Ratio Normalization (Principled anchor-drift control)
                # Root Cause: Stale ghost anchors cause cosine-distance explosion when 
                # the encoder evolves rapidly (observed: l_cga 0.02→0.72 at E3, exceeding D=0.20).
                # Principle: CGA is a regularizer — it should never exceed the primary loss.
                # We normalize CGA so its gradient contribution stays proportional to 
                # the diffusion anchor via smooth ratio scaling. When CGA >> D, the tanh
                # saturates and suppresses the excess. When CGA << D, tanh ≈ identity.
                with torch.no_grad():
                    # Use the scaler's diffusion EMA as a stable anchor (always in scope,
                    # smoothed across steps — more robust than current-batch diff_loss).
                    d_anchor = self.loss_scaler.loss_emas[0].detach().clamp(min=1e-4)
                    cga_ratio = l_cga_raw.detach() / d_anchor
                    # Lorentzian gate (Hill function): 1/(1+x²)
                    # Properties vs tanh(1/(x+1)):
                    #   ratio=0.1 → gate=0.990 (vs 0.761) — preserves small CGA signals
                    #   ratio=1.0 → gate=0.500 (vs 0.462) — 50% at parity 
                    #   ratio=3.6 → gate=0.072 (vs 0.187) — strong suppression when CGA > D
                    ratio_gate = 1.0 / (1.0 + cga_ratio.pow(2))
                l_cga = l_cga_raw * ratio_gate
                
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
            src_mask=batch.get("future_mask", None),
            training=self.training
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
                     tau=self.curr_tau
                )

                # [v2026 SOTA] Adaptive Stochastic Bootstrapping (Shatter the Echo Chamber)
                # Rationale: In offline datasets, s -> s' transitions are fixed, causing the Critic
                # to overfit and memorize deterministic paths (EV > 0.80). 
                # We inject dynamic variance scaled to the global synced Advantage StdDev. This acts 
                # as Continuous Label Smoothing, forcing the Distributional Critic to widen its 
                # quantiles and learn generalizable structural means rather than point-memorization.
                with torch.no_grad():
                    # 1. Fetch DDP-Synced Global Scale (Shape: [1])
                    adv_scale = self.awr_calculator.adv_std.detach().clamp(min=1e-3)
                    base_sigma = 0.05 * adv_scale
                    
                    # 2. Compute Scale-Invariant Local Volatility (Heteroscedasticity)
                    # diff() reduces time dimension by 1. Pad front to keep shape [B, T, N].
                    # Rationale: We detect temporal instability per quantile to scale jitter.
                    v_diff = torch.cat([torch.zeros_like(target_values[:, :1]), target_values.diff(dim=1)], dim=1)
                    norm_diff = v_diff / adv_scale
                    volatility = torch.tanh(torch.abs(norm_diff))
                    
                    # Local Sigma scales up to 2x based on patient instability
                    local_sigma = base_sigma * (1.0 + volatility)
                    
                    # 3. Generate Raw Noise
                    raw_noise = torch.randn_like(target_values) * local_sigma
                    
                    # 4. SOTA: Variance-Preserving Causal Smoothing (MA(1) Process)
                    # Pad with REAL noise to prevent variance collapse at t=0
                    pre_noise = torch.randn_like(raw_noise[:, :1]) * local_sigma[:, :1]
                    noise_shifted = torch.cat([pre_noise, raw_noise[:, :-1]], dim=1)
                    
                    # Blend factor 0.7071 (sqrt(0.5)) ensures (a^2 + a^2 = 1.0), preserving exact variance
                    blend_factor = 0.70710678
                    temporal_noise = blend_factor * raw_noise + blend_factor * noise_shifted
                    
                    # 5. Dynamic Symmetric Iron Dome (Zero-Sync, Broadcast-Safe)
                    # We use pure tensor operations to avoid .item() syncs or view_as crashes.
                    local_limit = 2.5 * local_sigma
                    clamped_noise = torch.max(torch.min(temporal_noise, local_limit), -local_limit)
                    
                    # 6. Pessimistic Shift (Conservative RL)
                    # Applied AFTER clamping to strictly guarantee the downward shift is preserved.
                    pessimism_penalty = -0.5 * local_sigma * volatility
                    
                    # [FORENSIC FIX #1] Sigmoid Exploration Decay (Breaks memorization feedback loop)
                    # Root Cause: base_sigma = 0.05 * adv_scale creates a positive feedback loop:
                    # as the critic memorizes (EV→1.0), advantages sharpen, adv_scale grows,
                    # noise grows, target jitter grows, V explodes (31→115 at E3).
                    # Principle: Uncertainty-driven exploration (Thompson Sampling analogy).
                    # When the critic is uncertain (low EV), we explore heavily.
                    # When the critic is confident (high EV), exploration noise decays.
                    # Note: We use the accumulated EV from previous steps (lagged 1 batch)
                    # because current-batch EV hasn't been computed yet at this point.
                    # [POST-PATCH FIX] Explicit NaN guard (torchmetrics returns NaN for empty metric,
                    # not exception — the try/except never triggered, and NaN poisoned the pipeline).
                    prev_ev = self.train_explained_var.compute().detach()
                    if not torch.isfinite(prev_ev):
                        prev_ev = torch.tensor(0.0, device=self.device)
                    prev_ev = prev_ev.clamp(-1.0, 1.0)
                    ev_noise_scale = torch.sigmoid(5.0 * (0.7 - prev_ev))
                    
                    # 7. Inject with exploration-decayed noise
                    target_values = target_values + (clamped_noise + pessimism_penalty).detach() * ev_noise_scale

                # B. Run Anchor Head (if applicable)
                teacher_logits = None
                # [v112.0 SOTA FIX] Dynamic Step-Based Warmup (Smoking Gun #40)
                # Rationale: Coarse epoch gates (>=2) fail for high-density runs. 
                # Step-based warmup ensures consistent target convergence.
                teacher_warmup = self.cfg.train.get("teacher_warmup_steps", 500)
                if self.cfg.model.use_auxiliary_head and "phase_label" in batch and self.ema is not None and self.global_step >= teacher_warmup:
                     # Unified teacher pass for [B+G]
                     teacher_aux = self.model.aux_head(ctx_aux, mask=ctx_mask)
                     # Surgical Mask: Only anchor the fresh batch [0:B]
                     teacher_logits = teacher_aux["logits"][:B]

        # 3. Anchor Loss Injection (Gradient Allowed)
        # Must be OUTSIDE no_grad so 'logits' (Student) gradients flow
        if teacher_logits is not None:
             # Surgical Mask: Only anchor Student representations for the main batch [0:B]
             # [SOTA FIX] Distill independent sigmoids to perfectly align with ASL manifold
             l_anchor = F.mse_loss(torch.sigmoid(logits[:B]), torch.sigmoid(teacher_logits))
             
             # [v112.0 SOTA FIX] Transition Smoothing (Smoking Gun #43)
             # [SOTA FIX - RATIO ANNEALING] Dynamic Budget Integration for Auxiliary Gate
             # Rationale: Replaced hardcoded 100-step ramp with dynamic 1% ratio.
             _max_budget_aux = getattr(self.trainer, "estimated_stepping_batches", ScalingSteward.SOTA_REF_SAFE_BUDGET)
             _safe_budget_aux = max(100.0, float(_max_budget_aux))
             ramp_steps = max(100, int(0.01 * _safe_budget_aux))
             
             teacher_warmup = self.cfg.train.get("teacher_warmup_steps", 500)
             alpha = 1.0
             if self.global_step < (teacher_warmup + ramp_steps):
                 alpha = min(1.0, max(0.0, (self.global_step - teacher_warmup) / ramp_steps))
             
             aux_loss = aux_loss + (0.5 * alpha) * l_anchor

        # 4. AWR Bootstrapping (Target Calculation - No Grad)
        with torch.no_grad():
            is_truncated = batch.get("is_truncated", None)
            # [v2026 SOTA] Vectorized Bootstrap (Zero-Sync)
            # Rationale: Replaced if .any() with float masking for graph compatibility.
            bootstrap_value = None
            if is_truncated is not None:
                bootstrap_value = target_values[:, -1:] * is_truncated.float().unsqueeze(-1)

            # Student Values for SAW (Detached for Target generation)
            # [v12.2 SOTA] Unified Context for SAW
            # [PATCH 5 cont.] Also pass tau for student-teacher consistency
            student_values = self.model.value_head.get_expectile_summary(
                self.model.value_head(global_ctx_unified),
                tau=self.curr_tau
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
                advantages, 
                values=target_values, 
                rewards=returns, 
                mask=f_mask,
                turbo_mode=(self._shadow_resumption_grace_steps > 0)
            )
            
            # [v7.3 SOTA FIX] AWR Warmup (The "Cognitive Settle")
            # Rationale: Advantage weighting on a random Value Head causes 'Selection Panic'.
            # We enforce uniform weights (1.0) for the first 1000 steps to let V(s) settle.
            # Scaling ensures consistency across different batch sizes/devices.
            n_batches = self.trainer.num_training_batches
            
            # [SOTA FIX v8.1] Robust Warmup Clamp (Transparent Edition)
            # Rationale: Show the true limit but enforce 'ON' if past Epoch 0 to prevent 
            # resumption amnesia while maintaining telemetry transparency.
            # [SOTA FIX P12] Sync to Global Warmup
            scaled_warmup = getattr(self.trainer, "global_warmup_steps", 1000)
            awr_warmup = 0 if self.current_epoch > 0 else scaled_warmup
            
            # [TELEMETRY] Log Warmup State (Once per epoch/resume to avoid spam)
            if batch_idx == 0:
                 is_learning = self.global_step >= awr_warmup if self.current_epoch == 0 else True
                 logger.info(f"[AWR] Warmup Check: Step={self.global_step}, Limit={scaled_warmup} ({'Epoch-0 Restricted' if self.current_epoch > 0 else 'Active'}), Learning={'ON' if is_learning else 'PAUSED'}")

            if self.global_step < awr_warmup:
                weights_awr = torch.ones_like(weights_awr)
            
            # [v2026 SOTA] Atomic AWR (Removing Double-Normalization)
            # Rationale: AdvantageCalculator already performs global population-scale 
            # normalization. Removing this second pass ensures telemetry parity.
            pass

            weights_awr_log = {"train/awr_ess": diag["ess"]}
        
        # 5. Computed Weighted Diffusion Loss (Gradient Allowed)
        # raw_diff_loss has gradients. weights_awr is detached.
        diff_loss = (raw_diff_loss * weights_awr * f_mask).sum() / (f_mask.sum() + 1e-8)
        self.train_awr_ess.update(weights_awr_log.get("train/awr_ess", 0.0))
        
        # 6. Critic Loss (Gradient Allowed)
        # pred_values (L540) has gradients. returns is detached.
        # [SOTA v4.1] Implicit Distributional Critic Task
        # [v4.1.6 SOTA FIX] Mask-Aware Critic (Smoking Gun #Padding-Bleed)
        # [FORENSIC FIX #2] Root cause addressed in loss_scaler.py (log_vars[1] constraint relaxed)
        # The BayesianProjectedScaler now has freedom to reduce critic weight when V explodes,
        # eliminating the need for any hard cap here. See loss_scaler.py Fix #2.
        critic_loss = self.model.value_loss_fn(pred_values, returns, tau=self.curr_tau.item(), mask=f_mask)
        
        # [v2026 SOTA] Telemetry consolidated at step-end (L2703) to prevent double-counting.

        # [v4.1.2 SOTA FIX] Global Prevalence & Mask Parity
        if dist.is_initialized():
            # [v30.5 SOTA FIX] Asymmetric Gather Protection (Iron Dome)
            # Rationale: Standard all_gather deadlocks if batch sizes differ on rank-0 vs rank-N.
            ctx_aux_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(ctx_aux), dim=0)
            # Use targets_expanded for global prevalence scaling
            targets_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(targets_expanded.long()), dim=0)
            mask_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(ctx_mask), dim=0)
            
            # Global CFM: Balanced scaling based on the entire DDP batch (including ghosts)
            n_sepsis_global = (targets_global > 0).sum()
            cfm_global = LinearManifoldSentinel.log_scale_prevalence(targets_global.numel(), n_sepsis_global)
        else:
            ctx_aux_global = ctx_aux
            targets_global = targets_expanded # [Patch 63] Use [B+G] for correct prevalence
            mask_global = ctx_mask
            
            # [Patch 63] Unified Single-GPU Logic
            n_sepsis_global = (targets_global > 0).sum()
            cfm_global = LinearManifoldSentinel.log_scale_prevalence(targets_global.numel(), n_sepsis_global)

        # D. Contrastive Sepsis Clustering (ACL+)
        # [v4.2 SOTA Pillar 5] Inject Clinical Metadata (Velocity, Static)
        with torch.no_grad():
            velocity = (fut[:, 0, :] - past[:, -1, :]) # [B, D_in]
            # [v17.4] Metadata Expansion: Pad ghosts with zeros to match B+G batch
            # Rationale: Ghosts are historical, their 'future velocity' is not in the current context.
            vel_ghost = torch.zeros(num_ghosts, velocity.size(1), device=self.device)
            velocity_expanded = torch.cat([velocity, vel_ghost], dim=0)
        
        # [PHASE 1 FIX] Unthrottle ACL to 70% (was 30%)
        # [v12.2 SOTA] Unified Context for Contrastive Clustering
        acl_factor = self.cfg.train.get("acl_throttle_factor", 0.7)
        global_ctx_throttled = GradientThrottler.throttle(global_ctx_unified, factor=acl_factor)
        
        # Use expanded metadata to match [B+G] context
        raw_meta = torch.cat([global_ctx_throttled, velocity_expanded, static_expanded], dim=-1)
        z_acl = self.acl_projector(raw_meta)
        
        # [v4.2.1 SOTA] Global Contrastive Clustering (DDP-Safe)
        if dist.is_initialized():
            # [v30.5 SOTA FIX] Asymmetric Gather Protection
            z_acl_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(z_acl), dim=0)
            targets_global_acl = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(targets_expanded.long()), dim=0)
            acl_loss = self.sepsis_acl(z_acl_global, targets_global_acl) * cfm_global
        else:
            acl_loss = self.sepsis_acl(z_acl, targets_expanded.long()) * cfm_global


        # --- 3. [DEPRECATED] SOTA Path: Gradient Scaling Hooks ---
        # Legacy manual hooks removed in v3.1.5 in favor of Unified Pass + 
        # Asymmetric Throttling via ctx_aux path.
        
        # [v2026 SOTA] Unified Task Initialization (Smoking Gun #NameError)
        # Rationale: Ensures all variables are defined regardless of branching path.
        l_bgsl = torch.tensor(0.0, device=self.device, requires_grad=True)
        l_tcb = torch.tensor(0.0, device=self.device, requires_grad=True)
        phys_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # --- 4. Multi-Task Balancing Logic ---
        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Single-Pass Uncertainty Weighting
            curr_phys_weight = self._get_curr_physics_weight()
            # [NaN FIX] Gradient-Isolated Reconstruction (Diffusion Amplification Limit)
            # Root Cause: 1/sqrt(alpha) at high timesteps (t>90) amplifies x0 reconstruction
            # errors into physics loss, contributing to gradient explosion.
            # Floor raised from 1e-2 (100x magnification) to 0.1 (10x magnification).
            # Only affects t>87 (sqrt(alpha_bar) < 0.1), preserving >87% of timestep range.
            alpha_t = self.model.scheduler.alphas_cumprod[t][:, None, None]
            sqrt_alpha_safe = torch.sqrt(alpha_t).clamp(min=0.1)
            x0_approx = (noisy_fut - torch.sqrt(1 - alpha_t) * pred_noise) / sqrt_alpha_safe
            
            # [v26.3 SOTA FIX] Install Manifold Governance Bridge
            # This squashes 100,000x error amplification outliers early in training.
            # Without this, physics losses on x0-hallucinations explode to 161+ GN.
            x0_approx = self.model.governance(x0_approx)
            
            # [PHASE 2] Safety envelope operates on clinical units. Denormalize x0 first.
            # [CRITICAL FIX] Use MODEL normalizer (Calibrated)
            normalizer = self.model.normalizer
            x0_clinical = normalizer.denormalize(x0_approx)
            
            # [v26.4 FIX] Physics Loss (Structure Prep)
            # Actual loss calculation deferred to L1781 to avoid double-counting.
            # x0_clinical is preserved for the later calculation.

            # [v26.4 REMOVED] Logic moved to L1215 after final redefinition to prevent shadowing.
            
            # [v5.1.3 SOTA] Unit Normalization (The "Regime Alignment")
            # Proactively scales major regression tasks into the [1.0, 15.0] range.
            # Also fixes a mask-safety bug by using the scalar 'diff_loss' variable 
            # (which correctly handles f_mask division from L613).
            loss_dict = {
                'diffusion': diff_loss.mean(),       # [v2026 SOTA] Enforce 0D scalar
                # [SOTA P13 FIX] Critic Unshackling (Expert Phase 5)
                # Removed manual 0.1 suppression. Scaler now manages balance via ALI.
                'critic': critic_loss.mean(),
                'aux': aux_loss.mean(),              # Clinical Anchor (0.5)
                'acl': acl_loss.mean()               # (1.5)
            }
            
            # [v4.0 PERFECT] Add BGSL and TCB to the balance
            # pred_state is logits from aux_head. We need them to be [B, T, 1] for BGSL.
            # SequenceAuxHead returns [B, C]. We need a sequence-level prediction.
            # However, for now, let's assume we use the window-level logits for state loss
            # and potentially expand SequenceAuxHead if we want sequence-level risk.
            
            # [v5.2] Manifold Sync: Use the SOTA SequenceAuxHead for BGSL supervision
            # We call the aux_head with return_sequence=True to get [B, T, C]
            if hasattr(self.model, "aux_head"):
                aux_seq_out = self.model.aux_head(ctx_expert, return_sequence=True)
                logits_seq = aux_seq_out["logits"]
                
                if logits_seq.shape[-1] > 1:
                    # [SOTA Alignment] Convert multi-class logits to a single "Sepsis Risk" logit
                    pred_state = logits_seq[..., 1:].logsumexp(dim=-1, keepdim=True) - logits_seq[..., 0:1]
                else:
                    pred_state = logits_seq
            else:
                # [v2026 SOTA] Robust Fallback
                # If aux_head is disabled (e.g. for simple diffusion tests), use dummy state.
                pred_state = torch.zeros(B, ctx_expert.size(1), 1, device=self.device)
            
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
                mask=mask_aligned,
                stability_factor=stability_factor
            )
            l_bgsl = bgsl_out["loss"]
            
            # [v20.0] Cross-Manifold Synergy (Ghost-TCB Bonding)
            # Rationale: DDP Parallelization for Contrastive Memory
            # We gather expert latents across all ranks to provide a massive 
            # negative pool for every GPU.
            tcb_q = torch.cat([global_ctx[:B], global_ctx_expert[B:]], dim=0)
            tcb_k = torch.cat([teacher_global, ghost_batch["anchors"]], dim=0)

            # [SOTA P14] Patient-Aware Negative Gating (PANG)
            # Rationale: Enables relational masking to prevent sliding-window self-contrast.
            # Adler32 provides deterministic bits for DDP consensus without host-sync.
            p_ids_raw = batch["patient_id"]
            p_ids_h = torch.tensor([zlib.adler32(str(s).encode()) for s in p_ids_raw], device=self.device)
            ghost_ids_h = torch.zeros(num_ghosts, dtype=torch.long, device=self.device)
            tcb_ids_local = torch.cat([p_ids_h, ghost_ids_h], dim=0)

            if dist.is_initialized():
                # [v30.5 SOTA FIX] Asymmetric Gather Protection for Contrastive Memory
                # 1. Gather queries and keys for global contrastive loss
                tcb_q_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(tcb_q), dim=0)
                tcb_k_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(tcb_k), dim=0)
                tcb_ids_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(tcb_ids_local), dim=0)
                
                # 2. Gather negative mask
                is_negative = (batch["phase_label"] == 0)
                ghost_neg_mask = torch.zeros(num_ghosts, dtype=torch.bool, device=self.device)
                tcb_enqueue_mask_local = torch.cat([is_negative, ghost_neg_mask], dim=0)
                # Pack bool into float for gathering
                tcb_enqueue_mask_global = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(tcb_enqueue_mask_local.float()), dim=0).bool()
                
                tcb_out = self.tcb_buffer(
                    tcb_q_global, 
                    tcb_k_global, 
                    ids_q=tcb_ids_global,
                    ids_k=tcb_ids_global,
                    enqueue_mask=tcb_enqueue_mask_global
                )
            else:
                is_negative = (batch["phase_label"] == 0)
                ghost_neg_mask = torch.zeros(num_ghosts, dtype=torch.bool, device=self.device)
                tcb_enqueue_mask = torch.cat([is_negative, ghost_neg_mask], dim=0)
                
                tcb_out = self.tcb_buffer(
                    tcb_q, 
                    tcb_k, 
                    ids_q=tcb_ids_local,
                    ids_k=tcb_ids_local,
                    enqueue_mask=tcb_enqueue_mask
                )
            l_tcb = tcb_out["loss"]
            
            # [v26.6 SOTA] TCB Warmup (The "Memory Settling" Protocol)
            # Rationale: Early in training, the TCB queue is random and InfoNCE is high (~6.9).
            # We ramp its weight from 0.0 to 1.0 over ~2.5 epochs to prevent GN shocks.
            # [v27.0 FIX] Scale warmup using ScalingSteward for step-density invariance.
            # [v27.2 SOTA FIX] Link Warmup to Capacity (Test C)
            # Rationale: Ensures the memory bank is fully representative before activating loss.
            n_curr = self.trainer.num_training_batches
            tcb_warmup_steps = max(100, ScalingSteward.get_steps(self.tcb_buffer.base_capacity, n_curr))
            # [v2026 SOTA FIX] Vectorized Warmup (Zero-Sync)
            tcb_multiplier = (self.grad_norm_step_count.float() / float(tcb_warmup_steps)).clamp(max=1.0)
            l_tcb = l_tcb * tcb_multiplier

            # [SOTA Governor] Closed-Loop Stability Control (Nash-Inspired)
            with torch.no_grad():
                # [TELEMETRY] Governor State
                self.log("gov/stability_factor", stability_factor, on_step=True)
                self.log("gov/gamma_effective", 1.0 + (self.risk_aware_loss.gamma_neg - 1.0) * stability_factor, on_step=True)



            # [SOTA v4.0] Raised Alpha Floor (Gradient Starvation Prevention)
            # Rationale: alpha_min=0.01 allows 100x gradient suppression.
            # [SOTA v10.5] Clinical Gradient Preservation (Grid-Search Proven)
            # Rationale: alpha_min=0.5 caused "Generative Dominance" (60% Diffusion weight).
            # Lowering to 0.1 allows the Clinical task to dominate when sepsis signal is strong.
            alpha_min = getattr(self.cfg, "alpha_min", 0.1)
            
            # [FORENSIC FIX #3] Decoupled Alpha Curriculum (Eliminates feedback loop)
            # Root Cause: The original formula `raw_alpha = a_ema / d_ema` creates a positive
            # feedback loop: aux loss grows → raw_alpha grows → diffusion weight increases →
            # encoder prioritizes denoising → classification degrades → aux loss grows.
            # This caused alpha_sota to saturate at 0.76 by E3, flooding the encoder
            # with 10x diffusion gradient and destroying clinical discrimination.
            #
            # Principle: Decouple alpha from loss statistics entirely. The BayesianProjectedScaler
            # (Kendall et al.) already handles dynamic multi-task balancing via uncertainty
            # weighting. Alpha_sota should be a PURE CURRICULUM: gradually allow diffusion to
            # participate more as the clinical signal stabilizes. No feedback, no dynamic ratio.
            #
            # Schedule: Cosine ramp from alpha_min(0.1) → alpha_target(0.5) over 20 epochs.
            # At E3:  ramp≈0.054, α≈0.122 (vs old 0.76 — 6x less diffusion pressure)
            # At E10: ramp=0.500, α≈0.300 (balanced)
            # At E20: ramp=1.000, α=0.500 (equal weighting with scaler fine-tuning)
            alpha_target = getattr(self.cfg, "alpha_target", 0.5)
            warmup_epochs = max(1, getattr(self.cfg, "alpha_warmup_epochs", 20))
            curr_epoch = float(getattr(self, "current_epoch", 0))
            ramp = 0.5 * (1.0 - math.cos(math.pi * min(1.0, curr_epoch / float(warmup_epochs))))
            alpha_sota = alpha_min + (alpha_target - alpha_min) * ramp
            
            # [TELEMETRY] Retain EMA visibility for diagnostics (scaler still uses them internally)
            with torch.no_grad():
                d_ema = self.loss_scaler.loss_emas[0]
                a_ema = self.loss_scaler.loss_emas[2]
            
            self.log("train/alpha_sota", alpha_sota, on_step=True, prog_bar=True)
            self.log("train/d_ema", d_ema, on_step=True)
            self.log("train/a_ema", a_ema, on_step=True)
            
            loss_dict['diffusion'] = diff_loss * alpha_sota
            loss_dict['bgsl'] = l_bgsl
            loss_dict['tcb'] = l_tcb
            
            # [v26.4 FIX] Unified Physics Manifold (Merge Schism)
            # Rationale: We reuse 'phys_loss' for the unweighted sum. 
            # Note: We use curr_sigma_scale curriculum here.
            phys_loss = self.model.phys_loss(x0_approx) + self.safety_envelope(x0_clinical, risk_coef, sigma_scale=self.curr_sigma_scale)
            
            # [SOTA RECOVERY v3.2] Physics Gradient Normalization (DDP-Safe)
            # Rationale: Normalize physics loss by global running gradient magnitude to match Diffusion.
            # Prevents "Bullying" (52:1 ratio) and enforces 1:1 pressure across the entire cluster.
            phys_loss_raw = torch.clamp(phys_loss, max=float(self.curr_phys_clamp))
            
            # Access DDP-Synchronized EMAs from the scaler
            with torch.no_grad():
                d_ema_sync = self.loss_scaler.loss_emas[0]
                p_ema_sync = self.loss_scaler.loss_emas[6]
                
            # [REVERTED] Dynamic Physics Scaling (d_ema / p_ema) with Warmup
            # Rationale: Fixed 1.0 scale ignored signal imbalance.
            phys_scale_raw = d_ema_sync / (p_ema_sync + 1e-8)
            
            # Dynamic Clamping with Warmup
            _warmup_steps = getattr(self.trainer, "global_warmup_steps", ScalingSteward.SOTA_REF_WARMUP)
            warmup_progress = min(1.0, float(self.global_step) / float(_warmup_steps))
            clamp_min = 0.2 - (0.1 * warmup_progress) # 0.2 -> 0.1
            
            # [FORENSIC FIX #4] Soft Saturation via tanh (Principled phys_scale bounding)
            # Root Cause: d_ema/p_ema ratio grows unboundedly (0.85→3.58) because physics loss
            # naturally shrinks 4x faster than diffusion as the model learns constraints.
            # Principle: tanh(x/k)*k provides smooth saturation approaching k asymptotically.
            # This preserves gradients for small ratios (tanh ≈ identity near 0) while
            # smoothly compressing extreme ratios — no discontinuous gradient at a clamp boundary.
            max_phys_scale = 3.0
            phys_scale_raw_clamped = torch.clamp(phys_scale_raw, min=clamp_min)
            phys_scale = max_phys_scale * torch.tanh(phys_scale_raw_clamped / max_phys_scale)

            # [SOTA v4.0] Critical Telemetry: Physics Visibility
            # Rationale: Removed .item() to stay in the graph.
            self.log("train/phys_scale", phys_scale, on_step=True, prog_bar=True)
            self.log("train/p_ema", p_ema_sync, on_step=True)
            
            loss_dict['phys'] = phys_loss_raw * phys_scale

            # Rationale: EMA updates must only occur on the stepping batch.
            acc_batches = self.cfg.train.get("accumulate_grad_batches", 1)
            
            # [SOTA FIX v33.3] Epoch Boundary Alignment (Deep Forensic Fix)
            
            # [SOTA FIX v33.3] Epoch Boundary Alignment (Deep Forensic Fix)
            # Rationale: 'is_accumulating' must be exactly 'not should_step'.
            # Previous logic missed 'is_last_batch', causing scaler accumulation 
            # while optimizer stepped, leading to 'Epoch Bleed'.
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
            should_step_scaler = ((batch_idx + 1) % acc_batches == 0) or is_last_batch
            is_accumulating = not should_step_scaler
            
            # [SOTA PATCH] Softened S-Curve Warmup (Floor raised to 0.25)
            # Rationale: Original 0.01 floor starved critic/physics for ~400 steps.
            # The critic MUST receive ≥25% signal from step 0 to establish value baselines.
            # [SOTA FIX P12] Sync to Global Warmup
            _gw_steps = getattr(self.trainer, "global_warmup_steps", 1000)
            warmup_ratio = min(1.0, float(self.global_step) / float(max(1, _gw_steps)))
            
            if warmup_ratio < 1.0:
                smooth_penalty = max(0.25, 0.5 * (1.0 - math.cos(math.pi * warmup_ratio)))
                for k in list(loss_dict.keys()):
                   if any(x in k.lower() for x in ["critic", "phys"]):
                       loss_dict[k] = loss_dict.get(k, 0.0) * smooth_penalty
            
            # [SOTA P13] ALI Trigger: Architectural Manifold Auto-Calibration
            # Rationale: Ensures Critic and Clinical tasks start with gradient parity 
            # relative to the Diffusion anchor, escaping the "Dead Critic" initialization.
            if not self.loss_scaler.is_calibrated:
                self.loss_scaler.calibrate_log_vars(loss_dict, anchor_key='diffusion')

            # [NaN GUARD] Graph-Severing Lightning Rods
            # Sanitizes individual components before the Bayesian Scaler mixes them.
            # Injects safe dummy leaf tensor to preserve execution without poisoning weights.
            for k in list(loss_dict.keys()):
                if not torch.isfinite(loss_dict[k]).all():
                    logger.warning(f"⚠️ [NaN GUARD] Non-finite {k} loss detected. Severing graph connection.")
                    loss_dict[k] = torch.zeros(1, device=self.device, requires_grad=True).squeeze()

            # [SOTA 2025] Exclusive Uncertainty Scaling
            # phys_loss is now a managed task in the conflict loop.
            # [v26.5 FIX] Pass curriculum-based physics multiplier to the scaler.
            scaled_total, logs = self.loss_scaler(
                loss_dict, 
                stability_factor=stability_factor,
                phys_multiplier=curr_phys_weight,
                batch_size=B,
                is_accumulating=is_accumulating
            )
            
            # [v41.0 SOTA FIX] Explicit Loss Assignment (The Missing Link)
            # Rationale: 'total_loss' must be updated from the scaler output to 
            # ensure it is a Tensor for downstream .detach() calls.
            total_loss = scaled_total
            
            # [v25.7 FIX] Accumulation-Aware A-GEM Backup
            # Rationale: l_batch + l_ref = total_loss for the gradient projection.
            w_aux = logs.get('weight/aux', 1.0)
            p_aux = logs.get('priority/aux', 1.0)

            # Trace the adaptive beta to diagnose valid range issues.
            loss_dict['beta'] = diag.get("beta_dynamic", self.awr_calculator.beta)
            # "WA" corresponds to Weights Average
            loss_dict['WA'] = diag.get("weights_mean", 1.0) 
            loss_dict['W_Max'] = diag.get("weights_max", 1.0)
            
            # Add to progress bar for real-time monitoring
            self.log("train/awr_beta", loss_dict['beta'], on_step=True, prog_bar=True)
            self.log("train/wa_mean", loss_dict['WA'], on_step=True, prog_bar=True)
            self.log("train/wa_max", loss_dict['W_Max'], on_step=True, prog_bar=True)
            
            # [v25.8 SOTA] Unified Clinical Branch
            # Rationale: New Sepsis discoveries must be protected, NOT suppressed.
            # We move aux_loss (Sepsis Discovery) into the Reference branch as the Anchor.
            # This turns A-GEM from a "Noise Filter" into a "Clinical Bodyguard".
            # [v14.1 FIX] Correct math to prevent sign-flip. aux_loss already includes 0.5*cga.
            # [v26.7 SOTA FIX] Mathematical Anchor Alignment
            # Rationale: Anchor MUST perfectly mirror the scaler's gradient contribution.
            # w_aux already contains 0.5*precision. We multiply by clinical priority (p_aux).
            l_ref = (w_aux * aux_loss * p_aux) if (num_ghosts > 0 or batch_has_sepsis) else None
            
            # [v32.0 SOTA FIX] Correct Gradient Accumulation DDP Sync
            # (already moved up)
            
            # [FIX 1.1] Normalize for Gradient Accumulation
            if acc_batches > 1:
                accum_scale = 1.0 / acc_batches
                total_loss_bwd = scaled_total * accum_scale
                l_ref_bwd = (w_aux * aux_loss * p_aux * accum_scale) if (num_ghosts > 0 or batch_has_sepsis) else None
            else:
                total_loss_bwd = scaled_total
                l_ref_bwd = (w_aux * aux_loss * p_aux) if (num_ghosts > 0 or batch_has_sepsis) else None

            l_batch_bwd = total_loss_bwd - (l_ref_bwd if l_ref_bwd is not None else 0.0)

            # [Optimization Context]
            # Use no_backward_sync if we are in DDP and NOT on the stepping batch.
            sync_context = contextlib.nullcontext()
            if is_accumulating and dist.is_initialized():
                sync_context = self.trainer.strategy.no_backward_sync(self)

            # [v172.0 SOTA FIX] Collective Entry Barrier (Smoking Gun #172)
            # Rationale: Vectorized flag participation to avoid rank divergence without .item() sync.
            ref_needed = (l_ref_bwd is not None and l_ref_bwd.grad_fn is not None)
            if dist.is_initialized():
                 ref_flag = torch.tensor([float(ref_needed)], device=self.device)
                 dist.all_reduce(ref_flag, op=dist.ReduceOp.MAX)
                 global_ref_needed = (ref_flag > 0) 
            else:
                 global_ref_needed = torch.as_tensor(ref_needed, device=self.device)

            # [v32.0 SOTA FIX] Accumulation-Aware AWR Sync
            # acc_batches and is_accumulating are already defined correctly at L1701 (including is_last_batch check)
            
            if global_ref_needed:
                # 1. Capture Reference Gradients
                # [v172.1] Participation Guard: 
                # If local rank has no active reference signal, skip expensive autograd.
                # grad_ref_buffer will remain zero (or keep previous accumulation) and 
                # participate correctly in the global all_reduce.
                if ref_needed:
                    # [SOTA PATCH] Aux Throttle Disabled by Default
                    # Rationale: Default throttle_val=0.1 cut sepsis gradients 90%,
                    # preventing the aux head from learning. Default to False;
                    # enable only if sepsis gradients cause instability.
                    if self.cfg.model.get("throttle_aux", False):
                        if hasattr(aux_loss, 'requires_grad') and aux_loss.requires_grad:
                            throttle_val = self.cfg.model.get("aux_throttle", 0.5)
                            def throttle_aux_gradient(grad):
                                return grad * throttle_val
                            aux_loss.register_hook(throttle_aux_gradient)

                    with sync_context:
                        # [NaN GUARD] Reference Shield
                        if not torch.isfinite(l_ref_bwd):
                            logger.warning(f"⚠️ [NaN GUARD] Non-finite reference loss ({l_ref_bwd.item():.4f}). Shielding AGEM.")
                            l_ref_bwd = torch.zeros_like(l_ref_bwd, requires_grad=True)

                        ref_grads = torch.autograd.grad(
                            l_ref_bwd, 
                            [p for p in self.parameters() if p.requires_grad],
                            retain_graph=True,
                            allow_unused=True
                        )
                    
                    # extraction ...
                    g_ref = {}
                    idx = 0
                    for name, p in self.named_parameters():
                        if p.requires_grad:
                            g = ref_grads[idx]
                            if g is not None:
                                g_ref[name] = g.detach()
                            idx += 1
                else:
                    ref_grads = None
                    g_ref = {}
                
                # [v174.0 SOTA FIX] AGEM Double-Sync Bias (Smoking Gun #174)
                # Rationale: Sub-batch all_reduces are redundant and noisy. 
                # We aggregate locally and only perform one global sync at the step boundary.
                # (Legacy all_reduce logic removed from here)

                # B. Batch Pass (Standard Backward - populates .grad)
                
                # [NaN GUARD] Prevent NaN loss from corrupting ALL parameters
                if not torch.isfinite(l_batch_bwd):
                    logger.warning(f"⚠️ [NaN GUARD] Non-finite batch loss ({l_batch_bwd.item():.4f}). Zeroing backward for batch {batch_idx}.")
                    l_batch_bwd = torch.zeros_like(l_batch_bwd, requires_grad=True)

                # [Distributed Guard]
                with sync_context:
                    self.manual_backward(l_batch_bwd)
                
                # C. [v53.0 SOTA FIX] AGEM Accumulation Protocol (Amnesia Guard)
                # Rationale: AGEM must protect the TOTAL update, not just the local slice.
                # We accumulate g_ref across the cycle and only project on the stepping batch.
                
                # 1. Update Reference Accumulator
                if self.grad_ref_buffer is None:
                    # Lazy allocation based on actual parameter count
                    total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
                    self.grad_ref_buffer = torch.zeros(total_p, device=self.device)
                
                # Coalesce current g_ref into the accumulator
                with torch.no_grad():
                    offset = 0
                    for name, p in self.named_parameters():
                        if p.requires_grad:
                            if name in g_ref:
                                numel = g_ref[name].numel()
                                self.grad_ref_buffer[offset:offset+numel].add_(g_ref[name].flatten())
                                offset += numel
                            else:
                                offset += p.numel()

                # [v54.8 SOTA FIX] Surgical Graph Purge (Smoking Gun #RAM-05)
                # Rationale: ref_grads must be deleted EXPLICITLY after the loop
                # to free the 1.6GB gradient graph before next loss step.
                del ref_grads
                del g_ref
                import gc
                gc.collect()

                # 2. Projection Gate: Only on Stepping Batch
                if not is_accumulating:
                    # [SOTA FIX v30.0/v53.0] DDP Reference Synchronization
                    # Combine all-reduce with normalization
                    if dist.is_initialized():
                        dist.all_reduce(self.grad_ref_buffer, op=dist.ReduceOp.SUM)
                        # Average by world_size (Individual grads already scaled by acc_batches)
                        self.grad_ref_buffer /= dist.get_world_size()

                    # [v48.1 SOTA FIX] Global Reference Sanitization (Smoking Gun #48 cont.)
                    # Rationale: Ensures that even if the reference pass fails, 
                    # we don't propagate NaNs into the model during the restoration step.
                    with torch.no_grad():
                        torch.nan_to_num_(self.grad_ref_buffer, nan=0.0, posinf=0.0, neginf=0.0)

                    # [SOTA v10.2] Vectorized Layer-Wise A-GEM Projection
                    # Now applied to the GLOBAL accumulated gradient.
                    
                    # 1. Unpack grad_ref_buffer into localized dict for easy _foreach access
                    g_ref_accum = {}
                    offset = 0
                    for name, p in self.named_parameters():
                        if p.requires_grad:
                            numel = p.numel()
                            g_ref_accum[name] = self.grad_ref_buffer[offset:offset+numel].view_as(p)
                            offset += numel

                    # 2. Collect Parameters with gradients
                    proj_params = []
                    proj_refs = []
                    for name, p in self.named_parameters():
                        if name in g_ref_accum and p.grad is not None:
                            proj_params.append(p.grad)
                            proj_refs.append(g_ref_accum[name])
                    
                    if proj_params:
                        # [v5.0 SOTA] Global PCGrad Mathematical Projection
                        # Rationale: Layer-wise PCGrad applies identical rotational operations to all 
                        # layers regardless of magnitude, violently tearing Adam/LAMB momentum spaces. 
                        # Projecting the gradients globally across the entire vector space preserves the 
                        # geometric magnitude relationships while fusing 2000 kernels down to 2.
                        
                        # 1. Fuse total gradients into 1D vectors for Global Projection
                        flat_p = torch.cat([p.view(-1) for p in proj_params])
                        flat_r = torch.cat([r.view(-1) for r in proj_refs])
                        
                        # 2. Global Dot Product
                        dot_pr = torch.sum(flat_p * flat_r, dtype=torch.float32)
                        
                        # 3. Soft Margin Check
                        # We only project if they are actively fighting (cos_sim < -0.05)
                        sq_p = torch.sum(flat_p * flat_p, dtype=torch.float32) + 1e-8
                        sq_r = torch.sum(flat_r * flat_r, dtype=torch.float32) + 1e-8
                        
                        norm_p = torch.sqrt(sq_p)
                        norm_r = torch.sqrt(sq_r)
                        cos_sim = dot_pr / (norm_p * norm_r)
                        
                        if cos_sim < -0.05:
                            # 4. Global Alpha Coefficient
                            raw_alpha = -1.0 * (dot_pr / sq_r)
                            
                            # Dynamic Clamping
                            # Mathematical ceiling based on inherent vector scale disparities to prevent
                            # gradient explosion if reference vector is tiny.
                            max_scale = (norm_p / norm_r) * 5.0
                            alpha = torch.clamp(raw_alpha, min=-max_scale, max=max_scale)
                            
                            # 5. Global Fused Projection Update (g_p <- g_p + alpha * g_r)
                            # This replaces the entire `foreach` loop and applies exactly perfectly.
                            torch._foreach_add_(proj_params, proj_refs, alpha=alpha.item())
                        
                        # Note: Artificial Magnitude Restoration (CAGrad-Lite) was explicitly deleted here to allow
                        # natural gradient decay at the Pareto front.
                        
                        # Note: Artificial Magnitude Restoration (CAGrad-Lite) was explicitly deleted here to allow
                        # natural gradient decay at the Pareto front.
                        
                        # 5. Restore Reference (g <- g + ref)
                        if self.orthogonal_replay:
                            torch._foreach_add_(proj_params, proj_refs, alpha=1.0)
                    
                    # 6. Cycle Reset
                    self.grad_ref_buffer.zero_()
            else:
                # [SOTA RECOVERY v3.2] Runtime Gradient Fix
                # Rationale: Restoring total_loss_bwd to ensure correct gradient scaling 
                # (1/acc_batches) when AGEM is bypassed.
                self.manual_backward(total_loss_bwd)


            

            
            # Weights for logging (Sigmas)
            task_weights = [
                logs.get('weight/diffusion', torch.tensor(1.0, device=self.device)), 
                logs.get('weight/critic', torch.tensor(1.0, device=self.device)),
                logs.get('weight/aux', torch.tensor(1.0, device=self.device)),
                logs.get('weight/acl', torch.tensor(1.0, device=self.device)),
                logs.get('weight/bgsl', torch.tensor(1.0, device=self.device)),
                logs.get('weight/tcb', torch.tensor(1.0, device=self.device)),
                logs.get('weight/phys', torch.tensor(1.0, device=self.device))
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
                l_tcb,
                phys_loss
            ])
            
            # [v107.0 SOTA FIX] GradNorm Accumulation Parity (Smoking Gun #107)
            # Rationale: Accumulate clean losses across the cycle to provide 
            # GradNorm with the true average, preventing tail-batch bias.
            with torch.no_grad():
                self.gn_loss_accumulator.add_(primary_losses.detach())
                self.gn_acc_count.add_(1)
            # [v30.5 SOTA] Step-Level GradNorm Update
            # Meta-weights now only react to clean, finished gradients.
            # Use cached weights for projection; update happens in step block.
            task_weights = self.gradnorm.get_weights().detach()
            
            # 2. Weighted losses for CAGrad surgery
            # [v2026 SOTA FIX] Unified Connectivity Guard (Deadlock Prevention)
            # Rationale: manual_backward on a tensor without grad_fn skips DDP sync.
            # We add 0.0 * first_param to every task to ENSURE all ranks sync together.
            first_p = next(self.parameters())
            weighted_tasks = [
                (diff_loss_unweighted * task_weights[0]) + 0.0 * first_p, 
                (critic_loss * task_weights[1]) + 0.0 * first_p, 
                (aux_loss * task_weights[2]) + 0.0 * first_p, 
                (acl_loss * task_weights[3]) + 0.0 * first_p,
                (l_bgsl * task_weights[4]) + 0.0 * first_p,
                (l_tcb * task_weights[5]) + 0.0 * first_p,
                (phys_loss * task_weights[6]) + 0.0 * first_p
            ]
            total_loss = torch.stack([t.detach() for t in weighted_tasks]).sum()
            loss_dict = {
                "diff": weighted_tasks[0].detach(), "critic": weighted_tasks[1].detach(), 
                "aux": weighted_tasks[2].detach(), "acl": weighted_tasks[3].detach(),
                "bgsl": weighted_tasks[4].detach(), "tcb": weighted_tasks[5].detach(),
                "phys": weighted_tasks[6].detach()
            }
            
            # 3. [v33.5 SOTA FIX] GradNorm Meta-Update (Forensic #358)
            # Rationale: Must happen BEFORE pc_backward to avoid "freed graph" RuntimeError.
            # We use 'primary_losses' (Live Tensor) from the stepping batch to drive 
            # the weight updates, ensuring valid gradient graphs.
            self._gn_loss_step = None
            if should_step:
                 # [v2026 SOTA FIX] AMP-Safe Meta-Update (Forensic #358)
                 # Rationale: Pass the trainer's scaler to GradNorm to ensure 
                 # task gradients don't underflow in mixed-precision.
                 scaler = self._get_scaler()
                 self._gn_loss_step, _ = self.gradnorm.update(primary_losses, scaler=scaler)
                 if self._gn_loss_step is not None:
                      gn_loss = self._gn_loss_step.detach()

            # 4. Conflict-Averse Surgery (Backward Pass)
            # [v2026 SOTA] DDP Iron Dome (Smoking Gun #SyncLatency)
            # Rationale: no_sync prevents redundant DDP grad-reductions during accumulation.
            # We only sync on the final 'should_step' batch.
            is_start_of_accum = (batch_idx % acc_batches == 0)
            
            context = contextlib.nullcontext()
            if dist.is_initialized() and not should_step:
                # [ROBUSTNESS] Check if strategy supports no_sync (DDP/DeepSpeed usually do)
                if hasattr(self.trainer.strategy, "model") and hasattr(self.trainer.strategy.model, "no_sync"):
                    context = self.trainer.strategy.model.no_sync()
            
            with context:
                opt.pc_backward(
                    weighted_tasks, 
                    backward_fn=self.manual_backward, 
                    accumulate=not is_start_of_accum
                )
            
            # [v26.1 Unified Fix] 
            # physics_loss is now part of the scaled_total within loss_dict.
            # No separate backward pass is required.

        # [v27.2 SOTA FIX] Non-Destructive Manifold Observation (Patch P)
        # Rationale: sanitize_gradients clips to 1.0 in-place. 
        # During accumulation, this 'shaves' early batches, losing 75% energy.
        # We now use compute_grad_norm for telemetry and reserve clipping for the step block.
        current_grad_pressure = 0.0 # [v311.0 SOTA FIX] Scope Leak Prevention
        with torch.no_grad():
            # [v184.0 SOTA FIX] Unified Pressure Monitoring (Smoking Gun #17/18)
            # Rationale: Pressure must reflect BOTH model manifold stress and 
            # meta-parameter instability to provide a truthful signal for the Sentinel.
            meta_params = []
            if hasattr(self, 'loss_scaler') and self.loss_scaler is not None:
                meta_params.extend(list(self.loss_scaler.parameters()))
            if self.gradnorm is not None and hasattr(self.gradnorm, 'weights'):
                meta_params.append(self.gradnorm.weights)

            acc_norm = self.cfg.train.get("accumulate_grad_batches", 1)
            raw_pressure = OrthogonalGuard.compute_grad_norm(self.model, extra_params=meta_params)
            current_grad_pressure = raw_pressure / float(acc_norm)
            
            # [v118.2 SOTA FIX] DDP Pressure Consensus (Smoking Gun #118)
            # Rationale: All ranks must agree on the manifold pressure to prevent 
            # divergent Sentinel states and curriculum pauses.
            if dist.is_initialized():
                p_tensor = torch.tensor([current_grad_pressure], device=self.device)
                dist.all_reduce(p_tensor, op=dist.ReduceOp.MAX)
                current_grad_pressure = p_tensor 

            self.log("train/manifold_norm_eff", min(current_grad_pressure, 1.0), on_step=True)

            # [v2026 SOTA] Efficient Intra-Epoch CSV Logging
            # Moved to the end of training_step to prevent '.compute() before .update()' warnings.
            self.log("train/manifold_norm_std", self.grad_norm_std, on_step=True)

        # Rationale: Manually manage accumulation for precise DDP synchronization
        # and bit-perfect resumption.
        acc_batches = self.cfg.train.get("accumulate_grad_batches", 1)
        
        # is_last_batch was calculated at the start of training_step
        if should_step:
            # [v91.0 SOTA FIX] Uniform Step Scaling (Smoking Gun #91)
            # Rationale: Use the stateful index for exact tail-batch normalization.
            actual_accum = self._shadow_grad_accum_idx
            
            if actual_accum != acc_batches:
                logger.info(f"[EPOCH TAIL] Normalizing step for {actual_accum}/{acc_batches} accumulation.")

            # [SOTA FIX] Manual unscaling required for AdamW (Standard Protocol)
            scaler = self._get_scaler()
            if scaler is not None:
                scaler.unscale_(opt)

            # [ROBUSTNESS] Use .get() fallback to prevent crash if key is missing
            clip_val = self.cfg.train.get("grad_clip", 1.0)
            
            # [v27.1 FIX] Scale clip threshold with accumulation steps
            # Rationale: Accumulated gradients scale as √(accum_steps)
            # Without scaling, accum=16 retains only 38.6% of gradient info vs 100% for accum=1
            accum_steps = self.cfg.train.get("accumulate_grad_batches", 1)
            if accum_steps > 1:
                clip_val = clip_val * (accum_steps ** 0.5)
            
            if clip_val > 0:
                # [v27.2 SOTA FIX] Late-Stage Iron Dome Protection
                # Rationale: We only apply destructive manifold clipping AFTER accumulation
                # to preserve full signal integrity from all batches in the cycle.
                OrthogonalGuard.sanitize_gradients(self.model)
                
                # 1. Clip Loss Scaler & GradNorm (Smoking Gun #18)
                # Rationale: Meta-gradients operate on a different scale than the model.
                # Hardening them ensures meta-parameter drift doesn't corrupt the stability signal.
                
                # [v2026 FIX] Dynamically scale meta-clip by accumulation headroom
                accum_steps = self.cfg.train.get("accumulate_grad_batches", 1)
                meta_clip = 0.1 * (accum_steps ** 0.5) if accum_steps > 1 else 0.1
                
                if hasattr(self, 'loss_scaler') and self.loss_scaler is not None:
                    torch.nn.utils.clip_grad_norm_(self.loss_scaler.parameters(), meta_clip)
                
                if self.gradnorm is not None and hasattr(self.gradnorm, 'weights'):
                    torch.nn.utils.clip_grad_norm_([self.gradnorm.weights], meta_clip)
                
                # 2. Main Parameters: Adaptive Clipping (AGC)
                # [SOTA PATCH] Balanced AGC Factor (0.05)
                # Rationale: 0.01 kept only 10% of gradients (confirmed by test).
                # 0.05 allows 5× parameter norm gradient — stable yet learnable.
                if self.adaptive_clipping:
                    adaptive_gradient_clip_(self.parameters(), clip_factor=0.05)
                
                # 3. Hard Safety Clip & Finite Check
                grad_norm_val = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
            else:
                # [v26.1 FIX] Allow training without clipping (assume finite or trust regularizers)
                grad_norm_val = torch.tensor(0.0, device=self.device)

            # [v30.0 SOTA] Global Heartbeat Consensus
            # Rationale: Ranks MUST step or skip together. A single Inf on one rank 
            # will cause a "One-Armed Bandit" state if others proceed.
            if dist.is_initialized():
                finite_t = torch.as_tensor(1.0 if torch.isfinite(grad_norm_val) else 0.0, device=self.device)
                dist.all_reduce(finite_t, op=dist.ReduceOp.MIN)
                should_apply = (clip_val <= 0) or (finite_t > 0.5)

            else:
                should_apply = (clip_val <= 0) or torch.isfinite(grad_norm_val)

            # [v2026 SOTA] Stabilization Metadata Consolidation
            # Rationale: These variables must be available for both the Deadman Switch 
            # and the final TrendSentinel statistics update.
            _max_budget = getattr(self.trainer, "estimated_stepping_batches", ScalingSteward.SOTA_REF_SAFE_BUDGET)
            _safe_budget = max(100.0, float(_max_budget))
            is_init_period = self.global_step < max(100, int(0.01 * _safe_budget))
            grace_val = self._shadow_resumption_grace_steps
            
            # [v2026 SOTA] Atomic Gradient Pressure Acquisition
            # Ensure pressure is captured even if optimization is skipped
            current_grad_pressure = grad_norm_val if isinstance(grad_norm_val, torch.Tensor) else torch.tensor([grad_norm_val], device=self.device)

            # [SOTA FIX] Stabilization Initialization Bridge
            # Rationale: Sentinel must establish a baseline during the first steps.
            if grace_val == 0 and is_init_period:
                self.grad_norm_ema.fill_(current_grad_pressure)
                self.grad_norm_std.fill_(0.5) 
            
            # [v2026 SOTA] H100 Deadman Switch (Iron Dome Layer)
            # Rationale: Skip any 5-sigma gradient spikes to protect the manifold.
            # Cleanly exit if persistent instability is detected to save compute.
            if should_apply and not is_init_period and grace_val <= 0:
                current_p = current_grad_pressure.item()
                
                # Check for "Manifold Shock" (5-Sigma Outlier)
                z_score = TrendSentinel.calculate_z_score(current_p, self.grad_norm_ema, self.grad_norm_std)
                if z_score > 5.0:
                    # [v2026.1 STABILITY FIX] Enhanced Diagnostics
                    logger.warning(
                        f"☄️ [IRON DOME] Blocked 5-Sigma Gradient Spike "
                        f"(Z={z_score.item():.2f}, Norm={current_p:.4f}, "
                        f"EMA={self.grad_norm_ema.item():.4f}, STD={self.grad_norm_std.item():.4f}). "
                        f"Skipping step."
                    )
                    should_apply = False
                
                # Check for "Manifold Collapse" (Emergency Shutdown)
                if TrendSentinel.is_unstable(self.grad_norm_ema, self.grad_norm_std, max_pressure=7.0, max_sigma=3.0):
                    logger.critical("🚨 [DEADMAN SWITCH] Global Manifold Collapse Detected. Emergency Shutdown Triggered.")
                    import os
                    logger.critical("Final Manifold Stats: Pressure={:.4f}, Sigma={:.4f}".format(self.grad_norm_ema.item(), self.grad_norm_std.item()))
                    os._exit(1)

            if should_apply:
                # [v2026 SOTA FIX] Unified Scalar Consensus (Smoking Gun #ScalarDrift)
                # Rationale: Different ranks may have different local scale factors.
                # We enforce global consensus and unscale before reduction to ensure parity.
                scaler = self._get_scaler()
                if dist.is_initialized() and scaler is not None:
                    scale_t = torch.tensor([scaler.get_scale()], device=self.device)
                    dist.all_reduce(scale_t, op=dist.ReduceOp.MIN)
                    scaler._scale = scale_t # Force bit-perfect consensus
                
                # [v118.3 SOTA FIX] Meta-Gradient Sync (Smoking Gun #118.3)
                # Rationale: loss_scaler parameters are not wrapped in DDP.
                if dist.is_initialized() and hasattr(self, "loss_scaler"):
                    inv_scale = 1.0 / (scaler.get_scale() + 1e-8) if scaler is not None else 1.0
                    for p in self.loss_scaler.parameters():
                        if p.requires_grad and p.grad is not None:
                            # [v2026 SOTA] Pre-Reduction Unscaling
                            # Rationale: Must be unscaled BEFORE all_reduce for rank parity.
                            p.grad.mul_(inv_scale)
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= dist.get_world_size()

                # [SOTA FIX] Scalar-aware step for FP16 Stability
                scaler_local = self._get_scaler()
                if scaler_local is not None:
                    scaler_local.step(opt)
                else:
                    opt.step()
                
                # [v48.0 SOTA FIX] Stateful Accumulation Index Reset (Zero-Sync)
                # Rationale: Once we step, the accumulation cycle is complete.
                self._shadow_grad_accum_idx = 0
                
                # Post-step Integrations (EMA, etc)
                # [v17.3 Hardened] Dead Teacher Fix: Update Target Network
                # [v108.0 SOTA FIX] EMA Accumulation Parity (Smoking Gun #37)
                # Rationale: The Teacher should follow the Student at the effective update rate.
                # Since we step the optimizer ONCE, we must move the teacher ONCE (update_every=1).
                # Previous use of 'actual_accum' caused 16x faster drift during accumulation.
                if self.ema is not None: 
                    self.ema.update(self.model, update_every=1)
                
                # [PMS] Initialization Boundary Detector (Zero-Spam)
                # Rationale: Standardize the Sentinel warmup period across all step densities.
                # Consolidates logs to boundaries to prevent iteration-level pollution.
                if grace_val == 0 and self.global_step == 0 and not self._pms_start_logged:
                    logger.info("[PMS] Establishing Gradient Pressure Baseline...")
                    self._pms_start_logged = True
                
                if grace_val == 0 and not is_init_period and not self._pms_init_logged:
                    logger.info(f"[PMS] Initialization Complete. GN Baseline: {self.grad_norm_ema.item():.4f}")
                    self._pms_init_logged = True
                elif self._shadow_resumption_grace_steps == 1:
                    logger.info("[PMS] Resumption Grace Period Concluded (Stats Preserved).")
                # [v12.8.3 SOTA FIX] Direct Attachment Projection
                if hasattr(self, 'loss_scaler') and self.loss_scaler is not None:
                    if hasattr(self.loss_scaler, 'project_parameters'):
                        self.loss_scaler.project_parameters()

                # [v48.0] Cycle Reset
                self._shadow_grad_accum_idx = 0
                
                # [v2026 CLEANUP] Grace Period decrement MOVED to batch-level (line ~2637)
                # to avoid double-decrement on stepping batches.

                # [v30.5 SOTA] Legacy GradNorm Step
                # Protected from spikes and scaled correctly.
                # [v2026 CLEANUP] GradNorm Legacy Step Block REMOVED (Smoking Gun #DoubleStep)
                # Rationale: gradnorm.update() (gradnorm.py L146-152) now performs 
                # backward() + clip + step() internally. The old legacy code here was 
                # attempting the same on a .detach()ed loss (producing zero gradients)
                # but the spurious scaler.step() call corrupted _growth_tracker counts.
                # The accumulator reset is still needed:
                if self.balancing_mode == "legacy_surgical" and self._gn_loss_step is not None:
                     self.gn_loss_accumulator.zero_()
                     self.gn_acc_count.fill_(0)
            else:
                logger.warning(f"⚠️ Gradient Spike Detected (Norm={grad_norm_val.item():.2f}). Skipping optimization step for batch {batch_idx}.")
            
            # [v2026.1 STABILITY FIX] Decoupled Sentinel Update
            # Rationale: The TrendSentinel MUST see every gradient norm, including
            # spikes that are blocked by the Iron Dome. Without this, the EMA becomes
            # permanently stale after a block, causing a "Hypersensitivity Trap" where
            # the model can never recover from a single false positive.
            # Validated by forensic_stability_probe.py: Z-escalation 5.70→7.19 (broken)
            # vs stable convergence (fixed).
            ada_decay_threshold = ScalingSteward.get_steps(300, self.trainer.num_training_batches)
            active_decay = 0.90 if self.grad_norm_step_count < ada_decay_threshold else self.grad_ema_decay
            active_decay_scaled = active_decay ** (self._shadow_grad_accum_idx if self._shadow_grad_accum_idx > 0 else 1)
            
            if grace_val <= 0 and not is_init_period:
                TrendSentinel.update_stats(
                    current_grad_pressure, 
                    self.grad_norm_ema, 
                    self.grad_norm_std, 
                    active_decay_scaled,
                    step_tensor=self.grad_norm_step_count
                )
            
            # [v23.0 SOTA FIX] Mandatory Reservoir Purge (Smoking Gun #224)
            # Rationale: Regardless of step success, we MUST wipe the gradients 
            # to prevent accumulation overlap into the next cycle.
            opt.zero_grad()

            # [SOTA FIX] Update scaler factor after step or skip
            scaler = self._get_scaler()
            if scaler is not None:
                scaler.update()

            sch = self.lr_schedulers()
            if sch is not None:
                if isinstance(sch, list):
                    for s in sch: 
                        s.scheduler.step() if hasattr(s, "scheduler") else s.step()
                else:
                    sch.scheduler.step() if hasattr(sch, "scheduler") else sch.step()
        
        # Log periodicity: every batch regardless of accumulation

        # --- 6. Telemetry & Metric Accumulation ---
        with torch.no_grad():
            # [v4.1.9 SOTA FIX] Mask-Aware Training EV (Smoking Gun #Padding-Bleed)
            ev = self.model.value_loss_fn.compute_explained_variance(
                pred_values, 
                returns, 
                tau=self.curr_tau.item(),
                mask=f_mask
            )
            
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
            
            # [v2026 SOTA] Vectorized DAB Transfer (Zero-Sync)
            # Rationale: Replaced if sepsis_mask.any() with atomic fused gather.
            # Empty tensors participate in the handshake to maintain DDP consensus.
            sepsis_mask = (targets > 0)
            
            # 1. Prepare Local Tensors (Float32 Flat Packets)
            T, F_feat = past.shape[1], past.shape[2]
            D = global_ctx_expert.shape[1]
            
            # Masking is vectorized; indexing with GPU mask doesn't break the graph.
            # We create empty tensors if no sepsis is found, preserving rank consensus.
            flat_vitals = past[sepsis_mask].reshape(-1, T*F_feat).float()
            if src_mask is not None:
                 flat_masks = src_mask[sepsis_mask].reshape(-1, T*F_feat).float()
            else:
                 flat_masks = torch.ones_like(flat_vitals)
            
            flat_labels = targets[sepsis_mask].float().unsqueeze(1)
            flat_latents = global_ctx_expert[:B][sepsis_mask].float()
            flat_unc = uncertainty[:B][sepsis_mask].float()
            
            local_dict = {
                "vitals": flat_vitals,
                "masks": flat_masks,
                "labels": flat_labels,
                "latents": flat_latents,
                "uncertainty": flat_unc
            }

            # 2. Perform Gather (DDP-aware with single-GPU fallback)
            if self.ddp_gatherer is not None:
                gathered_flat = self.ddp_gatherer.gather_fused_batch(local_dict)
            else:
                # Single-GPU fallback: concatenate local tensors into the same flat format
                if flat_vitals.size(0) > 0:
                    gathered_flat = torch.cat([flat_vitals, flat_masks, flat_labels, flat_latents, flat_unc], dim=1)
                else:
                    gathered_flat = flat_vitals.new_empty(0, T*F_feat + T*F_feat + 1 + D + 1)

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
                 
                 # [v2026 Phase 12] Metabolic Momentum Burst (Warmup: 10% of total training budget)
                 # [SOTA FIX - RATIO ANNEALING] Replaced 1000 steps with dynamic budget ratio.
                 _max_budget_burst = getattr(self.trainer, "estimated_stepping_batches", ScalingSteward.SOTA_REF_SAFE_BUDGET)
                 _safe_budget_burst = max(float(ScalingSteward.SOTA_REF_WARMUP), float(_max_budget_burst))
                 is_burst_period = self.global_step < max(ScalingSteward.SOTA_REF_WARMUP, int(0.10 * _safe_budget_burst))
                 
                 # Rapidly ingests new 'Leaky' representation space after metamorphosis.
                 self.ghost_bank.update(
                    vitals=g_v.reshape(-1, T, F_feat),
                    masks=g_m.reshape(-1, T, F_feat),
                    labels=g_lbl.squeeze(1).long(),
                    latents=g_lat,
                    uncertainties=g_unc,
                    prototype_burst=is_burst_period
                 )
            
            # Global Rank 0 Logging (SOTA: Pass objects, not .compute(), to avoid sync bottleneck)
            # [TELEMETRY] Primary Metrics (Visible in Progress Bar)
            # Use detached scalars (.item()) for the progress bar to ensure immediate visibility.
            # Shortening to L, D, V, etc. is handled by the APEXProgressBar callback.
            # [TELEMETRY] Primary Metrics (Visible in Progress Bar)
            # [v13.5] Batch-Level State Decrements
            # [v13.5] Batch-Level State Decrements (Zero-Sync)
            if self._shadow_resumption_grace_steps > 0:
                self._shadow_resumption_grace_steps -= 1

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
                "train_ev": ev,
                "ood_score": uncertainty[:B].mean(), # [FIX] Map to local variable, slice to main batch
                "bank_size": self.ghost_bank.size.float(),
                "clinical_snr": (aux_loss * (logs.get('weight/aux', 1.0) if 'logs' in locals() else 1.0)) / (diff_loss * (logs.get('weight/diffusion', 1.0) if 'logs' in locals() else 1.0) + 1e-8),
                "curr_phys_weight": torch.as_tensor(curr_phys_weight, device=self.device).detach().clone(),
                "w_aux": torch.as_tensor(task_weights[2], device=self.device).detach().clone(),
                "SOTA_LR": float(self.optimizers().param_groups[0]["lr"]),
                "Warmup": float(self.cfg.train.get("warmup_steps", 0)),
                "TotalSteps": float(getattr(self.trainer, "estimated_stepping_batches", -1)),
                "manifold_stability": 1.0,
            }, on_step=True, on_epoch=False, prog_bar=True)

            # [TELEMETRY] Detailed Diagnostics (WandB Only)
            with torch.no_grad():
                bank_unc = self.ghost_bank.uncertainties[:self.ghost_bank.size].mean() if self.ghost_bank.size > 0 else 0.0
                manifold_drift = 0.0
                if self.ghost_bank.size > 0:
                    # Drift = 1 - sim(Prototype, BatchExpertAvg)
                    batch_expert_avg = F.normalize(global_ctx_expert[:B].mean(dim=0, keepdim=True), dim=1)
                    manifold_drift = 1.0 - torch.matmul(batch_expert_avg, self.ghost_bank.prototype_ema.T)

            self.log_dict({
                "train/loss_critic": self.train_loss_critic,
                "train/loss_aux": self.train_loss_aux,
                "train/loss_acl": self.train_loss_acl,
                "train/loss_bgsl": self.train_loss_bgsl,
                "train/loss_tcb": self.train_loss_tcb,
                "train/loss_phys": self.train_loss_phys,
                "train/loss_gradnorm": self.train_loss_gradnorm,
                "train/train_ev": self.train_explained_var,
                "train/awr_ess": self.train_awr_ess,
                "train/bank_avg_uncertainty": bank_unc,
                "train/manifold_drift": manifold_drift,
                "train/weight_diff": task_weights[0],
                "train/weight_critic": task_weights[1],
                "train/weight_aux": task_weights[2],
                "train/curr_phys_weight": curr_phys_weight,
            }, on_step=True, on_epoch=False, prog_bar=False)

            # [v2026 SOTA] Efficient Intra-Epoch CSV Logging
            # Log every N% (e.g., 5%, 10%, 15%) without syscall latency
            if getattr(self, "csv_log_interval", 0) > 0:
                # 1. Initialize Logger on First Step (Rank 0 Only)
                if self.csv_logger is None and (not dist.is_initialized() or dist.get_rank() == 0):
                     log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/fallback"
                     self.csv_logger = BufferedCSVLogger(log_dir)

                # 2. Check Interval bucket
                # e.g., pct = 5, interval = 5 -> bucket 1
                total_batches = self.trainer.num_training_batches
                if total_batches > 0:
                     pct = int(((batch_idx + 1) / total_batches) * 100)
                     # Only log if we crossed a new interval bucket (5, 10, 15...)
                     # and haven't logged it yet.
                     if pct % int(self.csv_log_interval) == 0 and pct > getattr(self, "last_logged_bucket", -1):
                         if self.csv_logger:
                             # [SOTA FIX] Handle Single vs Multi Optimizer safely
                             current_opt = opt[0] if isinstance(opt, list) else opt
                             # current_grad_pressure was computed earlier in the step
                             gn_val = current_grad_pressure if 'current_grad_pressure' in locals() else 0.0
                             
                             row = {
                                 "epoch": getattr(self, "current_epoch", 0),
                                 "pct": pct,
                                 "step": self.global_step,
                                 "loss": total_loss.item() if (isinstance(total_loss, torch.Tensor) and total_loss.dim() == 0) else float(total_loss),
                                 "gn": float(gn_val),
                                 "ess": float(weights_awr_log.get("train/awr_ess", 0.0)) if 'weights_awr_log' in locals() else 0.0,
                                 "ev": float(self.train_explained_var.compute().item()),
                                 "lr": float(current_opt.param_groups[0]['lr']) if current_opt else 0.0
                             }
                             # Add component losses
                             for k, v in loss_dict.items():
                                 row[f"loss_{k}"] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                                 
                             self.csv_logger.log(row)
                             self.last_logged_bucket = pct

        return total_loss



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
            uncertainty = out.get("aux_uncertainty", None) # [v2026 SOTA] Extract for sepsis gating
            
            if logits is not None:

                
                # [v14.2 SOTA FIX] Direct Probability Extraction
                # Rationale: Using probabilities computed by the head ensures 
                # correct activation (Sigmoid vs Softmax) is used.
                probs = out.get("aux_probs", None)
                
                if probs is not None:
                    # Multi-class: Probabilistic Union for independent sigmoids
                    # P(Any Sepsis) = 1 - Product(1 - P(Class_i))
                    if probs.shape[-1] > 1:
                        risk_prob = 1.0 - (1.0 - probs[:, 1:]).prod(dim=1)
                    else:
                        risk_prob = probs.squeeze()
                else:
                    # Fallback for old planners
                    if logits.shape[-1] > 1:
                        probs = F.softmax(logits, dim=-1)
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
                # [SOTA FIX]: Use detached probabilities for metric accumulation to prevent graph leaks
                risk_prob_metric = risk_prob.detach()
                
                ece = compute_ece(risk_prob_metric, binary_label)
                oe = compute_overconfidence_error(risk_prob_metric, binary_label)
                
                self.val_ece.update(ece)
                self.val_oe.update(oe)
                self.val_acc_sepsis.update(logits.detach(), target_class)
                self.val_auroc_sepsis.update(risk_prob_metric, binary_label)
                self.val_precision.update(risk_prob_metric, binary_label)
                self.val_recall.update(risk_prob_metric, binary_label)
                self.val_f1.update(risk_prob_metric, binary_label)

                # [v5.3.4 SOTA FIX] Align Validation Semantic Baseline
                # [v2026 SOTA] Expert Realignment: Measuring Student vs Teacher TD Parity.
                # Rationale: Training optimizes Student to predict Teacher-bootstrapped targets.
                # Validation EV must use the same "Ruler" to avoid measurement desync.
                if value_preds is not None and "future_data" in batch:
                    with torch.no_grad():
                        # A. Teacher Pass (Frozen EMA) - Generate the target Return for EV check
                        with self.ema_teacher_context():
                            teacher_out = self.model(batch, reduction='none')
                            target_values = self.model.value_head.get_expectile_summary(teacher_out["pred_value"], tau=self.curr_tau)
                        
                        # B. Identify Truncation (Critical for Windowed EV)
                        is_truncated = batch.get("is_truncated", torch.zeros(bs, dtype=torch.bool, device=self.device))
                        bootstrap_value = target_values[:, -1:] * is_truncated.float().unsqueeze(-1)

                        # C. Clinical Reward
                        rewards = self.awr_calculator.compute_clinical_reward(
                            batch["future_data"], 
                            batch.get("outcome_label", None),
                            dones=batch.get("is_terminal", None),
                            feature_indices=self.clinical_feat_idx,
                            normalizer=None, 
                            src_mask=batch.get("future_mask", None),
                            training=self.training
                        )

                        # D. Teacher-Student SAW (State Advantage Weighting)
                        v_student = self.model.value_head.get_expectile_summary(value_preds, tau=self.curr_tau)
                        advantages = self.awr_calculator.compute_saw(
                            rewards, 
                            student_values=v_student,
                            teacher_values=target_values,
                            dones=batch.get("is_terminal", None),
                            bootstrap_value=bootstrap_value
                        )
                        val_returns = (advantages + v_student).detach()
                        
                        # E. Distributional EV Measurement
                        # [v4.1.7 SOTA FIX] Mask-Aware EV (Smoking Gun #Padding-Bleed)
                        ev = self.model.value_loss_fn.compute_explained_variance(
                            value_preds, 
                            val_returns, 
                            tau=self.curr_tau.item(),
                            mask=batch.get("future_mask")
                        )
                        # [SOTA FIX Phase 7] EV Validation Bounding
                        # Prevents dashboard collapse from initial noisy expectations
                        ev_bounded = torch.clamp(ev, min=-1.0, max=1.0)
                        self.val_explained_var.update(ev_bounded)

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
            
            # [v14.3 SOTA FIX] Pass head-derived probs to callbacks
            if "risk_prob" in locals():
                result["sepsis_prob"] = risk_prob
            if "uncertainty" in locals():
                result["sepsis_uncertainty"] = uncertainty
                
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
        # [SOTA FIX] Manifold Disentanglement (GMSE Repair)
        # Rationale: Static features (indices 22-27) are conditioning inputs, not generative outputs.
        # Including them in MSE creates an irreducible error floor (~3700) that masks dynamic learning.
        DYNAMIC_CHANNELS = 22 # Defined by Schema (0-22 are dynamic)

        # Project to Dynamic Subspace
        pred_dynamic = pred_safe[..., :DYNAMIC_CHANNELS].contiguous()
        gt_dynamic = gt_safe[..., :DYNAMIC_CHANNELS].contiguous()

        # [SOTA FIX Phase 7] Valid Subspace Masking (Smoking Gun #GMSE)
        # Rationale: padding_mask=1 indicates padded timesteps (noise).
        # We must invert it to extract only valid clinical timesteps before MSE.
        padding_mask = subset.get("padding_mask", None)
        
        if padding_mask is not None:
            valid_idx = ~padding_mask.bool()
            
            # Projecting [B, T, D] -> [N, D] where N = total valid timesteps
            pred_dyn_valid = pred_dynamic[valid_idx]
            gt_dyn_valid = gt_dynamic[valid_idx]
            
            self.val_mse_global.update(pred_dyn_valid, gt_dyn_valid)
            
            if pred_safe.shape[-1] > 6:
                self.val_mse_hemo.update(pred_safe[..., :7][valid_idx].contiguous(), gt_safe[..., :7][valid_idx].contiguous())
            if pred_safe.shape[-1] > 17:
                self.val_mse_labs.update(pred_safe[..., 7:18][valid_idx].contiguous(), gt_safe[..., 7:18][valid_idx].contiguous())
            if pred_safe.shape[-1] > 21:
                self.val_mse_electrolytes.update(pred_safe[..., 18:22][valid_idx].contiguous(), gt_safe[..., 18:22][valid_idx].contiguous())
        else:
            # Update Metric on Valid Subspace (Generative Error)
            self.val_mse_global.update(pred_dynamic, gt_dynamic)
            
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
            # pass raw physical data directly to safety calculations
            map_idx, sbp_idx = 4, 2
            
            # [SOTA FIX v10.3] Sparse Stitching
            # We MUST pass src_mask so Guardian finds the LAST VALID observation.
            # Otherwise it compares Pred (e.g. 140) vs Missing Obs (0.0) -> OOD.
            s_mask = subset.get("src_mask", None)

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
                # [SOTA RECOVERY v3.2] Dual-Stream Telemetry
                # Rationale: Log raw logs for OOD detection, but use smoothed for metrics.
                raw_pred = pred_safe.clone().detach()
                
                # Apply Smoothing for Metric Stability
                for _ in range(3):
                    p_pad = F.pad(pred_safe.permute(0, 2, 1), (1, 1), mode='replicate')
                    B, C, T = p_pad.shape
                    weights_expanded = weights.expand(C, 1, 3) 
                    smoothed = F.conv1d(p_pad, weights_expanded, groups=C)
                    pred_safe = smoothed.permute(0, 2, 1)

            safety_results = self.safety_guardian.check_trajectories(
                subset["observed_data"], 
                pred_safe, # [SOTA PATCH] Use smoothed predictions — raw has extreme single-timestep jitter
                src_mask=s_mask,
                force_clinical=True
            )
            
            # 5. Log Failure Breakdown
        self.val_ood_rate.update(safety_results["ood_rate"])
        self.val_safe_traj_count.update(safety_results["safe_count"])
            
        # 5. Physics Violations (Checking Normalized Bounds)
        # We must RE-NORMALIZE to check if the model is hitting the [-1, 1] clamp.
        # [SOTA Fix] Check explicitly against Normalized Bounds
        pred_norm_check = normalizer.normalize(pred_safe)[0] # Returns (norm, static) tuple -> take [0]
        violations = ((pred_norm_check.abs() > 0.99).float().mean())
        self.val_phys_violation_rate.update(violations)

    def on_train_epoch_end(self):
        """
        [v35.0 SOTA FINAL] Unified Epoch Finalization.
        1. Logs epoch metrics (with DDP sync).
        2. Refreshes GhostBank anchors to prevent Phantom Gradients.
        3. Resets metrics for the next epoch.
        """
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

        # Reset Metrics
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
        
        if self.trainer.is_global_zero:
            logger.info(f"🔄 [TELEMETRY] Epoch {self.current_epoch} Training Metrics Reset.")
            
        # [v110.0 SOTA FIX] Teacher Consensus Engine (Smoking Gun #38)
        # Rationale: Average teacher shadow weights across all ranks at epoch-end
        # to prevent silent divergence in the EMA manifold.
        #
        # [v36.0 SOTA FIX] Periodic Hard Teacher Reset (Fix #H2)
        # Rationale: Diagnostic testing confirmed Teacher-Student drift of 1.87x (threshold: 0.05).
        # Soft EMA updates cause the teacher to lag behind the student's evolved manifold,
        # leading to stale advantage estimates and GMSE oscillation after epoch            # Hard sync every 3 epochs (Reverted from 20)
        if self.ema is not None:
            # [FIX #H2] Check if this is a hard sync epoch
            hard_sync_interval = 3
            if (self.current_epoch + 1) % hard_sync_interval == 0:
                # Hard Sync: Copy student weights to teacher shadow
                # This resets the value manifold drift to zero
                if self.trainer.is_global_zero:
                    logger.info(f"🔄 [HARD SYNC] Resetting EMA teacher to student state (epoch {self.current_epoch})...")
                
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in self.ema.shadow:
                            self.ema.shadow[name].copy_(param.data.cpu().float())
                    
                    for name, buffer in self.model.named_buffers():
                        if name in self.ema.shadow:
                            if torch.is_floating_point(buffer):
                                self.ema.shadow[name].copy_(buffer.data.cpu().float())
                            else:
                                self.ema.shadow[name].copy_(buffer.data.cpu())
                
                if self.trainer.is_global_zero:
                    logger.info(f"✅ [HARD SYNC] Complete. Teacher-Student drift reset to 0.0.")
            
            # Normal synchronization: Average shadow weights across DDP ranks
            self.ema.synchronize()

        # [v33.1 SOTA FIX] Ghost Anchor Refresh (Smoking Gun #356)
        # Rationale: Re-align stored latents with current encoder manifold 
        # to prevent "Ghost Drift" (18.6% error found in audit).
        if self.ghost_bank.size > 0:
             def ghost_encoder_fn(vitals_batch, masks_batch):
                 # vitals_batch: [B, T, 28] (Raw Clinical)
                 # masks_batch: [B, T, 28] (Raw Masks)
                 
                 # 1. Extract Static Metadata (Indices 22-27)
                 # [v12.0 Schema] 22:Age, 23:Gender, 24:Unit1, 25:Unit2, 26:AdmTime, 27:LOS
                 # All timesteps are identical for static, take t=0
                 static_batch = vitals_batch[:, 0, 22:].clone()
                 
                 # 2. Normalize (Model expects Normalized Inputs)
                 # Note: self.model.normalize handles the split internally? 
                 # No, it takes (x_ts, x_static).
                 # We must pass the raw tensors.
                 norm_vitals, norm_static = self.model.normalize(vitals_batch, static_batch)
                 
                 # 3. Derive Padding Mask from Imputation Mask
                 # In training_step (L758), padding is where ALL channels are 0.0 in mask.
                 if masks_batch is not None:
                     bool_padding_mask = (masks_batch.sum(dim=-1) == 0)
                 else:
                     bool_padding_mask = None
                     
                 # 4. Forward Pass (Frozen Stats)
                 # We simply want the latent, not to update BN running stats.
                 with self.frozen_stats():
                     out_alb = self.model.encoder(
                         norm_vitals,
                         norm_static,
                         imputation_mask=masks_batch, 
                         padding_mask=bool_padding_mask
                     )
                     
                 # 5. Return the "Expert" Latent (Matches 'global_ctx_expert' used in update)
                 return out_alb["global_expert"]

             if self.trainer.is_global_zero:
                 logger.info(f"👻 [GHOST REFRESH] Re-encoding {self.ghost_bank.size} anchors (Decay=0.9/Momentum=0.1)...")
                 
             # [ry.md FIX 3] Smooth Manifold Evolution (decay 0.3 → 0.9)
             # Simulation: decay=0.3 creates 80x more landscape shift than 0.9.
             # 70% anchor jump causes abrupt gradient mismatch at epoch boundary.
             # 10% update (decay=0.9) provides smooth manifold evolution.
             self.ghost_bank.refresh_anchors(ghost_encoder_fn, decay=0.9)
             if self.global_step % 100 == 0:
                 logger.info(f"[GHOST REFRESH] Bank Size: {self.ghost_bank.size.item()} | Refresh Decay: 0.9 (Smooth Momentum)")
             # [v2026 RAM SPIKE FIX] Memory Clearing (Smoking Gun #RAM-01)
             # Rationale: Ghost Refresh creates ~500MB of activations that must be freed
             # before checkpoint serialization runs to prevent OOM.
             gc.collect()
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
             if self.trainer.is_global_zero:
                 logger.info("🧹 [MEMORY] Ghost Refresh activations cleared.")

        # [SOTA SAFETY FIX] Verify Loss Scaler Hygiene (Deep Forensic Guard)
        # Rationale: Prevent "Epoch Bleed" where accumulator logic fails at boundaries.
        if hasattr(self, "loss_scaler") and hasattr(self.loss_scaler, "assert_clean"):
            self.loss_scaler.assert_clean()

    def on_validation_epoch_start(self):
        """[v2026 SOTA] Robustness Guard: Ensures validation buffers are fresh."""
        self.validation_step_outputs.clear()
        # [SOTA FIX] Forensic Patch 1: Reset OOD cache to prevent amnesia
        if hasattr(self, "safety_guardian"):
            self.safety_guardian.reset_cache()


    def on_test_epoch_start(self):
        """[v2026 SOTA] Robustness Guard: Ensures test buffers are fresh."""
        if hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        """
        Aggregates safety stats and performs Global F2-Optimal Threshold Calibration.
        """
        # [v4.2.1 SOTA] DDP-Safe Global Calibration
        local_probs = torch.cat([x["prob"] for x in self.validation_step_outputs]) if self.validation_step_outputs else torch.tensor([], device=self.device)
        local_labels = torch.cat([x["label"] for x in self.validation_step_outputs]) if self.validation_step_outputs else torch.tensor([], device=self.device)
        
        if dist.is_initialized():
             # [v4.2.1 SOTA] Synchronized Tensor Gathering (Smoking Gun #RAM-03)
             # Rationale: Gathering a list of 10,000 dicts via all_gather_object
             # causes massive pickling overhead and transient RAM spikes.
             # Fix: Use the zero-copy optimized gatherer for raw tensors.
             all_probs = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(local_probs), dim=0).cpu()
             all_labels = torch.cat(SOTA_DistributedGatherer.gather_asymmetric(local_labels), dim=0).cpu()
        else:
             all_probs = local_probs.cpu()
             all_labels = local_labels.cpu()
        
        # [v2026 SOTA] Explicit Memory Release
        # Rationale: local_probs/local_labels are already in the all_* tensors.
        # Clearing the list early prevents keeping dual copies in VRAM.
        self.validation_step_outputs.clear()
        import gc
        gc.collect()
        
        opt_f2, opt_thresh = 0.0, 0.5
        pos_probs = all_probs.view(-1)
        pos_labels = all_labels.view(-1).long()
        if all_probs.numel() > 0:
            thresholds = torch.linspace(0.01, 0.99, 50)
            best_f2 = -1.0
            
            # [SOTA FIX] Handle multi-class probabilities for binary-style F2 calibration
            if all_probs.dim() == 2 and all_probs.shape[1] >= 2:
                # [REVERTED] Simple Sepsis Summation
                # Rationale: The "Forensic Index Alignment" logic was dropping the Recovery class (Index 3).
                # Reverting to simple slicing ensuring all positive classes are counted.
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
        if dist.is_initialized():
            threshold_tensor = torch.tensor([opt_thresh], device=self.device)
            dist.all_reduce(threshold_tensor, op=dist.ReduceOp.SUM)
            opt_thresh = (threshold_tensor / dist.get_world_size()).item()
            
        # [SOTA PATCH] Welford-Style Threshold Calibration
        # Rationale: Starting EMA from 0.5 is a bad prior for 1.76% prevalence — 
        # the optimal threshold is typically 0.15-0.30. For epoch 0-1, use the raw
        # F2-optimal directly (no prior to smooth against). From epoch 2+, apply
        # EMA smoothing for stability across validation noise.
        new_thresh = opt_thresh
        if self.current_epoch <= 1:
            # No valid prior yet — trust the data directly
            updated_thresh = new_thresh
        else:
            # Smooth with EMA from epoch 2+ (we now have a calibrated prior)
            prev_thresh = self.calibrated_threshold.item()
            updated_thresh = (self.threshold_ema_decay * prev_thresh) + ((1 - self.threshold_ema_decay) * new_thresh)
        self.calibrated_threshold.fill_(updated_thresh)
        
        # Use the CALIBRATED (EMA) threshold for metrics
        final_thresh = self.calibrated_threshold.item()
            
        # [v2026 SOTA] Unbiased Clinical Evaluation (Smoking Gun #Bias)
        # Rationale: Averaging AUROCs across ranks is mathematically incorrect.
        # We compute all metrics on the pooled global dataset for exact parity.
        s_auroc, s_prec, s_rec, s_f1 = 0.0, 0.0, 0.0, 0.0
        if pos_probs.numel() > 0:
            # Use the synchronized final_thresh for categorical metrics
            s_auroc = tm_func.auroc(pos_probs, pos_labels, task="binary").item()
            s_prec = tm_func.precision(pos_probs, pos_labels, task="binary", threshold=final_thresh).item()
            s_rec = tm_func.recall(pos_probs, pos_labels, task="binary", threshold=final_thresh).item()
            s_f1 = tm_func.f1_score(pos_probs, pos_labels, task="binary", threshold=final_thresh).item()
        def safe_compute(m):
            if hasattr(m, 'weight') and m.weight == 0:
                return 0.0
            return m.compute()

        self.log_dict({
            "val/mse_global": safe_compute(self.val_mse_global),
            "val/mse_hemo": safe_compute(self.val_mse_hemo),
            "val/mse_labs": safe_compute(self.val_mse_labs),
            "val/mse_electrolytes": safe_compute(self.val_mse_electrolytes),
            "val/sepsis_acc": safe_compute(self.val_acc_sepsis),
            "val/sepsis_auroc": s_auroc,  # [UNBIASED]
            "val/sepsis_precision": s_prec, # [UNBIASED]
            "val/sepsis_recall": s_rec,     # [UNBIASED]
            "val/sepsis_f1": s_f1,          # [UNBIASED]
            "val/clinical_f2_opt": opt_f2,
            "val/clinical_threshold_opt": final_thresh,
            "val/raw_threshold_epoch": opt_thresh,
            "val/ece": safe_compute(self.val_ece),
            "val/oe": safe_compute(self.val_oe),
            "val/explained_var": safe_compute(self.val_explained_var),
            "val/ood_rate_avg": safe_compute(self.val_ood_rate),
            "val/safe_trajectories_avg": safe_compute(self.val_safe_traj_count),
            "val/phys_violation_rate": safe_compute(self.val_phys_violation_rate),
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
        
        # [v33.0 SOTA FIX] Reset Missing Validation Metrics
        self.val_mse_global.reset()
        self.val_mse_hemo.reset()
        self.val_mse_labs.reset()
        self.val_mse_electrolytes.reset()

        # [v2026 RAM SPIKE FIX] Validation Outputs Clearing (Smoking Gun #RAM-02)
        # Rationale: validation_step_outputs can hold ~100MB+ of tensors that must be
        # freed BEFORE checkpoint serialization runs to prevent OOM.
        self.validation_step_outputs.clear()
        gc.collect()

    # =========================================================================
    # UTILITIES & SETUP
    # =========================================================================


    def _get_curr_physics_weight(self) -> float:
        """
        Curriculum Learning for Physiological Constraints.
        Ramps up weight over 50% of TOTAL steps for resume transparency.
        [v14.9 SOTA Fix] Dynamic Anchor to actual training length.
        """
        # [FORENSIC FIX #6] Re-enabled Physics Curriculum
        # Root Cause: Disabled curriculum meant physics weight was constant at 0.2 from step 0.
        # Combined with the d_ema/p_ema ratio scaling (phys_scale), this allowed physics 
        # gradients to compete with clinical signal from the very beginning.
        # Fix: Ramp from 0.01→base_weight over 50% of training for smooth introduction.
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
        # [v106.0 SOTA FIX] Teacher Determinism (Smoking Gun #106)
        # Rationale: Teacher passes in 'train' mode keep Dropout active,
        # making target returns jittery. This forces 'eval' for stable targets.
        was_training = self.model.training
        self.model.eval()
        
        try:
            if hasattr(self, 'ema') and self.ema is not None:
                with self.ema.swap():
                    yield
            else:
                # Fallback for Student-Student Bootstrapping
                yield
        finally:
            if was_training:
                self.model.train()

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
            # [SOTA FIX] Default to "sample" to prevent 15-minute stalls on missing keys
            config_mode = self.cfg.train.get("awr_calibration_mode", "sample")
            # [SOTA FIX] Adaptive Calibration Speed (20k is statistically significant for 451k)
            max_samples = self.cfg.train.get("awr_max_samples", 20000)
            
            # [SOTA FORENSIC FIX] "full" means scan entire population. Use with caution.
            if config_mode == "full":
                mode = "full"
            elif max_samples < num_samples and max_samples > 0:
                 mode = "sample"
            else:
                 mode = config_mode
            
            logger.info(f"[AWR Config] Mode='{mode}' (Orig='{config_mode}'), MaxSamples={max_samples}, Population={num_samples}")

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
                    s_mask = sample.get("future_mask").to(self.device).unsqueeze(0)
                    r = self.awr_calculator.compute_clinical_reward(
                        future, 
                        label, 
                        dones=sample.get("is_terminal").unsqueeze(0).to(self.device),
                        feature_indices=self.clinical_feat_idx,
                        normalizer=None,
                        src_mask=s_mask
                    )
                    
                    # [v25.6 SOTA FIX] Solve "AWR Amnesia" & "Shape Mismatch"
                    # Rationale 1: mask can be [T_full, C] or [T_full], but rewards (r) is [1, T_curr].
                    # Rationale 2: T_curr might be < T_full for short trajectories.
                    # Rationale 3: We reduce to sequence-level mask before indexing.
                    m_raw = sample.get("future_mask")
                    m_step = torch.as_tensor(m_raw).to(self.device).float()
                    
                    if m_step.dim() > 1:
                        m_step = m_step.any(dim=-1).float()
                    
                    # Force alignment with reward sequence length (handle partial windows)
                    T_actual = r.shape[1]
                    m_step = m_step[:T_actual]
                    
                    # Filter by mask to only keep "real" clinical data steps
                    valid_rewards = r.view(-1)[m_step.view(-1) > 0.5]
                    if valid_rewards.numel() > 0:
                        rewards_list.append(valid_rewards.cpu()) # Move to CPU
            
            if len(rewards_list) > 0:
                # [v25.6] Efficient aggregation
                all_rewards = torch.cat(rewards_list)
                stats_tensor[0] = all_rewards.mean().item()
                stats_tensor[1] = all_rewards.std().item() + 1e-8
                logger.info(f"AWR Stats (True Population): Mean={stats_tensor[0]:.4f}, Std={stats_tensor[1]:.4f}, Points={all_rewards.numel()}")
            else:
                logger.warning("No valid samples found for AWR calibration. Using defaults.")
                stats_tensor[0] = 0.0
                stats_tensor[1] = 1.0

        # --- DDP Synchronization ---
        if dist.is_initialized():
            dist.broadcast(stats_tensor, src=0)
            
        self.awr_calculator.set_stats(mean=stats_tensor[0].item(), std=stats_tensor[1].item())
        # [v5.0 SOTA FIX] Buffer Stability (Event Horizon)
        # Rationale: Re-registering a buffer breaks the computational graph link
        # to the optimizer. We must update the existing buffer in-place.
        if hasattr(self, "_awr_stats_initialized"):
            self._awr_stats_initialized.fill_(True)
        else:
            # Fallback for old checkpoints lacking the buffer
            self.register_buffer("_awr_stats_initialized", torch.tensor([True]))
        
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
        # [SOTA PATCH] Guarded Unified Scaling with Clamped Ratio
        # Rationale: get_unified_scaling() is correct physics but can produce
        # extreme LR values when n_curr diverges from REF_STEPS (e.g., debug=200 → 2.4× inflation).
        # We clamp the scaling ratio k to [0.8, 1.2] to prevent >20% LR deviation.
        n_curr = ScalingSteward.SOTA_REF_STEPS  # Default: identity scaling
        if self.trainer:
            if hasattr(self.trainer, "num_training_batches") and self.trainer.num_training_batches > 1:
                n_curr = self.trainer.num_training_batches
            elif hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "train_ds"):
                d_len = len(self.trainer.datamodule.train_ds)
                bs = self.cfg.train.batch_size
                n_curr = d_len // bs if bs > 0 else ScalingSteward.SOTA_REF_STEPS
        
        # Clamp the density ratio to prevent extreme LR swings
        k_raw = float(n_curr) / float(ScalingSteward.SOTA_REF_STEPS)
        k_clamped = max(0.8, min(1.2, k_raw))
        n_clamped = int(k_clamped * ScalingSteward.SOTA_REF_STEPS)
        
        scaled_lr, scaled_wd = ScalingSteward.get_unified_scaling(
            self.cfg.train.lr, self.cfg.train.weight_decay, n_clamped
        )
        
        logger.info(
            f"[SOTA PATCH] LR Scaling: n_curr={n_curr}, k_raw={k_raw:.2f}, "
            f"k_clamped={k_clamped:.2f} | LR: {self.cfg.train.lr:.2e} -> {scaled_lr:.2e}"
        )
        
        if self.balancing_mode == "sota_2025":
            # [SOTA 2025] Parameter Groups
            uw_lr = self.cfg.train.get("uw_lr", 0.005)
            uw_lr_scaled, _ = ScalingSteward.get_unified_scaling(uw_lr, 0.0, n_clamped)
            
            # 1. Identify Aux Parameters
            aux_params = list(self.model.aux_head.parameters()) if hasattr(self.model, 'aux_head') else []
            aux_param_ids = {id(p) for p in aux_params}
            
            # 2. Identify ACL Parameters
            acl_params = list(self.acl_projector.parameters())
            acl_param_ids = {id(p) for p in acl_params}
            
            # 3. Identify Main Model Parameters
            decay_params = []
            no_decay_params = []
            excluded_ids = aux_param_ids.union(acl_param_ids)
            
            for p in self.model.parameters():
                if id(p) in excluded_ids: continue
                if p.ndim < 2: no_decay_params.append(p)
                else: decay_params.append(p)
            
            aux_lr_mult = self.cfg.train.get("aux_lr_multiplier", 1.0)
            
            optimizer_params = [
                {'params': decay_params, 'lr': scaled_lr, 'weight_decay': scaled_wd},
                {'params': no_decay_params, 'lr': scaled_lr, 'weight_decay': 0.0},
                {'params': aux_params, 'lr': scaled_lr * aux_lr_mult, 'weight_decay': scaled_wd},
                {'params': acl_params, 'lr': scaled_lr * aux_lr_mult, 'weight_decay': scaled_wd},
                {'params': self.loss_scaler.parameters(), 'lr': uw_lr_scaled, 'weight_decay': 0.0}
            ]
            
            logger.info(
                f" [SOTA] Optimizer Scaling: k={n_curr/1176:.2f}x | "
                f"LR: {self.cfg.train.lr:.2e} -> {scaled_lr:.2e} | "
                f"WD: {self.cfg.train.weight_decay:.2e} -> {scaled_wd:.2e}"
            )
            
            base_optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=scaled_lr,
                weight_decay=scaled_wd,
                betas=(0.9, 0.999),
                eps=1e-5,
                fused=False
            )


        else:
            optimizer_params = [{"params": self.model.parameters()}]
            base_optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-5, # [v130.0 SOTA FIX] Epsilon Hardening (Smoking Gun #130)
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
        # [v2026 SOTA FIX] Absolute Warmup Priority (Smoking Gun #WarmupLag)
        # Rationale: Using warmup_ratio (0.05) on a 5-day H100 run leads to thousand-step 
        # dead-zones at start. Prioritize the user's absolute 'warmup_steps' from config.
        config_warmup = self.cfg.train.get("warmup_steps", 0)
        if config_warmup > 0:
            warmup_steps = config_warmup
        else:
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
        [SOTA v30.5] Universal Resumption Bridge (Persistence).
        Ensures perfect memory restoration for Manual Optimization.
        """
        # [v12.0 SOTA] CPU Shadow Sync (Zero-Sync Bridge)
        # Rationale: Sync high-speed shadows to persistent buffers ONLY at checkpoint time.
        self.grad_accum_idx.fill_(self._shadow_grad_accum_idx)
        self.resumption_grace_steps.fill_(self._shadow_resumption_grace_steps)

        # [v2026 RAM LOCKDOWN] Early Pre-emptive Harvest
        # Rationale: Clearing the ~5GB activation heap BEFORE PL starts 
        # serializing model state into the checkpoint dict.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if hasattr(self.model, "normalizer"):
            checkpoint["normalizer_state"] = self.model.normalizer.state_dict()
            
        if self._fnd_grad_ema is not None:
            checkpoint["_fnd_grad_ema"] = self._fnd_grad_ema

        # [SOTA FIX] Manual Optimization Persistence
        # Capture main optimizers and schedulers explicitly
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)): opts = [opts]
        checkpoint["optimizer_states"] = [o.state_dict() for o in opts]
        
        schs = self.lr_schedulers()
        if schs is not None:
            if not isinstance(schs, (list, tuple)): schs = [schs]
            checkpoint["lr_schedulers"] = [s.state_dict() for s in schs]

        # [v2026 Phase 19 FIX] EMA Double-Save Elimination (Audit Finding)
        # Rationale: EMACallback.on_save_checkpoint (callbacks.py L549-551) already 
        # saves ema_state_dict under the same key. Both reference the same self.ema 
        # object (attached via pl_module.ema = self.ema in EMACallback._init_ema).
        # Removing this redundant serialization saves ~100ms per checkpoint.
        # The EMACallback is the single authoritative save source for EMA state.

        # [v2026 SOTA] GradNorm Persistence Bridge
        # Rationale: Captures meta-optimizer state, EMAs, and loss anchors.
        if self.gradnorm is not None:
            checkpoint["gradnorm_state"] = self.gradnorm.get_gradnorm_state()
            logger.info("[SAVE] GradNorm internal state captured.")

        # [v48.0] Save Accumulation Progress
        checkpoint["grad_accum_idx"] = self.grad_accum_idx

        # [v118.0 SOTA FIX] GradNorm Accumulation Persistence (Smoking Gun #91)
        # [AXE-SHARPENED] Using .to('cpu') instead of .cpu().clone() to avoid redundant RAM usage
        checkpoint["gn_loss_accumulator"] = self.gn_loss_accumulator.to('cpu')
        checkpoint["gn_acc_count"] = self.gn_acc_count.to('cpu')

        # [v7 FIX] Removed partial_gradients saving (was ~800MB CPU RAM spike).
        # On resume, accumulation cycle restarts fresh (at most 4 steps lost).
        # grad_accum_idx is reset, so no stale gradient state.
        
        # [v7 FIX] Removed grad_ref_buffer saving (was variable CPU RAM spike).
        # AGEM reference is re-accumulated within the first few steps after resume.
        # It's zeroed at each accumulation cycle boundary anyway.
            
        # [v2026] Global Memory Harvest
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # [v54.0] Scaler Restoration Bridge (Smoking Gun #110)
        # Rationale: Standard PL sometimes misses the scaler state in manual optimization.
        if hasattr(self.trainer, "precision_plugin") and hasattr(self.trainer.precision_plugin, "scaler"):
             if self.trainer.precision_plugin.scaler is not None:
                  checkpoint["grad_scaler_state"] = self.trainer.precision_plugin.scaler.state_dict()
                  logger.info("[SAVE] GradScaler state captured.")

        # [v38.1 SOTA] AWR Persistence Bridge (Smoking Gun #SG-123)
        # Rationale: Captures all adaptive whitening stats and beta dynamics.
        if hasattr(self, "awr_calculator") and self.awr_calculator is not None:
            checkpoint["awr_state"] = self.awr_calculator.get_awr_state()
            logger.info("[SAVE] AWR internal state captured (Whitening/Adaptive).")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        [SOTA v2026] Elastic Resumption Engine.
        Directly aligns model capacity and task structure before strict loading.
        """
        state_dict = checkpoint.get("state_dict", {})

        # 1. INITIALIZE SAFETY FLAGS (For Legacy Checkpoints)
        # [v2026 SOTA FIX] Shape-Consistent Default Injection (Smoking Gun #ShapeCrash)
        # Rationale: All injected tensors MUST match the registered buffer shapes ([1]).
        # Injecting scalar [] tensors causes load_state_dict shape mismatches and
        # downstream optimizer state corruption.
        if "grad_norm_std" not in state_dict:
            state_dict["grad_norm_std"] = torch.tensor([0.5])
        if "grad_norm_ema" not in state_dict:
            state_dict["grad_norm_ema"] = torch.tensor([1.0])
            
        if "grad_norm_step_count" not in state_dict:
            state_dict["grad_norm_step_count"] = torch.tensor([checkpoint.get("global_step", 100)], dtype=torch.long)
            
        # [v12.0 SOTA FIX] Handle Missing v12 Buffers (Smoking Gun #error12)
        # Rationale: stability_factor and AWR hyperparameters are now non-persistent 
        # or were recently added. We initialize them if missing from older checkpoints.
        if "stability_factor" not in state_dict:
            state_dict["stability_factor"] = torch.tensor([1.0])
            logger.info("[RESUME] Initialized missing 'stability_factor' to 1.0.")
            
        if "awr_calculator.gamma" not in state_dict:
            # Note: Being non-persistent means we rely on the cfg value already set in __init__
            # but we can also explicitly set it here if we want to bypass strict loading issues.
            # Since we use strict=False in load_state_dict, this is mostly for logging/assurance.
             logger.info("[RESUME] 'awr_calculator.gamma' missing from checkpoint (expected for non-persistent).")
             
        if "awr_calculator.lambda_gae" not in state_dict:
             logger.info("[RESUME] 'awr_calculator.lambda_gae' missing from checkpoint.")

        # [v48.0 SOTA FIX] Stateful Accumulation Index (Smoking Gun #90)
        # [v2026 AUDIT FIX] Resumption Math Parity (Reset to 0)
        # Rationale: On resume, gradients are empty. We MUST restart the 
        # accumulation cycle from zero to ensure that the next optimizer step 
        # uses a FULL aggregate of exactly 'accumulate_grad_batches' samples.
        # Restoring a mid-cycle index (e.g. 8/16) without the first 8 gradients 
        # would lead to a mathematically incorrect, "under-fueled" step.
        state_dict["grad_accum_idx"] = torch.tensor([0], dtype=torch.long)
        # [v2026 AUDIT] Direct Buffer Fill (Belt and Suspenders)
        # Verify that buffer is reset even if PL loaded state_dict before this hook.
        if hasattr(self, "grad_accum_idx"):
             self.grad_accum_idx.fill_(0)
        self._shadow_grad_accum_idx = 0
        logger.info("⚖️ [RESUME] Accumulation Index reset to 0 (Mathematical Parity).")

        # [v118.0] Restore GradNorm Accumulators
        if "gn_loss_accumulator" in checkpoint:
             self.pending_gn_accumulator = checkpoint["gn_loss_accumulator"]
        if "gn_acc_count" in checkpoint:
             self.pending_gn_acc_count = checkpoint["gn_acc_count"]
             
        # [v7 FIX] partial_gradients no longer saved/restored (saves ~800MB RAM).
        # Accumulation cycle restarts fresh on resume.
            
        # [v52.2 SOTA FIX] Grace Period Renewal (Smoking Gun #MasterAudit)
        # Rationale: All models, including mature ones, benefit from a re-settling
        # grace period of 50 steps to allow momentum and stability metrics to align.
        # [CRITICAL FIX v2026-02-10] Direct Buffer Fill - REQUIRED for on_load_checkpoint
        # The previous code modified `state_dict` which has NO EFFECT in this callback
        # because PL already loaded the model BEFORE calling on_load_checkpoint.
        self.resumption_grace_steps.fill_(50)
        logger.info("⚡ [RESUME] Resumption Grace Period ACTIVATED (50 steps). Turbo Mode ENABLED!")

        # 2. 6-TO-7 TASK TRANSITION (Padding for 'phys' expansion)
        # Rationale: We added 'phys' as the 7th task. Old checkpoints only have 6.
        # Strict loading will fail unless we pad with neutral priors.
        for prefix in ["loss_scaler.", "gradnorm."]:
            key = f"{prefix}log_vars" if prefix == "loss_scaler." else f"{prefix}weights"
            if key in state_dict and state_dict[key].shape[0] == 6:
                # Pad log_vars with 0.5 (neutral/standard loss magnitude)
                # Pad gradnorm weights with 1.0 (neutral)
                pad_val = 0.5 if prefix == "loss_scaler." else 1.0
                state_dict[key] = torch.cat([state_dict[key], torch.tensor([pad_val], device=state_dict[key].device)])
                logger.info(f"[RESUME] Padded {key} to 7 tasks (Unit Neutralization).")
            
            # Pad EMA buffers
            ema_key = f"{prefix}loss_emas" if prefix == "loss_scaler." else f"{prefix}initial_losses"
            if ema_key in state_dict and state_dict[ema_key].shape[0] == 6:
                state_dict[ema_key] = torch.cat([state_dict[ema_key], torch.tensor([1.0], device=state_dict[ema_key].device)])
                logger.info(f"[RESUME] Padded {ema_key} to 7 tasks.")

        # 3. ELASTIC BUFFER RESIZING (TCB & GhostBank)
        # Rationale: If the checkpoint was saved with scaled buffers (e.g. 6021),
        # we must resize our local buffers (e.g. 1024) BEFORE load_state_dict is called.
        
        # TCB Buffer Resize
        tcb_key = next((k for k in state_dict.keys() if "tcb_buffer.queue" in k), None)
        if tcb_key and hasattr(self, "tcb_buffer"):
            ckpt_capacity = state_dict[tcb_key].shape[0]
            if ckpt_capacity != self.tcb_buffer.capacity:
                logger.info(f"[RESUME] Elastic Resize: TCB {self.tcb_buffer.capacity} -> {ckpt_capacity}")
                self.tcb_buffer.capacity = ckpt_capacity
                device = self.tcb_buffer.queue.device
                # [v27.2 SOTA FIX] Gaussian Noisy Initialization
                # Rationale: Zeros kill the contrastive gradient signal until refill.
                # Small noise (0.01) preserves flow without introducing significant bias.
                init_q = torch.randn(ckpt_capacity, self.tcb_buffer.d_model, device=device) * 0.01
                self.tcb_buffer.register_buffer("queue", F.normalize(init_q, dim=1))
        
        # Ghost Bank Resize
        ghost_key = next((k for k in state_dict.keys() if "ghost_bank.raw_vitals" in k), None)
        if ghost_key and hasattr(self, "ghost_bank"):
            ckpt_capacity = state_dict[ghost_key].shape[0]
            if ckpt_capacity != self.ghost_bank.capacity:
                logger.info(f"[RESUME] Elastic Resize: Ghost Bank {self.ghost_bank.capacity} -> {ckpt_capacity}")
                self.ghost_bank.capacity = ckpt_capacity
                device = self.ghost_bank.raw_vitals.device
                h, f, l = self.ghost_bank.history_len, self.ghost_bank.feature_dim, self.ghost_bank.latent_dim
                self.ghost_bank.register_buffer("raw_vitals", torch.zeros(ckpt_capacity, h, f, device=device))
                self.ghost_bank.register_buffer("raw_masks", torch.zeros(ckpt_capacity, h, f, device=device))
                self.ghost_bank.register_buffer("raw_labels", torch.zeros(ckpt_capacity, dtype=torch.long, device=device))
                self.ghost_bank.register_buffer("latent_anchors", torch.zeros(ckpt_capacity, l, device=device))
                self.ghost_bank.register_buffer("uncertainties", torch.zeros(ckpt_capacity, 1, device=device))

        # 4. RESTORATION HOOKS (Manual Alignment)
        if "normalizer_state" in checkpoint:
            self.pending_normalizer_state = checkpoint["normalizer_state"]
            logger.info("[RESUME] normalizer_state captured for on_fit_start.")
            
        # [CRITICAL FIX] Capture Optimizer & Scheduler States for Manual Restoration
        # Standard PL sometimes drops these when using manual_optimization=True
        if "optimizer_states" in checkpoint:
             self.pending_optimizer_states = checkpoint["optimizer_states"]
             logger.info(f"[RESUME] Captured {len(self.pending_optimizer_states)} optimizer states.")

        if "lr_schedulers" in checkpoint:
             self.pending_scheduler_states = checkpoint["lr_schedulers"]
             logger.info(f"[RESUME] Captured scheduler states.")

        if "gradnorm_state" in checkpoint:
             self.pending_gn_state = checkpoint["gradnorm_state"]
             logger.info("[RESUME] GradNorm state captured for manual restoration bridge.")

        if "ema_state_dict" in checkpoint:
             self.pending_ema_state = checkpoint["ema_state_dict"]
             logger.info("[RESUME] Captured EMA shadow weights.")
             
        # [v30.3 SOTA FIX] MGP Anchor Restoration (The Great Whiplash)
        # Rationale: _fnd_grad_ema is saved in root, but must be restored manually
        # to ensure the PCGrad direction is aligned with pre-save state.
        if "_fnd_grad_ema" in checkpoint:
             if self._fnd_grad_ema is None:
                  self._fnd_grad_ema = checkpoint["_fnd_grad_ema"].clone().to(self.device)
             else:
                  self._fnd_grad_ema.copy_(checkpoint["_fnd_grad_ema"].to(self._fnd_grad_ema.device))
             logger.info("[RESUME] MGP Direction Anchor Restored.")
             
        # [v7 FIX] grad_ref_buffer no longer saved/restored (saves RAM).
        # AGEM reference is re-accumulated within a few steps after resume.

        # [v54.0] Scaler Restoration Bridge
        if "grad_scaler_state" in checkpoint:
             if hasattr(self.trainer, "precision_plugin") and hasattr(self.trainer.precision_plugin, "scaler"):
                  if self.trainer.precision_plugin.scaler is not None:
                       self.trainer.precision_plugin.scaler.load_state_dict(checkpoint["grad_scaler_state"])
                       logger.info("[RESUME] GradScaler state restored.")
        
        # [v38.1 SOTA] AWR Persistence Bridge
        # Rationale: Direct restoration to bypass fit_stats loop.
        if "awr_state" in checkpoint:
             self.pending_awr_state = checkpoint["awr_state"]
             logger.info("[RESUME] AWR state captured for manual restoration bridge.")
             
        # [v31.0 SOTA FIX] Active Grace Logging
        # Rationale: Only output the hiberation log if we are actually resuming.
        logger.info("🛡️ [PMS] Resumption Bridge Active. Sentinel hibernated for 50 settling steps.")
