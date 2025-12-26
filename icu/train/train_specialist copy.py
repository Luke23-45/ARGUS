"""
scripts/train_specialist.py
--------------------------------------------------------------------------------
Phase 2: Specialist (APEX-MoE) Training Engine (Hydra Refactor).

This script executes the "Specialization" phase of the training pipeline.
It transforms a pre-trained Generalist (Phase 1) into a Mixture-of-Experts.

Workflow:
1. **Bootstrapping**: Loads the Generalist checkpoint to initialize weights.
2. **Phase-Locking**: Freezes the History Encoder and Sepsis Router.
3. **Gradient Surgery**: Training loop uses Hard Gating to split batches based
   on Ground Truth outcome labels (Stable vs Crash).
4. **Soft-Gating Inference**: Validation loop uses Probabilistic Blending
   to evaluate the combined system.

Robustness Features:
- Class Imbalance Handling via `crash_weight`.
- Differential Gradient updates (only Experts evolve).
- Architecture-Aware Logging.
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # PyTorch 2.0+ unified AMP
import numpy as np
from tqdm.auto import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Project Imports ---
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
# Note: 'apex_moe_planner.py' must be restored to 'icu/models/'
from icu.models.apex_moe_planner import APEX_MoE_Planner
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn, ensure_data_ready
from icu.utils.train_utils import (
    EMA,
    RotationalSaver as AtomicSaver, # Renaming to match usage if needed, or update usage
    setup_logger,
    set_seed,
    count_parameters,
    UnifiedLogger,
    configure_robust_optimizer
)
from icu.utils.advantage_calculator import ICUAdvantageCalculator
# Ensure AtomicSaver alias works if not present in utils (AtomicSaver was renamed/merged to RotationalSaver?)
# Checking train_generalist, it imported RotationalSaver.
# train_specialist used AtomicSaver. I need to be sure.
# In icu/utils/train_utils.py (formerly training_utils.py), check if AtomicSaver exists.


# ==============================================================================
# SPECIALIZED TRAINING LOOP (GRADIENT SURGERY)
# ==============================================================================
def train_one_epoch_apex(
    model: APEX_MoE_Planner,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    awr_calculator: ICUAdvantageCalculator,  # Added
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """
    Executes one epoch of APEX training with SOTA practices.
    
    Features:
    - Device-aware AMP (works on CPU and CUDA)
    - Gradient Surgery for expert specialization
    - Gradient norm logging for debugging
    """
    model.train()
    
    # Metrics accumulator
    stats = {
        "loss": [],
        "loss_stable": [],
        "loss_crash": [],
        "n_stable": [],
        "n_crash": [],
        "grad_norm": []  # Added for gradient health monitoring
    }
    
    # Device-aware autocast
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    pbar = tqdm(dataloader, desc=f"Ep {epoch} [APEX Tuning]", leave=False)
    
    for batch in pbar:
        # 1. Move Data
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # 2. Advantage Calculation (SOTA v2.0)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=device_type, enabled=(device_type == "cuda")):
            with torch.no_grad():
                # We need advantage weights BEFORE the gated forward 
                # to avoid redundant expert passes.
                # We get the baseline pred_value from the frozen generalist head.
                past_norm = model.normalize(batch["observed_data"])
                _, global_ctx = model.encoder(past_norm, batch["static_context"])
                pred_value = model.value_head(global_ctx).squeeze(-1)
                
                clinical_reward = awr_calculator.compute_reward_torch(batch["future_data"], batch["outcome_label"])
                advantage = clinical_reward - pred_value
                weights, ess = awr_calculator.calculate_weights(advantage)
            
            # forward() splits the batch internally and returns breakdown
            # It now accepts awr_weights for specialized gradient surgery
            loss_dict = model(batch, awr_weights=weights)
            loss = loss_dict['loss']
            
            # Add value alignment loss (even if frozen, we log it)
            # In Phase 2, we usually don't train ValueNet, but we monitor stability.
            v_loss = F.mse_loss(pred_value, clinical_reward)
            
        # 3. Backward
        scaler.scale(loss).backward()
        
        # 4. Unscale and log gradient norm BEFORE clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        # 5. Optimizer Step
        scaler.step(optimizer)
        scaler.update()
        
        # 6. LR Step (per-batch for warmup schedulers)
        scheduler.step()
        
        # 7. EMA Update
        ema.update(model)
        
        # 8. Logging Data Extraction
        current_loss = loss.item()
        s_loss = loss_dict['stable_loss'].item()
        c_loss = loss_dict['crash_loss'].item()
        n_stab = loss_dict['stable_batch_size'].item()
        n_crash = loss_dict['crash_batch_size'].item()
        
        stats["loss"].append(current_loss)
        if n_stab > 0: stats["loss_stable"].append(s_loss)
        if n_crash > 0: stats["loss_crash"].append(c_loss)
        stats["n_stable"].append(n_stab)
        stats["n_crash"].append(n_crash)
        stats["grad_norm"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        stats["ess"] = stats.get("ess", []) + [ess]
        
        # Dynamic Progress Bar
        pbar.set_postfix(
            loss=f"{current_loss:.3f}",
            ess=f"{ess:.1%}",
            gn=f"{stats['grad_norm'][-1]:.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}"
        )

    # Aggregate stats carefully
    avg_stats = {}
    for k, v in stats.items():
        if len(v) > 0:
            avg_stats[k] = np.mean(v)
        else:
            avg_stats[k] = 0.0
            
    return avg_stats


@torch.no_grad()
def validate_apex(
    ema: EMA, 
    model: APEX_MoE_Planner, 
    dataloader: DataLoader, 
    device: torch.device, 
    logger: logging.Logger, 
    sample_steps: int = 50,
    limit_batches: int = 50  # Gap 17: Now configurable
) -> tuple:
    """
    Performs Phase 2 Validation using Soft Gating and EMA weights.
    Evaluating how well the Combined Team performs on clinical metrics (MSE/MAE).
    
    Args:
        limit_batches: Number of batches to validate on (for speed)
    """
    ema.apply_shadow(model)
    model.eval()
    mse_errors = []
    mae_errors = []
    
    pbar = tqdm(dataloader, desc="Validation", total=min(limit_batches, len(dataloader)), leave=False)
    for i, batch in enumerate(pbar):
        if i >= limit_batches: break
        
        # 1. Move Data
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        gt_future = batch['future_data']
        
        # 2. Sample (Soft Gated Inference)
        # Uses P(Crash) to blend experts
        pred_future = model.sample(batch, num_steps=sample_steps)
        
        # 3. Compute Errors
        mse = ((pred_future - gt_future) ** 2).mean().item()
        mae = (pred_future - gt_future).abs().mean().item()
        mse_errors.append(mse)
        mae_errors.append(mae)
        
        pbar.set_postfix(mse=f"{mse:.4f}", mae=f"{mae:.4f}")
     
    ema.restore(model)
    return np.mean(mse_errors), np.mean(mae_errors)


# ==============================================================================
# MAIN ENGINE (HYDRA)
# ==============================================================================
@hydra.main(version_base=None, config_path="../../conf", config_name="specialist")
def main(cfg: DictConfig):
    # 1. Setup Environment - Use configurable paths from cfg.paths
    # -------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.info(f"Hydra CWD: {os.getcwd()}")
    
    # Resolve all directory paths from config
    output_dir = Path(cfg.paths.output_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.log_dir)
    backup_dir = Path(cfg.paths.backup_dir) if cfg.paths.backup_dir else None
    
    # Create all directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger to the configured log directory
    setup_logger(cfg.run_name, str(log_dir))
    
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Unified Logger (WandB + TB) - use output_dir for tensorboard
    exp_logger = UnifiedLogger(cfg, str(output_dir))
    
    logger.info(f"[START] Starting APEX-MoE Specialist Training: {cfg.run_name}")
    logger.info(f" Mode: Phase 2 (Fine-Tuning)")
    logger.info(f" Output Dir: {output_dir}")
    logger.info(f" Checkpoint Dir: {checkpoint_dir}")
    logger.info(f" Log Dir: {log_dir}")
    logger.info(f" Backup Dir: {backup_dir}")
    logger.info(f" Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Initialize Data
    logger.info("[DATA] Ensuring Data is Ready (Orchestrator)...")
    ensure_data_ready(
        dataset_dir=cfg.dataset.dataset_dir,
        hf_repo_id=cfg.dataset.hf_repo,
        force_download=cfg.dataset.get("force_download", False)
    )
    
    logger.info("[DATA] Initializing Dataset...")
    # SOTA dataset handles missing files automatically via 'hf_repo' arg if needed
    train_ds = ICUSotaDataset(
        dataset_dir=cfg.dataset.dataset_dir,
        hf_repo_id=cfg.dataset.hf_repo,
        split="train",
        augment_noise=cfg.dataset.augment_noise
    )
    val_ds = ICUSotaDataset(
        dataset_dir=cfg.dataset.dataset_dir,
        split="val",
        augment_noise=0.0
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=robust_collate_fn,
        worker_init_fn=set_seed(cfg.seed, worker=True)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=robust_collate_fn
    )

    # 3. Model Bootstrapping (The Critical Step)
    logger.info("[MODEL] Bootstrapping APEX-MoE Model...")
    
    # A. Init Empty Generalist Structure
    base_cfg = ICUConfig(**cfg.model)
    generalist = ICUUnifiedPlanner(base_cfg)
    
    # B. Load Pretrained Weights
    # For Phase 2, pretrained_path IS required.
    pretrained_path = cfg.training.pretrained_path
    if pretrained_path is None or not os.path.exists(pretrained_path):
        logger.error(f"[ERROR] Phase 1 Checkpoint not found at: {pretrained_path}. Please verify 'training.pretrained_path' in config.")
        sys.exit(1)
        
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    # Robustly handle different save structures
    state_dict = checkpoint.get('model', checkpoint)
    
    generalist.load_state_dict(state_dict, strict=True)
    logger.info(" Loaded Phase 1 Weights.")
    
    # C. Initialize Specialist (Wraps Generalist)
    # This clones the backbone into two experts and freezes the encoder/router
    # C. Initialize Specialist (Wraps Generalist)
    # This clones the backbone into two experts and freezes the encoder/router
    crash_weight = cfg.training.get("crash_weight", 5.0)
    lambda_reg = cfg.training.get("lambda_reg", 0.01) # [SOTA] Default tether weight
    model = APEX_MoE_Planner(generalist, crash_loss_weight=crash_weight, lambda_reg=lambda_reg)
    model.to(device)
    
    if cfg.compile_model and torch.__version__ >= '2.0':
        model = torch.compile(model)
        logger.info("[COMPILED] Model compiled for speed.")
    
    # 4. Training Components
    # -------------------------------------------------------------------------
    # CRITICAL: Optimizer must only see unfrozen parameters (The Experts)
    # The Encoder and Router are frozen inside APEX_MoE_Planner.__init__
    # 4a. Optimizer (SOTA Hygiene: Decay Exclusion)
    # configure_robust_optimizer handles filtering trainable params internally
    optimizer = configure_robust_optimizer(
        model=model,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        use_fused=True
    )
    
    # 4b. Scheduler: Warmup + Cosine (per-step)
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = int(total_steps * 0.05)  # 5% warmup
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps, 
        eta_min=cfg.training.min_lr
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    # 4c. Mixed Precision Scaler
    scaler = GradScaler('cuda') if device.type == 'cuda' else GradScaler('cpu')
    
    # 4d. EMA for stable inference
    ema = EMA(model, decay=cfg.training.get('ema_decay', 0.9999))
    
    # 4e. AWR Optimization Engine
    awr_calculator = ICUAdvantageCalculator(
        beta=cfg.training.get("awr_beta", 0.5),
        max_weight=cfg.training.get("awr_max_weight", 20.0)
    )

    # 4f. Checkpoint Saver (uses checkpoint_dir and optional backup_dir)
    from icu.utils.train_utils import RotationalSaver
    saver = RotationalSaver(
        save_dir=str(checkpoint_dir), 
        remote_dir=str(backup_dir) if backup_dir else None,
        keep_last_n=3,
        snapshot_every_n=50
    )
    
    # 4f. Resume from checkpoint
    # Priority: 1) Explicit resume_from in config, 2) Auto-detect latest in checkpoint_dir
    start_epoch = 1
    resume_path = None
    
    if cfg.paths.resume_from:
        # Explicit resume path from config
        resume_path = Path(cfg.paths.resume_from)
        if resume_path.exists():
            logger.info(f"[RESUME] Using explicit checkpoint: {resume_path}")
        else:
            logger.warning(f"[RESUME] Explicit checkpoint not found: {resume_path}")
            resume_path = None
    
    if resume_path is None:
        # Auto-detect latest checkpoint in checkpoint_dir
        latest_ckpts = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if latest_ckpts:
            resume_path = sorted(latest_ckpts)[-1]
            logger.info(f"[RESUME] Auto-detected checkpoint: {resume_path}")
    
    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"[RESUME] Resuming from epoch {start_epoch}")
    
    trainable_count, total_count = count_parameters(model)
    logger.info(f" APEX Model Ready.")
    logger.info(f" Frozen Params (Perception): {(total_count - trainable_count)/1e6:.2f}M")
    logger.info(f" Trainable Params (Experts): {trainable_count/1e6:.2f}M")
    logger.info(f" Scheduler: {warmup_steps} warmup steps, {total_steps - warmup_steps} cosine steps")

    # 5. Training Loop
    best_val_mse = float('inf')
    
    logger.info(f"[LOOP] Starting Phase 2 Loop for {cfg.training.epochs} epochs...")
    
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        start_time = time.time()
        
        # --- TRAIN ---
        metrics = train_one_epoch_apex(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            logger=logger,
            awr_calculator=awr_calculator, # Passed
            grad_clip=cfg.training.grad_clip
        )
        
        # --- VALIDATE (Soft Gating) ---
        val_mse, val_mae = validate_apex(
            ema=ema, 
            model=model, 
            dataloader=val_loader, 
            device=device, 
            logger=logger,
            limit_batches=cfg.training.get('val_batches', 50)
        )
        
        # --- LOGGING ---
        epoch_time = time.time() - start_time
        
        # Log Logic: Monitor Crash Expert engagement
        n_crash = metrics.get('n_crash', 0)
        n_stable = metrics.get('n_stable', 0)
        crash_rate = n_crash / (n_crash + n_stable + 1e-6)
        
        # Log to Unified Logger
        metrics_dict = {
            "train/loss": metrics.get('loss', 0),
            "train/loss_stable": metrics.get('loss_stable', 0),
            "train/loss_crash": metrics.get('loss_crash', 0),
            "train/crash_rate": crash_rate,
            "train/grad_norm": metrics.get('grad_norm', 0),
            "train/ess": metrics.get('ess', 0), # Log ESS
            "val/mse": val_mse,
            "val/mae": val_mae,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        }
        exp_logger.log_metrics(metrics_dict, step=epoch)
        
        logger.info(
            f"Ep {epoch:03d} | "
            f"Loss: {metrics.get('loss', 0):.4f} (Stb: {metrics.get('loss_stable', 0):.4f}, Crsh: {metrics.get('loss_crash', 0):.4f}) | "
            f"GradNorm: {metrics.get('grad_norm', 0):.2f} | "
            f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | "
            f"Crash Rate: {crash_rate:.1%}"
        )
        
        # --- SAVING ---
        is_best = val_mse < best_val_mse
        
        # Save with RotationalSaver API: (state_dict, epoch, is_best)
        saver.save(
            state_dict={
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),  # Persist scaler
                'config': base_cfg,
                'best_val_mse': best_val_mse if not is_best else val_mse
            },
            epoch=epoch,
            is_best=is_best
        )
        
        if is_best:
            best_val_mse = val_mse
            logger.info(f"[BEST] New Best Specialist! MSE: {best_val_mse:.4f}")

    logger.info("[DONE] APEX-MoE Specialization Complete.")
    exp_logger.finish()


if __name__ == "__main__":
    main()