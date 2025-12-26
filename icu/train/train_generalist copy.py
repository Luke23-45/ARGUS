"""
scripts/train_generalist.py
--------------------------------------------------------------------------------
Phase 1: Generalist Training Engine (Hydra Refactor).

This script trains the 'ICUUnifiedPlanner' from scratch on the full dataset.
It establishes the shared latent space and the 'Sepsis Router' that will be
frozen and used in Phase 2.

Key Features (Updated to SOTA Practices):
1. **Robust Loop**: Handles AMP, Gradient Clipping, EMA updates, and LR scheduling.
2. **Clinical Validation**: Runs full inference (sampling) during validation.
3. **Atomic Persistence**: Uses the AtomicSaver to prevent corrupted checkpoints.
4. **Dataset Normalization**: Computes global min/max from train dataset.
5. **Efficiency**: Optional torch.compile for PyTorch 2+ speedups.
"""
import sys
import os
import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn, ensure_data_ready
from icu.utils.train_utils import (
    setup_logger, set_seed, count_parameters, EMA,
    configure_robust_optimizer, RotationalSaver, SurgicalCheckpointLoader,
    UnifiedLogger
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_dataset_stats(dataset: ICUSotaDataset, input_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes global min/max across all vitals channels for normalization.
    
    Args:
        dataset: The training dataset.
        input_dim: Number of vitals channels (from model config).
    """
    mins = torch.full((input_dim,), float('inf'))
    maxs = torch.full((input_dim,), float('-inf'))
    
    for sample in tqdm(dataset, desc="Computing Stats", leave=False):
        if sample is None:
            continue
        vitals = torch.cat([sample['observed_data'], sample['future_data']], dim=0)  # [T_total, D]
        mins = torch.minimum(mins, vitals.min(dim=0)[0])
        maxs = torch.maximum(maxs, vitals.max(dim=0)[0])
    
    return mins, maxs

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,  # Can be SequentialLR or any LRScheduler
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    grad_clip: float = 1.0
) -> dict:
    """
    Executes one training epoch with SOTA practices.
    
    Features:
    - Device-aware AMP (works on CPU and CUDA)
    - Gradient norm logging (before clipping)
    - Per-step LR scheduling (for warmup + cosine)
    """
    model.train()
    stats = {
        "loss": [],
        "diff": [],      # Diffusion Loss
        "aux": [],       # Auxiliary Sepsis Loss
        "grad_norm": []  # Gradient health monitoring
    }
    
    # Determine device type for autocast
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    pbar = tqdm(dataloader, desc=f"Ep {epoch} [Train]", leave=False)
    
    for batch in pbar:
        # 1. Move Data
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # 2. Forward (Mixed Precision - device aware)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=device_type, enabled=(device_type == "cuda")):
            # Model outputs dict: {'loss', 'diffusion_loss', 'aux_loss'}
            loss_dict = model(batch)
            loss = loss_dict['loss']
            
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
        
        # 7. EMA Update (after optimizer step)
        ema.update(model)
        
        # 8. Logging
        current_loss = loss.item()
        stats["loss"].append(current_loss)
        stats["diff"].append(loss_dict["diffusion_loss"].item())
        stats["aux"].append(loss_dict["aux_loss"].item())
        stats["grad_norm"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        pbar.set_postfix(
            loss=f"{current_loss:.4f}",
            aux=f"{loss_dict['aux_loss'].item():.4f}",
            gn=f"{stats['grad_norm'][-1]:.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}"
        )
    
    return {k: np.mean(v) for k, v in stats.items()}

@torch.no_grad()
def validate_clinical(
    ema: EMA, 
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device,
    logger: logging.Logger, 
    sample_steps: int = 50,
    limit_batches: int = 50  # Gap 17: Now configurable
) -> tuple:
    """
    Performs CLINICAL validation using EMA weights.
    Samples actual vitals and computes MSE/MAE against ground truth.
    
    Args:
        limit_batches: Number of batches to validate on (for speed)
    """
    ema.apply_shadow(model)  # Load EMA weights temporarily
    model.eval()
    mse_errors = []
    mae_errors = []
    
    pbar = tqdm(dataloader, desc="Validation", total=min(limit_batches, len(dataloader)), leave=False)
    for i, batch in enumerate(pbar):
        if i >= limit_batches: break
        
        # 1. Move Data
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        gt_future = batch['future_data']
        
        # 2. Sample (Inference Mode)
        pred_future = model.sample(batch, num_steps=sample_steps)
        
        # 3. Compute Errors
        mse = ((pred_future - gt_future) ** 2).mean().item()
        mae = (pred_future - gt_future).abs().mean().item()
        mse_errors.append(mse)
        mae_errors.append(mae)
        
        pbar.set_postfix(mse=f"{mse:.4f}", mae=f"{mae:.4f}")
    
    ema.restore(model)  # Restore original weights
    return np.mean(mse_errors), np.mean(mae_errors)

# ==============================================================================
# MAIN ENGINE (HYDRA)
# ==============================================================================

@hydra.main(version_base=None, config_path="../../conf", config_name="generalist")
def main(cfg: DictConfig):
    print("DEBUG: main() started")
    
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
    print("DEBUG: Logger setup complete")
    
    set_seed(cfg.seed)
    print("DEBUG: Seed set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Unified Logger (WandB + TB) - use output_dir for tensorboard
    print(f"DEBUG: Initializing UnifiedLogger (wandb={cfg.logging.use_wandb})...")
    exp_logger = UnifiedLogger(cfg, str(output_dir))
    print("DEBUG: UnifiedLogger init complete")
    
    logger.info(f"[START] Starting Generalist Training: {cfg.run_name}")
    logger.info(f" Output Dir: {output_dir}")
    logger.info(f" Checkpoint Dir: {checkpoint_dir}")
    logger.info(f" Log Dir: {log_dir}")
    logger.info(f" Backup Dir: {backup_dir}")
    logger.info(f" Device: {device}")
    logger.info(f" Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Initialize Data
    logger.info("[DATA] Ensuring Data is Ready (Orchestrator)...")
    ensure_data_ready(
        dataset_dir=cfg.dataset.dataset_dir,
        hf_repo_id=cfg.dataset.hf_repo,
        force_download=cfg.dataset.get("force_download", False)
    )
    
    logger.info("[DATA] Initializing Dataset...")
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
    
    # Gap 19: Improved data loading with persistent workers
    use_workers = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        collate_fn=robust_collate_fn,
        worker_init_fn=set_seed(cfg.seed, worker=True),
        persistent_workers=use_workers,  # Keep workers alive between epochs
        prefetch_factor=2 if use_workers else None  # Prefetch next batch
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        num_workers=min(2, cfg.num_workers), 
        pin_memory=True, 
        collate_fn=robust_collate_fn,
        persistent_workers=use_workers and cfg.num_workers >= 2,
        prefetch_factor=2 if use_workers else None
    )
    logger.info(f" Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 3. Compute Dataset Stats for Normalizer
    logger.info("[STATS] Computing Dataset Statistics...")
    vitals_min, vitals_max = compute_dataset_stats(train_ds, cfg.model.input_dim)
    logger.info(f" Vitals Min: {vitals_min.tolist()} | Max: {vitals_max.tolist()}")

    # 4. Initialize Model
    logger.info("[MODEL] Building Model...")
    # Instantiate config from Hydra DictConfig
    model_cfg = ICUConfig(**cfg.model)
    
    model = ICUUnifiedPlanner(model_cfg).to(device)
    
    # Set normalizer buffers
    model.normalizer.vitals_min.copy_(vitals_min.to(device))
    model.normalizer.vitals_max.copy_(vitals_max.to(device))
    model.normalizer.initialized.fill_(1)
    
    # Gap 20: Log model architecture summary
    logger.info(f"[MODEL] Architecture Summary:")
    logger.info(f"  - Encoder Layers: {model_cfg.encoder_layers}")
    logger.info(f"  - Backbone Layers: {model_cfg.n_layers}")
    logger.info(f"  - Model Dim: {model_cfg.d_model}, Heads: {model_cfg.n_heads}")
    logger.info(f"  - History: {model_cfg.history_len}h -> Predict: {model_cfg.pred_len}h")
    
    if cfg.compile_model and torch.__version__ >= '2.0':
        model = torch.compile(model)
        logger.info("[COMPILED] Model compiled for speed.")
    
    # 5. Training Components
    # -------------------------------------------------------------------------
    # 5a. Optimizer
    # 5a. Optimizer (SOTA Hygiene: Decay Exclusion)
    optimizer = configure_robust_optimizer(
        model=model,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        use_fused=True
    )
    
    # 5b. Scheduler: Warmup + Cosine (per-step, not per-epoch)
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = int(total_steps * 0.05)  # 5% warmup
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,  # Start at 1% of LR
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
    
    # 5c. Mixed Precision Scaler
    scaler = GradScaler('cuda') if device.type == 'cuda' else GradScaler('cpu')
    
    # 5d. EMA for stable inference
    ema = EMA(model, decay=cfg.training.get('ema_decay', 0.9999))
    
    # 5e. Checkpoint Saver (uses checkpoint_dir and optional backup_dir)
    saver = RotationalSaver(
        save_dir=str(checkpoint_dir), 
        remote_dir=str(backup_dir) if backup_dir else None,
        keep_last_n=3,
        snapshot_every_n=50
    )
    
    # 5f. Resume from checkpoint
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
    
    trainable, total = count_parameters(model)
    logger.info(f" Params: {trainable/1e6:.2f}M Trainable / {total/1e6:.2f}M Total")
    logger.info(f" Scheduler: {warmup_steps} warmup steps, {total_steps - warmup_steps} cosine steps")

    # 6. Training Loop
    best_val_mse = float('inf')
    
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        start_time = time.time()
        
        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            logger=logger,
            grad_clip=cfg.training.grad_clip
        )
        
        # --- VALIDATE (Using EMA) ---
        val_mse, val_mae = validate_clinical(
            ema=ema, 
            model=model, 
            dataloader=val_loader, 
            device=device, 
            logger=logger,
            limit_batches=cfg.training.get('val_batches', 50)
        )
        
        # --- LOGGING ---
        epoch_time = time.time() - start_time
        
        # Log to Unified Logger (WandB/TB)
        metrics_dict = {
            "train/loss": train_metrics['loss'],
            "train/diff_loss": train_metrics['diff'],
            "train/aux_loss": train_metrics['aux'],
            "train/grad_norm": train_metrics['grad_norm'],  # Gap 7: Gradient health
            "val/mse": val_mse,
            "val/mae": val_mae,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        }
        exp_logger.log_metrics(metrics_dict, step=epoch)
        
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Loss: {train_metrics['loss']:.4f} (Diff: {train_metrics['diff']:.4f}, Aux: {train_metrics['aux']:.4f}) | "
            f"GradNorm: {train_metrics['grad_norm']:.2f} | "
            f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # --- SAVING ---
        is_best = val_mse < best_val_mse
        
        # Save checkpoint with RotationalSaver API: (state_dict, epoch, is_best)
        saver.save(
            state_dict={
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),  # Gap 4: Persist scaler
                'config': model_cfg,
                'best_val_mse': best_val_mse if not is_best else val_mse
            },
            epoch=epoch,
            is_best=is_best
        )
        
        if is_best:
            best_val_mse = val_mse
            logger.info(f"[BEST] New Best Model! MSE: {best_val_mse:.4f}")

    logger.info("[DONE] Phase 1 Generalist Training Complete.")
    exp_logger.finish()

if __name__ == "__main__":
    main()

    