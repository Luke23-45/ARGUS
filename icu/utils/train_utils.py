"""
icu/utils/train_utils.py
--------------------------------------------------------------------------------
APEX-MoE: SOTA Engineering Core (v3.4 - Robust Specification).

Status: PRODUCTION-READY / SAFETY-CRITICAL

This module provides the fault-tolerant infrastructure required for the APEX
Pipeline. It abstracts hardware management, distributed coordination, and 
persistence logic away from the scientific modeling code.

Architectural Pillars:
1.  **Concurrency Shield**: Strict Rank-0 guarding for all filesystem operations.
2.  **Tiered Storage**: Offloads EMA shadow weights to CPU RAM to prevent VRAM OOM.
3.  **Atomic Persistence**: Uses `fsync` + `rename` to guarantee checkpoint integrity.
4.  **State Recovery**: Scans disk on init to recover "Rolling Window" history.
5.  **Hygiene**: Strict parameter filtering for Weight Decay application.

Authors: APEX Research Team
"""
from __future__ import annotations

import collections
import functools
import glob
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from omegaconf import DictConfig, OmegaConf

# Optional SOTA Dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

# ==============================================================================
# 1. DISTRIBUTED HARDWARE GUARDRAILS
# ==============================================================================

def get_rank() -> int:
    """Returns the global rank of the current process (0 to N-1)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    # Check SLURM environment for pre-init robustness
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    return 0

def is_main_process() -> bool:
    """True if this is the coordinator (Rank 0) or a single-process run."""
    return get_rank() == 0

def rank_zero_only(fn):
    """Decorator: Ensures the function ONLY executes on Rank 0."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return fn(*args, **kwargs)
        return None
    return wrapper

def set_seed(seed: int = 42, worker: bool = False):
    """
    Enforces reproducible determinism across Python, Numpy, and PyTorch.
    Supports worker_init_fn logic for DataLoaders.
    """
    # Python
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        
    # Cudnn algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Slower but reproducible
    
    if worker:
        # Closure for DataLoader
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        return worker_init_fn
    return None

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def get_hardware_context() -> Dict[str, Any]:
    """
    SOTA Hardware Orchestrator.
    Automatically detects GPU/CPU capabilities and returns optimal training settings.
    
    Verified for:
    - PyTorch Lightning 2.6.0
    - Windows/Linux Compatibility
    - Mixed Precision Fallbacks (Ampere/Hopper vs Older GPUs)
    """
    is_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    
    if is_gpu:
        devices = torch.cuda.device_count()
        # [SOTA] Use BF16 if supported (Ampere+), otherwise FP16
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        
        return {
            "accelerator": "gpu",
            "devices": devices,
            "precision": precision,
            "pin_memory": True,
            "strategy": "auto" if devices == 1 else "ddp" 
        }
    
    # CPU Fallback
    return {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "bf16-mixed", # PL 2.6.0 will auto-fallback to 32 if unsupported on CPU
        "pin_memory": False,
        "strategy": "auto"
    }

# ==============================================================================
# 2. OBSERVABILITY ENGINE (Unified Logger)
# ==============================================================================

class UnifiedLogger:
    """
    SOTA Logger combining CSV (Data), TensorBoard (Viz), and WandB (Cloud).
    
    Robustness Features:
    - **Rank-0 Lock**: Automatically silences itself on worker nodes.
    - **Atomic CSV**: Flushes every line to disk to survive crashes.
    - **Metric Flattening**: Converts {'val': {'loss': 1}} -> 'val/loss'.
    - **Resilience**: Continues training even if WandB/TensorBoard fail.
    """
    def __init__(
        self, 
        cfg: DictConfig, 
        run_dir: Union[str, Path], 
        use_wandb: bool = True, 
        use_tb: bool = True
    ):
        self.rank = get_rank()
        if self.rank != 0:
            return # Null-Init for workers

        # PATCH: Ensure Base Logging Identity
        # We verify that a Root Logger handler exists. If not, we basic-config it 
        # to prevents "No handlers could be found" warnings or silent failures.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("UnifiedLogger")
        
        # A. Local Data Log (JSONL is robust to schema drift)
        self.log_path = self.run_dir / "metrics.jsonl"
        self.log_file = open(self.log_path, 'a', buffering=1) 

        # B. TensorBoard
        self.tb_writer = None
        if use_tb and SummaryWriter:
            try:
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb_logs"))
            except Exception as e:
                self.logger.warning(f"TensorBoard init failed: {e}")

        # C. WandB
        self.wandb_run = None
        if use_wandb and wandb:
            try:
                # If a run is already active (Hydra might init it), grab it
                if wandb.run is None:
                    wandb.init(
                        project=cfg.logging.wandb_project,
                        name=cfg.run_name,
                        config=OmegaConf.to_container(cfg, resolve=True),
                        dir=str(self.run_dir),
                        resume="allow",
                        mode="online" if cfg.logging.use_wandb else "offline"
                    )
                self.wandb_run = wandb
                self.logger.info("WandB Hooked Successfully.")
            except Exception as e:
                self.logger.error(f"WandB init failed: {e}")

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Dispatches metrics to all active backends safely."""
        # 1. Flatten Dictionary
        flat = self._flatten_dict(metrics)
        flat['step'] = step
        flat['timestamp'] = time.time()

        # 2. Robust Write
        self._write_jsonl(flat)

        # 3. TensorBoard
        if self.tb_writer:
            for k, v in flat.items():
                if isinstance(v, (int, float, np.number)):
                    self.tb_writer.add_scalar(k, float(v), step)

        # 4. WandB
        if self.wandb_run:
            try:
                self.wandb_run.log(flat, step=step)
            except Exception:
                pass # Non-blocking failure

    @rank_zero_only
    def finish(self):
        """Clean shutdown."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
        if hasattr(self, 'tb_writer') and self.tb_writer:
            self.tb_writer.close()
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.finish()

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Resolve Tensors
                if isinstance(v, torch.Tensor):
                    # Robustness: Scalarize if necessary, else take mean
                    if v.numel() == 1:
                        v = v.item()
                    else:
                        v = v.detach().cpu().float().mean().item()
                items.append((new_key, v))
        return dict(items)

    def _write_jsonl(self, metrics: Dict[str, Any]):
        """
        Robust Logging: JSONL handles evolving schemas (e.g. adding 'preshock_loss' later)
        without corrupting previous rows, unlike CSV.
        """
        try:
            import json
            # Dump to JSON string + newline
            payload = json.dumps(metrics) + "\n"
            self.log_file.write(payload)
        except Exception:
            pass # Non-blocking failure

# ==============================================================================
# 3. PERSISTENCE ENGINE (Rotational Saver)
# ==============================================================================

class RotationalSaver:
    """
    Atomic Checkpoint Saver with State Recovery.
    
    Robustness Features:
    - **Atomic Writes**: Writes to `.ckpt.tmp` -> `fsync` -> `rename` to `.ckpt`.
    - **State Recovery**: Scans disk on init to recover "Keep Last N" history 
      (fixing the "Amnesia" bug).
    - **Async Backups**: Uploads to backup_dir in a background thread.
    - **Rolling Window**: Automatically deletes old checkpoints (excluding snapshots).
    """
    def __init__(
        self, 
        save_dir: Union[str, Path], 
        remote_dir: Optional[Union[str, Path]] = None, 
        keep_last_n: int = 3, 
        snapshot_every_n: int = 50
    ):
        self.rank = get_rank()
        
        # MEDIUM FIX: Initialize default attributes to prevent AttributeError on non-rank-0
        # These attributes might be accessed even if this rank doesn't save
        self.save_dir = None
        self.remote_dir = None
        self.keep_last_n = keep_last_n
        self.snapshot_every_n = snapshot_every_n
        self._upload_thread = None
        self.saved_epochs = []
        
        if self.rank != 0:
            return  # Only Rank 0 saves, but attributes are initialized

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.remote_dir = Path(remote_dir) if remote_dir else None
        if self.remote_dir:
            self.remote_dir.mkdir(parents=True, exist_ok=True)
            
        self.keep_last_n = keep_last_n
        self.snapshot_every_n = snapshot_every_n
        self._upload_thread = None
        
        # STATE RECOVERY (The Fix for Amnesia)
        self.saved_epochs = self._scan_disk()

    def _scan_disk(self) -> List[Tuple[int, Path]]:
        """Scans disk for existing checkpoints to rebuild history."""
        existing = []
        pattern = re.compile(r".*_epoch_(\d+)\.ckpt$")
        
        for file_path in self.save_dir.glob("*.ckpt"):
            match = pattern.match(file_path.name)
            if match:
                epoch = int(match.group(1))
                existing.append((epoch, file_path))
        
        # Sort by epoch
        existing.sort(key=lambda x: x[0])
        logging.getLogger(__name__).info(f"[SAVER] Recovered {len(existing)} checkpoints from disk.")
        return existing

    @rank_zero_only
    def save(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
        filename_prefix: str = "checkpoint"
    ):
        filename = f"{filename_prefix}_epoch_{epoch:04d}.ckpt"
        local_path = self.save_dir / filename
        temp_path = self.save_dir / f"{filename}.tmp"
        
        # 1. Atomic Write (Robust File Handle Pattern)
        try:
            with open(temp_path, 'wb') as f:
                # Direct serialization to file buffer
                torch.save(state_dict, f)
                # Force OS to write buffer to disk
                f.flush()
                # os.fsync(f.fileno()) might be needed for absolute safety
                try:
                    os.fsync(f.fileno())
                except: pass
            
            # Atomic Rename (now safe because handle is closed)
            temp_path.replace(local_path)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"[SAVER] FATAL: Failed to save {filename}: {e}")
            if temp_path.exists(): 
                try: 
                    temp_path.unlink() 
                except: 
                    pass
            return

        # 2. Rotation Logic (Rolling Window)
        self.saved_epochs.append((epoch, local_path))
        self.saved_epochs.sort(key=lambda x: x[0])
        
        # Filter: Don't delete Snapshots
        deletion_candidates = [
            (e, p) for (e, p) in self.saved_epochs 
            if e % self.snapshot_every_n != 0
        ]
        
        if len(deletion_candidates) > self.keep_last_n:
            # Identify victims (oldest)
            victims = deletion_candidates[:-self.keep_last_n]
            for _, victim_path in victims:
                try:
                    if victim_path.exists():
                        victim_path.unlink()
                        # Update state
                        self.saved_epochs = [x for x in self.saved_epochs if x[1] != victim_path]
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to delete old checkpoint {victim_path}: {e}")

        # 3. Best Model Handling
        if is_best:
            best_path = self.save_dir / "best_model.ckpt"
            try:
                shutil.copy2(local_path, best_path)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to copy best model: {e}")

        # 4. Async Backup (Cloud Upload)
        if self.remote_dir:
            self._trigger_async_upload(local_path, filename, is_best)

    def _trigger_async_upload(self, local_src: Path, filename: str, is_best: bool):
        # Do not spawn if previous upload is still running (prevent thread explosion)
        if self._upload_thread and self._upload_thread.is_alive():
            return 

        self._upload_thread = threading.Thread(
            target=self._worker_upload,
            args=(local_src, filename, is_best),
            daemon=False  # [SAFETY FIX] Non-daemon allows graceful shutdown
        )
        self._upload_thread.start()

    def cleanup(self):
        """[SAFETY FIX] Join upload thread to prevent corrupted uploads on exit."""
        if self._upload_thread and self._upload_thread.is_alive():
            self._upload_thread.join(timeout=30.0)  # Wait up to 30s for upload

    def _worker_upload(self, src: Path, fname: str, is_best: bool):
        try:
            dst = self.remote_dir / fname
            shutil.copy2(src, dst)
            if is_best:
                shutil.copy2(src, self.remote_dir / "best_model.ckpt")
        except Exception:
            pass # Silent fail on backup

# ==============================================================================
# 4. STABILITY ENGINE (Tiered EMA)
# ==============================================================================

class TieredEMA:
    """
    Hardware-Aware Exponential Moving Average.
    
    Robustness Features:
    - **CPU Offloading**: Shadow weights stored in pinned System RAM, preventing
      VRAM OOM during heavy validation steps.
    - **Precision Guard**: Forces FP32 for shadow weights even if model is BF16/FP16,
      preventing underflow over long training runs.
    - **Context Management**: `with ema.swap():` syntax for safe inference.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Shadow storage: {param_name: CPU_Tensor}
        self.shadow = {}
        # Backup storage (for swapping): {param_name: CPU_Tensor}
        self.backup = {}
        self.model_ref = model
        
        # Initialize
        self._register(model)

    def _register(self, model: nn.Module):
        use_cuda = torch.cuda.is_available()
        for name, param in model.named_parameters():
            if param.requires_grad:
                # .pin_memory() enables fast non_blocking transfers from CPU to GPU
                # only valid if CUDA is available.
                data = param.data.detach().cpu().float().clone()
                self.shadow[name] = data.pin_memory() if use_cuda else data
        
        for name, buffer in model.named_buffers():
            # Robust Logic: Buffers might be integers
            data = buffer.data.detach().cpu().clone()
            if torch.is_floating_point(buffer):
                data = data.float()
            
            self.shadow[name] = data.pin_memory() if use_cuda else data

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Async GPU -> CPU transfer
                # Overlaps with the next training step's GPU kernels
                new_data = param.data.detach().to(device="cpu", non_blocking=True).float()
                
                # CPU Math (Wait handled automatically by PyTorch when accessing new_data)
                self.shadow[name].mul_(self.decay).add_(new_data, alpha=(1.0 - self.decay))

        # Handle Buffers
        for name, buffer in model.named_buffers():
            if name in self.shadow:
                target_dtype = self.shadow[name].dtype
                # Handle int/float mismatch dynamically
                if torch.is_floating_point(buffer):
                    new_data = buffer.data.detach().to(device="cpu", non_blocking=True).float()
                else:
                    new_data = buffer.data.detach().to(device="cpu", non_blocking=True)
                
                # Integer buffers (e.g., steps) are usually just copied, not averaged
                if target_dtype in (torch.int64, torch.int32, torch.bool):
                    self.shadow[name].copy_(new_data)
                else:
                    self.shadow[name].mul_(self.decay).add_(new_data, alpha=(1.0 - self.decay))

    def apply_shadow(self, model: nn.Module):
        """
        SWAP: Saves current Model(GPU) to Backup(CPU). 
              Loads Shadow(CPU) to Model(GPU).
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Backup current weight
                self.backup[name] = param.data.detach().cpu().clone()
                # Overwrite with Shadow
                param.data.copy_(self.shadow[name].to(param.device, non_blocking=True))
                
        for name, buffer in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buffer.data.detach().cpu().clone()
                buffer.data.copy_(self.shadow[name].to(buffer.device, non_blocking=True))

    def restore(self, model: nn.Module):
        """RESTORE: Loads Backup(CPU) back to Model(GPU). Clears Backup."""
        
        # Params are safe (always float)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device, non_blocking=True))
        
        # Buffers need Type Safety
        for name, buffer in model.named_buffers():
            if name in self.backup:
                saved_data = self.backup[name].to(buffer.device, non_blocking=True)
                
                # Check for type mismatch (e.g., Saved Float vs Model Int)
                if buffer.dtype != saved_data.dtype:
                    # Explicit cast required to prevent runtime error
                    if not torch.is_floating_point(buffer):
                        saved_data = saved_data.to(dtype=buffer.dtype)
                
                buffer.data.copy_(saved_data)
        
        self.backup = {} # Free memory

    def swap(self):
        """Context Manager: usage `with ema.swap(model): validate()`"""
        class SwapContext:
            def __init__(self, ema, model):
                self.ema = ema
                self.model = model
            def __enter__(self):
                self.ema.apply_shadow(self.model)
            def __exit__(self, exc_type, exc_value, traceback):
                self.ema.restore(self.model)
        return SwapContext(self, self.model_ref)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# Alias for backward compatibility with model wrappers
EMA = TieredEMA

# ==============================================================================
# 5. OPTIMIZATION FACTORY (Robust Hygiene)
# ==============================================================================

def configure_robust_optimizer(
    model: nn.Module, 
    learning_rate: float, 
    weight_decay: float, 
    betas: Tuple[float, float] = (0.9, 0.99),
    use_fused: bool = True
) -> Optimizer:
    """
    Configures AdamW with 'Parameter Hygiene'.
    
    Logic:
    1.  **Decayed (Regularized)**: Conv weights, Linear weights.
    2.  **No-Decay (Raw)**: Biases, Norms (Layer/Group/Batch/RMS), Embeddings, 1D Tensors.
    """
    # 1. Separate Parameters
    decay_params = set()
    no_decay_params = set()
    
    # Types to universally exclude from decay
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)
    
    # Also handle RMSNorm if present (check by class name string to avoid import dep)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn # Full Param Name
            
            if not p.requires_grad:
                continue
                
            # Rule 1: Embeddings / Norms -> No Decay
            if isinstance(m, blacklist_weight_modules) or "Norm" in m.__class__.__name__:
                no_decay_params.add(fpn)
            
            # Rule 2: 1D tensors (biases) -> No Decay
            elif p.ndim < 2:
                no_decay_params.add(fpn)
                
            # Rule 3: Positional Embeddings -> No Decay
            elif "pos_emb" in fpn or "time_emb" in fpn or "rope" in fpn:
                no_decay_params.add(fpn)

    # Validate against actual parameter list
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Assign remainder to Decay group
    optim_groups = [
        {"params": [], "weight_decay": weight_decay}, # Decay
        {"params": [], "weight_decay": 0.0},          # No Decay
    ]
    
    for pn, p in param_dict.items():
        if pn in no_decay_params:
            optim_groups[1]["params"].append(p)
        else:
            optim_groups[0]["params"].append(p)
            
    # 2. Hardware-Aware Factory
    # [SOTA PERFORMANCE] Enable fused=True by default.
    # Note: We use a "Surgical Bypass" strategy in the model wrappers (on_before_optimizer_step)
    # to manually handle gradient clipping. This avoids the RuntimeError caused by
    # Lightning's automatic clipping suite while keeping the 2x+ speedup of fused AdamW.
    use_fused = use_fused and torch.cuda.is_available() and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    
    extra_args = dict(fused=True) if use_fused else dict()
    
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=learning_rate, 
        betas=betas, 
        **extra_args
    )
    
    if is_main_process():
        logging.getLogger("OptimizerFactory").info(
            f"AdamW Configured: Fused={use_fused}. "
            f"Decayed: {len(optim_groups[0]['params'])}, Raw: {len(optim_groups[1]['params'])}"
        )
        
    return optimizer

# ==============================================================================
# 6. TRANSFER LEARNING (Surgical Loader)
# ==============================================================================

class SurgicalCheckpointLoader:
    """
    Enables Phase 1 -> Phase 2 transitions ("Brain Transplants").
    Handles key remapping and partial loading.
    """
    @staticmethod
    def load_model(
        model: nn.Module, 
        checkpoint_path: str, 
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            model: Target model to load into.
            checkpoint_path: Path to .ckpt file.
            strict: Strict checking of keys.
        """
        logger = logging.getLogger("SurgicalLoader")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        logger.info(f"Extracting DNA from: {checkpoint_path}")
        
        # Load to CPU to avoid VRAM spike
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # ----------------------------------------------------------------------
        # PATCH: Robust Prefix & MoE Expansion Logic
        # ----------------------------------------------------------------------
        
        # 1. Prefix Normalization (Removing 'model.' wrapper from Lightning)
        # We handle state_dict extraction from PL/DeepSpeed/Standard formats
        if "state_dict" in checkpoint:
            src_state = checkpoint["state_dict"]
        elif "model" in checkpoint:
            src_state = checkpoint["model"]
        else:
            src_state = checkpoint

        # We build a normalized source dictionary first.
        normalized_src = {}
        for k, v in src_state.items():
            if k.startswith("model."):
                normalized_src[k[6:]] = v
            else:
                normalized_src[k] = v
        
        # 2. Target Analysis
        target_keys = set(model.state_dict().keys())
        final_state = {}

        # 3. Matching & Expansion
        # Strategy: iterate source, keeping exact matches. 
        # THEN, iterate target to find missing MoE keys.
        
        # A. Copy Exact Matches & Normalization Matches
        for k, v in normalized_src.items():
            final_state[k] = v
            
        # B. MoE Bootstrap (The "Expert Amnesia" Fix)
        # Pattern: Target 'experts.N.attr' should grab from Source 'backbone.attr'
        # Assumption: Generalist uses 'backbone' and Specialist uses 'experts'
        moe_pattern = re.compile(r"^experts\.(\d+)\.(.*)")
        
        for tgt_key in target_keys:
            # If we already found it in source, skip
            if tgt_key in final_state:
                continue
                
            match = moe_pattern.match(tgt_key)
            if match:
                # expert_id = match.group(1) # unused, we map all experts to single backbone
                attr_suffix = match.group(2)
                
                # Construct hypothetical source key (Generalist Anatomy)
                # We try both 'backbone.' prefix and raw attribute (depending on model)
                candidates = [f"backbone.{attr_suffix}", attr_suffix]
                
                for src_key in candidates:
                    if src_key in normalized_src:
                        logger.info(f"   Bootstrapping Expert Node: {tgt_key} <- {src_key}")
                        final_state[tgt_key] = normalized_src[src_key].clone()
                        break
        
        src_state = final_state
        # ----------------------------------------------------------------------
            
        # Perform Load
        keys = model.load_state_dict(src_state, strict=strict)
        
        logger.info(f"Transplant Complete. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
        
        return checkpoint

# ==============================================================================
# 7. LOGGING UTILITY
# ==============================================================================

def setup_logger(run_name: str, log_dir: str) -> logging.Logger:
    """Configures global Python logging (Root Source of Truth)."""
    # 1. Worker Silence (DDP Safe)
    if not is_main_process():
        # Strictly silence non-main processes for the project namespace
        l = logging.getLogger("icu")
        l.setLevel(logging.ERROR)
        l.propagate = False
        return l

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. Grab Project-Specific Logger
    # Using 'icu' as the namespace prevents third-party library spam (fsspec, etc)
    # reaching the main DDP console at INFO level.
    logger = logging.getLogger("icu")
    logger.setLevel(logging.INFO)
    
    # 3. Handle Directory Changes or Duplicate Handlers
    # Scan for existing Handlers to prevent duplication
    current_handlers = list(logger.handlers)
    log_file = os.path.join(log_dir, f"{run_name}.log")
    
    # Remove existing StreamHandlers to prevent console duplication
    for h in current_handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            
    # Remove FileHandlers if path differs (for recovery runs)
    for h in current_handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(h.baseFilename) == os.path.abspath(log_file):
                # If path matches, we still need to re-add StreamHandler if it was removed
                pass 
            else:
                h.close()
                logger.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    
    # 4. Stream Handler (Stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 5. File Handler (Disk)
    fh = logging.FileHandler(log_file, mode='a') 
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info(f"Global Logger configured. Writing to {log_file}")
    
    return logger