#!/usr/bin/env python3
"""
LR Finder Script for ICU Diffusion
===================================

This script performs a "Range Test" to find the optimal Learning Rate.
Ref: Cyclic Learning Rates for Training Neural Networks (Smith, 2017)

Method:
1. Start with a very small LR (e.g., 1e-7).
2. Exponentially increase LR after each batch.
3. Observe Loss.
4. Stop when Loss explodes.
5. Optimal LR is typically 5-10x smaller than the point of explosion (steepest descent).
"""

import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Imports from project
try:
    from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
    from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"[ERROR] Could not import project modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LR_Finder")

def run_lr_finder(cfg: DictConfig, start_lr=1e-7, end_lr=10.0, num_iter=100):
    """
    Runs the LR Finder logic.
    """
    # 1. Setup Data (Train Split Only)
    # Robust Path Finding
    candidates = [
        "icu_research/data",
        "data",
        str(ROOT_DIR / "data"),
        str(ROOT_DIR / "icu_research" / "data")
    ]
    
    data_dir = None
    for c in candidates:
        if (Path(c) / "train" / "train.lmdb").exists():
            data_dir = c
            break
            
    if data_dir is None:
        logger.error(f"Could not find valid data directory with train/train.lmdb. Checked: {candidates}")
        return [], []
    
    logger.info(f"Loading Dataset from {data_dir}...")
    dataset = ICUSotaDataset(
        dataset_dir=data_dir,
        split="train",
        augment_noise=0.0
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        collate_fn=robust_collate_fn,
        num_workers=0 # Keep simple for test
    )
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ICUConfig(**cfg.model)
    model = ICUUnifiedPlanner(model_cfg).to(device)
    model.train()
    
    # 3. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=cfg.training.weight_decay)
    
    # 4. Range Test Loop
    lrs = []
    losses = []
    
    # Calculate multiplier
    curr_lr = start_lr
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    logger.info(f"Starting LR Range Test: {start_lr:.1e} -> {end_lr:.1e} over {num_iter} steps.")
    
    pbar = tqdm(total=num_iter)
    avg_loss = 0.0
    best_loss = float('inf')
    smoothing = 0.05
    
    iter_idx = 0
    
    # Loop over batches (cycling if needed)
    while iter_idx < num_iter:
        for batch in dataloader:
            if iter_idx >= num_iter: 
                break
                
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward
            optimizer.zero_grad()
            loss_dict = model(batch)
            loss = loss_dict['loss']
            
            # Robustness check
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Loss diverges! Stopping.")
                break
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Record
            loss_val = loss.item()
            
            # Smoothed loss
            if iter_idx == 0:
                avg_loss = loss_val
            else:
                avg_loss = smoothing * loss_val + (1 - smoothing) * avg_loss
            
            # Stop if loss explodes
            if iter_idx > 10 and avg_loss > 4 * best_loss:
                logger.info("Loss exploded. Stopping early.")
                break
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            lrs.append(curr_lr)
            losses.append(avg_loss)
            
            # Update LR
            curr_lr *= lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
            
            iter_idx += 1
            pbar.update(1)
            pbar.set_description(f"LR: {curr_lr:.1e} | Loss: {avg_loss:.4f}")
        
    pbar.close()
    
    return lrs, losses

def plot_lr_finder(lrs, losses, output_file="lr_finder_plot.png"):
    """Plots LR vs Loss."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("LR Finder: Loss vs Learning Rate")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        # Suggest valid point
        min_loss_idx = np.argmin(losses)
        suggested_lr = lrs[min_loss_idx] / 10.0 # Common heuristic
        
        plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f"Suggested LR (~{suggested_lr:.1e})")
        plt.legend()
        
        plt.savefig(output_file)
        logger.info(f"Plot saved to {output_file}")
        
        # Print recommendation
        print("\n" + "="*50)
        print("RECOMMENDATION")
        print("="*50)
        print(f"Minimum Loss: {min(losses):.4f} at LR: {lrs[min_loss_idx]:.1e}")
        print(f"Suggested Max LR (10x smaller): {suggested_lr:.1e}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to plot: {e}")

def main():
    # Load default config
    # Start with simple defaults if no config file provided
    conf_path = ROOT_DIR / "conf" / "generalist.yaml"
    if conf_path.exists():
        cfg = OmegaConf.load(conf_path)
    else:
        logger.error("Configuration file not found!")
        return

    lrs, losses = run_lr_finder(cfg)
    plot_lr_finder(lrs, losses, output_file="icu_research/scripts/lr_finder_results.png")

if __name__ == "__main__":
    main()
