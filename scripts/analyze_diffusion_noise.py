#!/usr/bin/env python3
"""
Diffusion Noise Schedule Analyzer
=================================

Visualizes the Forward Diffusion Process on Real ICU Data.
Goal: Ensure the signal is properly destroyed by t=T, but not too early.

If signal destroyed at t=10:  Schedule is too aggressive (Reduce beta_min/max).
If signal remains at t=T:     Schedule is too weak (Increase beta_max or T).
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

try:
    from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
    from icu.datasets.dataset import ICUSotaDataset
except ImportError as e:
    print(f"[ERROR] Project modules not found: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Noise_Analyzer")

def analyze_noise(cfg_path: Path):
    logger.info("Setting up Noise Analysis...")
    cfg = OmegaConf.load(cfg_path)
    
    # 1. Load 1 Sample
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
        logger.error(f"Could not find valid data directory. Checked: {candidates}")
        return

    dataset = ICUSotaDataset(data_dir, split="train")
    sample = dataset[0] # Get first sample
    
    # Extract vitals (L, C)
    vitals = sample['vitals']
    vitals_t = torch.tensor(vitals, dtype=torch.float32).unsqueeze(0) # [1, L, C]
    
    # 2. Initialize Model (for Schedule)
    model_cfg = ICUConfig(**cfg.model)
    planner = ICUUnifiedPlanner(model_cfg)
    
    # Access internal noise scheduler
    # We use the DDPMScheduler logic inside the planner
    scheduler = planner.noise_scheduler
    
    # 3. Diffuse and Record
    timesteps = [0, 10, 25, 50, 75, 99]
    noisy_trajectories = []
    
    logger.info(f"Diffusing sample at steps: {timesteps}")
    
    for t in timesteps:
        if t == 0:
            noisyiveness = vitals_t
        else:
            t_batch = torch.full((1,), t, dtype=torch.long)
            noise = torch.randn_like(vitals_t)
            noisyiveness = scheduler.add_noise(vitals_t, noise, t_batch)
            
        noisy_trajectories.append(noisyiveness.squeeze(0).numpy())
        
    # 4. Plot
    plot_path = "icu_research/scripts/diffusion_process_viz.png"
    visualize_diffusion(noisy_trajectories, timesteps, plot_path)

def visualize_diffusion(trajs, steps, save_path):
    """Plots the degradation of the Heart Rate (channel 0) channel."""
    
    plt.figure(figsize=(15, 6))
    
    # Assuming Channel 0 is HR
    # Plot only first channel for clarity
    for idx, (traj, t) in enumerate(zip(trajs, steps)):
        color = plt.cm.viridis(idx / len(steps))
        plt.plot(traj[:, 0], label=f"t={t}", color=color, linewidth=2)
        
    plt.title("Forward Diffusion Process: Heart Rate Channel", fontsize=14)
    plt.xlabel("Time (steps in window)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    logger.info(f"Visualization saved to {save_path}")
    print(f"\n[OUTPUT] Plot generated: {save_path}")
    print("Check this plot. If t=99 looks like pure Gaussian noise (mean 0, var 1), schedule is good.")

if __name__ == "__main__":
    conf = ROOT_DIR / "conf" / "generalist.yaml"
    if conf.exists():
        analyze_noise(conf)
    else:
        logger.error("Config not found.")
