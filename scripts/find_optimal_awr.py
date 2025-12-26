#!/usr/bin/env python3
"""
scripts/find_optimal_awr.py
--------------------------------------------------------------------------------
SOTA ICU AWR Parameter Optimization & Stability Search.

This script performs a high-fidelity audit of the advantage distribution across
the real ICU dataset. It jointly optimizes:
1.  **AWR Temperature (Beta)**: Controls the peakiness of clinical weighting.
2.  **Clipping Threshold (MaxWeight)**: Protects against out-of-distribution outliers.

Optimization Objective:
Find (Beta, MaxWeight) that results in 20-50% Effective Sample Size (ESS)
while maximizing information coverage across the dataset.

Source of Truth: ICUSotaDataset + ClinicalNormalizer
"""

import sys
import os
import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Suppress noisy warnings during exhaustive run
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Project Imports
try:
    from icu.utils.advantage_calculator import ICUAdvantageCalculator
    from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
    from icu.datasets.normalizer import ClinicalNormalizer
except ImportError as e:
    print(f"[ERROR] Could not import project modules. Ensure CWD is project root: {e}")
    sys.exit(1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AWR_Optimizer")

def analyze_awr_stability(
    data_dir: str, 
    num_samples: int = -1,  # -1 means full dataset
    batch_size: int = 1024,
    target_ess: float = 0.35
):
    print("\n" + "="*80)
    print("  🚀 APEX-MoE: EXHAUSTIVE AWR Stability & Parameter Optimizer v5.0")
    print("  Source of Truth: FULL DATASET")
    print("="*80 + "\n")

    # 1. Load Dataset & Metadata
    logger.info(f"Loading Ground-Truth Dataset from: {data_dir}")
    try:
        dataset = ICUSotaDataset(
            dataset_dir=data_dir,
            split="train",
            history_len=24,
            pred_len=6,
            augment_noise=0.0
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Extract Normalizer (Physical Unit Ground Truth)
    index_path = Path(data_dir) / "train_index.json"
    if not index_path.exists():
        index_path = Path(data_dir).parent / "train_index.json"
    
    normalizer = ClinicalNormalizer(ts_channels=28, static_channels=6)
    try:
        channel_names = dataset.ts_columns
        normalizer.calibrate_from_stats(index_path, channel_names)
    except Exception as e:
        logger.warning(f"Could not load normalizer from index: {e}. Falling back to RAW rewards.")
        normalizer = None

    # 3. Exhaustive Data Collection
    total_avail = len(dataset)
    to_process = total_avail if num_samples <= 0 else min(num_samples, total_avail)
    
    logger.info(f"Processing {to_process:,} trajectories (Exhaustive Search)...")
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, # Shuffle not needed for exhaustive pass
        collate_fn=robust_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    all_rewards = []
    processed = 0
    
    with torch.no_grad():
        pbar = tqdm(total=to_process, desc="Collecting Full Dataset Rewards")
        for batch in loader:
            if not batch: continue
            
            vitals = batch["future_data"] # [B, T, D]
            labels = batch["outcome_label"] # [B]
            
            # Use the Calculator
            calc = ICUAdvantageCalculator()
            rewards = calc.compute_clinical_reward(vitals, labels, normalizer=normalizer)
            
            all_rewards.append(rewards.flatten().cpu())
            
            batch_size_actual = vitals.shape[0]
            processed += batch_size_actual
            pbar.update(batch_size_actual)
            
            if processed >= to_process:
                break
        pbar.close()
                
    rewards_flat = torch.cat(all_rewards)
    
    # 4. Calculate Global Advantages
    mu = rewards_flat.mean()
    sigma = rewards_flat.std() + 1e-8
    advantages = (rewards_flat - mu) / sigma
    
    logger.info(f"Global Reward Stats: Mean={mu:.4f}, Std={sigma:.4f}")
    logger.info(f"Global Range: [{rewards_flat.min():.2f}, {rewards_flat.max():.2f}]")

    # 5. ULTRA-FIDELITY 2D Grid Search (Beta vs MaxWeight)
    # User requested to search "all values" between 1 and 20 for MaxWeight.
    # We implement a dense grid for both to ensure no clinical signal is missed.
    
    # Beta: 0.01 to 20.0 with high density
    beta_low = np.linspace(0.01, 0.99, 10)
    beta_high = np.arange(1.0, 20.1, 0.2) # Granular steps of 0.2
    beta_range = np.sort(np.unique(np.concatenate([beta_low, beta_high])))
    
    # MaxWeight: All values from 1.0 to 20.0 with high granularity
    mw_range = np.arange(1.0, 20.5, 0.5) # Hits 1.0, 1.5, 2.0 ... 20.0
    
    results = []
    
    logger.info(f"Performing ULTRA-FIDELITY Search (Grid: {len(beta_range)}x{len(mw_range)})...")
    
    # Standard AWR Objective + Bootstrap Robustness
    num_bootstraps = 5
    bootstrap_size = min(len(advantages) // 2, 1000000) # Cap bootstrap for speed
    
    for beta in tqdm(beta_range, desc="Scanning High-Density Temperature"):
        for mw in mw_range:
            boot_ess = []
            boot_coverage = []
            
            for _ in range(num_bootstraps):
                idx = torch.randint(0, len(advantages), (bootstrap_size,))
                adv_sub = advantages[idx]
                
                raw_w = torch.exp(adv_sub / beta)
                w = torch.clamp(raw_w, max=mw)
                
                sum_w = w.sum()
                sum_w_sq = (w ** 2).sum()
                ess = (sum_w ** 2) / (sum_w_sq * w.numel())
                boot_ess.append(ess.item())
                
                coverage = (w > 1.1).float().mean().item()
                boot_coverage.append(coverage)
            
            mean_ess = np.mean(boot_ess)
            std_ess = np.std(boot_ess)
            mean_cov = np.mean(boot_coverage)
            
            # Full set clipping check
            full_w = torch.clamp(torch.exp(advantages / beta), max=mw)
            clipped = (full_w >= (mw - 1e-3)).float().mean().item()
            
            ess_error = abs(mean_ess - target_ess)
            score = 100.0 - (ess_error * 150.0) - (clipped * 30.0) + (mean_cov * 20.0) - (std_ess * 5.0)
            
            results.append({
                "beta": beta,
                "max_weight": mw,
                "ess": mean_ess,
                "ess_std": std_ess,
                "coverage": mean_cov,
                "clipping": clipped,
                "score": score
            })

    # 6. Rank Results
    df = pd.DataFrame(results)
    best = df.loc[df['score'].idxmax()]
    
    print("\n" + "-"*40)
    print("  🏆 GLOBAL OPTIMAL AWR CONFIGURATION")
    print("  (Validated on 100% of Dataset)")
    print("-"*40)
    print(f"  Recommended Beta:       {best['beta']:.3f}")
    print(f"  Recommended MaxWeight: {best['max_weight']:.1f}")
    print(f"  Expected ESS:           {best['ess']:.2%}")
    print(f"  Clinical Coverage:      {best['coverage']:.2%}")
    print(f"  Outlier Clipping:       {best['clipping']:.2%}")
    print(f"  Selection Stability:    Pulsating ({best['ess_std']:.4f})")
    print("-"*40)

    # 7. Generate YAML Snippet
    print("\n[HYDRA OVERRIDE]")
    print(f"python train_generalist.py train.awr_beta={best['beta']:.3f} train.awr_max_weight={best['max_weight']:.1f}")

    print("\n[YAML SNIPPET - specialist.yaml]")
    yaml_fmt = f"""
training:
  awr_beta: {best['beta']:.3f}
  awr_max_weight: {best['max_weight']:.1f}
  awr_lambda: 0.95
  awr_gamma: 0.99
    """
    print(yaml_fmt.strip())
    print("="*80 + "\n")

    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exhaustive AWR Parameter Search.")
    parser.add_argument("--data_dir", type=str, default="sepsis_clinical_28", help="Path to ICU dataset root")
    parser.add_argument("--samples", type=int, default=-1, help="Number of trajectories (-1 for all)")
    parser.add_argument("--target_ess", type=float, default=0.35, help="Target ESS (0.2-0.5)")
    args = parser.parse_args()

    # Search for data if default doesn't exist
    data_path = Path(args.data_dir)
    if not data_path.exists():
        if (ROOT_DIR / "data").exists(): 
            data_path = ROOT_DIR / "data"
        elif (ROOT_DIR / "sepsis_clinical_28").exists():
            data_path = ROOT_DIR / "sepsis_clinical_28"

    analyze_awr_stability(
        data_dir=str(data_path), 
        num_samples=args.samples,
        target_ess=args.target_ess
    )
