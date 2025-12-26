#!/usr/bin/env python3
"""
scripts/optimize_icu_awr.py
--------------------------------------------------------------------------------
SOTA ICU AWR Parameter Optimization Tool.

This script analyzes a sample of the ICU dataset to find the optimal 
AWR temperature (beta) that balances gradient diversity (ESS) with 
advantage exploitation.

Target: ESS ~ 50% (Standard for robust importance weighting).
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Project Imports
try:
    from icu.utils.advantage_calculator import ICUAdvantageCalculator
    from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
except ImportError as e:
    print(f"[ERROR] Could not import project modules. Ensure CWD is project root: {e}")
    sys.exit(1)

def optimize_beta(data_dir: str, target_ess: float = 0.5, num_samples: int = 1000):
    print("=" * 70)
    print(f"ICU-AWR v2.0: Beta Optimization (Target ESS: {target_ess:.1%})")
    print("=" * 70)

    # 1. Load Dataset
    print(f"[DATA] Loading train split from {data_dir}...")
    try:
        ds = ICUSotaDataset(
            dataset_dir=data_dir,
            split="train",
            augment_noise=0.0
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    loader = torch.utils.data.DataLoader(
        ds, 
        batch_size=min(num_samples, len(ds)), 
        shuffle=True, 
        collate_fn=robust_collate_fn
    )
    
    # Get a large enough batch for representative stats
    batch = next(iter(loader))
    vitals = batch["future_data"]
    labels = batch["outcome_label"]
    
    print(f"[COMPUTE] Analyzing {vitals.shape[0]} episodes...")

    # 2. Setup Calculator
    calc = ICUAdvantageCalculator()
    
    # 3. Compute Clinical Advantages
    # We use a zero baseline for distribution analysis since value nets 
    # are trained against these rewards.
    rewards = calc.compute_clinical_reward(vitals, labels)
    
    # Whiten locally for the analysis
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    advantages = (rewards - mean) / std
    
    print(f"  Reward Mean: {mean:.4f}")
    print(f"  Reward Std:  {std:.4f}")
    print(f"  Reward Range: [{rewards.min():.2f}, {rewards.max():.2f}]")

    # 4. Scan Beta Search Space
    print(f"\n[SCAN] Searching for optimal Beta in range [0.01, 10.0]...")
    
    betas = np.linspace(0.01, 5.0, 500)
    best_beta = 1.0
    best_diff = 1.0
    final_ess = 0.0
    
    for beta in betas:
        calc.beta = beta
        # Calculate weights (no whitening needed here as we whitened above)
        weights, ess = calc.calculate_weights(advantages)
        
        diff = abs(ess - target_ess)
        if diff < best_diff:
            best_diff = diff
            best_beta = beta
            final_ess = ess

    print(f"\n[RESULT] Optimization Complete!")
    print(f"  Recommended Beta: {best_beta:.3f}")
    print(f"  Achieved ESS:     {final_ess:.2%}")
    print(f"  Max Weight:       {calc.max_weight}")

    print("\n--- HYDRA OVERRIDE ---")
    print(f"python train_generalist.py train.awr_beta={best_beta:.3f}")
    
    print("\n--- YAML SNIPPET ---")
    print("training:")
    print(f"  awr_beta: {best_beta:.3f}")
    print(f"  awr_max_weight: {calc.max_weight}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ICU AWR Hyperparameters")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to ICU dataset root")
    parser.add_argument("--ess", type=float, default=0.5, help="Target Effective Sample Size (0.0-1.0)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of trajectories to sample")
    
    args = parser.parse_args()
    
    # Correct path if data_dir is relative
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        # Check if icu_research/data exists
        if (ROOT_DIR / "data").exists():
            data_path = (ROOT_DIR / "data")
        elif (ROOT_DIR / "icu_research" / "data").exists():
            data_path = (ROOT_DIR / "icu_research" / "data")
            
    optimize_beta(str(data_path), target_ess=args.ess, num_samples=args.samples)
