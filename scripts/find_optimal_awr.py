#!/usr/bin/env python3
"""
scripts/find_optimal_awr.py
--------------------------------------------------------------------------------
SOTA ICU AWR Parameter Optimization & Stability Search (Vectorized Edition).

This script performs a high-fidelity audit of the advantage distribution across
the real ICU dataset. It uses Vectorized Grid Search to jointly optimize:
1.  AWR Temperature (Beta): Controls the peakiness of clinical weighting.
2.  Clipping Threshold (MaxWeight): Protects against out-of-distribution outliers.

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
    target_ess: float = 0.35,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("\n" + "="*80)
    print("  🚀 APEX-MoE: EXHAUSTIVE AWR Stability & Parameter Optimizer v5.0")
    print("  Source of Truth: FULL DATASET")
    print(f"  Compute Engine: {device.upper()}")
    print("="*80 + "\n")

    # 1. Load Dataset & Metadata
    logger.info(f"Loading Ground-Truth Dataset from: {data_dir}")
    try:
        dataset = ICUSotaDataset(
            dataset_dir=data_dir,
            split="train",
            history_len=24,
            pred_len=6,
            augment_noise=0.0 # Strict evaluation mode
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Extract Normalizer (Physical Unit Ground Truth)
    # We load this to ensure the AdvantageCalculator has access to physics bounds if needed.
    index_path = Path(data_dir) / "train_index.json"
    if not index_path.exists():
        index_path = Path(data_dir).parent / "train_index.json"
    
    normalizer = ClinicalNormalizer(ts_channels=28, static_channels=6)
    try:
        # Check if dataset has column metadata, else fall back to canonical
        channel_names = getattr(dataset, 'ts_columns', None) 
        if not channel_names:
            logger.warning("Dataset metadata missing 'ts_columns'. Using internal CANONICAL spec.")
            from icu.datasets.dataset import CANONICAL_COLUMNS
            channel_names = CANONICAL_COLUMNS
            
        normalizer.calibrate_from_stats(index_path, channel_names)
        logger.info("Normalizer calibrated successfully.")
    except Exception as e:
        logger.warning(f"Could not load normalizer from index: {e}. Proceeding with RAW logic.")
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
    
    calc = ICUAdvantageCalculator()
    all_rewards = []
    processed = 0
    
    with torch.no_grad():
        pbar = tqdm(total=to_process, desc="Collecting Rewards")
        for batch in loader:
            if not batch: continue
            
            # Move to device for fast calculation
            vitals = batch["future_data"].to(device) # [B, T, D]
            
            # Robust label handling
            if "outcome_label" in batch:
                labels = batch["outcome_label"].to(device)
            elif "phase_label" in batch:
                # Fallback: Treat Sepsis (Phase 2) and Pre-Shock (Phase 1) as positive
                labels = (batch["phase_label"] >= 1).float().to(device)
            else:
                labels = torch.zeros(vitals.shape[0], device=device)

            # Compute Rewards
            # Note: We pass normalizer=None because 'vitals' from dataset are RAW.
            # The calculator handles raw thresholds internally.
            
            # [FIX] Extract Mask for Padding Safety
            # We are evaluating FUTURE predictions, so we must use future_mask (length 6)
            # NOT src_mask (history length 24).
            src_mask = batch.get("future_mask", None)
            if src_mask is not None:
                src_mask = src_mask.to(device)
            
            rewards = calc.compute_clinical_reward(vitals, labels, normalizer=None, src_mask=src_mask)
            
            # Aggregation: Mean reward per trajectory is standard for AWR baselines
            # unless we have a dense value function.
            # [FIX] Masked Mean to ignore padding
            if src_mask is not None:
                # Ensure mask is [B, T]
                if src_mask.dim() == 3: src_mask = src_mask.any(dim=-1)
                mask_sums = src_mask.float().sum(dim=1) + 1e-8
                traj_rewards = (rewards * src_mask.float()).sum(dim=1) / mask_sums
            else:
                traj_rewards = rewards.mean(dim=1)
                
            all_rewards.append(traj_rewards.cpu())
            
            processed += vitals.shape[0]
            pbar.update(vitals.shape[0])
            
            if processed >= to_process:
                break
        pbar.close()
                
    rewards_flat = torch.cat(all_rewards).to(device)
    
    # 4. Distribution Analysis (Sanity Check)
    mu = rewards_flat.mean()
    sigma = rewards_flat.std() + 1e-8
    
    logger.info("-" * 40)
    logger.info(f"Global Reward Stats (N={len(rewards_flat):,}):")
    logger.info(f"  Mean: {mu:.4f} | Std: {sigma:.4f}")
    logger.info(f"  Min:  {rewards_flat.min():.4f} | Max: {rewards_flat.max():.4f}")
    
    # Check for degeneracy
    if sigma < 1e-5:
        logger.critical("CRITICAL FAILURE: Reward distribution is degenerate (variance ~ 0).")
        logger.critical("Check AdvantageCalculator logic or label integrity.")
        return

    # Standardize Advantages (Cold Start Assumption: V(s) = mu)
    # A_norm = (R - mu) / sigma
    advantages = (rewards_flat - mu) / sigma
    
    # 5. VECTORIZED ULTRA-FIDELITY GRID SEARCH
    # Instead of Python loops, we broadcast tensors for GPU acceleration.
    
    # Beta: 0.05 to 5.0 (Log-space search is often better, but linear covers dense region)
    betas = torch.linspace(0.05, 5.0, steps=100, device=device)
    # MaxWeight: 1.0 to 50.0
    max_weights = torch.linspace(1.0, 50.0, steps=50, device=device)
    
    logger.info(f"Performing Vectorized Search ({len(betas)} Betas x {len(max_weights)} Cliffs)...")
    
    # Expand Dimensions for Broadcasting
    # Advantages: [N, 1, 1]
    # Betas:      [1, B, 1]
    # MaxWeights: [1, 1, M]
    
    A = advantages.view(-1, 1, 1)      # [N, 1, 1]
    B_vec = betas.view(1, -1, 1)       # [1, 100, 1]
    M_vec = max_weights.view(1, 1, -1) # [1, 1, 50]
    
    # Step A: Compute Raw Weights [N, B, 1]
    # W_raw = exp(A / Beta)
    # Note: We do this first to avoid re-computing exp() for every max_weight
    logger.info("  > Step 1: Exponentiation...")
    W_raw = torch.exp(A / B_vec)
    
    # Step B: Apply Clipping [N, B, M]
    # This might be memory intensive. If N is huge, we chunk it.
    # N=50k, B=100, M=50 => 250M elements (1GB float32). Safe on most GPUs.
    logger.info("  > Step 2: Vectorized Clipping & ESS...")
    
    results = []
    
    # Iterate over betas to keep memory reasonable (chunking by Beta)
    for i, beta in enumerate(tqdm(betas, desc="Scanning Hyperplane")):
        # W_slice: [N, 1] -> broadcast against M_vec: [1, M]
        w_slice = W_raw[:, i, :] # [N, 1]
        
        # Clip against all MaxWeights simultaneously
        # clipped_w: [N, M]
        clipped_w = torch.minimum(w_slice, M_vec.squeeze(0))
        
        # Compute Statistics along dimension 0 (Samples)
        sum_w = clipped_w.sum(dim=0)        # [M]
        sum_w_sq = (clipped_w ** 2).sum(dim=0) # [M]
        N = clipped_w.shape[0]
        
        # ESS = (Sum W)^2 / (Sum W^2)
        ess = (sum_w ** 2) / (sum_w_sq + 1e-8)
        ess_norm = ess / N # Normalized ESS [0, 1]
        
        # Coverage: Fraction of samples with weight > 1.0 (Information gain)
        # Using a soft threshold slightly above 1.0 to detect "informative" samples
        coverage = (clipped_w > 1.05).float().mean(dim=0)
        
        # Clipping Rate: Fraction of samples hitting the cap
        # We check if w >= max_weight - epsilon
        is_clipped = (clipped_w >= (M_vec.squeeze(0) - 1e-3)).float().mean(dim=0)
        
        # Move stats to CPU for list storage
        # [FIX] Flatten to ensure 1D arrays (M,)
        ess_norm_cpu = ess_norm.cpu().numpy().flatten()
        cov_cpu = coverage.cpu().numpy().flatten()
        clip_cpu = is_clipped.cpu().numpy().flatten()
        mw_cpu = max_weights.cpu().numpy().flatten()
        
        beta_val = beta.item()
        
        for j in range(len(max_weights)):
            # Scoring Function (The "Heart" of the Logic)
            # We penalize ESS deviation from target (0.35)
            # We reward Coverage (more data used is better)
            # We penalize excessive Clipping (wasted signal)
            
            curr_ess = ess_norm_cpu[j]
            curr_mw = mw_cpu[j]
            curr_cov = cov_cpu[j]
            curr_clip = clip_cpu[j]
            
            # Hard Constraints
            if curr_ess < 0.15 or curr_ess > 0.60:
                score = -1000.0 # Disqualify
            else:
                ess_error = abs(curr_ess - target_ess)
                # Score = Stability - Penalty + Bonus
                score = 100.0 \
                        - (ess_error * 200.0) \
                        - (curr_clip * 20.0) \
                        + (curr_cov * 30.0)
            
            results.append({
                "beta": beta_val,
                "max_weight": curr_mw,
                "ess": curr_ess,
                "coverage": curr_cov,
                "clipping": curr_clip,
                "score": score
            })

    # 6. Rank Results
    df = pd.DataFrame(results)
    # Filter out failed runs
    df = df[df['score'] > -900]
    
    if df.empty:
        logger.error("No configurations met the minimum stability criteria (ESS 15%-60%).")
        logger.error("Advice: Your data might be too noisy. Try increasing AWR Lambda.")
        return

    best = df.loc[df['score'].idxmax()]
    
    print("\n" + "-"*40)
    print("  🏆 GLOBAL OPTIMAL AWR CONFIGURATION")
    print(f"  (Optimized on {len(rewards_flat)} samples)")
    print("-" * 40)
    print(f"  Recommended Beta:       {best['beta']:.3f}")
    print(f"  Recommended MaxWeight:  {best['max_weight']:.1f}")
    print(f"  Expected ESS:           {best['ess']:.2%} (Target: {target_ess:.0%})")
    print(f"  Clinical Coverage:      {best['coverage']:.2%}")
    print(f"  Outlier Clipping:       {best['clipping']:.2%}")
    print(f"  Optimization Score:     {best['score']:.2f}")
    print("-" * 40)

    # 7. Generate YAML Snippet
    print("\n[HYDRA OVERRIDE]")
    print(f"python train_generalist.py train.awr_beta={best['beta']:.3f} train.awr_max_weight={best['max_weight']:.1f}")

    print("\n[YAML SNIPPET - specialist.yaml]")
    yaml_fmt = f"""
training:
  # Optimized via find_optimal_awr.py
  awr_beta: {best['beta']:.3f}
  awr_max_weight: {best['max_weight']:.1f}
  awr_lambda: 0.95
  awr_gamma: 0.99
    """
    print(yaml_fmt.strip())
    print("="*80 + "\n")
    
    # Save to file for automated pipelines
    output_file = Path("optimized_awr_params.json")
    with open(output_file, 'w') as f:
        json.dump(best.to_dict(), f, indent=4)
    logger.info(f"Optimal parameters saved to {output_file}")

    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized AWR Parameter Search.")
    parser.add_argument("--data_dir", type=str, default="sepsis_clinical_28", help="Path to ICU dataset root")
    parser.add_argument("--samples", type=int, default=-1, help="Number of trajectories (-1 for all)")
    parser.add_argument("--target_ess", type=float, default=0.35, help="Target ESS (0.2-0.5)")
    parser.add_argument("--batch_size", type=int, default=4096, help="Inference batch size")
    args = parser.parse_args()

    # Search for data if default doesn't exist
    data_path = Path(args.data_dir)
    if not data_path.exists():
        # Try common project paths
        if (ROOT_DIR / "data").exists(): 
            data_path = ROOT_DIR / "data"
        elif (ROOT_DIR / "sepsis_clinical_28").exists():
            data_path = ROOT_DIR / "sepsis_clinical_28"
    
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    analyze_awr_stability(
        data_dir=str(data_path), 
        num_samples=args.samples,
        batch_size=args.batch_size,
        target_ess=args.target_ess
    )