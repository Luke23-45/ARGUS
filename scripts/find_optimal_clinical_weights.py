#!/usr/bin/env python3
"""
scripts/find_optimal_clinical_weights.py
--------------------------------------------------------------------------------
APEX-MoE ICU Pipeline: Clinical Weight Optimization & Audit (v2.0 Vectorized)
Author: APEX Research Team
Context: Imbalanced Sepsis Learning (2.9% Scarcity)

Objective: 
Rather than assuming 'pos_weight', this script audits the entire training 
distribution to find the mathematical 'Golden Ratio' for Weighted Cross-Entropy.

Optimization Theory:
1.  **Effective Sample Size (ESS)**: Ensures the minority class (Sepsis) has
    enough 'Gradient Mass' to influence the Transformer backbone.
    Formula: ESS = (Sum W)^2 / (Sum W^2)
2.  **Balanced Odds**: Calculates weights such that Expected Loss(Healthy) 
    equals Expected Loss(Sepsis) in the cold-start phase.
3.  **Signal Amplification**: Scales the auxiliary head based on task difficulty.
"""

import os
import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Project imports
try:
    # Use the robust SOTA dataset for the most accurate audit
    from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
except ImportError as e:
    print(f"[CRITICAL] Could not import ICU modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WeightOptimizer")
warnings.filterwarnings("ignore")

def calculate_ess(n_neg: int, n_pos: int, pos_weight: float) -> float:
    """
    Calculates Effective Sample Size (ESS) for a given binary weighting scheme.
    Theory: Kish's ESS formula applied to class weights.
    W_neg = 1.0, W_pos = pos_weight
    """
    sum_w = (n_neg * 1.0) + (n_pos * pos_weight)
    sum_w_sq = (n_neg * 1.0**2) + (n_pos * pos_weight**2)
    
    if sum_w_sq == 0: return 0.0
    return (sum_w ** 2) / sum_w_sq

def calculate_optimal_weights(total_windows: int, sepsis_windows: int) -> Dict[str, Any]:
    """
    Applies imbalanced learning formulas to find the optimization sweet spot.
    """
    healthy_windows = total_windows - sepsis_windows
    prevalence = sepsis_windows / total_windows if total_windows > 0 else 0
    
    if sepsis_windows == 0:
        logger.warning("No Sepsis cases found in audit! Defaulting to neutral weights.")
        return {
            "prevalence": 0,
            "recommended_pos_weight": 1.0,
            "recommended_aux_scale": 0.1,
            "imbalance_ratio": float('inf'),
            "ess_ratio": 1.0
        }

    # 1. Standard Balanced Weight (Inverse Class Frequency)
    # W_pos = N_neg / N_pos
    # This theoretically equalizes the expected gradient magnitude from both classes.
    pos_weight = healthy_windows / sepsis_windows
    
    # 2. Aux Loss Scaling
    # As scarcity increases, the Router needs more 'Loudness' to learn the signal.
    # Logarithmic scaling prevents explosion for extremely rare events.
    # Base 0.1, scales up as task gets harder (pos_weight increases).
    aux_scale = 0.1 * (1.0 + np.log10(max(1.0, pos_weight / 5.0)))
    
    # 3. ESS Audit
    # Check if this weight destroys the effective batch size
    # A very high weight concentrates all gradient signal on a few examples (Low ESS)
    ess = calculate_ess(healthy_windows, sepsis_windows, pos_weight)
    ess_ratio = ess / total_windows

    return {
        "prevalence": prevalence,
        "theoretical_pos_weight": pos_weight,
        "recommended_pos_weight": round(pos_weight, 2),
        "recommended_aux_scale": round(aux_scale, 3),
        "imbalance_ratio": healthy_windows / sepsis_windows,
        "ess_absolute": int(ess),
        "ess_ratio": ess_ratio
    }

def main():
    parser = argparse.ArgumentParser(description="APEX-MoE Clinical Weight Optimizer")
    parser.add_argument("--dataset_dir", type=str, default="sepsis_clinical_28")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--history_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=-1) # -1 = Full Audit
    parser.add_argument("--batch_size", type=int, default=4096) # Vectorized speed
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    logger.info(f"🔍 AUDITING CLINICAL WINDOWS: {args.dataset_dir}/{args.split}")
    
    # 1. Load Dataset (Robust SOTA Loader)
    try:
        dataset = ICUSotaDataset(
            dataset_dir=args.dataset_dir,
            split=args.split,
            history_len=args.history_len,
            pred_len=args.pred_len,
            augment_noise=0.0, # Audit raw data
            augment_mask_prob=0.0
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Vectorized Audit Loop
    total_available = len(dataset)
    audit_samples = total_available if args.max_samples <= 0 else min(args.max_samples, total_available)
    
    logger.info(f"Auditing {audit_samples:,} sliding windows (Vectorized)...")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True, # Random sampling if max_samples < total
        num_workers=args.num_workers,
        collate_fn=robust_collate_fn,
        pin_memory=True
    )
    
    total_processed = 0
    sepsis_count = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Engine: {device}")
    
    pbar = tqdm(total=audit_samples, desc="Scanning Distribution")
    
    for batch in loader:
        if not batch: continue
        
        # Resolve Labels (Robust Fallback)
        if "outcome_label" in batch:
            labels = batch["outcome_label"]
        elif "phase_label" in batch:
            # Sepsis is Phase 1 (Pre-Shock) or 2 (Shock)
            labels = (batch["phase_label"] >= 1).float()
        else:
            labels = torch.zeros(batch["future_data"].shape[0])
            
        # Count Positives
        # Move to GPU for fast summation if available, else CPU
        labels = labels.to(device)
        batch_pos = (labels > 0.5).sum().item()
        
        sepsis_count += batch_pos
        current_batch_size = labels.shape[0]
        total_processed += current_batch_size
        pbar.update(current_batch_size)
        
        if total_processed >= audit_samples:
            break
            
    pbar.close()
    
    # 3. Calculate Optimization
    results = calculate_optimal_weights(total_processed, sepsis_count)
    results["total_audited"] = total_processed
    results["sepsis_detected"] = sepsis_count

    # 4. Report Generation
    print("\n" + "═" * 80)
    print("🧠 APEX-MoE CLINICAL WEIGHT OPTIMIZATION REPORT (SOTA v2.0)")
    print("═" * 80)
    
    print(f"\n📊 DYNAMIC AUDIT RESULTS (n={results['total_audited']:,}):")
    print(f"   Sepsis-Positive Windows: {results['sepsis_detected']:,}")
    print(f"   Window Prevalence:       {results['prevalence']*100:.3f}%")
    print(f"   Imbalance Ratio:         1 : {results['imbalance_ratio']:.1f}")
    
    print(f"\n⚖️  STABILITY ANALYSIS:")
    print(f"   Theoretical Weight:      {results['theoretical_pos_weight']:.4f}")
    print(f"   Effective Sample Size:   {results['ess_absolute']:,} ({results['ess_ratio']:.1%} of data)")
    
    if results['ess_ratio'] < 0.10:
        print("   ⚠️  WARNING: Low ESS (<10%). Gradients may be unstable. Consider Focal Loss.")
    else:
        print("   ✅  STATUS: Gradient Signal is Healthy (ESS > 10%).")
    
    print(f"\n🎯 RECOMMENDED HYPERPARAMETERS:")
    print(f"   pos_weight:           {results['recommended_pos_weight']}")
    print(f"   aux_loss_scale:       {results['recommended_aux_scale']}")
    
    print(f"\n📋 HYDRA OVERRIDE SNIPPET:")
    print(f"   train.pos_weight={results['recommended_pos_weight']} train.aux_loss_scale={results['recommended_aux_scale']}")
    
    print("\n" + "═" * 80 + "\n")
    
    # Optional: Save to JSON for pipeline automation
    out_file = Path("optimal_weights.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()