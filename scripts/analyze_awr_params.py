#!/usr/bin/env python3
"""
AWR Hyperparameter Recommendation Script
=========================================

This script analyzes the ICU dataset to recommend optimal AWR (Advantage Weighted Regression)
hyperparameters based on empirical data statistics.

AWR Formula Reference:
    weight_i = exp(advantage_i / β)
    
Where:
    - β (temperature): Controls how sharply advantages affect weights
    - max_weight: Clips extreme weights to prevent gradient explosion

Best Practices (from literature):
    1. β = std(advantages) is a common and effective heuristic
    2. max_weight typically 10-100, prevents exp() overflow
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

try:
    import lmdb
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)


def load_icu_samples(data_dir: Path, max_samples: int = 2000) -> list:
    """
    Load samples accurately using the ICU dataset structure:
    1. Read {split}_index.json for metadata
    2. Open {split}.lmdb for binary data
    3. Reconstruct numpy arrays
    """
    split = "train"
    subdir = data_dir / split
    lmdb_path = subdir / f"{split}.lmdb"
    index_path = subdir / f"{split}_index.json"
    
    if not lmdb_path.exists() or not index_path.exists():
        print(f"[ERROR] Could not find LMDB or Index at {subdir}")
        return []

    print(f"[DATA] Loading index from {index_path}...")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    episodes = index.get('episodes', [])
    if not episodes:
        print("[WARN] No episodes found in index.")
        return []
        
    import random
    if len(episodes) > max_samples:
        random.seed(42)
        episodes = random.sample(episodes, max_samples)
        
    print(f"[DATA] Opening LMDB at {lmdb_path}...")
    # Open with lock=False to avoid Windows file lock issues
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False)
    
    samples = []
    
    with env.begin() as txn:
        for ep in episodes:
            try:
                mods = ep['modalities']
                
                # Fetch Vitals
                v_meta = mods['vitals']
                v_bytes = txn.get(v_meta['key'].encode('ascii'))
                vitals = np.frombuffer(v_bytes, dtype=v_meta['dtype']).reshape(v_meta['shape'])
                
                # Fetch Labels (Outcome)
                outcome = 0
                if 'labels' in mods:
                    l_meta = mods['labels']
                    l_bytes = txn.get(l_meta['key'].encode('ascii'))
                    labels = np.frombuffer(l_bytes, dtype=l_meta['dtype']).reshape(l_meta['shape'])
                    outcome = np.max(labels) # 1 if sepsis ever occurred
                
                samples.append({
                    'future_data': vitals,
                    'outcome_label': outcome
                })
            except Exception:
                continue
                
    env.close()
    return samples


def compute_trajectory_returns(samples: list) -> np.ndarray:
    """Compute returns based on stability and smoothness."""
    returns = []
    for sample in samples:
        future = np.array(sample.get('future_data', []))
        if len(future) == 0:
            continue
            
        # 1. Stability (Inverse of variance across channels)
        future_var = np.mean(np.var(future, axis=0))
        stability = 1.0 / (1e-6 + future_var)
        
        # 2. Smoothness (Inverse of mean absolute difference)
        if len(future) > 1:
            diffs = np.diff(future, axis=0)
            smoothness = 1.0 / (1e-6 + np.mean(np.abs(diffs)))
        else:
            smoothness = 1.0
            
        # Result: higher for stable, smooth trajectories
        trajectory_return = np.log(1e-6 + stability) + np.log(1e-6 + smoothness)
        returns.append(trajectory_return)
        
    return np.array(returns)


def compute_advantages(returns: np.ndarray) -> np.ndarray:
    """Advantage = Return - Mean Return."""
    return returns - np.mean(returns)


def recommend_awr_params(advantages: np.ndarray) -> dict:
    """Recommend Beta (temp) as std(advantages)."""
    adv_std = np.std(advantages)
    
    # Temperature Recommendation: Use standard deviation
    recommended_beta = max(0.1, round(adv_std, 2))
    
    # Max Weight: Cap at ~exp(3) for stability
    recommended_max_weight = round(min(np.exp(3.0), 50.0), 1)
    
    return {
        "beta": recommended_beta,
        "max_weight": recommended_max_weight,
        "stats": {
            "std": round(adv_std, 4),
            "mean": round(np.mean(advantages), 4),
            "max": round(np.max(advantages), 4),
            "min": round(np.min(advantages), 4),
            "count": len(advantages)
        }
    }


def main():
    print("=" * 70)
    print("AWR HYPERPARAMETER ANALYSIS")
    print("Data-Driven Recommendations for ICU Diffusion Training")
    print("=" * 70)
    
    # Standard ICU research data root
    data_dir = Path("icu_research/data")
    if not data_dir.exists():
        data_dir = Path("data")
        
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found. Searched: icu_research/data, data")
        return

    samples = load_icu_samples(data_dir)
    
    if not samples:
        print("[ERROR] No samples loaded. Check if dataset is built.")
        return
        
    print(f"[COMPUTE] Analyzing {len(samples)} episodes...")
    returns = compute_trajectory_returns(samples)
    advantages = compute_advantages(returns)
    rec = recommend_awr_params(advantages)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    stats = rec['stats']
    print(f"  Samples:    {stats['count']}")
    print(f"  Adv Std:    {stats['std']}")
    print(f"  Adv Range:  [{stats['min']}, {stats['max']}]")
    
    print("\n--- RECOMMENDED CONFIG ---")
    print(f"  awr_temperature: {rec['beta']}")
    print(f"  awr_max_weight:  {rec['max_weight']}")
    
    print("\n--- YAML SNIPPET ---")
    print("```yaml")
    print("training:")
    print(f"  awr_temperature: {rec['beta']}")
    print(f"  awr_max_weight: {rec['max_weight']}")
    print("```")


if __name__ == "__main__":
    main()
