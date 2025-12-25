
"""
scripts/harden_metadata.py
--------------------------------------------------------------------------------
Surgically updates dataset index files with Robust Quantiles (P01/P99).
This ensures the ClinicalNormalizer (v3.1) can perform Winsorization 
without requiring a full data re-ingestion.

Technique: Reservoir Sampling (Memory-Safe Percentile Estimation)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from icu.datasets.dataset import ICUTrajectoryDataset, CANONICAL_COLUMNS

def calculate_robust_quantiles(dataset: ICUTrajectoryDataset, reservoir_size: int = 200000):
    """
    Scans the dataset and collects samples to estimate 1st and 99th percentiles.
    Uses per-channel Reservoir Sampling for mathematically unbiased estimation.
    """
    ts_channels = 28
    # Reservoir for each channel
    reservoirs = [[] for _ in range(ts_channels)]
    n_seen_c = [0] * ts_channels
    
    print(f"Scanning dataset ({len(dataset)} windows) for robust quantiles...")
    
    # Sample every 2nd window to balance speed and coverage (~225k windows)
    stride = 2 
    
    for i in tqdm(range(0, len(dataset), stride)):
        try:
            sample = dataset[i]
            if sample is None: continue
            
            # Observed data: [T, C]
            v = sample["observed_data"]
            n_steps = v.shape[0]
            
            # Sample 5 points per window to capture temporal variance
            for _ in range(5):
                t_idx = np.random.randint(0, n_steps)
                vals = v[t_idx]
                
                for c in range(ts_channels):
                    val = vals[c].item()
                    # Filter anomalies and missing values
                    if not np.isnan(val) and not np.isinf(val):
                        n_seen_c[c] += 1
                        if len(reservoirs[c]) < reservoir_size:
                            reservoirs[c].append(val)
                        else:
                            # Standard Reservoir Replacement logic (Algorithm R)
                            r = np.random.randint(0, n_seen_c[c])
                            if r < reservoir_size:
                                reservoirs[c][r] = val
        except Exception:
            continue

    p01 = []
    p99 = []
    
    print("\nComputing statistics...")
    for c in range(ts_channels):
        data = np.array(reservoirs[c])
        if len(data) > 0:
            p01.append(float(np.percentile(data, 1)))
            p99.append(float(np.percentile(data, 99)))
        else:
            print(f"Warning: No valid data for channel {c}")
            p01.append(0.0)
            p99.append(1.0)
            
    # Verification Summary
    print("\nEstimated Robust Quantiles (1st % | 99th %):")
    for i in range(min(10, ts_channels)):
        name = CANONICAL_COLUMNS[i]
        print(f"  {name:12}: {p01[i]:10.2f} | {p99[i]:10.2f}")
    print("  ...")
            
    return p01, p99

def harden_metadata(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    splits = ["train", "val"]
    
    for split in splits:
        index_file = dataset_path / f"{split}_index.json"
        if not index_file.exists():
            print(f"Skipping {split}: Index not found at {index_file}")
            continue
            
        print(f"\n--- Hardening {split.upper()} Metadata ---")
        
        # 1. Load Dataset
        ds = ICUTrajectoryDataset(
            dataset_dir=dataset_dir,
            split=split,
            validate_schema=True
        )
        
        # 2. Compute Quantiles
        p01, p99 = calculate_robust_quantiles(ds)
        
        # 3. Load and Update JSON
        with open(index_file, 'r') as f:
            full_index = json.load(f)
            
        if "metadata" not in full_index:
            full_index["metadata"] = {}
        if "stats" not in full_index["metadata"]:
            full_index["metadata"]["stats"] = {}
            
        stats = full_index["metadata"]["stats"]
        stats["ts_p01"] = p01
        stats["ts_p99"] = p99
        full_index["metadata"]["ts_columns"] = CANONICAL_COLUMNS
        
        # 4. Save
        with open(index_file, 'w') as f:
            json.dump(full_index, f, indent=2)
            
        print(f"Successfully hardened {index_file}")

if __name__ == "__main__":
    DATASET_DIR = "sepsis_clinical_28_raw"
    harden_metadata(DATASET_DIR)
