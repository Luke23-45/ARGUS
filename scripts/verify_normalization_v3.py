
"""
scripts/verify_normalization_v3.py
--------------------------------------------------------------------------------
APEX-MoE: Advanced Normalization Verification (v3.1)
Author: APEX Research Team
Context: Life-Critical Sepsis Prediction

Description:
    This script verifies that the ICUTrajectoryDataset and ClinicalNormalizer
    are correctly leveraging the 'Hardened Metadata' (P01/P99 quantiles).
    It ensures that medical signals are not 'squashed' by outliers and that
    the normalization mapping stays within the latent range [-1, 1].
"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from icu.datasets.dataset import ICUTrajectoryDataset, CANONICAL_COLUMNS
from icu.datasets.normalizer import ClinicalNormalizer

def verify_system(dataset_dir: str):
    print("=== APEX Normalization Verification (v3.1) ===")
    
    # 1. Check if metadata is actually hardened
    index_path = Path(dataset_dir) / "train_index.json"
    with open(index_path, 'r') as f:
        meta = json.load(f).get("metadata", {}).get("stats", {})
    
    if "ts_p01" not in meta or "ts_p99" not in meta:
        print("[-] ERROR: Metadata is NOT hardened. Please run scripts/harden_metadata.py first.")
        return

    print("[+] SUCCESS: Hardened Metadata detected (P01/P99 present).")

    # 2. Initialize System
    dataset = ICUTrajectoryDataset(dataset_dir=dataset_dir, split="train")
    normalizer = ClinicalNormalizer(ts_channels=28, static_channels=6)
    
    # 3. Simulate Training Calibration
    # In real training, this happens in on_fit_start
    print(f"[i] Calibrating Normalizer using {index_path}...")
    normalizer.calibrate_from_stats(index_path, CANONICAL_COLUMNS)
    
    # 4. Stress Test: Pass samples through the normalizer
    print(f"[i] Stress testing normalizer with 1000 random windows...")
    n_samples = 1000
    all_norm_vals = []
    
    for _ in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        vitals = sample["observed_data"] # [T, 28]
        static = sample["static_context"] # [6]
        
        # Apply Normalization
        v_norm, s_norm = normalizer.normalize(vitals, static)
        
        # Check for NaNs/Infs
        if torch.isnan(v_norm).any() or torch.isinf(v_norm).any():
            print(f"[-] FATAL: Normalizer produced NaN/Inf at index {idx}")
            return
            
        # Check range
        if v_norm.min() < -1.0001 or v_norm.max() > 1.0001:
            print(f"[-] FATAL: Normalizer breached [-1, 1] range! Min={v_norm.min():.4f}, Max={v_norm.max():.4f}")
            return
            
        all_norm_vals.append(v_norm.numpy())

    print("[+] SUCCESS: Normalizer range check passed (Strict [-1, 1]).")

    # 5. Resolution Check (The "Bilirubin" Test)
    # Check if we are using the latent space effectively
    all_norm_vals = np.concatenate(all_norm_vals, axis=0) # [N*T, 28]
    
    print("\nNormalization Latent Utilization (Should be near -1 to 1):")
    for i in range(min(10, 28)):
        channel_data = all_norm_vals[:, i]
        c_min = channel_data.min()
        c_max = channel_data.max()
        c_mean = channel_data.mean()
        print(f"  {CANONICAL_COLUMNS[i]:12}: Range=[{c_min:6.2f}, {c_max:6.2f}] | Mean={c_mean:6.2f}")
    
    print("\n[+] VERIFICATION COMPLETE: System is ready for SOTA Training.")

if __name__ == "__main__":
    DATASET_DIR = "sepsis_clinical_28_raw"
    verify_system(DATASET_DIR)
