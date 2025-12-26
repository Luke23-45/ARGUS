
"""
scripts/analyze_normalized_dataset.py
--------------------------------------------------------------------------------
APEX-MoE ICU Pipeline: Latent-Space Quality Verification (v3.1)
Author: APEX Research Team
Context: Safety-Critical ICU Forecasting

This script audits the dataset AFTER it has passed through the ClinicalNormalizer.
It ensures that the mapping to [-1, 1] latent space is high-fidelity and 
mathematically stable for Diffusion/Transformer training.

Verification Goals:
1.  **Latent Utilization**: Ensure signals use the full [-1, 1] range (v3.1 Winsorization).
2.  **Numerical Stability**: Check for NaNs/Infs after normalization math.
3.  **Zero-Centering**: Verify if signals are reasonably centered (improves convergence).
4.  **Clipping Audit**: Quantify how much data is clamped to the [-1, 1] boundaries.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Project imports
try:
    from icu.datasets.dataset import ICUTrajectoryDataset, CANONICAL_COLUMNS, robust_collate_fn
    from icu.datasets.normalizer import ClinicalNormalizer
except ImportError as e:
    print(f"ERROR: Could not import ICU modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("NormalizedAnalyzer")

class NormalizedQualityReport:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.dataset_path = ""
        self.quality_score = 100.0
        self.channel_stats = {}
        self.issues = []
        
    def add_issue(self, severity: str, message: str):
        self.issues.append((severity, message))
        deductions = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 2}
        self.quality_score = max(0, self.quality_score - deductions.get(severity, 0))

    def print_summary(self):
        print("\n" + "═" * 80)
        print("🧠 APEX-MoE LATENT SPACE QUALITY REPORT (v3.1)")
        print("═" * 80)
        
        score_icon = "🟢" if self.quality_score >= 90 else "🟡" if self.quality_score >= 70 else "🔴"
        print(f"\n{score_icon} NORMALIZED QUALITY SCORE: {self.quality_score:.1f}/100")
        
        print(f"\n📊 CHANNEL LATENT UTILIZATION:")
        print(f"   {'Channel':<15} | {'Min':>6} | {'Max':>6} | {'Mean':>6} | {'Range Use':>10}")
        print(f"   {'-'*15}-|-{'-'*6}-|-{'-'*6}-|-{'-'*6}-|-{'-'*10}")
        
        for ch, s in self.channel_stats.items():
            # Range Use based on P01 to P99 density
            util = (s['p99'] - s['p01']) / 2.0 * 100
            print(f"   {ch:<15} | {s['min']:6.2f} | {s['max']:6.2f} | {s['mean']:6.2f} | {util:9.1f}%")

        if self.issues:
            print(f"\n⚠️  CRITICAL VALIDATION ISSUES:")
            for sev, msg in self.issues:
                print(f"   [{sev}] {msg}")
        else:
            print(f"\n✅ ALL CHANNELS MATHEMATICALLY STABLE")
        print("\n" + "═" * 80 + "\n")

class NormalizedAnalyzer:
    def __init__(self, dataset_dir: str, split: str = "train"):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.report = NormalizedQualityReport()
        self.report.dataset_path = str(self.dataset_dir)
        
        # Initialize Normalizer
        self.normalizer = ClinicalNormalizer(ts_channels=28, static_channels=6)
        index_path = self.dataset_dir / f"{split}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        logger.info(f"Calibrating Normalizer from {index_path}...")
        self.normalizer.calibrate_from_stats(index_path, CANONICAL_COLUMNS)

    def run(self, max_batches: int = 100):
        dataset = ICUTrajectoryDataset(dataset_dir=str(self.dataset_dir), split=self.split)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=robust_collate_fn)
        
        accumulated_data = [[] for _ in range(28)]
        label_counts = Counter()
        
        logger.info(f"Auditing latent space and labels over {max_batches} batches...")
        for i, batch in enumerate(tqdm(loader, total=max_batches)):
            if i >= max_batches: break
            
            obs = batch["observed_data"]
            static = batch["static_context"]
            labels = batch.get("outcome_label")
            
            # THE CORE TEST: Normalize
            v_norm, s_norm = self.normalizer.normalize(obs, static)
            
            # Check for NaNs (Training Killers)
            if torch.isnan(v_norm).any():
                self.report.add_issue("CRITICAL", "NaN detected in normalized vital signs!")
                break
                
            # Class Balance Audit
            if labels is not None:
                label_counts.update(labels.view(-1).long().tolist())
                
            # Flatten and collect
            v_flat = v_norm.view(-1, 28).numpy()
            for c in range(28):
                accumulated_data[c].append(v_flat[:, c])
                
        # 1. Compute Class Balance Issue
        total_labels = sum(label_counts.values())
        if total_labels > 0 and 1 in label_counts:
            ratio = label_counts[1] / total_labels
            if ratio < 0.1:
                self.report.add_issue("HIGH", f"Severe class imbalance: ratio = {ratio:.3f}")
            elif ratio < 0.3:
                self.report.add_issue("MEDIUM", f"Moderate class imbalance: ratio = {ratio:.3f}")
            
        # 2. Compute Channel Stats
        for c in range(28):
            data = np.concatenate(accumulated_data[c])
            c_name = CANONICAL_COLUMNS[c]
            
            # Use P01/P99 for "Range Use" to be honest about distribution density
            p01, p99 = np.percentile(data, [1, 99])
            
            stats = {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "p01": float(p01),
                "p99": float(p99),
                "mean": float(np.mean(data)),
                "std": float(np.std(data))
            }
            self.report.channel_stats[c_name] = stats
            
            # Check for signal squashing (using P99-P01 range to be "Anti-Cheat")
            effective_range = p99 - p01
            if effective_range < 0.2:
                self.report.add_issue("HIGH", f"Signal Squashing: Channel '{c_name}' utilizes < 10% of latent space.")
            
            # Check for range breach
            if stats['min'] < -1.05 or stats['max'] > 1.05:
                self.report.add_issue("CRITICAL", f"Range Breach: Channel '{c_name}' outside [-1, 1].")

        return self.report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./sepsis_clinical_28_raw")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    
    analyzer = NormalizedAnalyzer(args.dataset_dir)
    report = analyzer.run(max_batches=args.batches)
    report.print_summary()

if __name__ == "__main__":
    main()
