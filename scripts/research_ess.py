
import os
import sys
import torch
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.datasets.normalizer import ClinicalNormalizer
from torch.utils.data import DataLoader

# Setup minimal logging to avoid clutter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RESEARCH")

def run_research():
    print("\n" + "="*80)
    print("APEX-MoE: DEEP RESEARCH - ESS STABILITY ANALYSIS")
    print("="*80)

    # 1. Setup Data
    data_dir = "sepsis_clinical_28"
    print(f"[1] Loading Real Sepsis Data from {data_dir}...")
    dataset = ICUSotaDataset(
        dataset_dir=data_dir,
        split="val",
        history_len=24,
        pred_len=6,
        augment_noise=0.0
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True, 
        collate_fn=robust_collate_fn
    )
    
    batch = next(iter(loader))
    print(f"    Batch acquired. Outcomes: {batch['outcome_label'].sum().item()} deaths out of {batch['outcome_label'].shape[0]}")

    # 2. Setup Normalizer (Calibrated from index)
    print("[2] Calibrating Normalizer...")
    normalizer = ClinicalNormalizer(ts_channels=28, static_channels=6)
    # The index usually contains the stats
    index_path = Path(data_dir) / "val_index.json"
    with open(index_path, 'r') as f:
        idx_data = json.load(f)
        ts_cols = idx_data["metadata"].get("ts_columns")
    
    normalizer.calibrate_from_stats(index_path, ts_cols)

    # 3. Advantage Pipeline
    print("[3] Simulating Advantage Execution Flow...")
    calculator = ICUAdvantageCalculator(
        beta=1.0, 
        adaptive_beta=True, 
        adaptive_clipping=True
    )

    with torch.no_grad():
        # Compute Clinical Rewards
        rewards = calculator.compute_clinical_reward(
            batch["future_data"], 
            batch["outcome_label"],
            dones=batch["is_terminal"],
            src_mask=batch["future_mask"],
            normalizer=normalizer
        )
        
        # [RESEARCH ONLY] Add artificial dense variance to verify step-level credit assignment
        # In real training, this comes from the Dense Sepsis rewards.
        rewards += torch.randn_like(rewards) * 0.1
        
        # Blind Critic Simulation
        v_preds = torch.zeros(rewards.shape[0], rewards.shape[1])
        advantages = calculator.compute_gae(rewards, v_preds)
        
        # [SOTA v3.1] Mask-Aware Weights (1,536 clinical moments)
        f_mask = batch["future_mask"]
        if f_mask.dim() == 3: f_mask = f_mask.any(dim=-1)
        
        weights, diag = calculator.calculate_awr_weights(
            advantages, 
            mask=f_mask
        )

    # 4. Deep Analysis
    print("\n" + "-"*40)
    print("ANALYSIS METRICS (SOTA v3.1)")
    print("-"*40)
    print(f"Pool Size:         {advantages.numel()} elements")
    print(f"Valid Moments:     {int(f_mask.sum().item())}")
    print(f"Advantage Std:     {diag.get('adv_std', 0.0):.4f}")
    print(f"ESS (Reported):    {diag['ess']:.6f}")
    print(f"Theoretical 1/N:   {1/f_mask.sum().item():.6f}")
    
    # Check for Exponential Tempering Integrity
    print(f"Max Weight:        {diag['weights_max']:.2f}")
    print(f"Beta (Dynamic):    {diag['beta_dynamic']:.4f}")
    
    # 5. Root Cause Investigation (Are weights per-step?)
    print("\n[5] Step-Level Granularity Audit...")
    # Check variance within the first patient trajectory
    traj_0_weights = weights[0]
    traj_0_mask = f_mask[0]
    valid_weights = traj_0_weights[traj_0_mask.bool()]
    
    if valid_weights.numel() > 1:
        w_variance = valid_weights.std() / (valid_weights.mean() + 1e-8)
        print(f"    Traj 0 Intra-Step Variance: {w_variance.item():.4f}")
        if w_variance > 1e-4:
            print("    VERIFIED: Weights are differentiated at the clinical moment level.")
        else:
            print("    WARNING: Weights are uniform within trajectory (Coarse weighting persists).")
    else:
        print("    Insufficient valid steps in Traj 0 for variance analysis.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_research()
