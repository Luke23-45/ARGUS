
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from torch.utils.data import DataLoader

def run_diagnostic():
    print("="*60)
    print("APEX-MoE: ESS Stability Diagnostic Suite")
    print("="*60)

    # 1. Setup Dataset
    data_dir = "sepsis_clinical_28"
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} not found.")
        return

    print(f"[1] Loading Validation Dataset from {data_dir}...")
    dataset = ICUSotaDataset(
        dataset_dir=data_dir,
        split="val",
        history_len=24,
        pred_len=6,
        augment_noise=0.0,
        validate_schema=True
    )

    loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True, 
        collate_fn=robust_collate_fn,
        num_workers=0
    )

    # 2. Setup Advantage Calculator
    print("[2] Initializing Advantage Calculator (SOTA 2025 Config)...")
    calculator = ICUAdvantageCalculator(
        beta=1.0, 
        adaptive_beta=True, 
        adaptive_clipping=True
    )

    # 3. Process Batch
    print("[3] Running clinical simulation...")
    batch = next(iter(loader))
    
    # Simulate a "Blind Critic" (Zero value predictions)
    # This is the worst-case for AWR as it forces the calculator to rely solely on rewards
    B, T, C = batch["future_data"].shape
    v_preds = torch.zeros(B, T + 1) # Including bootstrap dim
    
    # Compute Rewards
    rewards = calculator.compute_clinical_reward(
        batch["future_data"], 
        batch["outcome_label"],
        dones=batch["is_terminal"],
        src_mask=batch["future_mask"]
    )
    
    # Compute Advantages (GAE)
    advantages = calculator.compute_gae(rewards, v_preds)
    
    # Compute Weights
    weights, diag = calculator.calculate_awr_weights(advantages)

    print("\n" + "-"*30)
    print("SIMULATION RESULTS")
    print("-"*30)
    print(f"Batch Size:      {B}")
    print(f"Reward Range:    [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"Advantage Range: [{advantages.min():.3f}, {advantages.max():.3f}]")
    print(f"Advantage Std:   {advantages.std():.3f}")
    print(f"ESS (Reported):  {diag['ess']:.4f}")
    print(f"Weight Entropy:  {diag['weight_entropy']:.4f}")
    print(f"Max Weight:      {diag['weights_max']:.4f}")
    print(f"Beta:            {calculator.beta.item():.4f}")

    # 4. Stress Test: THE "NUCLEAR" OUTLIER
    print("\n[4] Stress Test: Injecting Single 'Super-Survivor' (+10.0 Advantage)...")
    adv_stress = advantages.clone()
    adv_stress[0, -1] = 10.0 # One massive outlier
    
    w_stress, diag_stress = calculator.calculate_awr_weights(adv_stress)
    print(f"ESS with 10.0 Outlier: {diag_stress['ess']:.4f}")
    
    # 5. Stress Test: THE "UNIFORM" BATCH
    print("\n[5] Stress Test: Uniform Adv (Zero variance)...")
    adv_uniform = torch.zeros_like(advantages)
    w_uni, diag_uni = calculator.calculate_awr_weights(adv_uniform)
    print(f"ESS with Uniform Adv: {diag_uni['ess']:.4f}")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_diagnostic()
