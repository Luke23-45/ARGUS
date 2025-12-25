#!/usr/bin/env python3
"""
Architecture Benchmarking Script
================================

Benchmarks different ICU Diffusion Model variants (Small, Base, Large)
to find the tradeoffs between Throughput (Speed) and Capacity.

Metrics:
1. Parameter Count.
2. Inference Speed (Samples/Sec).
3. VRAM Usage (estimated).
"""

import sys
import time
import torch
import torch.nn as nn
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

try:
    from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
except ImportError:
    print("[ERROR] Project modules not found.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_variant(name, model_cfg_dict, batch_size=64, device="cuda"):
    logger.info(f"Benchmarking Variant: {name}...")
    
    # 1. Setup Model
    cfg = ICUConfig(**model_cfg_dict)
    try:
        model = ICUUnifiedPlanner(cfg).to(device)
    except Exception as e:
        logger.error(f"Failed to init model {name}: {e}")
        return None

    # 2. Count Params
    params = count_parameters(model)
    
    # 3. Dummy Data
    # B, L, C = batch_size, input_dim
    # We need a batch dictionary
    dummy_batch = {
        "observed_data": torch.randn(batch_size, 50, cfg.input_dim, device=device),
        "observed_mask": torch.ones(batch_size, 50, cfg.input_dim, device=device),
        "future_data": torch.randn(batch_size, 50, cfg.input_dim, device=device),
        "future_mask": torch.ones(batch_size, 50, cfg.input_dim, device=device),
        "static_context": torch.randn(batch_size, cfg.static_dim, device=device), # Added static context
        "time_steps": torch.randint(0, 100, (batch_size,), device=device),
        "outcome_label": torch.randint(0, 1, (batch_size,), device=device)
    }
    
    # 4. Warmup
    for _ in range(5):
        _ = model(dummy_batch)
    
    # 5. Timing Loop
    num_steps = 50
    start_time = time.time()
    
    model.train() # Measure training throughput (backprop included?)
    # Usually benchmark inference throughput?
    # Let's benchmark Forward + Backward (Training Speed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss_dict = model(dummy_batch)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    samples_per_sec = (batch_size * num_steps) / total_time
    
    return {
        "name": name,
        "params_m": params / 1e6,
        "throughput": samples_per_sec,
        "batch_size": batch_size
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running benchmarks on: {device}")
    
    # Base Config (From valid YAML)
    base_dim = 512
    base_layers = 6
    base_heads = 8
    
    variants = [
        {
            "name": "Small (Edge)",
            "cfg": {
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "input_dim": 5, "static_dim": 2,
                "history_len": 50, "pred_len": 50 # Match dummy data
            }
        },
        {
            "name": "Base (Generalist)", # Current Default
            "cfg": {
                "d_model": 512,
                "n_layers": 6,
                "n_heads": 8,
                "input_dim": 5, "static_dim": 2,
                "history_len": 50, "pred_len": 50 # Match dummy data
            }
        },
        {
            "name": "Large (Specialist)",
            "cfg": {
                "d_model": 768,
                "n_layers": 8,
                "n_heads": 12,
                "input_dim": 5, "static_dim": 2,
                "history_len": 50, "pred_len": 50 # Match dummy data
            }
        }
    ]
    
    results = []
    
    print("\n" + "="*60)
    print(f"{'VARIANT':<20} | {'PARAMS (M)':<10} | {'SPEED (samp/s)':<15}")
    print("-" * 60)
    
    for v in variants:
        res = benchmark_variant(v["name"], v["cfg"], device=device.type)
        if res:
            results.append(res)
            print(f"{res['name']:<20} | {res['params_m']:<10.2f} | {res['throughput']:<15.2f}")
    
    print("="*60 + "\n")
    
    # Recommendation
    base = results[1]
    large = results[2]
    
    print("ANALYSIS:")
    if large["throughput"] > 0.5 * base["throughput"]:
        print(f"Large model is efficient ({large['throughput']:.0f} s/s). Consider using for Phase 2.")
    else:
        print(f"Large model is slow. Stick to Base for now.")

if __name__ == "__main__":
    main()
