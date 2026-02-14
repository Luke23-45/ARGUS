
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from icu.models.wrapper_apex import ICUSpecialistWrapper
from icu.models.diffusion import ICUConfig
from icu.models.components.loss_scaler import BayesianProjectedScaler

# --- Helper ---
def check_module(name, module):
    print(f"Checking {name}...")
    issues = 0
    try:
        for n, p in module.named_parameters():
            if p.ndim == 0:
                print(f"  [FAIL][SCALAR] {name}.{n} has shape {p.shape}")
                issues += 1
            elif p.ndim == 1 and p.shape[0] == 1:
                print(f"  [OK][Vector-1] {name}.{n} has shape {p.shape}")
            elif p.ndim == 1 and p.shape[0] > 1:
                 # Standard vector parameters are fine, but we're looking for 1D vectors that *could* have been scalars
                 pass
    except Exception as e:
        print(f"  Error checking {name}: {e}")
        issues += 1
    
    if issues == 0:
        print(f"  [PASS] {name} is clean.")
    else:
        print(f"  [FAIL] {name} has {issues} issues.")

print("--- Starting Verification Injection ---")

# 1. Prepare Dummy Config for Wrapper
# Need to mimic the structure expected by wrapper_apex.py
# Default ICUConfig fields + train section
base_cfg = ICUConfig()
train_cfg = {
    "balancing_mode": "sota_2025", # Triggers the scalar creation path
    "pretrained_path": "dummy_ckpt.ckpt", # Will fail load but init happens before load
    "lambda_reg": 0.01,
    "lambda_lb": 0.01,
    "lambda_diversity": 0.001,
    "use_loss_free_balancing": False,
    "aux_loss_scale": 0.1,
    "awr_beta": 0.5,
    "awr_max_weight": 20.0,
    "awr_lambda": 0.95,
    "awr_gamma": 0.99,
    "adaptive_beta": True,
    "adaptive_clipping": True,
    "gradnorm_alpha": 1.5,
    "phys_curriculum_start": 0.01,
    "phys_curriculum_end": 0.3,
    "phys_curriculum_epochs": 50,
    "self_cond_prob_start": 0.0,
    "self_cond_prob_end": 0.5,
    "warmup_steps": 1500,
    "awr_calibration_mode": "sample",
    "awr_max_samples": 5000,
    "seed": 42
}

# Convert to DictConfig
cfg_dict = {
    "model": base_cfg.__dict__, 
    "train": train_cfg
}
cfg = OmegaConf.create(cfg_dict)

# 2. Check Loss Scaler
print("\nInstantiating BayesianProjectedScaler...")
try:
    loss_scaler = BayesianProjectedScaler(num_tasks=7)
    check_module("BayesianProjectedScaler", loss_scaler)
except Exception as e:
    print(f"Loss Scaler Init Error: {e}")

# 3. Instantiate Wrapper
# We expect it to fail loading weights but pass init
print("\nInstantiating ICUSpecialistWrapper...")
try:
    from icu.utils.train_utils import SurgicalCheckpointLoader
    SurgicalCheckpointLoader.load_model = lambda model, path, strict=True: print(f"  [Mock] Loaded weights from {path}")
    
    wrapper = ICUSpecialistWrapper(cfg)
    check_module("ICUSpecialistWrapper", wrapper)
    
except Exception as e:
    print(f"Wrapper Init Error: {e}")

print("\n--- Verification Complete ---")
