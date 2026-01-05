
import sys
import os
sys.path.append(os.getcwd())
import torch
import hydra
from omegaconf import OmegaConf
from icu.models.wrapper_generalist import ICUGeneralistWrapper
from icu.utils.train_utils import set_seed

@hydra.main(config_path="../conf", config_name="generalist", version_base="1.3")
def main(cfg):
    print("Locked & Loaded: Verifying Phase 1 Fixes...")
    set_seed(cfg.seed)
    
    # 1. Instantiate Model
    model = ICUGeneralistWrapper(cfg)
    
    # Disable warmup so LR doesn't start at 0
    OmegaConf.set_struct(cfg, False)
    cfg.train.warmup_steps = 0
    cfg.train.warmup_ratio = 0.0
    OmegaConf.set_struct(cfg, True)
    
    # Mock Trainer
    class DummyTrainer:
        estimated_stepping_batches = 1000
    model.trainer = DummyTrainer()

    
    # 2. Check Scale-Aware Initialization
    print("\n[Check 1] Loss Scaler Initialization:")
    log_vars = model.loss_scaler.log_vars.detach().cpu().numpy()
    target_vars = [1.0, 0.5, -0.5, 0.0, 0.5, 0.5]
    print(f"  Current: {log_vars}")
    print(f"  Target:  {target_vars}")
    
    # Allow small float diffs or if device mismatch (but we detached)
    # Just check if aux (index 2) is negative
    if log_vars[2] < 0.0:
         print("  ✅ SUCCESS: Aux log_var is negative (higher weight).")
    else:
         print("  ❌ FAILURE: Aux log_var is not negative.")

    # 3. Check Optimizer Groups
    print("\n[Check 2] Optimizer Parameter Groups:")
    model.configure_optimizers()
    # configure_optimizers returns (optimizer, scheduler) or just optimizer or dict
    # In wrapper_generalist.py it creates base_optimizer then potentially wraps it
    # We need to access the optimizer instance. 
    # Since configure_optimizers in Lightning returns the object, let's call it.
    
    res = model.configure_optimizers()
    if isinstance(res, tuple):
        opt = res[0]
    elif isinstance(res, dict):
        opt = res["optimizer"]
    else:
        opt = res
        
    num_groups = len(opt.param_groups)
    print(f"  Number of Parameter Groups: {num_groups}")
    
    # We expect 5 groups now
    if num_groups >= 5:
        print("  ✅ SUCCESS: Found separate parameter groups.")
        # Check LR of group 1 (Aux) - Check indices from code
        # Group 0: Main
        # Group 1: Aux
        # Group 2: Expert
        # Group 3: ACL
        # Group 4: Scaler
        
        base_lr = cfg.train.lr
        aux_mult = cfg.train.aux_lr_multiplier
        aux_lr = opt.param_groups[1]['lr']
        
        print(f"  Base LR: {base_lr}")
        print(f"  Aux Group LR: {aux_lr}")
        
        if abs(aux_lr - (base_lr * aux_mult)) < 1e-8:
             print("  ✅ SUCCESS: Aux LR is correctly boosted.")
        else:
             print(f"  ❌ FAILURE: Aux LR mismatch. Expected {base_lr * aux_mult}, got {aux_lr}")
             
        # Debug: Print all groups
        for i, group in enumerate(opt.param_groups):
             print(f"    Group {i} LR: {group['lr']} | Params: {len(group['params'])}")
    else:
        print("  ❌ FAILURE: Expected at least 5 parameter groups.")

    print("\nPhase 1 Verification Complete.")

if __name__ == "__main__":
    main()
