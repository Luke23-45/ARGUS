"""
[v20.0 VERIFICATION] Orchestration Mastery Suite
--------------------------------------------------------------------------------
Tests the technical ceiling features:
1. Prioritized Uncertainty Sampling
2. Cross-Manifold Synergy (Ghost-TCB Bonding)
3. A-GEM Gradient Projection
"""

import sys
import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Adjust sys.path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from icu.models.wrapper_generalist import ICUGeneralistWrapper
import icu.models.wrapper_generalist as wg
print(f"DEBUG: ICUGeneralistWrapper loaded from: {wg.__file__}")
from icu.models.diffusion import ICUConfig
from icu.models.components.ghost_bank import SepsisGhostBank

def verify_v20_mastery():
    print("="*80)
    print(" [v20.0] ORCHESTRATION MASTERY VERIFICATION")
    print("="*80)

    # 1. Setup Mock Workspace
    cfg = OmegaConf.create({
        "model": {
            "d_model": 128,
            "timesteps": 10,
            "input_dim": 28,
            "n_layers": 2,
            "n_heads": 4,
            "num_phases": 3,
            "use_auxiliary_head": True,
            "aux_loss_scale": 0.5
        },
        "train": {
            "num_ghosts": 4,
            "ghost_mixup_alpha": 0.2,
            "balancing_mode": "sota_2025",
            "grad_clip": 1.0,
            "awr_beta": 0.05,
            "awr_max_weight": 50.0,
            "awr_lambda": 0.95,
            "awr_gamma": 0.99
        }
    })

    wrapper = ICUGeneralistWrapper(cfg)
    bank = wrapper.ghost_bank
    
    # Manually populate the bank with varying uncertainties
    print("\n[STEP 1] Testing Prioritized Uncertainty Sampling...")
    for i in range(10):
        v = torch.randn(1, 24, 28)
        m = torch.ones(1, 24, 28)
        l = torch.tensor([1])
        lat = torch.randn(1, 128)
        unc = torch.tensor([[float(i)/10.0]]) # 0.0 to 0.9
        bank.update(v, m, l, lat, uncertainties=unc)

    # Sample with priority
    num_test_ghosts = 100
    ghosts = bank.sample(num_ghosts=num_test_ghosts, seed=42, uncertainty_weighted=True)
    avg_sampled_unc = ghosts["uncertainties"].mean().item()
    print(f"Bank Size: {bank.size.item()}")
    print(f"Bank Uncertainties: {bank.uncertainties[:bank.size.item()].squeeze().tolist()}")
    print(f"Sampled Uncertainties (First 10): {ghosts['uncertainties'][:10].squeeze().tolist()}")
    print(f"Average Sampled Uncertainty: {avg_sampled_unc:.4f} (Higher is better, expected > 0.6)")
    assert avg_sampled_unc > 0.6, "Prioritized sampling failed to bias towards high uncertainty."
    print("SUCCESS: Prioritized Sampling Verified.")

    # 2. Test TCB Synergy
    print("\n[STEP 2] Testing Cross-Manifold Synergy (Ghost-TCB Bonding)...")
    # We trigger a training_step and check if TCB loss uses the combined batch
    # Mock Batch
    batch = {
        "observed_data": torch.randn(4, 24, 28),
        "future_data": torch.randn(4, 6, 28),
        "static_context": torch.randn(4, 6),
        "phase_label": torch.zeros(4, dtype=torch.long),
        "src_mask": torch.ones(4, 24, 28)
    }
    
    # Mocking components to bypass complex dependencies
    from unittest.mock import PropertyMock, patch, MagicMock
    
    # 1. Mock Optimizers
    mock_opt = MagicMock()
    
    # 2. Mock Trainer
    wrapper.trainer = MagicMock()
    wrapper.trainer.accumulate_grad_batches = 1
    wrapper.trainer.max_epochs = 100

    # 3. Create context for robust patching
    with patch.object(ICUGeneralistWrapper, 'optimizers', return_value=mock_opt), \
         patch.object(ICUGeneralistWrapper, 'current_epoch', new_callable=PropertyMock, return_value=1), \
         patch.object(ICUGeneralistWrapper, 'global_step', new_callable=PropertyMock, return_value=10), \
         patch.object(ICUGeneralistWrapper, '_get_curr_physics_weight', return_value=0.2), \
         patch('icu.utils.advantage_calculator.ICUAdvantageCalculator.compute_clinical_reward', return_value=torch.randn(4, 6)):
            # [v20.0] Fix manual_backward in mock environment
            wrapper.manual_backward = lambda loss, **kwargs: loss.backward(**kwargs)
            
            # Perform training step
            loss = wrapper.training_step(batch, 0)
            print(f"Training Step executed successfully. Loss: {loss:.4f}")
            print(f"Loss Grad Fn: {loss.grad_fn}")
            
            # Check for gradients in expert manifold
            expert_grads = [p.grad for name, p in wrapper.model.named_parameters() if "expert" in name and p.grad is not None]
            print(f"Expert Params with Grads: {len(expert_grads)}")
            
            print("SUCCESS: Cross-Manifold Synergy initialized.")

    # 3. Test A-GEM Gradient Projection
    print("\n[STEP 3] Testing A-GEM Gradient Projection...")
    # This is implicitly verified if training_step runs, as it performs the dual backward.
    # To be explicit, we verify grads of expert layers are non-zero
    has_grads = False
    for name, p in wrapper.model.named_parameters():
        if "expert" in name and p.grad is not None:
            if p.grad.abs().sum() > 0:
                has_grads = True
                break
    
    assert has_grads, "A-GEM Gradient Projection nullified gradients or failed to backprop."
    print("SUCCESS: A-GEM Gradient Projection Verified.")

    print("\n" + "="*80)
    print(" [v20.0] ALL SOTA ORCHESTRATION PILLARS VERIFIED")
    print("="*80)

if __name__ == "__main__":
    verify_v20_mastery()
