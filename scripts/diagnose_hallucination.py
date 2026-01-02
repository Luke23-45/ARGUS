import torch
import torch.nn as nn
import torch.nn.functional as F
from icu.models.wrapper_generalist import ICUGeneralistWrapper, DynamicClassBalancer
from icu.models.components.risk_aware_loss import RiskAwareAsymmetricLoss
from omegaconf import OmegaConf

def verify_dimensions():
    print("--- Dimension Diagnostic ---")
    
    # Mock Config
    cfg = OmegaConf.create({
        "model": {
            "num_phases": 3,
            "history_len": 24,
            "pred_len": 6,
            "input_dim": 28,
            "static_dim": 6,
            "d_model": 128,
            "encoder_layers": 2,
            "n_heads": 4,
            "use_auxiliary_head": True,
            "timesteps": 100,
            "use_imputation_masks": True,
            "use_attention_pooling": True,
            "dropout": 0.1,
            "stochastic_depth_prob": 0.0,
            "use_swiglu": True,
            "ffn_dim_ratio": 4,
            "n_layers": 2,
            "num_phases": 3
        },
        "train": {
            "balancing_mode": "sota_2025",
            "asl_gamma_neg": 4.0,
            "asl_gamma_pos": 0.0,
            "risk_multiplier": 2.0,
            "throttle_scale": 1.0,
            "grad_clip": 1.0
        },
        "seed": 42
    })

    # 1. Balancer Test
    balancer = DynamicClassBalancer(num_classes=3)
    targets = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
    balancer.update(targets)
    weights = balancer.get_weights()
    print(f"Balancer Weights Shape: {weights.shape} (Expected [3])")
    print(f"Weights value: {weights}")

    # 2. Loss Logic Test
    loss_fn = RiskAwareAsymmetricLoss()
    logits = torch.randn(5, 3, requires_grad=True)
    y_labels = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
    y_one_hot = F.one_hot(y_labels, num_classes=3).float()
    risk_coef = torch.rand(5)
    
    # Test broadcasing
    try:
        loss = loss_fn(logits, y_one_hot, risk_coef, class_weights=weights)
        print(f"Loss forward with [3] weights: OK. Loss={loss.item():.4f}")
        loss.backward()
        print("Backward pass: OK")
    except Exception as e:
        print(f"Loss forward FAILED: {e}")

    # 3. Mismatch Simulation (Is there a bug if sepsis is binary?)
    print("\n--- Binary Mismatch Simulation ---")
    logits_bin = torch.randn(5, 1, requires_grad=True)
    targets_bin = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(-1)
    
    # If balancer is still 3-class but head is binary
    try:
        loss_bin = loss_fn(logits_bin, targets_bin, risk_coef, class_weights=weights)
        print(f"Binary Loss with [3] weights: {loss_bin.shape} (Wait, RiskAwareAsymmetricLoss.mean() returns scalar)")
        print(f"Binary Loss value: {loss_bin.item():.4f}")
    except Exception as e:
        print(f"Binary Loss FAILED (Expected if weights are [3]): {e}")

    # Let's see the internal ASL loss before mean()
    # asl_loss is [B, C]. For binary C=1, asl_loss is [B, 1].
    # class_weights.unsqueeze(0) is [1, 3].
    # [B, 1] * [1, 3] = [B, 3].
    # loss.mean() on [5, 3] returns a scalar.
    # THIS IS THE SILENT BUG. It averages over 2 junk columns!
    
if __name__ == "__main__":
    verify_dimensions()
