import torch
import torch.nn as nn
import torch.nn.functional as F
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.models.components.contrastive_loss import AsymmetricContrastiveLoss
from icu.models.components.loss_scaler import UncertaintyLossScaler

def test_saw_math():
    print("\n--- Testing SAW Math ---")
    calc = ICUAdvantageCalculator(gamma=0.9, adaptive_beta=False)
    
    B, T = 2, 4
    rewards = torch.ones(B, T)
    student_v = torch.full((B, T), 0.5)
    teacher_v = torch.full((B, T), 0.8)
    dones = torch.zeros(B, T)
    dones[:, -1] = 1.0
    bootstrap = torch.tensor([[1.0], [1.0]])
    
    # A = r + gamma * teacher_v(t+1) * (1-d) - student_v(t)
    # Step 0: 1.0 + 0.9 * 0.8 * 1.0 - 0.5 = 1.0 + 0.72 - 0.5 = 1.22
    # Step 3 (Terminal): 1.0 + 0.9 * 1.0 * 0.0 - 0.5 = 1.0 + 0 - 0.5 = 0.5
    
    adv = calc.compute_saw(rewards, student_v, teacher_v, dones=dones, bootstrap_value=bootstrap)
    print(f"Advantages:\n{adv}")
    
    expected_step0 = 1.22
    expected_step3 = 0.5
    
    assert torch.allclose(adv[:, 0], torch.tensor(expected_step0), atol=1e-5)
    assert torch.allclose(adv[:, 3], torch.tensor(expected_step3), atol=1e-5)
    print("SAW Math Verified!")

def test_acl_stability():
    print("\n--- Testing ACL Stability ---")
    acl = AsymmetricContrastiveLoss(d_model=16, num_classes=3, centroid_reg=0.1)
    
    B, T = 4, 10
    z = torch.randn(B, T, 16, requires_grad=True)
    y = torch.randint(0, 3, (B, T))
    mask = torch.ones(B, T).bool()
    
    loss = acl(z, y, mask=mask)
    print(f"ACL Loss: {loss.item()}")
    
    loss.backward()
    print(f"z.grad norm: {z.grad.norm().item()}")
    print(f"Centroids grad norm: {acl.centroids.grad.norm().item()}")
    
    assert not torch.isnan(loss)
    assert z.grad is not None
    print("ACL Stability Verified!")

def test_loss_scaler_3task():
    print("\n--- Testing Uncertainty Scaler (3 Tasks) ---")
    scaler = UncertaintyLossScaler(num_tasks=3)
    loss_dict = {
        'diffusion': torch.tensor(1.0, requires_grad=True),
        'aux': torch.tensor(2.0, requires_grad=True),
        'acl': torch.tensor(0.5, requires_grad=True)
    }
    
    total, metrics = scaler(loss_dict)
    print(f"Total Loss: {total.item()}")
    print(f"Weights: {metrics}")
    
    total.backward()
    assert 'weight/acl' in metrics
    print("3-Task Scaler Verified!")

if __name__ == "__main__":
    try:
        test_saw_math()
        test_acl_stability()
        test_loss_scaler_3task()
        print("\n[SUCCESS] SOTA Upgrade Patches are numerically sound and functionally correct.")
    except Exception as e:
        print(f"\n[FAILURE] Audit failed: {e}")
        import traceback
        traceback.print_exc()
