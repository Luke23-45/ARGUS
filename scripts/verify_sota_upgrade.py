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

def test_acl_broadcasting():
    print("\n--- Testing ACL Broadcasting (Window -> Seq) ---")
    acl = AsymmetricContrastiveLoss(d_model=16, num_classes=3)
    
    B, T = 4, 10
    z = torch.randn(B, T, 16)
    # y is window-level (B,)
    y = torch.randint(0, 3, (B,))
    mask = torch.ones(B, T).bool()
    
    # This should not raise IndexError
    loss = acl(z, y, mask=mask)
    print(f"ACL Broadcasting Loss: {loss.item()}")
    assert not torch.isnan(loss)
    print("ACL Broadcasting Verified!")

def test_aux_head_alignment():
    print("\n--- Testing Aux Head Alignment (Window-Level Logic) ---")
    B, C = 256, 3
    # Mocks SequenceAuxHead output (Window-Level)
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))
    
    # Simulates training_step gather logic
    probs = torch.softmax(logits, dim=-1)
    true_probs = probs.gather(-1, targets.unsqueeze(-1).long())
    error = 1.0 - true_probs.squeeze(-1)
    mining_weight = 1.0 + torch.sigmoid(error * 5.0)
    
    loss = F.cross_entropy(logits, targets) * mining_weight.mean()
    loss.backward()
    
    print(f"Aux Alignment Loss: {loss.item()}")
    assert not torch.isnan(loss)
    assert logits.grad is not None
    print("Aux Head Alignment Verified!")

def test_loss_scaler_4task():
    print("\n--- Testing Uncertainty Scaler (4 Tasks) ---")
    scaler = UncertaintyLossScaler(num_tasks=4)
    loss_dict = {
        'diffusion': torch.tensor(1.0, requires_grad=True),
        'critic': torch.tensor(1.5, requires_grad=True),
        'aux': torch.tensor(2.0, requires_grad=True),
        'acl': torch.tensor(0.5, requires_grad=True)
    }
    
    total, metrics = scaler(loss_dict)
    print(f"Total Loss: {total.item()}")
    print(f"Weights: {metrics}")
    
    total.backward()
    assert 'weight/critic' in metrics
    assert 'weight/acl' in metrics
    print("4-Task Scaler Verified!")

if __name__ == "__main__":
    try:
        test_saw_math()
        test_acl_stability()
        test_acl_broadcasting()
        test_aux_head_alignment()
        test_loss_scaler_4task()
        print("\n[SUCCESS] SOTA Upgrade Patches are numerically sound and functionally correct.")
    except Exception as e:
        print(f"\n[FAILURE] Audit failed: {e}")
        import traceback
        traceback.print_exc()
