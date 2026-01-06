
import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.components.bgsl_loss import BGSLLoss
from icu.models.components.loss_scaler import UncertaintyLossScaler

def simulate_surgical_step(target_val=2.0, ls_alpha=0.1):
    """
    Simulates a single training step with the Pass 5.1 SURGICAL implementation.
    """
    print(f"\n--- Surgical Simulation: Target={target_val} (Phase), LS_Alpha={ls_alpha} ---")
    
    # 1. Initialize Components
    scaler = UncertaintyLossScaler(num_tasks=6)
    bgsl_fn = BGSLLoss(pos_weight=10.0, gamma=4.0)
    
    # 2. Mock Latents & Predictions
    B, T, D = 4, 24, 28
    # Logit=5.0 -> Prob=0.993. 
    # With LS=0.1, the Target is 0.95. 
    # The gradient should be small but non-zero, allowing the model to stay "awake".
    pred_logits = torch.full((B, T, 1), 5.0, requires_grad=True)
    
    # 3. Real Data Target (The Phase Label: 0=Stable, 1=Pre, 2=Shock)
    raw_phase_label = torch.full((B,), float(target_val))
    
    # 4. SURGICAL LOGIC: Binary Map + Label Smoothing
    true_state_binary = (raw_phase_label > 0).float()
    # 1.0 -> 0.95, 0.0 -> 0.05
    smoothed_target = true_state_binary * (1 - ls_alpha) + (ls_alpha / 2)
    true_state = smoothed_target.view(B, 1, 1).expand(-1, T, 1)
    
    # 5. Compute BGSL Loss (Pass 5.1 version has NO CLAMPS internally)
    past_vitals = torch.zeros(B, T, D)
    mask = torch.zeros(B, T).bool()
    
    bgsl_out = bgsl_fn(pred_logits, true_state, past_vitals, mask=mask)
    l_bgsl = bgsl_out["loss"]
    
    # 6. Mock other losses
    loss_dict = {
        'diffusion': torch.tensor(1.0),
        'critic': torch.tensor(0.5),
        'aux': torch.tensor(0.3),
        'acl': torch.tensor(0.4),
        'bgsl': l_bgsl,
        'tcb': torch.tensor(0.2)
    }
    
    # 7. Scaler Pass
    total_loss, metrics = scaler(loss_dict)
    
    print(f"BGSL Loss (Smoothed): {l_bgsl.item():.4f}")
    print(f"Total Scaled Loss (L): {total_loss.item():.4f}")
    
    # 8. Gradient Check
    total_loss.backward()
    grad_val = pred_logits.grad[0, 0, 0].item()
    print(f"Local Gradient on Logit: {grad_val:.6f}")
    
    if total_loss.item() > 0 and abs(grad_val) > 0:
        print("✅ SURGICAL SUCCESS: Loss is positive and signal is alive.")
    else:
        print("❌ FAILURE: Something is still blinding the model.")

if __name__ == "__main__":
    # Test Shock Case (2.0)
    simulate_surgical_step(target_val=2.0)
    # Test Stable Case (0.0)
    simulate_surgical_step(target_val=0.0)
