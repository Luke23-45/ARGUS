
import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.components.bgsl_loss import BGSLLoss
from icu.models.components.loss_scaler import UncertaintyLossScaler

def simulate_training_step(target_val=2.0, use_fix=False):
    """
    Simulates a single training step with the exact components used in the generalist.
    """
    print(f"\n--- Simulation: Target={target_val}, UseFix={use_fix} ---")
    
    # 1. Initialize Components
    # We use the exact registry from the generalist [diffusion, critic, aux, acl, bgsl, tcb]
    scaler = UncertaintyLossScaler(num_tasks=6)
    bgsl_fn = BGSLLoss(pos_weight=10.0, gamma=4.0)
    
    # 2. Mock Latents & Predictions
    B, T, D = 4, 24, 28
    # Model is highly confident (Logit=5.0 -> Prob=0.993)
    pred_logits = torch.full((B, T, 1), 5.0, requires_grad=True)
    
    # 3. Real Data Target (The Phase Label)
    # 2.0 = Shock, 1.0 = Pre-Shock, 0.0 = Stable
    raw_phase_label = torch.full((B,), float(target_val))
    
    # 4. The Transformation Logic (The Fix)
    if use_fix:
        # Pass 5 fix: Map to binary
        mapped_target = (raw_phase_label > 0).float()
    else:
        # Legacy: Pass raw label
        mapped_target = raw_phase_label
        
    # Expand for sequence-level BGSL
    true_state = mapped_target.view(B, 1, 1).expand(-1, T, 1)
    
    # 5. Compute BGSL Loss
    # We mock vitals and mask as flat
    past_vitals = torch.zeros(B, T, D)
    mask = torch.zeros(B, T).bool()
    
    bgsl_out = bgsl_fn(pred_logits, true_state, past_vitals, mask=mask)
    l_bgsl = bgsl_out["loss"]
    
    # 6. Mock other losses for the Scaler
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
    
    print(f"BGSL Loss (Raw): {l_bgsl.item():.4f}")
    print(f"Total Scaled Loss: {total_loss.item():.4f}")
    
    # 8. Gradient Audit
    total_loss.backward()
    grad_norm = pred_logits.grad.norm().item()
    print(f"Gradient Norm on Encoder: {grad_norm:.4f}")
    
    if total_loss.item() < 0:
        print("!!! RED ALERT: Total Loss is NEGATIVE. Model will Reward Hack. !!!")
    elif total_loss.item() > 0 and use_fix:
        print("✅ STABLE: Loss is positive. Optimization is healthy.")

if __name__ == "__main__":
    # Scenario A: The Crash Path (Target=2.0, No Fix)
    simulate_training_step(target_val=2.0, use_fix=False)
    
    # Scenario B: The Patch Path (Target=2.0, With Fix)
    simulate_training_step(target_val=2.0, use_fix=True)
    
    # Scenario C: The Standard Path (Target=1.0, With Fix)
    simulate_training_step(target_val=1.0, use_fix=True)
