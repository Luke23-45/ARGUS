
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.components.loss_scaler import UncertaintyLossScaler

def simulate_scale_dominance():
    print("🔬 Forensic Scale Dominance Audit (The 10,000x Gap Test)")
    
    # 1. Initialize Scaler (6 tasks)
    scaler = UncertaintyLossScaler(num_tasks=6)
    
    # 2. Simulate Parameters (The "Sepsis Head" vs "Diffusion Backbone")
    # We use small vectors to track gradient magnitudes
    param_sepsis = nn.Parameter(torch.ones(10))
    param_diffusion = nn.Parameter(torch.ones(10))
    
    optimizer = torch.optim.AdamW([
        {'params': [param_sepsis, param_diffusion]},
        {'params': scaler.parameters(), 'lr': 0.025} # Higher LR for scaler as per config
    ])
    
    # 3. Simulate The Scale Gap (From logs2.md)
    # Target: 7000 (MSE) vs 0.5 (BCE)
    diffusion_scale = 7000.0
    sepsis_scale = 0.5
    
    print(f"\nPhase 1: Starting Audit with Diffusion={diffusion_scale}, Sepsis={sepsis_scale}")
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Simulated Losses
        # Diffusion loss is tied to param_diffusion
        # Sepsis loss is tied to param_sepsis
        l_diff = (param_diffusion**2).sum() * diffusion_scale 
        l_sepsis = (param_sepsis**2).sum() * sepsis_scale
        
        loss_dict = {
            'diffusion': l_diff,
            'aux': l_sepsis, # Sepsis head
            'critic': torch.tensor(0.5),
            'acl': torch.tensor(0.5),
            'bgsl': torch.tensor(0.5),
            'tcb': torch.tensor(0.5)
        }
        
        total_loss, metrics = scaler(loss_dict)
        total_loss.backward()
        
        # Capture Metrics
        w_diff = metrics['weight/diffusion']
        w_sepsis = metrics['weight/aux']
        
        # Check Gradient Magnitude (The "Freeze" Check)
        g_diff = param_diffusion.grad.norm().item()
        g_sepsis = param_sepsis.grad.norm().item()
        
        print(f"Step {step}: W_Diff={w_diff:.6f}, W_Sepsis={w_sepsis:.6f} | G_Diff={g_diff:.2f}, G_Sepsis={g_sepsis:.4f}")
        
        optimizer.step()

    # 4. Interpret Results
    final_w_diff = metrics['weight/diffusion']
    final_w_sepsis = metrics['weight/aux']
    
    print("\n--- Final Interpretation ---")
    print(f"Diffusion Weight reduced to: {final_w_diff:.8f}")
    print(f"Sepsis Weight stayed at: {final_w_sepsis:.4f}")
    
    # A "Freeze" occurs if G_Sepsis / G_Diff < 1e-4
    ratio = (param_sepsis.grad.norm() / param_diffusion.grad.norm()).item()
    print(f"Gradient Ratio (Sepsis/Diff): {ratio:.8f}")
    
    if ratio < 0.001:
        print("🚨 ALERT: SCALE DOMINANCE DETECTED. The Sepsis signal is 1000x weaker than Diffusion.")
        print("This explains why the AUC decreases: the optimizer is 'hunting' for 0.1 reductions in the 7000-scale MSE,")
        print("completely ignoring the 0.01 improvements in the 0.5-scale BCE.")
    else:
        print("✅ SUCCESS: Uncertainty Scaler is successfully balancing the scales.")

if __name__ == "__main__":
    simulate_scale_dominance()
