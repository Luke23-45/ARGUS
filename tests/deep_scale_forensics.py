
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.components.loss_scaler import UncertaintyLossScaler

def cosine_similarity(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

def run_deep_forensics():
    print("🔬 DEEP FORENSIC AUDIT: Multi-Task Scale Interference v2.0")
    print("----------------------------------------------------------")
    
    # 1. Setup Shared Manifold (The "Backbone")
    # In a real DiT-1D, all tasks share the main feature extractor.
    dim = 256
    backbone = nn.Parameter(torch.randn(dim) * 0.1)
    
    # Task Heads (Projections from backbone)
    head_diff = nn.Parameter(torch.randn(dim, 10))   # Diffusion Head
    head_sepsis = nn.Parameter(torch.randn(dim, 1))  # Sepsis Head
    
    # 2. Scaler Initialization
    # We use the actual scaler logic from the project
    scaler = UncertaintyLossScaler(num_tasks=6)
    
    # Optimizer settings from generalist.yaml
    optimizer = torch.optim.AdamW([
        {'params': [backbone, head_diff, head_sepsis], 'lr': 1e-4}, # Model LR
        {'params': scaler.parameters(), 'lr': 0.025}                 # Scaler LR (High)
    ])

    # 3. Task Scales (Forensic Evidence from logs2.md & reort.md)
    scales = {
        'diffusion': 7500.0, # High: MSE in clinical units (unnormalized)
        'critic': 150.0,     # Mid: Advantage scaling
        'aux': 0.5,          # Low: BCE with Sepsis (The victim)
        'acl': 1.5,          # Mid: Contrastive
        'bgsl': 1.0,         # Mid: Dynamics
        'tcb': 2.0           # Mid: Buffer
    }
    
    # 4. Simulation Loop (500 Steps = Approx 2-3 Epochs of clinical data)
    history = {k: [] for k in ['w_diff', 'w_sepsis', 'g_ratio', 'cos_sim', 'loss_total']}
    
    print(f"Initial Scales: Diffusion={scales['diffusion']}, Sepsis={scales['aux']}")
    
    for step in range(500):
        optimizer.zero_grad()
        
        # --- Task 1: Sepsis (The Target) ---
        # Goal: Optimize sepsis_head to produce correct classification
        l_sepsis = (backbone @ head_sepsis).pow(2).sum() * scales['aux']
        
        # --- Task 2: Diffusion (The Dominator) ---
        # Goal: Reconstruct vitals
        l_diff = (backbone @ head_diff).pow(2).sum() * (scales['diffusion'] * 1e-3) # SIMULATED PATCH
        
        # --- Static Tasks (Mock) ---
        l_others = {k: torch.tensor(v) for k, v in scales.items() if k not in ['diffusion', 'aux']}
        
        loss_dict = {
            'diffusion': l_diff,
            'aux': l_sepsis,
            **l_others
        }
        
        # Standard Scaler Forward
        total_loss, metrics = scaler(loss_dict)
        
        # --- Gradient Analysis (BEFORE Backward) ---
        # We want to know what the "Ideal" Sepsis gradient is
        l_sepsis.backward(retain_graph=True)
        grad_sepsis_only = backbone.grad.clone()
        optimizer.zero_grad()
        
        # Now do the real Combined Backward
        total_loss.backward()
        grad_combined = backbone.grad.clone()
        
        # --- Metrics Collection ---
        w_diff = metrics['weight/diffusion']
        w_sepsis = metrics['weight/aux']
        
        # Gradient Ratio: How much of the backbone update is sepsis-driven?
        g_ratio = (grad_sepsis_only.norm() / grad_combined.norm()).item()
        
        # Cosine Similarity: Is the combined gradient even pointing in the sepsis direction?
        # If < 0, the diffusion update is actually HURTING the sepsis learning.
        sim = cosine_similarity(grad_sepsis_only, grad_combined)
        
        history['w_diff'].append(w_diff)
        history['w_sepsis'].append(w_sepsis)
        history['g_ratio'].append(g_ratio)
        history['cos_sim'].append(sim)
        history['loss_total'].append(total_loss.item())
        
        if step % 50 == 0:
            print(f"Step {step:03d} | W_Diff: {w_diff:.8f} | W_Sepsis: {w_sepsis:.4f} | Sim: {sim:.4f} | Ratio: {g_ratio:.8f}")
            
        optimizer.step()

    # 5. Final Forensic Interpretation
    print("\n" + "="*60)
    print("🎯 FINAL FORENSIC INTERPRETATION")
    print("="*60)
    
    final_sim = history['cos_sim'][-1]
    final_ratio = history['g_ratio'][-1]
    # Detach for numpy
    w_list = [w.detach().item() for w in history['w_diff']]
    w_history = np.array(w_list)
    
    convergence_speed = np.argmin(w_history > 1e-4) if min(w_list) > 1e-4 else 500
    
    print(f"1. Directional Alignment: Cosine Similarity = {history['cos_sim'][-1]:.4f}")
    if history['cos_sim'][-1] < -0.05:
        print("   🔴 CRITICAL: Active Sabotage (Negative Transfer). Diffusion is destroying Sepsis features.")
    elif history['cos_sim'][-1] < 0.1:
        print("   🟡 NEUTRAL: Representation Diversity (Orthogonality). Tasks use different feature manifolds.")
    else:
        print("   🟢 COOPERATIVE: Tasks are learning from shared features.")
        
    print(f"2. Selection Pressure: Gradient Ratio = {history['g_ratio'][-1]:.8f}")
    if history['g_ratio'][-1] < 0.001:
        print(f"   🔴 CRITICAL: Sepsis signal is {1/history['g_ratio'][-1]:.0f}x weaker than the combined update.")
    else:
        print(f"   🟢 SUCCESS: Sepsis signal is well-represented in the update.")
    
    print(f"3. Scaler Responsiveness: Weights converged to {history['w_diff'][-1]:.8f}")
    
    # Check if parity is achieved (Scaled Diff vs Sepsis)
    current_pressure = history['w_diff'][-1].detach().item() * (scales['diffusion'] * 1e-3)
    sepsis_pressure = history['w_sepsis'][-1].detach().item() * scales['aux']
    print(f"   Scaled Diffusion Pressure: {current_pressure:.2f}")
    print(f"   Scaled Sepsis Pressure: {sepsis_pressure:.2f}")

    if abs(np.log10(current_pressure / sepsis_pressure)) < 1.0:
        print("\nVerdict: PARITY ACHIEVED. Model is ready for 0.82 AUC run.")
    else:
        print("\nVerdict: Scale mismatch persists.")

if __name__ == "__main__":
    # Simulate FIXED state (Diffusion/1000)
    run_deep_forensics()
