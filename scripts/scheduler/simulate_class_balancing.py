import torch
import numpy as np
import sys
import os

# Add current directory to path so we can import from icu
sys.path.append(os.getcwd())

from icu.models.wrapper_generalist import DynamicClassBalancer

def simulate_class_balancing():
    print("="*60)
    print("DYNAMIC CLASS BALANCER SIMULATION")
    print("="*60)
    
    # 1. Initialize with a strong prior (Sepsis is rare: 1/35)
    # pos_weight = 34.85
    num_classes = 3
    prior_pos_weight = 34.85
    # Use beta=0.90 for fast simulation (normally 0.99 for production)
    balancer = DynamicClassBalancer(num_classes=num_classes, beta=0.90, prior_pos_weight=prior_pos_weight)
    
    print(f"Prior pos_weight: {prior_pos_weight}")
    initial_weights = balancer.get_weights()
    print(f"Initial Weights: {initial_weights.tolist()}")
    
    # Verification of prior
    ratio = initial_weights[1] / initial_weights[0]
    print(f"Initial Ratio (Pos/Neg): {ratio:.2f} (Expected: {prior_pos_weight:.2f})")
    
    history = {"weights": [], "counts": []}
    
    # 2. Phase 1: Standard ICU (97% Stable, 3% Sepsis)
    print("\nPhase 1: High Imbalance (97% Stable, 3% Sepsis) - 20 steps")
    for i in range(20):
        labels = torch.cat([torch.zeros(124), torch.ones(2), torch.ones(2) * 2])
        balancer.update(labels)
        history["weights"].append(balancer.get_weights().tolist())
        
    w = balancer.get_weights()
    print(f"Weights after Phase 1: {w.tolist()}")
    print(f"Current Ratio: {w[1]/w[0]:.2f}")
    
    # 3. Phase 2: Distribution Shift (50% Stable, 50% Sepsis) - 50 steps
    print("\nPhase 2: Distribution Shift (50% Stable, 50% Sepsis) - 50 steps")
    for i in range(50):
        labels = torch.cat([torch.zeros(64), torch.ones(32), torch.ones(32) * 2])
        balancer.update(labels)
        if i % 10 == 0:
            w_curr = balancer.get_weights()
            print(f"Step {i}: Ratio={w_curr[1]/w_curr[0]:.2f}")
    
    # 4. Phase 3: Recovery Phase (Sepsis becomes rare again) - 50 steps
    # This proves the system isn't 'stuck' and can recover its original sensitivity.
    print("\nPhase 3: Recovery (Back to 97% Stable, 3% Sepsis) - 50 steps")
    for i in range(50):
        labels = torch.cat([torch.zeros(124), torch.ones(2), torch.ones(2) * 2])
        balancer.update(labels)
        if i % 10 == 0:
            w_curr = balancer.get_weights()
            print(f"Step {i}: Ratio={w_curr[1]/w_curr[0]:.2f}")
        
    w = balancer.get_weights()
    print(f"\nFinal Weights after Phase 3: {w.tolist()}")
    final_ratio = w[1]/w[0]
    print(f"Final Ratio (Pos/Neg): {final_ratio:.2f} (Expected: Tend toward ~34.85)")

    # 5. Final Verification
    # Success means we successfully adapted BACK toward the prior.
    if final_ratio > 10.0:
        print("\n✅ SUCCESS: Weighting recovered sensitivity (Robust Feedback).")
    else:
        print("\n❌ FAILURE: Weighting failed to recover.")

if __name__ == "__main__":
    simulate_class_balancing()
