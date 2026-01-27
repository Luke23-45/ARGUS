"""
Full Manifold Volatility Simulation (Verifying Epoch 5 Breakdown)
================================================================

This script simulates the combined effects of the flaws found in wrapper_generalist.py:
1. Double Physics Backward (Lines 1228 and 1306)
2. Bypassed Curriculum Weight (Line 1300 redefines phys_loss without weight)
3. Epoch 5 Clamp Removal (Line 1304)

We simulate the Gradient Pressure (EMA) over 10 epochs.
"""

import numpy as np
# Removed matplotlib dependency for head-less environment

def simulate_training(
    double_backward: bool = True,
    bypass_curriculum: bool = True,
    clamp_epochs: int = 5,
    num_epochs: int = 10,
    batches_per_epoch: int = 1176,
    base_phys_weight: float = 0.5,
    natural_grad_norm: float = 1.5,
    natural_phys_norm: float = 4.0,
    decay: float = 0.95
):
    ema = 1.0
    history = []
    
    # Pre-calculate curriculum weights (ramping 0.01 -> base over 50% steps)
    total_steps = num_epochs * batches_per_epoch
    warmup_steps = total_steps * 0.5
    
    for epoch in range(num_epochs):
        for batch in range(batches_per_epoch):
            global_step = epoch * batches_per_epoch + batch
            
            # 1. Physics Curriculum Weight
            if global_step < warmup_steps:
                curr_phys_weight = 0.01 + (base_phys_weight - 0.01) * (global_step / warmup_steps)
            else:
                curr_phys_weight = base_phys_weight
            
            # 2. Simulate Physics Violation (Constant baseline for SpO2 etc.)
            raw_phys_norm = natural_phys_norm
            
            # 3. Apply Clamp (Pre-Epoch 5)
            # The clamp in code is on the LOSS, which sets GRADIENTS to 0 if exceeded.
            is_clamped = (epoch < clamp_epochs)
            if is_clamped:
                # If loss is large (e.g. 50 > 10), gradients are 0.
                # We assume the model starts with high violation.
                grad_phys = 0.0 
            else:
                # Clamp removed. Suddenly full force.
                # If bypass_curriculum is True, we use weight 1.0 instead of curr_phys_weight.
                effective_weight = 1.0 if bypass_curriculum else curr_phys_weight
                grad_phys = raw_phys_norm * effective_weight
            
            # 4. Main Gradients (Diffusion/Value)
            grad_main = natural_grad_norm
            
            # 5. Backward Pass 1 (Total Loss)
            # Code: total_loss_bwd = scaled_total + phys_loss
            # If bug is present, phys_norm is added here.
            total_norm_1 = np.sqrt(grad_main**2 + grad_phys**2)
            
            # EMA Update 1 (Line 1236)
            ema = decay * ema + (1 - decay) * total_norm_1
            
            # 6. Backward Pass 2 (Post-Surgery)
            if double_backward:
                # Gradients accumulate in .grad buffer
                # New norm is (G_main + 2*G_phys)
                # Note: code uses sanitize_gradients which clips to 1.0 locally,
                # but EMA Update 2 (if present) would see the unclipped norm.
                pass # In the fixed code, Update 2 is gone.
            
            # In our simulation, we track the EMA seen at the START of next epoch
        
        history.append(ema)
        
    return history

def run_study():
    epochs = 10
    
    # Scenario A: The Bugged Architecture (As found)
    # Note: Double backward affects the WEIGHTS (model jumps), 
    # but the EMA only sees Update 1 now since I removed Update 2.
    # HOWEVER, Update 1 itself is now seeing Unweighted Physics at Epoch 5.
    hist_bug = simulate_training(bypass_curriculum=True, natural_phys_norm=10.0) # Assume SpO2 violation is high
    
    # Scenario B: Fixed (Physics curriculum applied, No cliffs)
    hist_fix = simulate_training(bypass_curriculum=False, natural_phys_norm=10.0)
    
    print("=" * 60)
    print("EMA PRESSURE PROGESSION (End of Epoch)")
    print("-" * 60)
    print(f"{'Epoch':<10} | {'Bugged (EMA)':<15} | {'Fixed (EMA)':<15}")
    print("-" * 60)
    for i in range(epochs):
        shock_marker = " [SHOCK!]" if hist_bug[i] > 5.0 else ""
        print(f"{i:<10} | {hist_bug[i]:<15.4f}{shock_marker} | {hist_fix[i]:<15.4f}")
    
    print("\nCONCLUSION:")
    if hist_bug[5] > hist_fix[5]:
        print(f"Verified: Epoch 5 Jump is {hist_bug[5]/hist_fix[5]:.2f}x higher with bug.")
        if hist_bug[5] > 5.0:
            print("Verified: Bugged version TRIPS SHOCK at Epoch 6 start.")

if __name__ == "__main__":
    run_study()
