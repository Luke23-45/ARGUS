import torch
import torch.nn as nn
import torch.nn.functional as F
import math

## =============================================================================
## NANA SCIENTIFIC PROBE V3: THE CYBERNETIC ECHO CHAMBER (FEEDBACK COLLAPSE)
## =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleCritic(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1) # Predicts scalar Value
        )
    def forward(self, x):
        return self.net(x)

def run_scientific_probe_v3(use_huber=False, use_jitter_loop=True):
    print(f"\n--- NANA PROBE V3: {'HUBER (HEALED)' if use_huber else 'SQUARED (INFECTED)'} + {'JITTER-LOOP' if use_jitter_loop else 'CLEAN'} ---")
    
    critic = SimpleCritic().to(DEVICE)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    
    adv_scale = 1.0 # The "Advantage StdDev" from AWR
    
    print(f"{'Step':<5} | {'V-Loss':<13} | {'AdvScale':<12} | {'GradNorm':<12} | {'Status'}")
    print("-" * 75)

    for step in range(1, 101):
        x = torch.randn(1, 128, device=DEVICE)
        v_pred = critic(x)
        
        # 1. Simulate the "Cybernetic Jitter" from wrapper_generalist.py L1731
        # target = baseline + noise * adv_scale
        # In our case, baseline is 10.0 (arbitrary)
        noise = torch.randn_like(v_pred) * (0.05 * adv_scale if use_jitter_loop else 0.0)
        target = torch.tensor([[10.0]], device=DEVICE) + noise
        
        # 2. Compute Loss
        diff = target - v_pred
        if use_huber:
            # THE HEALED LOGIC: Linear gradients for large errors
            loss = F.huber_loss(v_pred, target, delta=1.0)
        else:
            # THE INFECTED LOGIC: Squared pressure (diff**2)
            loss = (diff**2).mean()
            
        # 3. Simulate AWR feedback loop
        # In a real run, increasing error leads to higher advantage variance
        # which increases adv_scale for the NEXT step.
        with torch.no_grad():
            error_mag = torch.abs(diff).item()
            # Positive Feedback: higher error -> higher adv_scale -> higher jitter -> higher error
            adv_scale = 0.9 * adv_scale + 0.1 * (1.0 + error_mag)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Measure GradNorm
        grad_norm = 0.0
        for p in critic.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item()
        
        status = "HEALTHY"
        if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > 10000:
            status = "!!! SYSTEM COLLAPSE !!!"
            print(f"{step:<5} | {loss.item():<13.2f} | {adv_scale:<12.4f} | {grad_norm:<12.4f} | {status}")
            break
            
        if step % 20 == 0:
            print(f"{step:<5} | {loss.item():<13.2f} | {adv_scale:<12.4f} | {grad_norm:<12.4f} | {status}")

if __name__ == "__main__":
    # Test 1: The Infected logic (Current v12 codebase)
    run_scientific_probe_v3(use_huber=False, use_jitter_loop=True)
    
    # Test 2: The Healed logic (Scientist's recommendation)
    run_scientific_probe_v3(use_huber=True, use_jitter_loop=True)
