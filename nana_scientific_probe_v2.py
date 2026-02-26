import torch
import torch.nn as nn
import torch.nn.functional as F
import math

## =============================================================================
## NANA SCIENTIFIC PROBE V2: THE GHOST-LOOP COLLAPSE
## =============================================================================

# Simulate FP16 constraints for realistic training dynamics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

class shared_foundation(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.layer = nn.Linear(d_model, d_model, device=DEVICE)
    def forward(self, x):
        return self.layer(x)

class BayesianProjectedScaler(nn.Module):
    def __init__(self, num_tasks=2, decay=0.99, log_compressed=True):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=DEVICE))
        self.register_buffer("loss_emas", torch.ones(num_tasks, device=DEVICE))
        self.decay = decay
        self.log_compressed = log_compressed
        self.step_count = 0
        self.warmup_steps = 10

    def forward(self, losses_tensor):
        precision = torch.exp(-self.log_vars)
        with torch.no_grad():
            self.step_count += 1
            curr_decay = 0.85 if self.step_count < self.warmup_steps else self.decay
            self.loss_emas.lerp_(losses_tensor.detach(), 1.0 - curr_decay)
        
        theta_loss = (0.5 * precision.detach() * losses_tensor).sum()
        if self.log_compressed:
            log_ema_losses = torch.log(self.loss_emas.detach() + 1.0)
            sigma_loss = (0.5 * precision * log_ema_losses + 0.5 * self.log_vars).sum()
        else:
            sigma_loss = (0.5 * precision * self.loss_emas.detach() + 0.5 * self.log_vars).sum()
        return theta_loss + sigma_loss, precision.detach()

def run_scientific_probe_v2():
    print(f"\n--- NANA PROBE V2: RUNNING (Device={DEVICE}, dtype={DTYPE}) ---")
    
    foundation = shared_foundation()
    head_stable = nn.Linear(256, 1, device=DEVICE)
    head_ghost = nn.Linear(256, 1, device=DEVICE) # Represents the Expert manifold
    
    scaler = BayesianProjectedScaler(decay=0.99, log_compressed=True)
    optimizer = torch.optim.Adam(list(foundation.parameters()) + list(scaler.parameters()), lr=1e-3)
    
    # Simulating the Ghost Bank
    ghost_bank_anchors = [torch.randn(1, 1, device=DEVICE)] * 10
    
    print(f"{'Step':<5} | {'V-Loss':<13} | {'ScalerWeight':<12} | {'GradNorm':<12} | {'Status'}")
    print("-" * 70)

    for step in range(1, 201):
        x = torch.randn(1, 256, device=DEVICE)
        feats = foundation(x)
        
        # Task 1: Stable
        p1 = head_stable(feats)
        loss_stable = F.mse_loss(p1, torch.zeros_like(p1))
        
        # Task 2: Expert/Ghost Loop (The Poisoned Manifold)
        p2 = head_ghost(feats)
        
        # Ghost Anchor logic: The "Label" for Task 2 depends on OLD predictions
        # If the model drifts, the anchor stays stale, creating massive error.
        anchor = ghost_bank_anchors[step % 10]
        loss_volatile = F.mse_loss(p2, anchor)
        
        # Update Ghost Bank with current (potentially divergent) state
        if step % 5 == 0:
            ghost_bank_anchors[step % 10] = p2.detach() * 1.5 # Simulate a diverging signal phase
            
        # Simulate FP16 Gradient Scaling (Standard in Pytorch Lightning)
        # Scaler logic
        losses = torch.stack([loss_stable, loss_volatile])
        total_loss, precisions = scaler(losses)
        
        optimizer.zero_grad()
        # Backprop through Scaler and Foundation
        total_loss.backward()
        
        # Check Backbone Grad Norm
        grad_norm = 0.0
        for p in foundation.parameters():
            if p.grad is not None:
                # Force bit-depth ceiling (FP16 limit)
                if DTYPE == torch.float16:
                    p.grad.data.clamp_(min=-65504, max=65504)
                
                gn = p.grad.detach().data.norm(2).item()
                grad_norm += gn
        
        status = "HEALTHY"
        if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > 10000:
            status = "!!! SYSTEM COLLAPSE !!!"
            print(f"{step:<5} | {loss_volatile.item():<13.2f} | {precisions[1].item():<12.4f} | {grad_norm:<12.4f} | {status}")
            break
            
        if step % 20 == 0:
            print(f"{step:<5} | {loss_volatile.item():<13.2f} | {precisions[1].item():<12.4f} | {grad_norm:<12.4f} | {status}")

if __name__ == "__main__":
    run_scientific_probe_v2()
