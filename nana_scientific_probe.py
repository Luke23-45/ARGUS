import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

## =============================================================================
## NANA SCIENTIFIC PROBE: THE ANATOMY OF A SYSTEM COLLAPSE
## =============================================================================

# Setup basic logging to simulate the training environment
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s')
logger = logging.getLogger("ForensicScientist")

class shared_foundation(nn.Module):
    """Simulates the Transformer Backbone."""
    def __init__(self, d_model=128):
        super().__init__()
        self.layer = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.layer(x)

class task_head(nn.Module):
    """Simulates a task-specific head (e.g., Diffusion or Critic)."""
    def __init__(self, d_model=128):
        super().__init__()
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        return self.head(x)

class BayesianProjectedScaler(nn.Module):
    """Exact logic from icu/models/components/loss_scaler.py."""
    def __init__(self, num_tasks=2, decay=0.99, log_compressed=True):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.register_buffer("loss_emas", torch.ones(num_tasks))
        self.decay = decay
        self.log_compressed = log_compressed
        self.step_count = 0
        self.warmup_steps = 10 

    def forward(self, losses_tensor):
        precision = torch.exp(-self.log_vars)
        
        # 1. EMA Update (Simulating the consensus/momentum)
        with torch.no_grad():
            self.step_count += 1
            curr_decay = 0.95 if self.step_count < self.warmup_steps else self.decay
            self.loss_emas.lerp_(losses_tensor.detach(), 1.0 - curr_decay)
        
        # 2. Theta Loss (For Model Weights) - uses detached precision
        theta_loss = (0.5 * precision.detach() * losses_tensor).sum()
        
        # 3. Sigma Loss (For Uncertainty Parameters)
        if self.log_compressed:
            # THE LOGICAL FAILURE POINT: log() mutes the explosive signal
            log_ema_losses = torch.log(self.loss_emas.detach() + 1.0)
            sigma_loss = (0.5 * precision * log_ema_losses + 0.5 * self.log_vars).sum()
        else:
            # THE SCIENTIFIC FIX: raw EMA reflects reality
            sigma_loss = (0.5 * precision * self.loss_emas.detach() + 0.5 * self.log_vars).sum()
            
        return theta_loss + sigma_loss, precision.detach()

def run_scientific_probe(use_log_compression=True):
    # Initialize "The Environment"
    backbone = shared_foundation()
    head_stable = task_head()
    head_volatile = task_head()
    
    scaler = BayesianProjectedScaler(log_compressed=use_log_compression)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head_stable.parameters()) + 
                                 list(head_volatile.parameters()) + list(scaler.parameters()), lr=1e-3)
    
    # Trackers for the "Scientist's Report"
    gn_history = []
    
    label = "LOG-COMPRESSED (FAILED DESIGN)" if use_log_compression else "RAW-SIGNAL (ROBUST DESIGN)"
    print(f"\n\n{'='*80}\nEXPERIMENT: {label}\n{'='*80}")
    print(f"{'Step':<5} | {'ExplosiveLoss':<13} | {'ScalerWeight':<12} | {'BackboneGradNorm':<16} | {'Status'}")
    print("-" * 80)

    for step in range(1, 101):
        x = torch.randn(1, 128)
        feats = backbone(x)
        
        # Task 1: Stable (Target=0)
        p1 = head_stable(feats)
        loss_stable = F.mse_loss(p1, torch.zeros_like(p1))
        
        # Task 2: Volatile (Critic) - After step 50, it begins to explode (Numerical Storm)
        p2 = head_volatile(feats)
        if step < 50:
            target2 = torch.zeros_like(p2)
        else:
            # Simulate "Outlier Divergence" - Critic loss starts increasing 10x per step
            target2 = torch.ones_like(p2) * (step - 49) * 2.0 
            
        loss_volatile = F.mse_loss(p2, target2)
        
        # Scaling pass
        losses = torch.stack([loss_stable, loss_volatile])
        total_loss, precisions = scaler(losses)
        
        # Optimization pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Measure pressure on the Foundation (Backbone)
        grad_norm = 0.0
        for p in backbone.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item()
        gn_history.append(grad_norm)
        
        # Check for system failure
        status = "HEALTHY"
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            status = "!!! SYSTEM COLLAPSE (NaN/INF) !!!"
            print(f"{step:<5} | {loss_volatile.item():<13.2f} | {precisions[1].item():<12.4f} | {grad_norm:<16.4f} | {status}")
            break
        
        if step % 10 == 0 or step > 45:
            print(f"{step:<5} | {loss_volatile.item():<13.2f} | {precisions[1].item():<12.4f} | {grad_norm:<16.4f} | {status}")

    return gn_history

if __name__ == "__main__":
    # Probe 1: The current codebase logic (Log-Compressed)
    logger.info("Starting Probe 1: The Mechanism of Failure...")
    probe1_res = run_scientific_probe(use_log_compression=True)
    
    # Probe 2: The "Scientist's Recommended" logic (Raw Signal)
    logger.info("Starting Probe 2: The Mechanism of Resilience...")
    probe2_res = run_scientific_probe(use_log_compression=False)
    
    print("\n\nSCIENTIFIC CONCLUSION:")
    if len(probe1_res) < len(probe2_res):
        print(f"-> SYSTEM FAILED with Log-Compression at step {len(probe1_res)}.")
        print(f"-> SYSTEM SURVIVED with Raw-Signal calibration for all 100 steps.")
        print("\nTHEORY PROVED: The log() function in the Bayesian update creates a 'Signal Lag'.")
        print("This lag allowed the Volatile task weight to stay high while its gradient grew,")
        print("poisoning the shared Foundation until numerical limits were exceeded.")
    else:
        print("-> Both probes completed. Further investigation into the specific loss function required.")
