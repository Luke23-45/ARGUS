import torch
import torch.nn as nn
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_AE")

class MockBayesianScaler(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.optimizer = torch.optim.Adam([self.log_vars], lr=0.1)

    def forward(self, losses):
        precision = torch.exp(-self.log_vars)
        # SOTA Bayesian Loss: 0.5 * precision * L + 0.5 * log_var
        return (0.5 * precision * losses + 0.5 * self.log_vars).sum()

def simulate_rank_divergence(steps=100, drift_scale=0.1):
    logger.info(f"Simulating Rank Divergence over {steps} steps (DDP 2 Ranks)...")
    
    # [BUGGY] Independent Ranks
    rank0_buggy = MockBayesianScaler()
    rank1_buggy = copy.deepcopy(rank0_buggy)
    
    # [PATCHED] Synced Ranks (Universal Bridge)
    rank0_patched = MockBayesianScaler()
    rank1_patched = copy.deepcopy(rank0_patched)
    
    divergence_buggy = []
    divergence_patched = []
    
    for i in range(steps):
        # Base losses (Rank-Synced)
        base_losses = torch.tensor([1.5, 0.8, 2.2])
        
        # Rank Drift (Slightly different batch slices)
        noise0 = torch.tensor([drift_scale, 0.0, -drift_scale])
        noise1 = torch.tensor([-drift_scale, 0.0, drift_scale])
        
        l0 = base_losses + noise0
        l1 = base_losses + noise1
        
        # --- 1. BUGGY PASS ---
        rank0_buggy.optimizer.zero_grad()
        rank0_buggy(l0).backward()
        rank0_buggy.optimizer.step()
        
        rank1_buggy.optimizer.zero_grad()
        rank1_buggy(l1).backward()
        rank1_buggy.optimizer.step()
        
        # --- 2. PATCHED PASS (Determinstic Consensus) ---
        # Sync input losses before weighting (Universal Bridge Pillar 1)
        avg_losses = (l0 + l1) / 2.0
        
        rank0_patched.optimizer.zero_grad()
        rank0_patched(avg_losses).backward()
        rank0_patched.optimizer.step()
        
        rank1_patched.optimizer.zero_grad()
        rank1_patched(avg_losses).backward()
        rank1_patched.optimizer.step()
        
        # Track Divergence
        divergence_buggy.append(torch.abs(rank0_buggy.log_vars - rank1_buggy.log_vars).max().item())
        divergence_patched.append(torch.abs(rank0_patched.log_vars - rank1_patched.log_vars).max().item())

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS: RANK DIVERGENCE")
    logger.info("="*60)
    logger.info(f"Buggy Final Divergence:   {divergence_buggy[-1]:.4f}")
    logger.info(f"Patched Final Divergence: {divergence_patched[-1]:.4f}")
    
    reduction = (1.0 - (divergence_patched[-1] / (divergence_buggy[-1] + 1e-8))) * 100
    logger.info(f"Divergence Reduction:     {reduction:.1f}%")

    if divergence_patched[-1] < 1e-6:
        logger.info("\u2705 TEST AE PASSED: Universal Bridge eliminates Rank Divergence.")
    else:
        logger.error("\u274c TEST AE: Divergence still detected in Patched mode.")
            
    return divergence_patched[-1]

if __name__ == "__main__":
    simulate_rank_divergence()
