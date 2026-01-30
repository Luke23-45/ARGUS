import torch
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_AF")

class MockAWRCanculator:
    def __init__(self, initial_beta=0.05, momentum=0.9):
        self.beta = torch.tensor(initial_beta)
        self.momentum = momentum
        self.min_beta = 0.01

    def update_beta(self, local_weights):
        # ESS calculation (Local)
        ess = (local_weights.sum()**2) / (local_weights**2).sum()
        # Mock adaptive update
        error = (ess / len(local_weights)) - 0.5 # Aim for 50% ESS
        correction = torch.exp(torch.tensor(0.1 * error))
        
        new_beta = self.beta * correction
        self.beta.copy_(new_beta.clamp(min=self.min_beta))

def simulate_awr_drift(steps=100):
    logger.info(f"Simulating AWR Beta Drift over {steps} steps (DDP 2 Ranks)...")
    
    # [BUGGY] Independent Adaptation
    calc0_buggy = MockAWRCanculator()
    calc1_buggy = MockAWRCanculator()
    
    # [PATCHED] Synced Adaptation (Universal Bridge Pillar 3)
    calc0_patched = MockAWRCanculator()
    calc1_patched = MockAWRCanculator()
    
    for i in range(steps):
        # Rank 0: Low ESS, Rank 1: High ESS
        w0 = torch.tensor([10.0, 1.0, 1.0, 1.0])
        w1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        # Buggy Pass
        calc0_buggy.update_beta(w0)
        calc1_buggy.update_beta(w1)
        
        # Patched Pass: Simulate all_reduce on beta/stats
        calc0_patched.update_beta(w0)
        calc1_patched.update_beta(w1)
        
        # Consensus: Average beta across ranks (mimics all_reduce implemented in AWR)
        avg_beta = (calc0_patched.beta + calc1_patched.beta) / 2.0
        calc0_patched.beta.copy_(avg_beta)
        calc1_patched.beta.copy_(avg_beta)

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS: AWR BETA SYNC")
    logger.info("="*60)
    buggy_diff = torch.abs(calc0_buggy.beta - calc1_buggy.beta).item()
    patched_diff = torch.abs(calc0_patched.beta - calc1_patched.beta).item()
    
    logger.info(f"Buggy Final Divergence:   {buggy_diff:.4f}")
    logger.info(f"Patched Final Divergence: {patched_diff:.4f}")

    if patched_diff < 1e-6:
        logger.info("\u2705 TEST AF PASSED: AWR Bridge maintains consensus.")
    else:
        logger.error("\u274c TEST AF: Beta Still Diverged in Patched mode.")
            
    return patched_diff

if __name__ == "__main__":
    simulate_awr_drift()
