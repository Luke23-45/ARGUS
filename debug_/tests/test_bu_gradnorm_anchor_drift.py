import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BU")

class MockGradNorm:
    def __init__(self, num_tasks=2, alpha=1.5):
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.initial_losses = torch.zeros(num_tasks)
        self.alpha = alpha
        self.step_count = 0

    def update(self, losses):
        # SIMULATE BUGGY FIXED ANCHOR (#117)
        if self.step_count == 0:
            self.initial_losses.copy_(losses)
        
        # After many steps, losses have drifted
        # rel_rates = L / L0
        rel_rates = losses / (self.initial_losses + 1e-8)
        avg_rate = rel_rates.mean()
        rel_rates = rel_rates / (avg_rate + 1e-8)
        
        # Target norms: E[||g||] * r^alpha
        # We assume norms.mean() = 1.0 for simplicity
        target_norms = 1.0 * (rel_rates ** self.alpha)
        
        self.step_count += 1
        return target_norms

def test_gradnorm_anchor_obsolescence():
    logger.info("Simulating GradNorm Anchor Obsolescence (#117)...")
    
    # Task 0: Solved (Loss decreases 10x)
    # Task 1: Hardening (Loss stays same or increases due to curriculum)
    L0 = torch.tensor([10.0, 10.0])
    gn = MockGradNorm()
    gn.update(L0) # Set anchor
    
    # After 5 epochs
    L_new = torch.tensor([1.0, 10.0])
    targets = gn.update(L_new)
    
    logger.info(f"Initial Losses: {L0.tolist()}")
    logger.info(f"Current Losses: {L_new.tolist()}")
    logger.info(f"Target Norms (Selection Pressure): {targets.tolist()}")
    
    # Analysis:
    # rel_rates = [1/10, 10/10] = [0.1, 1.0]
    # avg_rate = 0.55
    # norm_rel_rates = [0.1/0.55, 1.0/0.55] = [0.18, 1.81]
    # targets = [0.18^1.5, 1.81^1.5] = [0.07, 2.45]
    
    ratio = targets[1] / targets[0]
    logger.info(f"Pressure Ratio (Hard/Easy): {ratio:.2f}x")
    
    if ratio > 10:
        logger.error(f"❌ Smoking Gun #117 CONFIRMED! Fixed anchor causes extreme pressure imbalance ({ratio:.2f}x).")
        logger.warning("⚠️ Rationale: Solved tasks receive near-zero target norms, making their weights irrelevant. Hard tasks dominate the total gradient norm.")

if __name__ == "__main__":
    test_gradnorm_anchor_obsolescence()
