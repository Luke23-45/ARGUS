import torch
import torch.nn as nn
from icu.core.gradnorm import GradNormBalancer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BV")

def test_gradnorm_rank_divergence():
    logger.info("Simulating GradNorm Rank Divergence (#131)...")
    
    # Simulate 2 Ranks
    num_tasks = 2
    gn_rank0 = GradNormBalancer(num_tasks=num_tasks)
    gn_rank1 = GradNormBalancer(num_tasks=num_tasks)
    
    # Ensure they start identical
    with torch.no_grad():
        gn_rank0.weights.copy_(torch.tensor([1.0, 1.0]))
        gn_rank1.weights.copy_(torch.tensor([1.0, 1.0]))

    # Rank 0 sees Task 0 as "Hard", Task 1 as "Easy"
    L0_rank0 = torch.tensor([10.0, 1.0])
    # Rank 1 sees Task 0 as "Easy", Task 1 as "Hard" (Different shuffle)
    L0_rank1 = torch.tensor([1.0, 10.0])
    
    # [v131.0 BUG] Each rank updates its weights based ONLY on local losses
    gn_rank0.update(L0_rank0)
    gn_rank1.update(L0_rank1)
    
    w0 = gn_rank0.get_weights()
    w1 = gn_rank1.get_weights()
    
    logger.info(f"Rank 0 Task Weights: {w0.tolist()}")
    logger.info(f"Rank 1 Task Weights: {w1.tolist()}")
    
    diff = (w0 - w1).abs().mean().item()
    logger.info(f"Mean Difference across ranks: {diff:.4f}")
    
    if diff > 0.1:
        logger.error(f"❌ Smoking Gun #131 CONFIRMED! GradNorm weights diverge across ranks ({diff:.4f}).")
        logger.warning("⚠️ Rationale: Without global loss synchronization, each rank optimizes for a different task distribution, leading to gradient skew in DDP.")

if __name__ == "__main__":
    test_gradnorm_rank_divergence()
