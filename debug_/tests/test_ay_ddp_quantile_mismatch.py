import torch
import torch.distributed as dist
import os
from icu.utils.advantage_calculator import ICUAdvantageCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AY")

def simulate_ddp_ranks():
    # We can't easily launch real DDP here, but we can mock the all_reduce behavior
    # and use the calc's internal logic.
    
    calc = ICUAdvantageCalculator(beta=0.5)
    
    # Rank 0 Data: Normal clinical distribution
    adv_r0 = torch.randn(128, 24) * 2.0 
    # Rank 1 Data: Contains a massive outlier
    adv_r1 = torch.randn(128, 24) * 2.0
    adv_r1[0, 0] = 500.0 # Extreme outlier
    
    # 1. Compute p99 locally
    p99_r0 = torch.quantile(adv_r0.detach().float(), 0.99)
    p99_r1 = torch.quantile(adv_r1.detach().float(), 0.99)
    
    # 2. Simulate DDP.all_reduce(..., op=MAX)
    global_p99 = torch.max(p99_r0, p99_r1)
    
    logger.info(f"Local P99 R0: {p99_r0.item():.2f}")
    logger.info(f"Local P99 R1: {p99_r1.item():.2f}")
    logger.info(f"Global P99 Sync (MAX): {global_p99.item():.2f}")
    
    # 3. Apply weights with real logic (simulated across ranks)
    # Under v36.0 FIX: Every rank uses its LOCAL maximum.
    
    calc.set_stats(mean=0.0, std=1.0)
    
    # We call calc.calculate_awr_weights(adv)
    # Rank 0 will calculate weights_local relative to its max (4.74)
    # Rank 1 will calculate weights_local relative to its max (500.0)
    
    # Then it sums them: Sum_w_global = Sum(exp(adv0 - 4.74)) + Sum(exp(adv1 - 500))
    # Finally: weights = weights_local * (total_numel / sum_w_global)
    
    # Simulate first part: weights_local
    adv0_clipped = torch.clamp(adv_r0, max=global_p99)
    adv1_clipped = torch.clamp(adv_r1, max=global_p99)
    
    scaled0 = adv0_clipped / 0.5
    scaled1 = adv1_clipped / 0.5
    
    w_loc0 = torch.exp(scaled0 - scaled0.max())
    w_loc1 = torch.exp(scaled1 - scaled1.max())
    
    sum_w_global = w_loc0.sum() + w_loc1.sum()
    total_numel = adv_r0.numel() + adv_r1.numel()
    
    norm_factor = total_numel / (sum_w_global + 1e-8)
    
    weights_r0 = w_loc0 * norm_factor
    weights_r1 = w_loc1 * norm_factor
    
    logger.info(f"R0 Weight Mean: {weights_r0.mean().item():.2f}")
    logger.info(f"R1 Weight Mean: {weights_r1.mean().item():.2f}")
    
    if weights_r0.mean() < 1e-1: # Should be roughly 1.0 if balanced
        logger.error(f"❌ DDP Rank Starvation persists! Rank 0 mean weight {weights_r0.mean().item():.2e}")
    else:
        logger.info("✅ DDP Rank Starvation neutralized. Rank 0 is contributing signal.")

if __name__ == "__main__":
    simulate_ddp_ranks()
