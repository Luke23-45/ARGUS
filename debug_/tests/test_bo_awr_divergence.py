import torch
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BO")

class MockAWR:
    def __init__(self, beta=0.5, mom=0.999):
        self.beta = beta
        self.mom = mom
        self.adv_mean = torch.zeros(1)
        self.adv_std = torch.ones(1)
        self.max_weight = torch.tensor(20.0)

    def compute_weights_local_max(self, advantages, rank):
        # SIMULATE THE BUG (#97): Local Max Subtraction
        adv_flat = advantages.flatten()
        
        # Whitening EMA (#96 - Local Update)
        curr_mu = adv_flat.mean()
        curr_sigma = adv_flat.std() + 1e-8
        self.adv_mean = self.mom * self.adv_mean + (1.0 - self.mom) * curr_mu
        self.adv_std = self.mom * self.adv_std + (1.0 - self.mom) * curr_sigma
        
        norm_adv = (advantages - self.adv_mean) / self.adv_std
        scaled_adv = norm_adv / self.beta
        
        # LOCAL MAX (#97)
        local_max = scaled_adv.max()
        log_w = scaled_adv - local_max
        weights = torch.exp(log_w)
        
        # Global Sum Norm
        sum_w = weights.sum()
        numel = weights.numel()
        # Assume dist.all_reduce happened
        return weights, sum_w, numel

def test_awr_rank_divergence():
    logger.info("Simulating AWR Rank Divergence (#96, #97)...")
    
    # Setup
    awr0 = MockAWR()
    awr1 = MockAWR()
    
    # SCENARIO: Rank 0 gets Critical samples (High Reward/Adv), Rank 1 gets Stable samples (Low Adv)
    # Global Max is in Rank 0
    adv0 = torch.tensor([10.0, 5.0, 2.0]) 
    adv1 = torch.tensor([1.0, 0.5, 0.1])
    
    # 1. Whitening Phase
    # After many steps, Rank 0 and Rank 1 see different distributions
    # Mocking hours of drift:
    awr0.adv_mean = torch.tensor([5.0])
    awr0.adv_std = torch.tensor([2.0])
    
    awr1.adv_mean = torch.tensor([0.5])
    awr1.adv_std = torch.tensor([0.2])
    
    # 2. Compute Weights
    w0, sum0, n0 = awr0.compute_weights_local_max(adv0, 0)
    w1, sum1, n1 = awr1.compute_weights_local_max(adv1, 1)
    
    # 3. Global Normalization
    global_sum = sum0 + sum1
    global_numel = n0 + n1
    norm_factor = global_numel / global_sum
    
    w0_final = w0 * norm_factor
    w1_final = w1 * norm_factor
    
    logger.info(f"Rank 0 Final Weights: {w0_final}")
    logger.info(f"Rank 1 Final Weights: {w1_final}")
    
    # ANALYSIS
    # Global Rank of advantages: 10.0 > 5.0 > 2.0 > 1.0 > 0.5 > 0.1
    # Mathematically, exp(10) >> exp(1).
    # Rank 1's best sample (Adv=1.0) should have MUCH lower weight than Rank 0's best (Adv=10.0).
    
    best0 = w0_final[0].item()
    best1 = w1_final[0].item()
    
    logger.info(f"Weight Ratio (Best Rank 0 / Best Rank 1): {best0/best1:.4f}")
    
    if abs(best0 - best1) < 1.0: # If they are roughly equalized
        logger.error("❌ Smoking Gun #97 CONFIRMED! Local Max Subtraction equalized rank importance.")
        logger.warning("⚠️ Rationale: Rank 1's mediocre samples were 'promoted' to be as important as Rank 0's outliers.")

    if abs(awr0.adv_mean - awr1.adv_mean) > 1.0:
        logger.error(f"❌ Smoking Gun #96 CONFIRMED! Whitening stats drifted between ranks (Diff: {abs(awr0.adv_mean - awr1.adv_mean).item():.2f})")

if __name__ == "__main__":
    test_awr_rank_divergence()
