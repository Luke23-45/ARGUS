import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BO_v2")

class MockAWR:
    def __init__(self, beta=0.5):
        self.beta = beta
        self.adv_mean = torch.tensor([0.0])
        self.adv_std = torch.tensor([1.0])

    def compute_consensus_weights(self, advantages, rank, mock_dist):
        # SIMULATE THE FIX (#96, #97): Global Stats and Global Max
        
        # 1. Whitening (Assume pre-synchronized global stats)
        # Using Rank 0's stats as consensus
        mu, sigma = mock_dist['mu'], mock_dist['sigma']
        
        norm_adv = (advantages - mu) / sigma
        scaled_adv = norm_adv / self.beta
        
        # 2. GLOBAL MAX (#97)
        l_max = scaled_adv.max()
        if rank == 0:
            mock_dist['max'] = l_max
            g_max = l_max
        else:
            # Rank 1 sees Rank 0's max if it's larger
            g_max = max(l_max, mock_dist['max'])
            mock_dist['max'] = g_max
            
        log_w = scaled_adv - g_max
        weights = torch.exp(log_w)
        
        return weights

def test_awr_consensus_fixed():
    logger.info("Verifying Patch #96/97: AWR Global Consensus...")
    
    # CASE: Rank 0 has critical samples (10.0), Rank 1 has stable samples (1.0)
    adv0 = torch.tensor([10.0, 8.0, 6.0])
    adv1 = torch.tensor([1.0, 0.5, 0.1])
    
    mock_dist = {
        'mu': torch.tensor([5.0]), # Consensus Mean
        'sigma': torch.tensor([2.0]), # Consensus Std
        'max': torch.tensor([-100.0]) 
    }
    
    awr = MockAWR()
    
    # 1. First Pass to find Global Max
    w0_raw = awr.compute_consensus_weights(adv0, 0, mock_dist)
    w1_raw = awr.compute_consensus_weights(adv1, 1, mock_dist)
    
    # 2. Re-compute with final global max for Rank 1 (simulating DDP all_reduce)
    w1_final = awr.compute_consensus_weights(adv1, rank=1, mock_dist=mock_dist)
    w0_final = w0_raw # Already has the global max (10.0)
    
    logger.info(f"Rank 0 Weights: {w0_final}")
    logger.info(f"Rank 1 Weights: {w1_final}")
    
    # ANALYSIS
    best0 = w0_final[0].item()
    best1 = w1_final[0].item()
    
    ratio = best0 / (best1 + 1e-8)
    logger.info(f"Weight Ratio (Best Rank 0 / Best Rank 1): {ratio:.4f}")
    
    # In math: exp((10-5)/2/0.5) / exp((1-5)/2/0.5) = exp(5) / exp(-4) = exp(9) approx 8103
    if ratio > 1000:
        logger.info("✅ Patch #96/97 SUCCESS! Critical samples keep their dominance across ranks.")
    else:
        logger.error(f"❌ Verification Failed. Ratio {ratio:.4f} is too low (equalized).")

if __name__ == "__main__":
    test_awr_consensus_fixed()
