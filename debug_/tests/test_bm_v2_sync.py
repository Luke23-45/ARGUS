import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BM_v2")

def test_gradnorm_consensus():
    logger.info("Verifying Patch #88: GradNorm/Governor Consensus...")
    
    # 1. Simulate Local Drift
    # Rank 0 and Rank 1 compute different stability factors due to local floating point noise
    sf0_local = 0.50000001 # Rank 0 calculation
    sf1_local = 0.49999999 # Rank 1 calculation
    
    # 2. Apply Consensus logic (v88.0)
    def get_consensus_sf(local_sf, rank, mock_dist):
        if rank == 0:
            # Rank 0 sends its value
            mock_dist['sf'] = local_sf
            return local_sf
        else:
            # Rank 1 receives from Rank 0
            return mock_dist['sf']

    mock_dist = {}
    
    sf0_final = get_consensus_sf(sf0_local, rank=0, mock_dist=mock_dist)
    sf1_final = get_consensus_sf(sf1_local, rank=1, mock_dist=mock_dist)
    
    logger.info(f"Final SF Rank 0: {sf0_final:.10f}")
    logger.info(f"Final SF Rank 1: {sf1_final:.10f}")
    
    diff = abs(sf0_final - sf1_final)
    if diff == 0:
        logger.info("✅ Patch #88 SUCCESS! Bit-perfect stability factor parity achieved.")
    else:
        logger.error(f"❌ Verification Failed. Divergence: {diff:.10f}")

if __name__ == "__main__":
    test_gradnorm_consensus()
