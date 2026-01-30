import torch
from icu.models.components.ghost_bank import SepsisGhostBank
import collections
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AW")

def test_ghost_monoculture():
    logger.info("Simulating Ghost Monoculture (Smoking Gun #23)...")
    
    capacity = 100
    bank = SepsisGhostBank(capacity=capacity)
    
    # 1. Fill the bank with 100 samples
    # 5 samples are "hard" (uncertainty = 0.8)
    # 95 samples are "easy" (uncertainty = 0.1)
    vitals = torch.randn(capacity, 24, 28)
    masks = torch.ones(capacity, 24, 28)
    labels = torch.zeros(capacity, dtype=torch.long)
    latents = torch.randn(capacity, 512)
    uncertainties = torch.full((capacity, 1), 0.1)
    uncertainties[:5] = 0.8 # Top 5% are hard
    
    bank.update(vitals, masks, labels, latents, uncertainties)
    
    # 2. Sample many times and count frequency
    counts = collections.defaultdict(int)
    num_samples = 1000
    batch_size = 10
    
    for i in range(num_samples // batch_size):
        # Using different seeds to simulate training steps
        out = bank.sample(num_ghosts=batch_size, seed=i, uncertainty_weighted=True)
        # We don't have the indices back, so let's check the labels or latents
        # To identify them uniquely, let's use the first element of the latent
        for l in out["anchors"]:
             counts[l[0].item()] += 1
             
    # 3. Analyze distribution
    sorted_counts = sorted(counts.values(), reverse=True)
    logger.info(f"Top 5 sample frequencies: {sorted_counts[:5]}")
    logger.info(f"Bottom 5 sample frequencies: {sorted_counts[-5:] if len(sorted_counts) > 5 else []}")
    
    diversity = len(counts)
    logger.info(f"Unique samples seen: {diversity} / {capacity}")
    
    if diversity < 10:
        logger.error(f"❌ Ghost Monoculture Detected! Model is only seeing {diversity} unique ghosts.")
    else:
        logger.info(f"✅ Ghost diversity is acceptable ({diversity} unique samples).")

if __name__ == "__main__":
    test_ghost_monoculture()
