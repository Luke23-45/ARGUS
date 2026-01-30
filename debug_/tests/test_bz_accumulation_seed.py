import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BZ")

def test_accumulation_seed_stagnation():
    logger.info("Verifying Smoking Gun #136: Accumulation Seed Stagnation...")
    
    # Simulation Parameters
    accumulate_grad_batches = 4
    n_batches = 8
    
    # Internal state simulation
    global_step = 0
    seeds = []
    
    for batch_idx in range(n_batches):
        # Current logic for seed calculation (wrapper_generalist.py:743)
        # seed = self.cfg.seed + self.global_step
        seed = 42 + global_step
        seeds.append(seed)
        
        # PyTorch Lightning Step Logic:
        # global_step increments only AFTER optimizer.step()
        if (batch_idx + 1) % accumulate_grad_batches == 0:
            global_step += 1
            
    logger.info(f"Batch Indices: {list(range(n_batches))}")
    logger.info(f"Generated Seeds: {seeds}")
    
    unique_seeds_per_cycle = len(set(seeds[:accumulate_grad_batches]))
    
    # THE SMOKING GUN:
    # If accumulate_grad_batches=4, we expect 4 unique seeds.
    # But if PL increments step only at the end, we get 1 unique seed.
    if unique_seeds_per_cycle == 1:
        logger.error(f"❌ Smoking Gun #136 CONFIRMED! Seed is STAGNANT for {accumulate_grad_batches} batches.")
        logger.warning(f"⚠️ Rationale: All sub-batches in a cycle (4) use seed={seeds[0]}, sampling the same ghosts. Diversity is reduced by {accumulate_grad_batches}x.")

if __name__ == "__main__":
    test_accumulation_seed_stagnation()
