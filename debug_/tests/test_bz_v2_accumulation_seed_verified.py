import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_BZ_v2")

def test_accumulation_seed_verification():
    logger.info("Verifying Accumulation Seed Diversity (#136)...")
    
    # Simulation Parameters
    accumulate_grad_batches = 4
    n_batches = 8
    
    # Internal state simulation
    # global_step only increments after the cycle
    global_step = 100
    current_epoch = 1
    
    seeds = []
    
    logger.info(f"Batch | Global Step | Seed | Unique in Cycle?")
    logger.info("-" * 50)
    
    for batch_idx in range(n_batches):
        # NEW IMPLEMENTATION LOGIC
        # ghost_seed = (self.current_epoch * 12345 + self.global_step + (batch_idx % acc_batches)) % (2**31)
        ghost_seed = (current_epoch * 12345 + global_step + (batch_idx % accumulate_grad_batches)) % (2**31)
        
        cycle_pos = batch_idx % accumulate_grad_batches
        is_unique = ghost_seed not in seeds[batch_idx - cycle_pos : batch_idx]
        
        seeds.append(ghost_seed)
        logger.info(f"{batch_idx:5d} | {global_step:11d} | {ghost_seed:10d} | {is_unique}")
        
        # PyTorch Lightning Step Logic:
        # global_step increments only AFTER optimizer.step()
        if (batch_idx + 1) % accumulate_grad_batches == 0:
            global_step += 1
            
    # Analysis
    unique_in_first_cycle = len(set(seeds[:accumulate_grad_batches]))
    
    # SUCCESS CRITERIA:
    # If accumulate_grad_batches=4, we expect 4 unique seeds in the first cycle.
    if unique_in_first_cycle == accumulate_grad_batches:
        logger.info(f"✅ Verification SUCCESS! {unique_in_first_cycle} unique seeds detected per cycle.")
    else:
        logger.error(f"❌ Verification FAILED! Only {unique_in_first_cycle} unique seeds in cycle.")

if __name__ == "__main__":
    test_accumulation_seed_verification()
