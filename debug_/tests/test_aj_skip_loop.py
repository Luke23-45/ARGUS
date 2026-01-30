import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AJ")

def simulate_skip_loop():
    logger.info("Simulating Skip-Loop Trap...")
    
    p = nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    
    # Batch 1: Causes INF
    logger.info("Batch 1: Injecting INF gradient.")
    loss1 = p * float('inf')
    loss1.backward()
    
    # Simulation of PL training loop
    should_apply = torch.isfinite(p.grad).all().item()
    
    if should_apply:
        opt.step()
        opt.zero_grad()
    else:
        logger.warning("INF detected, skipping step but NOT calling zero_grad (The Bug).")
        # opt.zero_grad() # This is the missing fix
        
    # Batch 2: Normal gradients
    logger.info("Batch 2: Injecting normal gradient.")
    loss2 = p * 1.0
    loss2.backward()
    
    # Check if Inf still exists
    is_inf = torch.isinf(p.grad).any().item()
    logger.info(f"Is gradient INF on Batch 2? {is_inf}")
    
    if is_inf:
        logger.error("\u274c SMOKING GUN #5: Skip-Loop Trap! The model is stuck in INF-skip cycle.")
    else:
        logger.info("\u2705 Gradient cleared.")

if __name__ == "__main__":
    simulate_skip_loop()
