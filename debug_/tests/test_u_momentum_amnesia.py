import torch
import logging
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from icu.utils.train_utils import ScalingSteward

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_U")

def simulate_balancer_drift(n_steps=200, beta=0.99):
    """Simulates how much a balancer 'remembers' after one epoch."""
    # After N steps, the weight of the initial state is beta^N
    memory_remaining = beta ** n_steps
    return memory_remaining

def run_momentum_analysis():
    logger.info("Simulating Momentum Amnesia across Epoch Densities...")
    
    # 1. Base Case: 200 steps (The reference from ScalingSteward)
    mem_base = simulate_balancer_drift(n_steps=ScalingSteward.REF_STEPS, beta=0.99)
    logger.info(f"Ref Epoch (200 steps): Memory Retention = {mem_base*100:.1f}% (Healthy)")
    
    # 2. Dense Case (UNPATCHED): 4000 steps with fixed beta=0.99
    mem_dense = simulate_balancer_drift(n_steps=4000, beta=0.99)
    logger.info(f"Dense Epoch (4000 steps, Fixed Beta): Memory Retention = {mem_dense*100:.4f}% (Amnesia!)")
    
    # 3. Dense Case (PATCHED): 4000 steps with ScalingSteward
    beta_scaled = ScalingSteward.get_decay(0.99, 4000)
    mem_corrected = simulate_balancer_drift(n_steps=4000, beta=beta_scaled)
    logger.info(f"Dense Epoch (4000 steps, Scaled Beta): Memory Retention = {mem_corrected*100:.1f}% (Restored)")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    
    issues = []
    if mem_dense < 0.05: # Retention should be significant
        logger.warning(f"Confirmed: Fixed beta (0.99) fails at high density ({mem_dense*100:.4f}% memory).")
    
    if abs(mem_corrected - mem_base) < 1e-4:
        logger.info("\u2705 PATCH U VERIFIED: ScalingSteward perfectly restored epoch-level memory parity.")
        logger.info("\u2705 TEST U PASSED: Hardening patch is robust.")
    else:
        logger.error("\u274c TEST U FAILED: ScalingSteward did not restore memory parity.")
            
    return issues

if __name__ == "__main__":
    run_momentum_analysis()
