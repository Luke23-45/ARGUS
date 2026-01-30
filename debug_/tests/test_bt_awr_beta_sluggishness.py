import torch
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BT")

class MockAWR:
    def __init__(self, beta=0.1):
        self.beta = torch.tensor([beta])
        self.beta_momentum = 0.99
    
    def update_beta(self, ess):
        # CURRENT IMPLEMENTATION (Vulnerable to sluggishness)
        target_ess = 0.20
        error_ess = (target_ess - ess)
        
        # P-Controller with Anti-Windup
        correction = math.exp(10.0 * error_ess)
        correction = max(0.5, min(2.0, correction))
        new_beta = self.beta.item() * correction
        
        mom = self.beta_momentum
        self.beta.copy_((mom * self.beta) + ((1.0 - mom) * new_beta))

def test_awr_beta_sluggishness():
    logger.info("Simulating AWR Beta Sluggishness (#116)...")
    
    # SCENARIO: ESS drops to 0.01 (severe collapse/imbalance)
    # We want to see how many steps it takes to reach 0.18 (near target)
    awr = MockAWR(beta=0.1) # Starting at 0.1
    current_ess = 0.01
    
    steps = 0
    max_steps = 100
    while awr.beta.item() < 0.2 and steps < max_steps:
        # In reality, increasing beta increases ESS. 
        # For simplicity, we assume ESS remains low until beta reaches a certain point,
        # or we just track the beta GROWTH rate.
        awr.update_beta(current_ess)
        steps += 1
        # logger.info(f"Step {steps}: Beta {awr.beta.item():.4f}")
        
    logger.info(f"Steps to increase Beta from 0.1 to 0.2: {steps}")
    
    # 0.99 momentum update: beta_next = 0.99*beta + 0.01*(beta*2.0) = beta * 1.01
    # To double beta: 1.01^n = 2 => n = log(2)/log(1.01) approx 69 steps.
    
    if steps > 20:
        logger.error(f"❌ Smoking Gun #116 CONFIRMED! AWR Beta is too sluggish ({steps} steps to react).")
        logger.warning("⚠️ Rationale: If a dataset shuffle creates a hard batch, the model is blinded for 70 steps before the selection pressure relaxes.")

if __name__ == "__main__":
    test_awr_beta_sluggishness()
