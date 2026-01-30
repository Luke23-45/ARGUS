import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BS")

class MockBayesianScaler(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # Initializing at 0.0 means precision = exp(0) = 1.0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        # precision = exp(-log_var)
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars
        return weighted_losses.sum(), precision

def test_bayesian_cold_start_shock():
    logger.info("Simulating Bayesian Scale Cold Start (#110)...")
    
    # Task 0: Main (Loss=1), Task 1: Physics (Loss=100 initially - Outlier)
    losses = torch.tensor([1.0, 100.0])
    
    scaler = MockBayesianScaler(num_tasks=2)
    optimizer = torch.optim.Adam(scaler.parameters(), lr=0.1)
    
    # 1. First Step: Cold Start
    total_loss, precision = scaler(losses)
    logger.info(f"Initial Precision: {precision.detach()}")
    
    # Gradient on Task 1 is large
    total_loss.backward()
    
    # 2. Check if one step is enough to suppress the 100.0 loss
    optimizer.step()
    
    with torch.no_grad():
        _, precision_new = scaler(losses)
        logger.info(f"Precision after 1 step: {precision_new}")
        
    # ANALYSIS:
    # If the 100.0 loss persists, even after one step of 0.1 LR, 
    # precision might only drop slightly (e.g. from 1.0 to 0.9).
    # This means the 100x gradient persists for many steps, causing a GN spike.
    
    if precision_new[1] > 0.5:
        logger.error("❌ Smoking Gun #110 CONFIRMED! Bayesian Scaler reacts too slowly to initial shocks.")
        logger.warning("⚠️ Rationale: Critical tasks like 'phys' must have a lower initial priority or the scaler must have higher 'learning inertia' (momentum).")

if __name__ == "__main__":
    test_bayesian_cold_start_shock()
