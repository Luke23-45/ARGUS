import torch
from icu.models.components.loss_scaler import BayesianProjectedScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AZ")

def test_bayesian_limit_cycle():
    logger.info("Simulating Bayesian Limit Cycle (Smoking Gun #35)...")
    
    scaler = BayesianProjectedScaler(num_tasks=7)
    optimizer = torch.optim.SGD(scaler.parameters(), lr=0.1)
    
    # Simulate a task with very small, stable loss
    # It wants to decrease log_var to increase precision
    # Diffusion is task 0
    loss_val = torch.tensor(1e-4) # Tiny loss
    
    log_vars_history = []
    weights_history = []
    
    for i in range(100):
        optimizer.zero_grad()
        # Mocking the forward part relating to one task
        loss_dict = {'diffusion': loss_val}
        total_loss, logs = scaler(loss_dict)
        
        total_loss.backward()
        optimizer.step()
        scaler.project_parameters()
        
        log_vars_history.append(scaler.log_vars[0].item())
        weights_history.append(logs['weight/diffusion'])
        
    logger.info(f"Final Log-Var: {scaler.log_vars[0].item():.4f}")
    logger.info(f"Final Weight: {logs['weight/diffusion']:.4f}")
    
    # Check for Limit Cycle: Does it oscillate near the floor?
    last_10 = log_vars_history[-10:]
    is_oscillating = any(last_10[i] != last_10[i+1] for i in range(len(last_10)-1))
    
    if is_oscillating:
        logger.error("❌ Bayesian Limit Cycle detected! Parameters are oscillating at the floor.")
    elif scaler.log_vars[0].item() == -1.5:
        # Check if it's "stuck" but could be better
        # If loss is 1e-4, precision should be huge.
        # But it's restricted by the floor.
        logger.info("✅ Bayesian Scaler is stable at the floor (Saturation).")
    else:
        logger.info("✅ Bayesian Scaler converged smoothly.")

if __name__ == "__main__":
    test_bayesian_limit_cycle()
