import torch
import torch.nn as nn
import logging
from icu.models.components.loss_scaler import BayesianProjectedScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CA")

def test_ddp_scaler_hang_simulation():
    logger.info("Verifying Smoking Gun #137: DDP Scaler Hang (Task Mismatch)...")
    
    scaler = BayesianProjectedScaler(num_tasks=7)
    
    # Simulate Rank 0: Full Task Set
    loss_dict_r0 = {
        'diffusion': torch.tensor(1.0, requires_grad=True),
        'critic': torch.tensor(0.5, requires_grad=True),
        'aux': torch.tensor(0.1, requires_grad=True),
        'acl': torch.tensor(0.2, requires_grad=True)
    }
    
    # Simulate Rank 1: Sepsis-Free Batch (No aux/acl)
    loss_dict_r1 = {
        'diffusion': torch.tensor(1.1, requires_grad=True),
        'critic': torch.tensor(0.6, requires_grad=True)
    }
    
    # Mock DDP state
    # We can't easily launch real DDP in a test snippet, but we can inspect the logic.
    
    def simulate_forward(rank, l_dict):
        losses = []
        active_keys = []
        for i, key in enumerate(scaler.keys):
            if key in l_dict:
                losses.append(l_dict[key])
                active_keys.append((i, key))
        
        losses_tensor = torch.stack(losses)
        logger.info(f"Rank {rank} Task Count: {len(losses_tensor)}")
        return losses_tensor

    t_r0 = simulate_forward(0, loss_dict_r0)
    t_r1 = simulate_forward(1, loss_dict_r1)
    
    if t_r0.shape != t_r1.shape:
        logger.error(f"❌ Smoking Gun #137 CONFIRMED! Rank 0 has {t_r0.shape[0]} tasks, Rank 1 has {t_r1.shape[0]} tasks.")
        logger.warning("⚠️ DDP all_reduce will hang/deadlock because tensor shapes do not match across ranks.")
    else:
        logger.info("✅ No mismatch detected in this simulation (highly unlikely).")

if __name__ == "__main__":
    test_ddp_scaler_hang_simulation()
