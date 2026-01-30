import torch
import torch.nn as nn
import logging
from icu.models.components.loss_scaler import BayesianProjectedScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AR")

def test_bayesian_scaler_crash():
    logger.info("Simulating Bayesian Scaler Singularity (Smoking Gun #22)...")
    
    scaler = BayesianProjectedScaler(num_tasks=2)
    
    # Mock losses
    loss_dict = {
        'diffusion': torch.tensor(1.0, requires_grad=True),
        'critic': torch.tensor(0.5, requires_grad=True)
    }
    
    try:
        # This should trigger NameError: regularization is not defined
        total_loss, metrics = scaler(loss_dict)
        logger.info(f"Total Loss: {total_loss}")
        
    except NameError as e:
        logger.error(f"❌ SMOKING GUN #22: Bayesian Scaler crashed due to undefined variable: {e}")
    except Exception as e:
        logger.error(f"❌ Bayesian Scaler failed with error: {e}")

def test_uncertainty_floor():
    logger.info("Checking for uncertainty floors...")
    scaler = BayesianProjectedScaler(num_tasks=2)
    
    # Force very small log_vars (high precision)
    with torch.no_grad():
        scaler.log_vars.fill_(-10.0) 
    
    loss_dict = {
        'diffusion': torch.tensor(1.0, requires_grad=True),
        'critic': torch.tensor(0.5, requires_grad=True)
    }
    
    # Note: L124 in loss_scaler.py has a project_parameters hook, 
    # but the forward pass L88 applies exp(-log_var) before projection.
    # If the optimizer pushes log_var to -10, precision becomes exp(10) = 22026.
    # This causes immediate GN explosion if not clamped in forward.
    
    logger.warning("Uncertainty Floor is only enforced in project_parameters (post-step).")
    logger.warning("If the optimizer takes a large step into high-precision, the NEXT forward pass will explode before projection can save it.")

if __name__ == "__main__":
    test_bayesian_scaler_crash()
    test_uncertainty_floor()
