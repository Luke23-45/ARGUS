import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BM")

def test_gradnorm_desync():
    logger.info("Simulating GradNorm Desync (#88)...")
    
    # 1. Ranks see different local data -> different local gradients
    # Weight parameter 'w'
    w = torch.tensor([1.0], requires_grad=True)
    
    # Local Losses
    loss_rank0 = 10.0 * w
    loss_rank1 = 1.0 * w
    
    # Backward
    loss_rank0.backward()
    grad0 = w.grad.clone()
    w.grad.zero_()
    
    loss_rank1.backward()
    grad1 = w.grad.clone()
    w.grad.zero_()
    
    # Local Norms
    norm0 = torch.norm(grad0).item()
    norm1 = torch.norm(grad1).item()
    
    logger.info(f"Local Grad Norm Rank 0: {norm0:.4f}")
    logger.info(f"Local Grad Norm Rank 1: {norm1:.4f}")
    
    # 2. Stability Factor Calculation (Local)
    # Rationale: Higher GN -> lower stability_factor -> lower task weight
    sf0 = 1.0 / (1.0 + norm0)
    sf1 = 1.0 / (1.0 + norm1)
    
    logger.info(f"Stability Factor Rank 0: {sf0:.4f}")
    logger.info(f"Stability Factor Rank 1: {sf1:.4f}")
    
    # 3. Apply Local Weights to Local Grads
    weighted_grad0 = grad0 * sf0
    weighted_grad1 = grad1 * sf1
    
    # 4. DDP AllReduce (Sum and Average)
    global_grad = (weighted_grad0 + weighted_grad1) / 2.0
    
    # 5. The "Fighting" Check
    # If they were synced, global_grad would be (grad0 + grad1) / 2.0 * Global_SF
    # Global_SF = 1.0 / (1.0 + (norm0+norm1)/2.0)
    avg_norm = (norm0 + norm1) / 2.0
    global_sf_synced = 1.0 / (1.0 + avg_norm)
    global_grad_synced = ((grad0 + grad1) / 2.0) * global_sf_synced
    
    drift = torch.norm(global_grad - global_grad_synced).item()
    logger.info(f"Global Gradient Drift due to Desync: {drift:.4f}")
    
    if drift > 0.01:
        logger.error(f"❌ Smoking Gun #88 CONFIRMED! GradNorm Desync causes incoherent global updates: {drift:.4f}")
        logger.warning("⚠️ Rationale: Using local gradient norms for global governor parameters causes ranks to disagree on task importance. DDP sync is required for ALL metrics that affect task weighting.")

if __name__ == "__main__":
    test_gradnorm_desync()
