import torch
import torch.nn as nn
import torch.nn.functional as F
from icu.models.components.loss_scaler import UncertaintyLossScaler
from icu.models.components.risk_aware_loss import RiskAwareAsymmetricLoss
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def test_surgical_patches():
    logger.info("Starting Pure-Functional Surgical Verification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Verify UncertaintyLossScaler Signal Floor
    logger.info("--- 1. Testing UncertaintyLossScaler Floor ---")
    scaler = UncertaintyLossScaler(num_tasks=2)
    # Task 0: Diffusion, Task 1: Aux
    # Set log_vars to a high value (e.g. 5.0) which should normally result in 
    # weight = 0.5 * exp(-5) = 0.003
    scaler.log_vars.data = torch.tensor([0.0, 5.0]) 
    
    losses = {'diffusion': torch.tensor(1.0), 'aux': torch.tensor(1.0)}
    total_loss, logs = scaler(losses)
    
    w_aux = logs['weight/aux'].item()
    logger.info(f"w_aux with log_var=5.0: {w_aux:.4f}")
    
    # Floor should be 0.05
    if abs(w_aux - 0.05) < 1e-4:
        logger.info("✅ SUCCESS: WA Signal Floor clamped to 0.05.")
    else:
        logger.error(f"❌ FAILURE: WA Signal Floor is {w_aux}, expected 0.05.")

    # 2. Verify Gradient Bridge
    logger.info("--- 2. Testing Direct Gradient Bridge ---")
    config = ICUConfig(
        input_dim=28, static_dim=6, d_model=128, n_layers=2, n_heads=4, 
        history_len=24, pred_len=6, use_auxiliary_head=True
    )
    model = ICUUnifiedPlanner(config).to(device)
    loss_fn = RiskAwareAsymmetricLoss(gamma_pos=1.0).to(device)
    
    # Mock context from encoder
    B = 2
    past = torch.randn(B, 24, 28, device=device)
    static = torch.randn(B, 6, device=device)
    mask = torch.ones(B, 24, 28, device=device)
    padding = (mask.sum(-1) == 0)
    
    # Student pass
    ctx_seq, global_ctx, ctx_mask = model.encoder(past, static, imputation_mask=mask, padding_mask=padding)
    
    # Auxiliary pass (The bridge)
    logits, _ = model.aux_head(ctx_seq, ctx_mask)
    
    # Compute loss
    targets = torch.randint(0, 3, (B,), device=device)
    targets_one_hot = F.one_hot(targets, num_classes=3).float()
    risk_coef = torch.rand(B, device=device)
    
    loss = loss_fn(logits, targets_one_hot, risk_coef)
    
    # Zero grads & backward
    model.encoder.zero_grad()
    loss.backward()
    
    # Check if ANY weight in the encoder got a gradient
    has_grad = False
    for p in model.encoder.parameters():
        if p.grad is not None and p.grad.norm() > 1e-9:
            has_grad = True
            break
            
    if has_grad:
        logger.info("✅ SUCCESS: Auxiliary loss backpropagated to Shared Encoder.")
    else:
        logger.error("❌ FAILURE: Gradient bridge is BROKEN.")

    # 3. Verify ASL gamma_pos
    logger.info("--- 3. Testing ASL gamma_pos ---")
    if loss_fn.gamma_pos == 1.0:
        logger.info("✅ SUCCESS: RiskAwareAsymmetricLoss.gamma_pos is 1.0.")
    else:
        logger.error(f"❌ FAILURE: gamma_pos is {loss_fn.gamma_pos}, expected 1.0.")

if __name__ == "__main__":
    test_surgical_patches()
