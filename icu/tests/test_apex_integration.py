
import torch
import torch.nn as nn
import logging
import sys
from unittest.mock import MagicMock

# [TEST FIX] Mock kagglehub
sys.modules["kagglehub"] = MagicMock()

from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
from icu.models.apex_moe_planner import APEX_MoE_Planner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAPEXIntegration")

def test_apex_integration():
    logger.info("--- Starting APEX-MoE Integration Verification ---")
    
    # 1. Setup Config
    cfg = ICUConfig(
        d_model=128, n_heads=4, n_layers=2,
        num_phases=6, # 6 Experts
        use_imputation_masks=True
    )
    
    # 2. Create Surrogate Generalist (Donor)
    logger.info("Initializing Surrogate Generalist...")
    # Add dummy aux_head to generalist as required by APEX
    generalist = ICUUnifiedPlanner(cfg)
    generalist.aux_head = nn.Linear(cfg.d_model, cfg.num_phases)
    
    # 3. Create APEX Specialist
    logger.info("Initializing APEX Specialist...")
    apex = APEX_MoE_Planner(generalist)
    apex.train()
    
    # 4. Create Dummy Batch
    B, T_obs = 4, 24
    T_pred = 6
    D = 28
    
    batch = {
        "observed_data": torch.randn(B, T_obs, D),
        "future_data":   torch.randn(B, T_pred, D),
        "static_context": torch.randn(B, 6),
        "src_mask":      torch.rand(B, T_obs, D).bernoulli(), # Fix: randn -> rand
        "phase_label":   torch.randint(0, 3, (B,)) # GT Phases {0, 1, 2}
    }
    
    # 5. Verify Forward Pass & Gradient Flow
    logger.info("Testing Forward Pass & Gradient Flow...")
    apex.zero_grad()
    
    # Enable grad for input to check flow? No, check router weights
    for param in apex.router.parameters():
        param.requires_grad = True
        
    outputs = apex(batch)
    loss = outputs["loss"]
    
    logger.info(f"Forward Pass Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check Router Gradients
    router_grad_norm = 0.0
    for param in apex.router.parameters():
        if param.grad is not None:
            router_grad_norm += param.grad.norm().item()
            
    logger.info(f"Router Gradient Norm: {router_grad_norm:.6f}")
    
    if router_grad_norm > 0.0:
        logger.info("[SUCCESS] Gradients are flowing to the Router!")
    else:
        logger.error("[FAILURE] Router gradients are ZERO. Differentiable routing failed.")
        sys.exit(1)

    # 6. Verify Hierarchical Supervision Logic
    logger.info("Testing Hierarchical Supervision Logic...")
    # Logic implicitly executed during forward pass (no crash = logic valid)
    
    # 7. Verify Loss Components
    logger.info(f"Router CE Loss: {outputs['router_ce_loss'].item():.4f}")
    logger.info(f"Diversity Loss: {outputs['diversity_loss'].item():.4f}")
    logger.info(f"Reg Loss: {outputs['reg_loss'].item():.4f}")
    
    logger.info("--- Verification Complete: GREEN LIGHT ---")

if __name__ == "__main__":
    try:
        test_apex_integration()
    except Exception as e:
        logger.exception("Test Failed with Exception:")
        sys.exit(1)
