import torch
import torch.nn as nn
import torch.nn.functional as F
from icu.models.wrapper_generalist import ICUGeneralistWrapper
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def test_gradient_bridge():
    logger.info("Starting Gradient Bridge Verification...")
    
    # 1. Setup Mock Config
    cfg = OmegaConf.create({
        "model": {
            "num_phases": 3,
            "history_len": 24,
            "pred_len": 6,
            "input_dim": 28,
            "static_dim": 6,
            "d_model": 128,
            "encoder_layers": 2,
            "n_heads": 4,
            "use_auxiliary_head": True,
            "timesteps": 100,
            "use_imputation_masks": True,
            "use_attention_pooling": True,
            "dropout": 0.1,
            "stochastic_depth_prob": 0.0,
            "use_swiglu": True,
            "ffn_dim_ratio": 4,
            "n_layers": 2,
            "num_phases": 3
        },
        "train": {
            "balancing_mode": "sota_2025",
            "asl_gamma_neg": 4.0,
            "asl_gamma_pos": 1.0,
            "risk_multiplier": 2.0,
            "throttle_scale": 1.0,
            "grad_clip": 1.0,
            "pos_weight": 5.0,
            "horizon_warmup": 10,
            "horizon_ramp": 40,
            "phys_loss_weight": 0.2,
            "start_gamma": 0.8,
            "end_gamma": 0.99
        },
        "seed": 42
    })

    # 2. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ICUGeneralistWrapper(cfg).to(device)
    model.train()
    
    # [FIX] Mock PL context for standalone testing
    from unittest.mock import MagicMock
    model.trainer = MagicMock()
    model.trainer.current_epoch = 0
    model.trainer.accumulate_grad_batches = 1
    model.manual_backward = lambda loss: loss.backward()
    
    class MockOpt:
        def __init__(self, params):
            self.param_groups = [{"lr": 0.001, "params": list(params)}]
        def step(self): pass
        def zero_grad(self): pass
    
    model.optimizers = lambda: MockOpt(model.parameters())
    model.lr_schedulers = lambda: None
    
    # 3. Create Mock Batch
    B = 4
    batch = {
        "observed_data": torch.randn(B, 24, 28, device=device),
        "future_data": torch.randn(B, 6, 28, device=device),
        "static_context": torch.randn(B, 6, device=device),
        "phase_label": torch.randint(0, 3, (B,), device=device),
        "outcome_label": torch.rand(B, device=device),
        "src_mask": torch.ones(B, 24, 28, device=device),
        "future_mask": torch.ones(B, 6, 28, device=device)
    }

    # 4. Zero Grads
    model.zero_grad()
    
    # Audit Encoder Gradients before step
    for name, param in model.model.encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is None or (param.grad == 0).all()

    # 5. Execute Training Step (Manual Opt)
    # We call training_step manually to observe internal behavior
    model.training_step(batch, 0)
    
    # 6. Verify Encoder Gradients
    # Any gradient > 0 in the encoder proves the bridge is open.
    has_grad = False
    for name, param in model.model.encoder.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            has_grad = True
            break
            
    if has_grad:
        logger.info("✅ SUCCESS: Encoder received gradients from training_step.")
    else:
        logger.error("❌ FAILURE: Encoder has ZERO gradients after training_step.")
        
    # 7. Check loss weights (WA)
    w_aux = model.model.loss_scaler.log_vars[1]
    # log_var <= 2.3 for weight floor
    logger.info(f"Loss Scaler log_vars: {model.model.loss_scaler.log_vars.data}")
    
    if model.model.loss_scaler.log_vars[1] <= 2.31: # Allow eps
         logger.info("✅ SUCCESS: WA Signal Floor is active (log_var <= 2.3).")
    else:
         logger.error("❌ FAILURE: WA Signal Floor is NOT active.")

if __name__ == "__main__":
    test_gradient_bridge()
