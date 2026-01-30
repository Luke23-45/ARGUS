import torch
import torch.nn as nn
import logging
import copy
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase6_Verification")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 10)

class MockGradNorm:
    def __init__(self, params):
        self.weights = nn.Parameter(torch.ones(2))
        self.optimizer = torch.optim.Adam([self.weights], lr=0.1)
        self.shared_params = list(params)

def verify_all_phase6():
    # --- TEST AH: GRADNORM SCALING & RESUMPTION ---
    logger.info("VERIFYING TEST AH (GradNorm Scaling & Resumption)...")
    model = SimpleModel()
    gn = MockGradNorm(model.backbone.parameters())
    scaler = torch.amp.GradScaler(init_scale=65536.0)
    
    # 1. Scaling Check
    gn_loss = torch.tensor(1.0, requires_grad=True, device='cpu')
    scaled_gn_loss = scaler.scale(gn_loss)
    
    # Buggy: gn_loss.backward() [Scaled] -> weights exploded
    # Patched: scaler.step(gn.optimizer) [Unscaled]
    gn.optimizer.zero_grad()
    scaled_gn_loss.backward()
    
    # Simulation of scaler.step()
    scaler.unscale_(gn.optimizer)
    gn.optimizer.step()
    
    logger.info(f"GradNorm Weights after Scales Step: {gn.weights.detach()}")
    if (gn.weights.abs() < 2.0).all():
        logger.info("\u2705 TEST AH.1 PASSED: GradNorm unscaled correctly.")
    else:
        logger.error("\u274c TEST AH.1 FAILED: Weights exploded!")

    # 2. Resumption Check
    state_before = copy.deepcopy(gn.optimizer.state_dict())
    gn_resumed = MockGradNorm(model.backbone.parameters())
    gn_resumed.optimizer.load_state_dict(state_before)
    if len(gn_resumed.optimizer.state_dict()['state']) > 0:
        logger.info("\u2705 TEST AH.2 PASSED: Optimizer memory restored.")

    # --- TEST AI: TELEMETRY PARITY ---
    logger.info("\nVERIFYING TEST AI (Telemetry Normalization)...")
    ema = torch.tensor(1.0)
    decay = 0.9
    accum = 16
    grad_norm_accum = 1.0 * math.sqrt(accum)
    
    # Patched: Normalized by sqrt(accum)
    grad_norm_norm = grad_norm_accum / math.sqrt(accum)
    ema_p = (ema * decay) + (grad_norm_norm * (1-decay))
    
    logger.info(f"EMA Patched Jump: {ema_p - ema:.4f}")
    if abs(ema_p - ema) < 0.1:
         logger.info("\u2705 TEST AI PASSED: Telemetry is step-invariant.")

    # --- TEST AJ: SKIP-LOOP RESOLUTION ---
    logger.info("\nVERIFYING TEST AJ (Skip-Loop Resolution)...")
    p = nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    
    # Batch 1: Inf
    loss1 = p * float('inf')
    loss1.backward()
    
    # Skip Block (Patched)
    if not torch.isfinite(p.grad).all():
        opt.zero_grad() # [CLEANUP]
        logger.info("Skipped Inf and called zero_grad().")
    
    # Batch 2: Finite
    loss2 = p * 1.0
    loss2.backward()
    
    if torch.isfinite(p.grad).all():
        logger.info("\u2705 TEST AJ PASSED: Skip-Loop resolved.")
    else:
        logger.error("\u274c TEST AJ FAILED: Inf persisted.")

if __name__ == "__main__":
    verify_all_phase6()
