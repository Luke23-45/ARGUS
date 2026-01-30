import torch
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AI")

def simulate_telemetry_jitter():
    logger.info("Simulating Telemetry Jitter across Accumulation boundaries...")
    
    ema = torch.tensor(1.0)
    decay = 0.9
    
    # 1. Steady State with Accum=1
    # Grad norm average is ~1.0
    for _ in range(20):
        grad_norm = 1.0 + torch.randn(1).item() * 0.1
        ema.mul_(decay).add_(grad_norm, alpha=1.0 - decay)
    
    logger.info(f"EMA with Accum=1: {ema.item():.4f}")
    
    # 2. Switch to Accum=16
    # Accumulated gradient norm should be sqrt(16) * batch_norm if i.i.d
    # Original Grad = Sum(g_i). Norm(Sum) = sqrt(Sum Norm^2) = sqrt(16) * Norm
    accum = 16
    grad_norm_accum = 1.0 * math.sqrt(accum) 
    
    # [BUGGY] Feed absolute norm to EMA
    ema_buggy = ema.clone()
    ema_buggy.mul_(decay).add_(grad_norm_accum, alpha=1.0 - decay)
    
    # [PATCHED] Normalize by sqrt(accum)
    ema_patched = ema.clone()
    grad_norm_normalized = grad_norm_accum / math.sqrt(accum)
    ema_patched.mul_(decay).add_(grad_norm_normalized, alpha=1.0 - decay)
    
    logger.info(f"EMA Buggy (After switch to N=16): {ema_buggy.item():.4f} (Jump: {ema_buggy.item() - ema.item():.4f})")
    logger.info(f"EMA Patched (After switch to N=16): {ema_patched.item():.4f} (Jump: {ema_patched.item() - ema.item():.4f})")
    
    if abs(ema_buggy.item() - ema.item()) > 0.5:
        logger.error("\u274c SMOKING GUN #7: Telemetry Jitter detected! Sentinel will trigger false shock.")
    else:
        logger.info("\u2705 Telemetry stable.")

if __name__ == "__main__":
    simulate_telemetry_jitter()
