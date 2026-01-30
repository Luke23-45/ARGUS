import torch
import logging
import math
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_T")

class MockTrendSentinel:
    @staticmethod
    def update_stats(current_val, ema, std, decay, step_tensor=None):
        with torch.no_grad():
            if step_tensor is not None:
                step_tensor.add_(1)
                t = step_tensor.item()
                bias_correction = 1.0 - (decay ** t) if t > 0 else 1.0
            else:
                bias_correction = 1.0

            delta = current_val - ema.item()
            ema.add_(delta * (1.0 - decay))
            
            # Variance Update
            new_delta = current_val - ema.item()
            sq_diff = delta * new_delta
            var = std.item() ** 2
            new_var = decay * var + (1.0 - decay) * sq_diff
            std.fill_(math.sqrt(max(new_var, 1e-6)))
            
            return ema / bias_correction, std / bias_correction

def simulate_sawtooth_telemetry(accum_steps=16, total_batches=320, decay=0.99):
    logger.info(f"Simulating Telemetry Sawtooth: Accumulation={accum_steps}, Batches={total_batches}, Decay={decay}")
    
    # Buggy: Average every batch
    ema_b = torch.tensor(1.0)
    std_b = torch.tensor(0.5)
    
    # Patched: Average only on should_step
    ema_p = torch.tensor(1.0 * accum_steps) # Initialized to expected accum norm (Sum of 16)
    std_p = torch.tensor(0.5 * accum_steps)
    
    current_accum_grad = torch.zeros(1)
    
    for i in range(total_batches):
        # Stable gradient (Mean=1.0)
        batch_grad = torch.randn(1) * 0.1 + 1.0
        current_accum_grad += batch_grad
        
        # BUGGY: Update every mini-batch
        MockTrendSentinel.update_stats(current_accum_grad.item(), ema_b, std_b, decay)
        
        # PATCHED: Update only at end of cycle (Real Optimizer Step)
        if (i + 1) % accum_steps == 0:
            # We use Power-Law decay to maintain the same half-life as 1-step updates
            decay_p = decay ** accum_steps 
            MockTrendSentinel.update_stats(current_accum_grad.item(), ema_p, std_p, decay_p)
            current_accum_grad.fill_(0.0)

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    logger.info(f"Buggy STD (Mini-batch level): {std_b.item():.4f}")
    logger.info(f"Patched STD (Step level):     {std_p.item():.4f}")
    
    noise_reduction = (std_b.item() - std_p.item()) / (std_b.item() + 1e-8)
    logger.info(f"Noise Reduction: {noise_reduction*100:.1f}%")

    # Threshold: Patched should be much lower than the "Ramp Variance"
    # Variance of 1..16 ramp is ~21, STD ~4.6.
    # Variance of Sum(16) should be 16 * 0.1^2 = 0.16, STD ~0.4.
    if std_p.item() < 0.6: # Close to ideal 0.4
        logger.info("\u2705 TEST T PASSED: Step-level updates restored sensitivity.")
    elif std_p.item() < std_b.item() * 0.5:
        logger.info("\u2705 TEST T PASSED: Significant noise reduction achieved.")
    else:
        logger.warning("\u26a0\ufe0f TEST T: Noise still too high.")
            
    return noise_reduction

if __name__ == "__main__":
    simulate_sawtooth_telemetry()
