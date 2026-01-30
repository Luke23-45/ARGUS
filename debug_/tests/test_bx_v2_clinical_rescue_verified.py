import torch
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_BX_v2")

def test_clinical_rescue_verification():
    logger.info("Verifying Clinical Gradient Rescue (#132)...")
    
    calc = ICUAdvantageCalculator()
    
    # Range of MAP values (mmHg)
    map_vals = torch.tensor([65.0, 60.0, 50.0, 40.0, 20.0, 0.0], requires_grad=True)
    
    # New Implementation Logic (using the internal helper)
    penalty = calc._clinical_sigmoid(map_vals, 60.0, 0.5)
    
    # Calculate gradients
    penalty.sum().backward()
    grads = map_vals.grad.abs()
    
    logger.info("MAP (mmHg) | Penalty | Gradient (Signal)")
    logger.info("-" * 40)
    for i in range(len(map_vals)):
        m = map_vals[i].item()
        p = penalty[i].item()
        g = grads[i].item()
        logger.info(f"{m:10.1f} | {p:7.4f} | {g:14.8f}")
        
    # SUCCESS CRITERIA:
    # At MAP=40 (danger), gradient should be > 0.005
    gradient_at_danger = grads[3].item() # MAP=40
    
    # Old gradient at MAP=40 was ~0.000022
    improvement_ratio = gradient_at_danger / 0.00002271
    
    logger.info(f"Gradient at MAP=40: {gradient_at_danger:.8f}")
    logger.info(f"Signal Improvement Ratio: {improvement_ratio:.1f}x")
    
    if gradient_at_danger > 0.005:
        logger.info("✅ Verification SUCCESS! Clinical gradient is now robust in extreme shock.")
    else:
        logger.error(f"❌ Verification FAILED! Gradient signal too weak ({gradient_at_danger:.8f}).")

if __name__ == "__main__":
    test_clinical_rescue_verification()
