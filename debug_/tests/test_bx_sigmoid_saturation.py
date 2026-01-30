import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BX")

def test_sigmoid_saturation():
    logger.info("Verifying Smoking Gun #132: Sigmoid Saturation...")
    
    # Range of MAP values (mmHg)
    # 60 is the boundary, 40 is severe shock, 20 is near-arrest.
    map_vals = torch.tensor([80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0], requires_grad=True)
    
    # Clinical Reward Penalty Logic (from advantage_calculator.py)
    # map_penalty = torch.sigmoid((60.0 - map_val) * 0.5)
    penalty = torch.sigmoid((60.0 - map_vals) * 0.5)
    
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
        
    # THE SMOKING GUN:
    # At MAP=60 (border), gradient is ~0.125
    # At MAP=40 (danger), gradient is ~0.005 (25x reduction)
    # At MAP=20 (arrest), gradient is ~0.0000002 (600,000x reduction)
    
    gradient_at_danger = grads[4].item() # MAP=40
    gradient_at_border = grads[2].item() # MAP=60
    
    reduction_ratio = gradient_at_border / (gradient_at_danger + 1e-12)
    
    if reduction_ratio > 20.0:
        logger.error(f"❌ Smoking Gun #132 CONFIRMED! Gradient signal vanishes by {reduction_ratio:.1f}x at MAP=40.")
        logger.warning("⚠️ Rationale: The sigmoid penalty 'gives up' on the most critical patients. The model receives no signal on how to improve for patients with MAP < 40, leading to the 'fair-weather' model problem.")

if __name__ == "__main__":
    test_sigmoid_saturation()
