import torch
import torch.nn.functional as F
from icu.models.components.bgsl_loss import BGSLLoss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AS")

def test_bgsl_explosion():
    logger.info("Simulating BGSL Trend Explosion (Smoking Gun #25)...")
    
    criterion = BGSLLoss()
    
    # 1. Baseline: Normal patient dynamics
    p_base = torch.randn(4, 24, 1, requires_grad=True)
    t_base = torch.randint(0, 2, (4, 24, 1)).float()
    v_base = torch.randn(4, 24, 28) # Baseline vitals
    
    out_base = criterion(p_base, t_base, v_base)
    l_base = out_base['l_shock']
    l_base.backward()
    grad_base = p_base.grad.norm().item()
    logger.info(f"Baseline Shock Loss: {l_base.item():.4f} | Grad Norm: {grad_base:.4f}")
    
    # 2. Adversarial Case: Perfectly Static patient (Clinical Dead Zone)
    # v_static is exactly 0.0 everywhere
    p_static = torch.randn(4, 24, 1, requires_grad=True)
    t_static = torch.randint(0, 2, (4, 24, 1)).float()
    v_static = torch.zeros(4, 24, 28) 
    
    # Inject a CLEAR clinical jitter (e.g. MAP drop 80 -> 70)
    v_static[0, -1, 0] = 10.0 
    
    out_adv = criterion(p_static, t_static, v_static)
    l_adv = out_adv['l_shock']
    l_adv.backward()
    grad_adv = p_static.grad.norm().item()
    
    logger.info(f"Static Shock Loss: {l_adv.item():.4f} | Grad Norm: {grad_adv:.4f}")
    
    ratio = grad_adv / (grad_base + 1e-8)
    logger.info(f"Explosion Ratio: {ratio:.2f}x")
    
    if ratio > 10.0: # Tightened threshold
        logger.error("❌ BGSL Shock Gradient exploded over 10x due to static vitals!")
    else:
        logger.info("✅ BGSL handled static vitals within reasonable bounds.")

if __name__ == "__main__":
    test_bgsl_explosion()
