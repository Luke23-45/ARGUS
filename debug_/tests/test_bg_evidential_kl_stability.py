import torch
from icu.models.components.sequence_aux_head import EvidentialLoss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BG")

def test_evidential_kl_stability():
    logger.info("Simulating Evidential KL Stability (Smoking Gun #50)...")
    
    criterion = EvidentialLoss(num_classes=2)
    
    # 1. Extreme confidence (Alpha >> 1)
    alpha_huge = torch.tensor([[1e6, 1.0]]) # Very confident in class 0
    y = torch.tensor([[1.0, 0.0]]) # Actually class 0
    
    loss_huge = criterion(alpha_huge, y, epoch_num=10)
    logger.info(f"Loss with alpha=1e6 (y correct): {loss_huge.item():.4f}")
    
    # 2. Extreme Error (Alpha >> 1, y wrong)
    alpha_wrong = torch.tensor([[1e6, 1.0]])
    y_wrong = torch.tensor([[0.0, 1.0]]) # Actually class 1
    
    loss_wrong = criterion(alpha_wrong, y_wrong, epoch_num=10)
    logger.info(f"Loss with alpha=1e6 (y wrong): {loss_wrong.item():.4f}")
    
    # 3. Floating-point limits
    alpha_max = torch.tensor([[1e30, 1.0]]) # Nearing fp32 limit for lgamma
    try:
        loss_max = criterion(alpha_max, y, epoch_num=10)
        logger.info(f"Loss with alpha=1e30: {loss_max.item():.4f}")
        if torch.isnan(loss_max) or torch.isinf(loss_max):
            logger.error("❌ Evidential KL Overflow detected at alpha=1e30.")
    except Exception as e:
        logger.error(f"❌ Criterion crashed at alpha=1e30: {e}")

if __name__ == "__main__":
    test_evidential_kl_stability()
