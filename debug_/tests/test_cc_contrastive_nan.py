import torch
import logging
from icu.utils.stabilization import StableContrastiveLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CC")

def test_contrastive_nan_poisoning():
    logger.info("Verifying Smoking Gun #145: Contrastive Loss NaN Poisoning...")
    
    criterion = StableContrastiveLoss(d_model=16, num_classes=3)
    
    # 1. Normal Update
    features = torch.randn(5, 16)
    targets = torch.tensor([0, 1, 0, 1, 2])
    criterion.forward(features, targets)
    
    logger.info(f"Centroids after normal update: Finite={torch.isfinite(criterion.centroids).all().item()}")
    
    # 2. Poisoned Update (NaN features)
    poison_features = torch.full((5, 16), float('nan'))
    criterion.forward(poison_features, targets)
    
    is_finite = torch.isfinite(criterion.centroids).all().item()
    logger.info(f"Centroids after poisoned update: Finite={is_finite}")
    
    if not is_finite:
        logger.error("❌ Smoking Gun #145 CONFIRMED! Centroids are NaNs. The global sepsis manifold is permanently corrupted.")
    else:
        logger.info("✅ Centroids remained finite (unexpected).")

if __name__ == "__main__":
    test_contrastive_nan_poisoning()
