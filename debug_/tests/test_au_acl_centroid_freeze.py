import torch
import torch.nn.functional as F
from icu.models.components.contrastive_loss import AsymmetricContrastiveLoss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AU")

def test_acl_freeze():
    logger.info("Simulating ACL Centroid Stagnation (Smoking Gun #26)...")
    
    d_model = 128
    num_classes = 3
    criterion = AsymmetricContrastiveLoss(d_model=d_model, num_classes=num_classes)
    
    # 1. Warmup: Initialize centroids
    criterion.initialized.fill_(False)
    criterion(torch.randn(16, d_model), torch.zeros(16, dtype=torch.long))
    
    # 2. Simulate High Step Density (e.g., 20k steps per epoch)
    # n_curr = 20000 -> 0.99^(200/20000) = 0.99^0.01 = 0.9999
    criterion.momentum = 0.99999
    
    # 3. Record initial state AFTER initialization
    initial_centroids = criterion.centroids.clone()
    
    # 4. Present a completely new regime (New batch centers)
    # The batch centers are far from the initial centroids
    # Shifted to [10.0, 10.0, ...]
    batch_z = torch.randn(16, d_model) + 10.0 
    batch_y = torch.zeros(16, dtype=torch.long) # All class 0
    
    # Run multiple updates to see if centroids move
    for _ in range(100):
        criterion(batch_z, batch_y)
        
    final_centroids = criterion.centroids.clone()
    
    # Calculate total movement
    movement = (final_centroids - initial_centroids).norm().item()
    logger.info(f"Centroid Movement after 100 updates with high momentum: {movement:.8f}")
    
    if movement < 1e-4:
        logger.error("❌ ACL Centroids are effectively frozen! They failed to adapt to the new latent regime.")
    else:
        logger.info("✅ ACL Centroids showing signs of adaptivity.")

if __name__ == "__main__":
    test_acl_freeze()
