import torch
import torch.nn as nn
import logging
from unittest.mock import patch
from icu.models.components.contrastive_loss import AsymmetricContrastiveLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_CI")

def test_contrastive_mass_parity_fix():
    logger.info("Verifying Fix for Smoking Gun #166: Contrastive Class-Mass Parity...")
    
    latent_dim = 128
    num_classes = 3
    acl_r0 = AsymmetricContrastiveLoss(d_model=latent_dim, num_classes=num_classes)
    acl_r1 = AsymmetricContrastiveLoss(d_model=latent_dim, num_classes=num_classes)
    
    # Force identical initial centroids and initialize them
    initial_centroids = torch.randn(num_classes, latent_dim)
    acl_r0.centroids.copy_(initial_centroids)
    acl_r1.centroids.copy_(initial_centroids)
    acl_r0.initialized.fill_(True)
    acl_r1.initialized.fill_(True)
    
    # Mock distributed
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.all_reduce") as mock_all_reduce:
        
        # Simulated Class 1 data: Rank 0 (10 samples), Rank 1 (1 sample)
        features_r0 = torch.randn(10, latent_dim)
        y_r0 = torch.ones(10, dtype=torch.long)
        
        features_r1 = torch.randn(1, latent_dim)
        y_r1 = torch.ones(1, dtype=torch.long)
        
        r0_sum = features_r0.sum(dim=0)
        r1_sum = features_r1.sum(dim=0)
        
        def mock_all_reduce_side_effect(buffer, op=None):
            # Simulation: buffer is [num_classes, d_model + 1]
            # Accumulate global stats for Class 1
            buffer[1, :latent_dim] = r0_sum + r1_sum
            buffer[1, latent_dim] = 11.0
            return None
        
        mock_all_reduce.side_effect = mock_all_reduce_side_effect
        
        acl_r0.train()
        acl_r0(features_r0, y_r0)
        
        acl_r1.train()
        acl_r1(features_r1, y_r1)
        
        # Check for Parity
        diff_centroids = torch.abs(acl_r0.centroids - acl_r1.centroids).sum().item()
        logger.info(f"Centroid Difference: {diff_centroids:.6e}")
        
        if diff_centroids < 1e-6:
            logger.info("✅ Fix for Smoking Gun #166 VERIFIED! Contrastive Centroids matched perfectly across ranks.")
        else:
            logger.error(f"❌ Fix FAILED! Centroid divergence: {diff_centroids:.6e}")

if __name__ == "__main__":
    test_contrastive_mass_parity_fix()
