import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_O")

class MockAsymmetricContrastiveLoss(nn.Module):
    """Simplified AsymmetricContrastiveLoss from icu/models/components/contrastive_loss.py"""
    def __init__(self, d_model: int = 128, num_classes: int = 3, momentum: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.momentum = momentum
        self.temperature = 0.25
        
        # Buffer, not Parameter
        self.register_buffer('centroids', F.normalize(torch.randn(num_classes, d_model), dim=1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def update_centroids_local(self, features: torch.Tensor, targets: torch.Tensor):
        """Mock update without DDP sync (reflecting current bug in codebase)"""
        with torch.no_grad():
            unique_classes = torch.unique(targets)
            for c in unique_classes:
                if c < 0 or c >= self.num_classes: continue 
                
                mask_c = (targets == c)
                batch_center = features[mask_c].mean(dim=0)
                batch_center = F.normalize(batch_center, p=2, dim=0)
                
                if self.initialized.item():
                    self.centroids[c].mul_(self.momentum).add_(batch_center, alpha=1 - self.momentum)
                else:
                    self.centroids[c] = batch_center
            
            self.centroids.data = F.normalize(self.centroids, p=2, dim=1)
            self.initialized.fill_(True)

    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, self.centroids.t()) / self.temperature
        return F.cross_entropy(logits, targets.long())

def simulate_ddp_drift(n_steps=100, momentum=0.99):
    logger.info(f"Simulating DDP Centroid Drift over {n_steps} steps (momentum={momentum})...")
    
    # Simulate 2 Ranks
    rank0_acl = MockAsymmetricContrastiveLoss(momentum=momentum)
    rank1_acl = MockAsymmetricContrastiveLoss(momentum=momentum)
    
    # They start with DIFFERENT random centroids (reproducing no-seed-sync bug)
    logger.info(f"Initial Cosine Similarity: {F.cosine_similarity(rank0_acl.centroids, rank1_acl.centroids).mean().item():.4f}")
    
    # Shared Model (simplified as random features that slowly migrate)
    # Goal: Representations converge, but centroids might not.
    target_vectors = F.normalize(torch.randn(3, 128), dim=1) 
    
    similarities = []
    gradient_conflicts = []
    
    for step in range(n_steps):
        # Generate common representations (as if from common model)
        # Add noise to simulate slightly different views/batches
        batch_features = target_vectors[torch.randint(0, 3, (32,))] + torch.randn(32, 128) * 0.1
        batch_targets = torch.randint(0, 3, (32,)) # Simplified: targets don't perfectly match features but correlate
        
        # Rank 0 Update
        rank0_acl.train()
        rank0_acl.update_centroids_local(batch_features, batch_targets)
        
        # Rank 1 Update (Same features, potentially different batch noise in reality)
        rank1_acl.train()
        rank1_acl.update_centroids_local(batch_features, batch_targets)
        
        # Measure Drift between Ranks
        sim = F.cosine_similarity(rank0_acl.centroids, rank1_acl.centroids).mean().item()
        similarities.append(sim)
        
        # Calculate Gradient conflict
        # dL/dz = (p - y) @ centroids
        # If centroids are different, the gradients pushed back to the encoder will differ.
        if step % 10 == 0:
            feat = F.normalize(batch_features, p=2, dim=1)
            feat.requires_grad = True
            
            loss0 = rank0_acl(feat, batch_targets)
            loss0.backward()
            grad0 = feat.grad.clone()
            
            feat.grad = None
            loss1 = rank1_acl(feat, batch_targets)
            loss1.backward()
            grad1 = feat.grad.clone()
            
            # Cosine similarity of gradients between ranks
            # 1.0 = Perfect consensus, -1.0 = Total conflict
            g_sim = F.cosine_similarity(grad0.flatten(), grad1.flatten(), dim=0).item()
            gradient_conflicts.append(g_sim)
            
    return similarities, gradient_conflicts

def run_drift_analysis():
    logger.info("\n" + "="*60)
    logger.info("TEST O: ACL DDP CENTROID DIVERGENCE ANALYSIS")
    logger.info("Checking if missing DDP sync causes gradient conflict")
    logger.info("="*60)
    
    # Case 1: High Momentum (Slow update, slow drift)
    sims_high, grads_high = simulate_ddp_drift(n_steps=200, momentum=0.99)
    logger.info(f"[Momentum 0.99] Final Similarity: {sims_high[-1]:.4f}")
    logger.info(f"[Momentum 0.99] Final Grad Consensus: {grads_high[-1]:.4f}")
    
    # Case 2: Low Momentum (Fast update, fast divergence)
    sims_low, grads_low = simulate_ddp_drift(n_steps=200, momentum=0.8)
    logger.info(f"[Momentum 0.80] Final Similarity: {sims_low[-1]:.4f}")
    logger.info(f"[Momentum 0.80] Final Grad Consensus: {grads_low[-1]:.4f}")

    issues = []
    if sims_high[-1] < 0.9:
        issues.append(f"Significant Centroid Drift: Ranks only {sims_high[-1]*100:.1f}% aligned")
    if grads_high[-1] < 0.8:
        issues.append(f"Consensus Failure: Gradient consensus dropped to {grads_high[-1]:.4f}")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    if not issues:
        logger.info("\u2705 TEST O PASSED: Centroids remained synced (unexpected for current code)")
    else:
        logger.warning("\u26a0\ufe0f TEST O: Observations detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return issues

if __name__ == "__main__":
    run_drift_analysis()
