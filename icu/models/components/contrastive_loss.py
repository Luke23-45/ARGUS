import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricContrastiveLoss(nn.Module):
    """
    [SOTA 2025] Asymmetric Contrastive Loss (ACL) for Sparse Clinical Events.
    
    Logic:
    - Instead of simple classification, we cluster sepsis trajectories in latent space.
    - Sepsis patients are pulled toward a "Shock" centroid.
    - Stable patients are pulled toward a "Homeostasis" centroid.
    - Centroids are pushed apart.
    - L2 Guard: Penalizes centroid magnitude to prevent representation collapse.
    """
    def __init__(self, d_model: int, num_classes: int = 3, temperature: float = 0.1, centroid_reg: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.centroid_reg = centroid_reg
        
        # Learnable centroids for each class (Shock, Recovery, Stable)
        self.centroids = nn.Parameter(torch.randn(num_classes, d_model))
        nn.init.xavier_uniform_(self.centroids)

    def forward(self, z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (B, T, D) or (B, D) Latent representations
            y: (B, T) or (B,) Labels (Long)
            mask: (B, T) Optional boolean padding mask
        """
        if z.dim() == 3:
            # Flatten to (B*T, D) and (B*T,)
            z = z.reshape(-1, self.d_model)
            y = y.reshape(-1)
            if mask is not None:
                m = mask.reshape(-1)
                z = z[m]
                y = y[m]
        
        if z.shape[0] == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Normalize features and centroids for cosine similarity
        z_norm = F.normalize(z, p=2, dim=1)
        c_norm = F.normalize(self.centroids, p=2, dim=1)
        
        # Compute similarities (B', num_classes)
        logits = torch.matmul(z_norm, c_norm.T) / self.temperature
        
        # Contrastive Loss (CrossEntropy toward matching centroid)
        contrastive_loss = F.cross_entropy(logits, y.long())
        
        # L2 Guard on centroids (Prevent Drift)
        centroid_penalty = torch.norm(self.centroids, p=2, dim=1).mean() * self.centroid_reg
        
        return contrastive_loss + centroid_penalty
