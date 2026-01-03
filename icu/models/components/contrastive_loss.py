import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricContrastiveLoss(nn.Module):
    """
    [SOTA 2025] Asymmetric Contrastive Loss (ACL) with EMA Stabilization.
    
    Why this fixes Gradient Explosion:
    1. Removes learnable centroids (which oscillate wildly via SGD).
    2. Replaces them with EMA (Exponential Moving Average) updates.
    3. Enforces strict L2 normalization on the hypersphere.
    
    Adapted from 'StableContrastiveLoss' in stabilization.py.
    """
    def __init__(self, d_model: int, num_classes: int = 3, temperature: float = 0.25, centroid_reg: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = 0.99 # [SOTA] Fixed slow momentum for stability
        
        # Buffer, not Parameter -> No Gradients on Centroids directly
        # This prevents "fighting" between the encoder and the centroids
        self.register_buffer('centroids', F.normalize(torch.randn(num_classes, d_model), dim=1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def forward(self, z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (B, T, D) or (B, D) Latent representations
            y: (B, T) or (B,) Labels (Long)
            mask: (B, T) Optional boolean padding mask
        """
        # --- 1. Flatten & Masking Logic (Preserved) ---
        if z.dim() == 3:
            B, T, D = z.shape
            if y.dim() == 1 and y.shape[0] == B:
                y = y.unsqueeze(1).expand(B, T)
            z = z.reshape(-1, self.d_model)
            y = y.reshape(-1)
            if mask is not None:
                if mask.dim() == 3: mask = mask.any(dim=-1)
                m = mask.reshape(-1)
                z = z[m]
                y = y[m]
        
        if z.shape[0] == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # --- 2. Spherical Projection (Critical for stability) ---
        features = F.normalize(z, p=2, dim=1)
        
        # --- 3. EMA Update (Training Only) ---
        if self.training:
            with torch.no_grad():
                unique_classes = torch.unique(y)
                for c in unique_classes:
                    if c < 0 or c >= self.num_classes: continue 
                    
                    mask_c = (y == c)
                    batch_center = features[mask_c].mean(dim=0)
                    batch_center = F.normalize(batch_center, p=2, dim=0)
                    
                    if self.initialized.item():
                        self.centroids[c].mul_(self.momentum).add_(batch_center, alpha=1 - self.momentum)
                    else:
                        self.centroids[c] = batch_center
                        
                self.centroids.data = F.normalize(self.centroids, p=2, dim=1)
                self.initialized.fill_(True)
        
        # --- 4. Compute Logits & Loss ---
        # Range: [-1/temp, 1/temp]
        logits = torch.matmul(features, self.centroids.t()) / self.temperature
        contrastive_loss = F.cross_entropy(logits, y.long())
        
        return contrastive_loss
