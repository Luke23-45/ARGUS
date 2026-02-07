import torch
import torch.nn as nn
import torch.nn.functional as F
from icu.utils.train_utils import ScalingSteward
import torch.distributed as dist

class AsymmetricContrastiveLoss(nn.Module):
    """
    [SOTA 2025] Asymmetric Contrastive Loss (ACL) with EMA Stabilization.
    
    Why this fixes Gradient Explosion:
    1. Removes learnable centroids (which oscillate wildly via SGD).
    2. Replaces them with EMA (Exponential Moving Average) updates.
    3. Enforces strict L2 normalization on the hypersphere.
    """
    def __init__(self, d_model: int, num_classes: int = 3, temperature: float = 0.25, centroid_reg: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.register_buffer("base_momentum", torch.tensor(0.99).float())
        self.register_buffer("momentum", torch.tensor(0.99).float())
        
        self.register_buffer('centroids', F.normalize(torch.randn(num_classes, d_model), dim=1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def scale_dynamics(self, n_curr: int):
        if n_curr <= 0: return
        raw_momentum = ScalingSteward.get_decay(float(self.base_momentum), n_curr)
        self.momentum.fill_(min(0.999, raw_momentum))

    def forward(self, z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (B, T, D) or (B, D) Latent representations
            y: (B, T) or (B,) Labels (Long)
            mask: (B, T) Optional boolean padding mask
        """
        device = z.device
        
        # 1. Flatten & Masking Logic
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
        
        has_data = z.shape[0] > 0
        num_classes = self.centroids.shape[0]

        # [v23.0 SOTA FIX] Collective Entry Consensus
        # Rationale: Ranks with zero samples MUST reach the all_reduce below
        # to prevent distributed deadlocks (Smoking Gun #220).
        indices = torch.unique(y) if has_data else torch.tensor([], device=device, dtype=torch.long)
        
        with torch.no_grad():
            self.centroids.data.copy_(F.normalize(self.centroids, dim=1))
            
            # 1. Local Class Center Calculation
            unique_classes = indices.tolist()
            local_sync = torch.zeros(num_classes, self.d_model + 1, device=device)
            
            if has_data:
                z_norm = F.normalize(z, p=2, dim=1)
                for c in unique_classes:
                    if c < 0 or c >= num_classes: continue
                    mask_c = (y == c)
                    if mask_c.any():
                        local_sync[c, :self.d_model] = z_norm[mask_c].sum(dim=0)
                        local_sync[c, self.d_model] = float(mask_c.sum())

            # 2. Distributed Consensus (Mass-weighted)
            # CRITICAL: This MUST be called by all ranks simultaneously.
            if dist.is_initialized():
                dist.all_reduce(local_sync, op=dist.ReduceOp.SUM)
            
            # 3. EMA Update
            for c in range(num_classes):
                global_count = local_sync[c, self.d_model]
                if global_count > 0:
                    global_center = local_sync[c, :self.d_model] / global_count
                    global_center = F.normalize(global_center.unsqueeze(0), dim=1).squeeze(0)
                    
                    if self.initialized.item():
                        self.centroids[c].mul_(self.momentum).add_(global_center, alpha=1 - self.momentum)
                    else:
                        self.centroids[c].copy_(global_center)
                        
            if local_sync[:, self.d_model].sum() > 0:
                self.initialized.fill_(True)

        # Early return for empty ranks AFTER collectives
        if not has_data:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Standard Computation
        z = F.normalize(z, dim=1)
        logits = torch.matmul(z, self.centroids.t()) / self.temperature
        contrastive_loss = F.cross_entropy(logits, y.long())
        
        return contrastive_loss
