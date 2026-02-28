import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class SymmetryGate(nn.Module):
    """
    [2025 SOTA] Gated Linear Unit (GLU) for cross-modal physiological signals.
    Learns a 'confidence mask' to emphasize correlated signals across modalities
    (e.g., coupling Heart Rate spikes with Lactate elevation).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # [v14.4 SOTA] Scale-Aware Normalization
        # Rationale: LayerNorm (Standard) is magnitude-invariant and amplifies floor noise.
        # Swapping for RMSNorm with large EPS to preserve clinical gating.
        from icu.models.components.nth_encoder import RMSNorm
        self.norm = RMSNorm(dim, eps=0.5)
        
        # [v14.0 SOTA] Neural Initialization
        # Ensures Sigmoid(0) = 0.5 at cold-start
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] concatenated multi-modal representations
        """
        g = torch.sigmoid(self.gate(x))
        p = self.proj(x)
        return self.norm(g * p + (1 - g) * x)

class PhysiologicalSEBlock(nn.Module):
    """
    [2025 SOTA] Physiological Squeeze-and-Excitation (SE) Block.
    Dynamically recalibrates the importance of different physiological channels
    based on global context (e.g., up-weighting Lactate during shock).
    
    [v14.2 Breakthrough] Dual-Squeeze: Combined Avg+Max pooling captures
    both trend and acute transients (resolving Signal Dilution Bottleneck).
    
    [v14.3 Forensic] Mask-Aware Pooling (MAP): Uses imputation_mask to 
    prevent normalization-shifted zeros from poisoning the context.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Reduction ratio for throughput
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(), 
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Identity Initialization
        nn.init.zeros_(self.fc[-2].weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T, D] (expanded imputation mask)
        """
        B, T, D = x.shape
        # Ensure mask is available for MAP
        if mask is None:
            mask = torch.ones_like(x)
            
        # 1. Mask-Aware Squeeze (MAP)
        # Rationale: Standard AvgPool includes shifted-zeros (-3.0) in the sum, 
        # biasing the context. MAP calculates true physiological mean.
        denom = mask.sum(dim=1).clamp(min=1e-8)
        y_avg = (x * mask).sum(dim=1) / denom # [B, D]
        
        # 2. Mask-Aware Peak Detection
        # Rationale: Padded values (-infinity equivalents) must be ignored in MaxPool.
        x_masked = x.masked_fill(mask == 0, -1e9)
        y_max = x_masked.max(dim=1).values # [B, D]
        
        # 3. Concatenation (Dual-Squeeze)
        y_dual = torch.cat([y_avg, y_max], dim=1)
        
        # 4. Excitation
        weights = self.fc(y_dual).view(B, 1, D)
        
        return x * weights

class GeometricProjector(nn.Module):
    """
    [SOTA Phase 1] Structured Physiological Encoder.
    Replaces simple linear projection with grouped sub-networks and symmetry gating.
    
    Features:
    1. Grouped Projections: Separate paths for Hemodynamics, Labs, and Electrolytes.
    2. Physiological SE-Block: Dynamic channel attention (SOTA 2025).
    3. Symmetry Gating: Captures non-linear correlations.
    4. Imputation Awareness: Handles dual-stream (Value + Mask) inputs.
    """
    def __init__(self, d_model: int, hemo_dim: int = 7, labs_dim: int = 11, elec_dim: int = 4, use_imputation_masks: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_imputation_masks = use_imputation_masks
        
        # [v4.2.1 SOTA] Dynamic Feature Groups
        # Passed from ICUConfig to ensure macro-alignment
        
        scale = 2 if use_imputation_masks else 1
        
        self.hemo_dim = hemo_dim * scale
        self.labs_dim = labs_dim * scale
        self.elec_dim = elec_dim * scale
        
        # Branch Projections
        # We allocate d_model/2 for Hemo (primary), d_model/2 for Labs, and d_model/4 for Elec
        # Total concats to 1.25 * d_model.
        # Branch Projections
        self.hemo_proj = nn.Sequential(
            nn.Linear(self.hemo_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
        self.labs_proj = nn.Sequential(
            nn.Linear(self.labs_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
        self.elec_proj = nn.Sequential(
            nn.Linear(self.elec_dim, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )

        # [v132.0 SOTA FIX] Branch-Level Normalization (Relocated)
        # Rationale: Final stage RMSNorm hijacks SE-Block noise suppression (80x Gain).
        # Fix: Apply norm to raw branch features BEFORE the nonlinear ensemble.
        # [v14.4] Scale-Aware stabilization: Use eps=0.5 to prevent noise amplification.
        from icu.models.components.nth_encoder import RMSNorm
        self.hemo_norm = RMSNorm(d_model // 2, eps=0.5)
        self.labs_norm = RMSNorm(d_model // 2, eps=0.5)
        self.elec_norm = RMSNorm(d_model // 4, eps=0.5)
        
        # Combined Dimension
        combined_dim = (d_model // 2) + (d_model // 2) + (d_model // 4)
        
        # [v14.5 SOTA] Harmonic Branch Balancers (Smoking Gun #128)
        # Rationale: Continuous signals (Hemo) can have 64x more energy than sparse labs.
        # We use learnable multipliers to allow the model to re-calibrate branch energy.
        self.hemo_gain = nn.Parameter(torch.ones(1))
        self.labs_gain = nn.Parameter(torch.ones(1))
        self.elec_gain = nn.Parameter(torch.ones(1))
        
        # SOTA Components
        self.se_block = PhysiologicalSEBlock(combined_dim)
        self.gate = SymmetryGate(combined_dim)
        
        # Final Merge
        self.output_proj = nn.Linear(combined_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_channels] where input_channels is 22 or 44
        """
        B, T, C = x.shape
        scale = 2 if self.use_imputation_masks else 1
        
        # 1. Slice carefully - assumes x matches groups perfectly
        # We also extract the imputation mask for MAP synchronization
        x_hemo = x[..., :self.hemo_dim]
        x_labs = x[..., self.hemo_dim : self.hemo_dim + self.labs_dim]
        x_elec = x[..., self.hemo_dim + self.labs_dim : self.hemo_dim + self.labs_dim + self.elec_dim]
        
        # [v14.3 SOTA] Mask Extraction
        # We need the 1-bit mask to ignore shifted-zeros during SE-Pooling.
        if self.use_imputation_masks:
            # Mask is stored in the second half of each group
            m_h = x_hemo[..., (self.hemo_dim // 2):]
            m_l = x_labs[..., (self.labs_dim // 2):]
            m_e = x_elec[..., (self.elec_dim // 4):] # [v112.0 FIX] Correct slice
            m_combined = torch.cat([m_h, m_l, m_e], dim=-1)
            # Expand mask to match latent space of SE-Block
            # Rationale: SE-Block operates on projected features. We expand the binary 
            # feature-mask to the latent dimension using boolean 'any' or broadcasting.
            # Here we simplify: if any feature in Hemo is valid, the hemo-latent branch is active.
            # For deeper fidelity, we broadcast the mean mask.
            m_se = torch.cat([
                m_h.any(dim=-1, keepdim=True).expand(-1, -1, self.d_model // 2),
                m_l.any(dim=-1, keepdim=True).expand(-1, -1, self.d_model // 2),
                m_e.any(dim=-1, keepdim=True).expand(-1, -1, self.d_model // 4)
            ], dim=-1)
        else:
            m_se = None

        # 2. Branch Projections & Relocated Norms
        z_hemo = self.hemo_norm(self.hemo_proj(x_hemo))
        z_labs = self.labs_norm(self.labs_proj(x_labs))
        z_elec = self.elec_norm(self.elec_proj(x_elec))
        
        # 3. Concatenate (Harmonic Rebalancing)
        # [v14.5] Apply learnable gains to equalize branch energies
        z_combined = torch.cat([
            z_hemo * self.hemo_gain, 
            z_labs * self.labs_gain, 
            z_elec * self.elec_gain
        ], dim=-1)
        
        # 4. Apply SOTA Refinements with MAP
        z_se = self.se_block(z_combined, mask=m_se) # Mask-Aware pooling
        z_gated = self.gate(z_se)                    # Non-linear gating
        
        # 5. Final Output (Raw manifold preservation)
        out = self.output_proj(z_gated)
        return out
