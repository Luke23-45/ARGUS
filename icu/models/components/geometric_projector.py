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
        self.norm = nn.LayerNorm(dim)

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
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Squeeze (Global Context)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] (Note: SE usually operates on C, so we view T as 'Spatial')
        Wait, for Time Series, 'Channels' usually means the embedding dim D.
        We want to re-weight the features D at each timestep? 
        Or re-weight across the sequence?
        Standard SE for 1D CNNs reweights Channels D based on global time average.
        """
        # Input x is [B, T, D]
        # We want to re-weight D (embedding dimension)
        # Permute to [B, D, T] for AvgPool1d
        B, T, D = x.shape
        y = x.transpose(1, 2) # [B, D, T]
        
        # Squeeze: Global Average Pooling across Time T
        # Result: [B, D, 1]
        y_avg = self.avg_pool(y).view(B, D) 
        
        # Excitation: Learn weights
        # [B, D]
        weights = self.fc(y_avg).view(B, 1, D) # Broadcast across Time
        
        # Scale
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
    def __init__(self, d_model: int, use_imputation_masks: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_imputation_masks = use_imputation_masks
        
        # Dimensions based on ICUConfig v11.0
        # Hemo: 7, Labs: 11, Elec: 4
        # Dynamic Feature Groups
        # Indices 0-6: Hemodynamics (7)
        # Indices 7-17: Labs (11)
        # Indices 18-21: Electrolytes (4)
        
        scale = 2 if use_imputation_masks else 1
        
        self.hemo_dim = 7 * scale
        self.labs_dim = 11 * scale
        self.elec_dim = 4 * scale
        
        # Branch Projections
        # We allocate d_model/2 for Hemo (primary), d_model/2 for Labs, and d_model/4 for Elec
        # Total concats to 1.25 * d_model.
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
        
        # Combined Dimension
        combined_dim = (d_model // 2) + (d_model // 2) + (d_model // 4)
        
        # SOTA Components
        self.se_block = PhysiologicalSEBlock(combined_dim)
        self.gate = SymmetryGate(combined_dim)
        
        # Final Merge
        self.output_proj = nn.Linear(combined_dim, d_model)
        self.norm = nn.GroupNorm(8, d_model) # 8 groups for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_channels] where input_channels is 22 or 44
        """
        B, T, C = x.shape
        scale = 2 if self.use_imputation_masks else 1
        
        # Split into groups
        # Based on icu/models/diffusion.py indices 0..21
        hemo_idx = 7 * scale
        labs_idx = (7 + 11) * scale
        elec_idx = (7 + 11 + 4) * scale
        
        # Slice carefully - assumes x matches these groups perfectly
        x_hemo = x[..., :hemo_idx]
        x_labs = x[..., hemo_idx:labs_idx]
        x_elec = x[..., labs_idx:elec_idx]
        
        # 1. Branch Projections
        z_hemo = self.hemo_proj(x_hemo)
        z_labs = self.labs_proj(x_labs)
        z_elec = self.elec_proj(x_elec)
        
        # 2. Concatenate
        z_combined = torch.cat([z_hemo, z_labs, z_elec], dim=-1)
        
        # 3. Apply SOTA Refinements
        z_se = self.se_block(z_combined) # Context-aware reweighting
        z_gated = self.gate(z_se)        # Non-linear gating
        
        # 4. Final Output
        out = self.output_proj(z_gated)
        
        # Apply GroupNorm (needs [B, C, T] format)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        
        return out
