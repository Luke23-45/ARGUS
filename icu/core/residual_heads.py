"""
working/residual_heads.py
-------------------------
SOTA Residual Task Head (Clinical MTL 2025).
"""

import torch
import torch.nn as nn

class ClinicalResidualHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )
        self.norm_out = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x + self.block(x)
        return self.head(self.norm_out(feat))