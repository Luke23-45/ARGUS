"""
icu/models/components/ghost_bank.py
----------------------------------
DAB: Diversity-Aware Ghost Bank (v17.3 Hardened).

RATIONALE:
To survive the "Prevalence-Coupled Stagnation," the model needs constant clinical
pressure. This bank stores diverse historical sepsis cases and re-injects them
into every training batch.

Key Features:
1.  Encoder Plasticity: Stores raw trajectories to force re-encoding.
2.  Topological Anchoring: Stores z_ref (latent anchors) for CGA alignment.
3.  Diversity Filtering: Reject redundant cases (Cosine Similarity > 0.98).
4.  DDP Parity: Synchronized ghost selection across cluster nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SepsisGhostBank(nn.Module):
    def __init__(
        self, 
        capacity: int = 256, 
        history_len: int = 24, 
        feature_dim: int = 28, 
        latent_dim: int = 512,
        similarity_threshold: float = 0.98
    ):
        """
        Args:
            capacity: Max number of ghost trajectories to store.
            history_len: T_obs (standard: 24).
            feature_dim: Number of canonical features (standard: 28).
            latent_dim: Dimension of z_expert (standard: 512).
            similarity_threshold: Cosine similarity limit for redundancy.
        """
        super().__init__()
        self.capacity = capacity
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.similarity_threshold = similarity_threshold

        # [v17.3 Hardened] Replay-Aware Buffers
        # Storing raw trajectories forces the model to perform a full forward pass
        # through the Encoder, providing gradients to the shared foundation.
        self.register_buffer("raw_vitals", torch.zeros(capacity, history_len, feature_dim))
        self.register_buffer("raw_masks", torch.zeros(capacity, history_len, feature_dim))
        self.register_buffer("raw_labels", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("latent_anchors", torch.zeros(capacity, latent_dim))
        # [v17.4 GIST-Q] Uncertainty Buffer: Tracks the 'Hardness' of historical ghosts.
        # Rationale: Higher uncertainty implies a case the model hasn't mastered.
        self.register_buffer("uncertainties", torch.zeros(capacity, 1))
        
        # Metadata / Tracking
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_full", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update(
        self, 
        vitals: torch.Tensor, 
        masks: torch.Tensor, 
        labels: torch.Tensor,
        latents: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None
    ):
        """
        DAB: Diversity-Aware Update with Surgical Sifting.
        
        Args:
            vitals: [B, T, D]
            masks: [B, T, D]
            labels: [B]
            latents: [B, L] - Expert latent descriptors (z_expert)
            uncertainties: [B, 1] - Model uncertainty (for sifting)
            active_mask: [B] - Mask indicating which samples are real sepsis cases
        """
        if active_mask is not None:
            vitals = vitals[active_mask]
            masks = masks[active_mask]
            labels = labels[active_mask]
            latents = latents[active_mask]
            if uncertainties is not None:
                uncertainties = uncertainties[active_mask]
        
        if vitals.shape[0] == 0:
            return

        if uncertainties is None:
            uncertainties = torch.zeros(vitals.shape[0], 1, device=vitals.device)

        # Normalize latents for similarity check
        norm_latents = F.normalize(latents, dim=1)
        
        # Process each potential ghost
        for i in range(vitals.shape[0]):
            new_v = vitals[i]
            new_m = masks[i]
            new_lab = labels[i]
            new_l = latents[i]
            new_l_norm = norm_latents[i]
            
            # 1. Surgical Sifting (Diversity + Difficulty)
            # A case is accepted if it's DIVERSE or if it's a HARD case (high uncertainty)
            if self.size > 0:
                existing_latents = F.normalize(self.latent_anchors[:self.size], dim=1)
                similarities = torch.matmul(existing_latents, new_l_norm) # [Size]
                max_sim = similarities.max()
                
                # [v17.4] Sifting: If it's too similar, only keep if it's harder than average
                if max_sim > self.similarity_threshold:
                    avg_unc = self.uncertainties[:self.size].mean()
                    if uncertainties[i] <= avg_unc:
                        # Case is redundant and easy
                        continue
            
            # 2. Sequential Insertion (Ring Buffer)
            p = int(self.ptr)
            self.raw_vitals[p] = new_v
            self.raw_masks[p] = new_m
            self.raw_labels[p] = new_lab
            self.latent_anchors[p] = new_l
            self.uncertainties[p] = uncertainties[i]
            
            self.ptr.fill_((p + 1) % self.capacity)
            if not self.is_full:
                self.size.fill_(self.size + 1)
                if self.size == self.capacity:
                    self.is_full.fill_(True)

    def sample(self, num_ghosts: int, seed: int) -> Dict[str, torch.Tensor]:
        """
        Harmonic Summoning: Selection from the bank using a global deterministic seed.
        
        Returns:
            Dict containing vitals, masks, labels, and latent_anchors for the ghosts.
        """
        if self.size == 0:
            # Fallback for early training: return zeros
            device = self.raw_vitals.device
            return {
                "vitals": torch.zeros(num_ghosts, self.history_len, self.feature_dim, device=device),
                "masks": torch.zeros(num_ghosts, self.history_len, self.feature_dim, device=device),
                "labels": torch.zeros(num_ghosts, dtype=torch.long, device=device),
                "anchors": torch.zeros(num_ghosts, self.latent_dim, device=device),
                "uncertainties": torch.zeros(num_ghosts, 1, device=device),
                "valid": torch.zeros(num_ghosts, dtype=torch.bool, device=device)
            }
        
        # Use a local RNG with the global seed to ensure DDP parity
        rng = torch.Generator(device=self.raw_vitals.device)
        rng.manual_seed(seed)
        
        indices = torch.randint(0, int(self.size), (num_ghosts,), generator=rng, device=self.raw_vitals.device)
        
        return {
            "vitals": self.raw_vitals[indices],
            "masks": self.raw_masks[indices],
            "labels": self.raw_labels[indices],
            "anchors": self.latent_anchors[indices],
            "uncertainties": self.uncertainties[indices],
            "valid": torch.ones(num_ghosts, dtype=torch.bool, device=self.raw_vitals.device)
        }

    def extra_repr(self) -> str:
        return f"capacity={self.capacity}, size={self.size.item()}, is_full={self.is_full.item()}"
