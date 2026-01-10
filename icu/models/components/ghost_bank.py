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
        similarity_threshold: float = 0.98,
        prototype_ema_decay: float = 0.99
    ):
        """
        Args:
            capacity: Max number of ghost trajectories to store.
            history_len: T_obs (standard: 24).
            feature_dim: Number of canonical features (standard: 28).
            latent_dim: Dimension of z_expert (standard: 512).
            similarity_threshold: Cosine similarity limit for redundancy.
            prototype_ema_decay: Decay for the global manifold centroid.
        """
        super().__init__()
        self.capacity = capacity
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.similarity_threshold = similarity_threshold
        self.prototype_ema_decay = prototype_ema_decay

        # [v17.3 Hardened] Replay-Aware Buffers
        # Storing raw trajectories forces the model to perform a full forward pass
        # through the Encoder, providing gradients to the shared foundation.
        self.register_buffer("raw_vitals", torch.zeros(capacity, history_len, feature_dim))
        self.register_buffer("raw_masks", torch.zeros(capacity, history_len, feature_dim))
        self.register_buffer("raw_labels", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("latent_anchors", torch.zeros(capacity, latent_dim))
        # [v17.4 GIST-Q] Uncertainty Buffer: Tracks the 'Hardness' of historical ghosts.
        self.register_buffer("uncertainties", torch.zeros(capacity, 1))
        
        # [v18.0 SOTA] Prototype EMA: The "Global Sepsis Manifold" centroid.
        # Rationale: Provides a stable anchor even during sifting.
        self.register_buffer("prototype_ema", torch.zeros(1, latent_dim))
        
        # Metadata / Tracking
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_full", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _update_prototype(self, new_latents: torch.Tensor):
        """Updates the global manifold centroid using EMA."""
        if new_latents.shape[0] == 0:
            return
        
        batch_avg = new_latents.mean(dim=0, keepdim=True)
        if self.prototype_ema.abs().sum() == 0:
            self.prototype_ema.copy_(batch_avg)
        else:
            self.prototype_ema.mul_(self.prototype_ema_decay).add_(batch_avg, alpha=1 - self.prototype_ema_decay)
        
        # Rescale to unit hypersphere for stable similarity mapping
        self.prototype_ema.copy_(F.normalize(self.prototype_ema, dim=1))

    def _find_lvp_index(self) -> int:
        """
        [v19.0] Path A: Greedy DPP (Determinantal Point Process) Sifting.
        
        Rationale:
        Instead of just discarding 'easy' samples, we discard samples that 
        contribute the LEAST to the diversity (volume) of the bank.
        
        Math:
        We look for the index i that minimizes the conditional determinant 
        contribution. In a greedy sense, this is the item most 'spanned' 
        by the other members (highest redundancy).
        """
        if not self.is_full:
            return int(self.ptr)
            
        # Normalize all latents for kernel computation
        latents = F.normalize(self.latent_anchors[:self.size], dim=1) # [C, D]
        
        # 1. Compute Kernel (Similarity Matrix)
        # Using a linear kernel for computational efficiency in the training loop
        # SOTA: Could use RBF, but linear hypersphere distance is ideal for CLS tokens.
        K = torch.matmul(latents, latents.T) # [C, C]
        
        # 2. Greedy Redundancy Masking
        # We calculate the 'Redundancy Score' for each item.
        # R_i = sum(K_ij^2) for j != i. 
        # This represents how much of item i's information is 'leaked' 
        # into other items in the bank.
        
        # Subtract identity to ignore self-similarity (which is 1.0)
        redundancy_scores = (K ** 2).sum(dim=1) - 1.0
        
        # 3. Informative Weighting (GIST-Q Integration)
        # We don't want to discard a very redundant item if it's also very uncertain (hard).
        # Value = Redundancy / (Uncertainty + eps)
        # Item with HIGHEST score is the LVP (High redundancy, Low uncertainty).
        existing_uncertainties = self.uncertainties[:self.size].squeeze(-1)
        lvp_scores = redundancy_scores / (existing_uncertainties + 1e-6)
        
        lvp_idx = torch.argmax(lvp_scores)
        return int(lvp_idx)

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
        DAB+: Selective Prototypical Update.
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

        # Update global prototype with new incoming signal
        self._update_prototype(latents)

        # Normalize latents for similarity check
        norm_latents = F.normalize(latents, dim=1)
        
        # Process each potential ghost
        for i in range(vitals.shape[0]):
            new_v = vitals[i]
            new_m = masks[i]
            new_lab = labels[i]
            new_l = latents[i]
            new_l_norm = norm_latents[i]
            new_unc = uncertainties[i].item()
            
            inserted = False
            
            # 1. Similarity-Based Informative Replacement
            if self.size > 0:
                existing_latents = F.normalize(self.latent_anchors[:self.size], dim=1)
                similarities = torch.matmul(existing_latents, new_l_norm) # [Size]
                max_sim, twin_idx = similarities.max(dim=0)
                
                if max_sim > self.similarity_threshold:
                    # Redundant case: Only replace if the new one is significantly "harder" (higher uncertainty)
                    if new_unc > self.uncertainties[twin_idx].item() * 1.1:
                        idx = int(twin_idx)
                        inserted = True
                    else:
                        continue # Discard redundant easy case
            
            # 2. LVP Selection for Diverse Cases
            if not inserted:
                if not self.is_full:
                    idx = int(self.ptr)
                    self.ptr.fill_((idx + 1) % self.capacity)
                    if self.size < self.capacity:
                        self.size.fill_(self.size + 1)
                        if self.size == self.capacity:
                            self.is_full.fill_(True)
                    inserted = True
                else:
                    # Bank is full and case is diverse: Find the Least Valuable existing ghost
                    idx = self._find_lvp_index()
                    inserted = True

            # 3. Final Insertion
            if inserted:
                self.raw_vitals[idx] = new_v
                self.raw_masks[idx] = new_m
                self.raw_labels[idx] = new_lab
                self.latent_anchors[idx] = new_l
                self.uncertainties[idx] = uncertainties[i]

    def sample(self, num_ghosts: int, seed: int, mixup_alpha: float = 0.0, uncertainty_weighted: bool = False) -> Dict[str, torch.Tensor]:
        """
        Harmonic Summoning with Manifold Mixup (v19.0) and Prioritized Sampling (v20.0).
        Selection from the bank using a global deterministic seed.
        
        Args:
            num_ghosts: Number of ghosts to sample.
            seed: Deterministic seed for DDP parity.
            mixup_alpha: Alpha for Beta distribution. If > 0, performs Manifold Mixup.
            uncertainty_weighted: If True, uses Prioritized Uncertainty Sampling.
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
        
        # [v20.0] Prioritized Uncertainty Sampling
        # Rationale: Training on the "Hardest" historical cases accelerates 
        # discovery of critical sepsis boundaries.
        if uncertainty_weighted:
            # Temperature scale (0.1) to strongly bias towards higher uncertainty
            logits = self.uncertainties[:self.size].squeeze(-1) / 0.1
            probs = torch.softmax(logits, dim=0)
            idx1 = torch.multinomial(probs, num_ghosts, replacement=True, generator=rng)
        else:
            # Sample indices for 'Base Ghosts'
            idx1 = torch.randint(0, int(self.size), (num_ghosts,), generator=rng, device=self.raw_vitals.device)
        
        out = {
            "vitals": self.raw_vitals[idx1],
            "masks": self.raw_masks[idx1],
            "labels": self.raw_labels[idx1].float(),
            "anchors": self.latent_anchors[idx1],
            "uncertainties": self.uncertainties[idx1],
            "valid": torch.ones(num_ghosts, dtype=torch.bool, device=self.raw_vitals.device)
        }

        # [v19.0] Path B: Manifold Mixup
        if mixup_alpha > 0:
            # Sample indices for 'Partner Ghosts'
            idx2 = torch.randint(0, int(self.size), (num_ghosts,), generator=rng, device=self.raw_vitals.device)
            
            # Sample Lambda from Beta distribution
            # We use a manual Beta implementation since torch.distributions can be slow in inner loops
            # or just use torch._standard_gamma and transform if needed, but for small alpha, 
            # a simple uniform-based approximation or direct torch.distributions is fine.
            # Sample Lambda from Beta distribution deterministically across ranks
            # Rationale: Ensures all GPUs mix ghosts identically (Harmonic Summoning).
            # [SOTA FIX]: Use generator-backed sampling or ICDF transform.
            u = torch.rand((num_ghosts, 1), generator=rng, device=self.raw_vitals.device)
            dist = torch.distributions.Beta(torch.tensor([mixup_alpha], device=u.device), 
                                          torch.tensor([mixup_alpha], device=u.device))
            lam = dist.icdf(u)
            
            # Mix Latent Anchors and Labels (Clinical Continuity)
            # Rationale: We mix the 'Targets' for alignment, but keep 'Vitals' as real 
            # to avoid input poisoning. 
            # Note: idx1 ghosts are re-encoded in training_step, so their anchors 
            # will be compared against these MIXED targets.
            out["anchors"] = lam * self.latent_anchors[idx1] + (1 - lam) * self.latent_anchors[idx2]
            
            # Labels become soft [B, 1] or [B, C]
            # If labels are indices, we convert to float probability-like values
            out["labels"] = lam.squeeze(-1) * self.raw_labels[idx1].float() + (1 - lam).squeeze(-1) * self.raw_labels[idx2].float()
            
            # Uncertainty is also mixed (Mean-weighted)
            out["uncertainties"] = lam * self.uncertainties[idx1] + (1 - lam) * self.uncertainties[idx2]

        return out

    def extra_repr(self) -> str:
        return f"capacity={self.capacity}, size={self.size.item()}, is_full={self.is_full.item()}"
