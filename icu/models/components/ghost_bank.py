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
        [v25.5 SOTA] Vectorized Diversity-Aware Update.
        Eliminates the O(B) loop for massive throughput gains.
        """
        if active_mask is not None:
            vitals, masks, labels, latents = vitals[active_mask], masks[active_mask], labels[active_mask], latents[active_mask]
            if uncertainties is not None: uncertainties = uncertainties[active_mask]
        
        B_orig = vitals.shape[0]
        if B_orig == 0: return

        # Intra-Batch Redundancy Filtering (SOTA v25.7 Precision Guard)
        # Prevents filling the bank with identical samples from the same batch
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                norm_b = F.normalize(latents, dim=1).half()
                b_self_sim = torch.matmul(norm_b, norm_b.T)
                # Mask out identity diagonal
                b_self_sim.fill_diagonal_(0)
                # Convert back for indexing
                b_self_sim = b_self_sim.float()
            # Find samples that are too similar to earlier ones in the same batch
            keep_mask = torch.ones(B_orig, dtype=torch.bool, device=vitals.device)
            for i in range(B_orig):
                if keep_mask[i]:
                    # If any subsequent sample is too similar, mask it out
                    too_similar = b_self_sim[i, i+1:] > 0.99
                    if too_similar.any():
                        keep_mask[i+1:][too_similar] = False
            
            vitals, masks, labels, latents = vitals[keep_mask], masks[keep_mask], labels[keep_mask], latents[keep_mask]
            if uncertainties is not None: uncertainties = uncertainties[keep_mask]
            B = vitals.shape[0]

        if uncertainties is None:
            uncertainties = torch.zeros(B, 1, device=vitals.device)

        # 1. Update global prototype with new incoming signal
        self._update_prototype(latents)

        # 2. Sequential Bootstrap for Empty Bank
        if self.size == 0:
            num_fill = min(B, self.capacity)
            self.raw_vitals[:num_fill].copy_(vitals[:num_fill])
            self.raw_masks[:num_fill].copy_(masks[:num_fill])
            self.raw_labels[:num_fill].copy_(labels[:num_fill])
            self.latent_anchors[:num_fill].copy_(latents[:num_fill])
            self.uncertainties[:num_fill].copy_(uncertainties[:num_fill])
            self.size.fill_(num_fill)
            self.ptr.fill_(num_fill % self.capacity)
            if self.size == self.capacity: self.is_full.fill_(True)
            return

        # 3. Vectorized Similarity Check (v25.7 Precision Guard)
        # Using .half() for the similarity matrix reduces peak VRAM by 50% for this op.
        with torch.cuda.amp.autocast(enabled=False):
            norm_new = F.normalize(latents, dim=1).half()
            norm_old = F.normalize(self.latent_anchors[:self.size], dim=1).half()
            # Similarity Matrix [B, Size] in FP16
            sim_matrix = torch.matmul(norm_new, norm_old.T)
            max_sim, twin_idx = sim_matrix.max(dim=1)
            # Convert back to float for stable logical ops
            max_sim = max_sim.float()
        
        # Criteria A: Informative Replacement (Redundant but harder)
        is_redundant = max_sim > self.similarity_threshold
        target_unc = self.uncertainties[twin_idx].flatten()
        is_harder = uncertainties.flatten() > (target_unc * 1.1)
        to_replace = is_redundant & is_harder
        
        # Criteria B: Diverse Candidates (Non-redundant)
        is_diverse = ~is_redundant
        
        # [PHASE 1] Batched Informative Replacement
        if to_replace.any():
            r_idx = twin_idx[to_replace]
            self.raw_vitals[r_idx] = vitals[to_replace]
            self.raw_masks[r_idx] = masks[to_replace]
            self.raw_labels[r_idx] = labels[to_replace]
            self.latent_anchors[r_idx] = latents[to_replace]
            self.uncertainties[r_idx] = uncertainties[to_replace]

        # [PHASE 2] Batched Diverse Expansion
        if is_diverse.any():
            dv, dm, dl, dlat, dunc = vitals[is_diverse], masks[is_diverse], labels[is_diverse], latents[is_diverse], uncertainties[is_diverse]
            num_div = dv.shape[0]
            
            # Fill remaining space
            available = self.capacity - int(self.size)
            num_fill = min(num_div, available)
            if num_fill > 0:
                indices = (torch.arange(num_fill, device=dv.device) + int(self.ptr)) % self.capacity
                self.raw_vitals[indices] = dv[:num_fill]
                self.raw_masks[indices] = dm[:num_fill]
                self.raw_labels[indices] = dl[:num_fill]
                self.latent_anchors[indices] = dlat[:num_fill]
                self.uncertainties[indices] = dunc[:num_fill]
                self.ptr.fill_((int(self.ptr) + num_fill) % self.capacity)
                self.size.fill_(int(self.size) + num_fill)
                if self.size == self.capacity: self.is_full.fill_(True)
                
            # Replace LVPs if bank is full
            num_lvp = num_div - num_fill
            if num_lvp > 0 and self.is_full:
                # SOTA: Batched LVP Selection (v25.7 Precision Guard)
                with torch.cuda.amp.autocast(enabled=False):
                    lat_all = F.normalize(self.latent_anchors[:self.size], dim=1).half()
                    K = torch.matmul(lat_all, lat_all.T)
                    redundancy = (K.float() ** 2).sum(dim=1) - 1.0
                unc = self.uncertainties[:self.size].flatten()
                lvp_scores = redundancy / (unc + 1e-6)
                
                _, lvp_indices = torch.topk(lvp_scores, min(num_lvp, int(self.size)))
                num_to_replace = lvp_indices.shape[0]
                self.raw_vitals[lvp_indices] = dv[num_fill:num_fill+num_to_replace]
                self.raw_masks[lvp_indices] = dm[num_fill:num_fill+num_to_replace]
                self.raw_labels[lvp_indices] = dl[num_fill:num_fill+num_to_replace]
                self.latent_anchors[lvp_indices] = dlat[num_fill:num_fill+num_to_replace]
                self.uncertainties[lvp_indices] = dunc[num_fill:num_fill+num_to_replace]

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
