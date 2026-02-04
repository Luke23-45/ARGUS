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
from icu.utils.train_utils import ScalingSteward

class SepsisGhostBank(nn.Module):
    def __init__(
        self, 
        capacity: int = 256, 
        history_len: int = 24, 
        feature_dim: int = 28, 
        latent_dim: int = 512,
        similarity_threshold: float = 0.98,
        prototype_ema_decay: float = 0.99,
        latent_adapter_strength: float = 0.15 # [v29.6] Increased base to 0.15 (Abyssal #303)
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
        self.base_capacity = capacity # [v26.1 FIX] Store Base for Idempotency
        self.capacity = capacity
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.similarity_threshold = similarity_threshold
        self.prototype_ema_decay = prototype_ema_decay
        self.base_latent_adapter_strength = latent_adapter_strength # [v29.6 FIX] Store Base
        self.latent_adapter_strength = latent_adapter_strength

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
    def _update_prototype(self, new_latents: torch.Tensor, decay_override: Optional[float] = None):
        """
        [v161.0 SOTA FIX] Global Prototype Parity (Smoking Gun #161)
        Rationale: Updates must occur identically on all ranks. If only one rank
        has sepsis samples, we must still synchronize the result to prevent drift.
        """
        if torch.distributed.is_initialized():
            # 1. Coalesce local signal
            local_sum = new_latents.sum(dim=0, keepdim=True) if new_latents.shape[0] > 0 else torch.zeros(1, self.latent_dim, device=self.prototype_ema.device)
            local_count = torch.tensor([float(new_latents.shape[0])], device=self.prototype_ema.device)
            
            # 2. Synchronize across cluster
            # Buffer: [SUM_D1, ..., SUM_DN, COUNT]
            sync_buffer = torch.cat([local_sum.flatten(), local_count])
            torch.distributed.all_reduce(sync_buffer, op=torch.distributed.ReduceOp.SUM)
            
            global_sum = sync_buffer[:-1].view(1, -1)
            global_count = sync_buffer[-1].item()
            
            if global_count <= 1e-6:
                return
            batch_avg = global_sum / global_count
        else:
            if new_latents.shape[0] == 0:
                return
            batch_avg = new_latents.mean(dim=0, keepdim=True)
            
        # [v51.0 SOTA FIX] NaN-Resistant Manifold Prototype (Smoking Gun #51)
        if not torch.isfinite(batch_avg).all():
            return # Skip update for non-finite data
        
        if self.prototype_ema.abs().sum() == 0:
            self.prototype_ema.copy_(batch_avg)
        else:
            # [v2026 Phase 12 FIX] Momentum Burst (Smoking Gun #Phase12)
            eff_decay = decay_override if decay_override is not None else self.prototype_ema_decay
            self.prototype_ema.mul_(eff_decay).add_(batch_avg, alpha=1 - eff_decay)
        
        # Rescale to unit hypersphere for stable similarity mapping
        self.prototype_ema.copy_(F.normalize(self.prototype_ema, dim=1))

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies bank capacity and decay across step densities."""
        if n_curr <= 0: return
        
        # 1. Scale EMA Decays
        self.prototype_ema_decay = ScalingSteward.get_decay(0.99, n_curr)
        # Adapt mix rate: (1 - strength) is the retention factor.
        retention_ref = 1.0 - self.base_latent_adapter_strength
        retention_curr = ScalingSteward.get_decay(retention_ref, n_curr)
        self.latent_adapter_strength = 1.0 - retention_curr
        
        # 2. Scale Capacity Linearly
        # Note: We re-allocate buffers to maintain identical epoch-time coverage.
        # This is safe because on_train_start runs after on_load_checkpoint.
        new_capacity = ScalingSteward.get_steps(self.base_capacity, n_curr)
        if new_capacity != self.capacity:
            # Re-allocate with current data preservation
            old_raw_vitals = self.raw_vitals
            old_raw_masks = self.raw_masks
            old_raw_labels = self.raw_labels
            old_latent_anchors = self.latent_anchors
            old_uncertainties = self.uncertainties
            
            num_to_keep = min(self.capacity, new_capacity)
            self.capacity = new_capacity
            
            # [SOTA FIX] Capture device to prevent CPU mismatch after resize
            device = self.raw_vitals.device
            
            self.register_buffer("raw_vitals", torch.zeros(new_capacity, self.history_len, self.feature_dim, device=device))
            self.register_buffer("raw_masks", torch.zeros(new_capacity, self.history_len, self.feature_dim, device=device))
            self.register_buffer("raw_labels", torch.zeros(new_capacity, dtype=torch.long, device=device))
            self.register_buffer("latent_anchors", torch.zeros(new_capacity, self.latent_dim, device=device))
            self.register_buffer("uncertainties", torch.zeros(new_capacity, 1, device=device))
            
            # Copy old data
            self.raw_vitals[:num_to_keep] = old_raw_vitals[:num_to_keep]
            self.raw_masks[:num_to_keep] = old_raw_masks[:num_to_keep]
            self.raw_labels[:num_to_keep] = old_raw_labels[:num_to_keep]
            self.latent_anchors[:num_to_keep] = old_latent_anchors[:num_to_keep]
            self.uncertainties[:num_to_keep] = old_uncertainties[:num_to_keep]
            
            # Update pointers
            new_size = min(int(self.size), new_capacity)
            self.size.fill_(new_size)
            self.ptr.fill_(new_size % new_capacity)
            self.is_full.fill_(new_size == new_capacity)

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
        active_mask: Optional[torch.Tensor] = None,
        prototype_burst: bool = False
    ):
        """
        [v25.5 SOTA] Vectorized Diversity-Aware Update.
        Eliminates the O(B) loop for massive throughput gains.
        Fused with [v4.0] Device Safety Anchors.
        """
        if active_mask is not None:
            vitals, masks, labels, latents = vitals[active_mask], masks[active_mask], labels[active_mask], latents[active_mask]
            if uncertainties is not None: uncertainties = uncertainties[active_mask]
        
        # [v26.0 SAFETY CRITICAL] Sanity Gate: Reject Poisoned Updates
        # If the model explodes (NaN/Inf), we MUST NOT pollute the memory bank.
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            return

        B_orig = vitals.shape[0]
        if B_orig == 0: return

        # Intra-Batch Redundancy Filtering (SOTA v25.7 Precision Guard)
        # Prevents filling the bank with identical samples from the same batch
        with torch.no_grad():
            # [v25.7] Mixed Precision Similarity (VRAM Optimization)
            with torch.cuda.amp.autocast(enabled=False):
                norm_b = F.normalize(latents, dim=1).half()
                b_self_sim = torch.matmul(norm_b, norm_b.T)
                b_self_sim.fill_diagonal_(0)
                b_self_sim = b_self_sim.float()
                
            # Find samples that are too similar to earlier ones in the same batch
            keep_mask = torch.ones(B_orig, dtype=torch.bool, device=vitals.device)
            for i in range(B_orig):
                if keep_mask[i]:
                    too_similar = b_self_sim[i, i+1:] > 0.99
                    if too_similar.any():
                        keep_mask[i+1:][too_similar] = False
            
            vitals, masks, labels, latents = vitals[keep_mask], masks[keep_mask], labels[keep_mask], latents[keep_mask]
            if uncertainties is not None: uncertainties = uncertainties[keep_mask]
            B = vitals.shape[0]

        if uncertainties is None:
            uncertainties = torch.zeros(B, 1, device=vitals.device)

        # 1. Update global prototype with new incoming signal
        # [v2026 Phase 12] Apply aggressive burst if requested (decay=0.1)
        eff_decay = 0.1 if prototype_burst else None
        self._update_prototype(latents, decay_override=eff_decay)

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
        with torch.cuda.amp.autocast(enabled=False):
            norm_new = F.normalize(latents, dim=1).half()
            norm_old = F.normalize(self.latent_anchors[:self.size], dim=1).half()
            sim_matrix = torch.matmul(norm_new, norm_old.T)
            max_sim, twin_idx = sim_matrix.max(dim=1)
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

    @torch.no_grad()
    def refresh_anchors(self, encoder: nn.Module):
        """
        [v33.1 SOTA FIX] Anchor Refresh mechanism (Fix #356).
        Re-encodes all stored trajectories to prevent latent representation drift.
        
        Rationale: As the encoder weights change, stored 'latent_anchors' become 
        misaligned with the current manifold. Periodic refresh ensures 
        diversity-based rejection remains accurate.
        """
        if self.size == 0:
            return
            
        # Batched Refresh for VRAM efficiency
        batch_size = 64
        num_iters = (int(self.size) + batch_size - 1) // batch_size
        
        # [v33.1] Force Evaluation Mode for deterministic encoding
        # [v42.0 SOTA FIX] Polymorphic Support (Module vs Function)
        was_training = None
        if isinstance(encoder, torch.nn.Module):
             was_training = encoder.training
             encoder.eval()
        
        try:
            for i in range(num_iters):
                start = i * batch_size
                end = min(start + batch_size, int(self.size))
                
                v_batch = self.raw_vitals[start:end]
                m_batch = self.raw_masks[start:end]
                
                # Re-encode using CURRENT encoder weights
                # [v33.1 Hardened] We assume encoder is a callable that projects to latent space.
                # If it's the APEX Planner, we might need to pass static data too.
                # However, for the ghost bank, vitals/masks usually suffice for the base representation.
                # In wrapper_generalist, we'll pass a lambda that handles the details.
                new_anchors = encoder(v_batch, m_batch)
                
                # Standardize on unit hypersphere
                self.latent_anchors[start:end].copy_(F.normalize(new_anchors, dim=1))
        finally:
            # Restore training state
            if was_training is not None:
                encoder.train(was_training)
                
        # Re-initialize prototype to match new latent space
        if self.size > 0:
            self.prototype_ema.fill_(0.0)
            self._update_prototype(self.latent_anchors[:self.size])

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
                "valid": torch.zeros(num_ghosts, dtype=torch.bool, device=device),
                # [v39.0 SOTA FIX] Ghost Bank Incompleteness (Smoking Gun #46)
                # Rationale: wrapper_generalist expects 'static' key even if bank is empty.
                # Standard static dim is 6 (Columns 22-27 of vitals).
                "static": torch.zeros(num_ghosts, 6, device=device)
            }
        
        # Use a local RNG with the global seed to ensure DDP parity
        # [SOTA FIX] Always use CPU generator for sampling indices to ensure cross-device consistency.
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        
        # [v20.0] Prioritized Uncertainty Sampling
        if uncertainty_weighted:
            # [v23.0 SOTA FIX] Entropy Injection (Smoking Gun #23)
            # Rationale: Pure prioritization (temp=0.1) leads to seeing only 
            # 5% of the bank, causing representation collapse.
            # Fix: Inject 5% pure uniform noise to ensure exploration.
            logits = self.uncertainties[:self.size].squeeze(-1) / 0.1
            # Move probs to CPU for multinomial
            probs_prioritized = torch.softmax(logits, dim=0).cpu()
            
            # 5% uniform base
            probs_uniform = torch.ones_like(probs_prioritized) / probs_prioritized.shape[0]
            probs = 0.95 * probs_prioritized + 0.05 * probs_uniform
            
            idx1 = torch.multinomial(probs, num_ghosts, replacement=True, generator=rng)
        else:
            # Sample indices for 'Base Ghosts'
            idx1 = torch.randint(0, int(self.size), (num_ghosts,), generator=rng, device='cpu')
        
        # Ensure indices are on the correct device for buffer lookup
        idx1 = idx1.to(self.raw_vitals.device)
        
        out = {
            "vitals": self.raw_vitals[idx1],
            "masks": self.raw_masks[idx1],
            "labels": self.raw_labels[idx1].float(),
            "anchors": self.latent_anchors[idx1],
            "uncertainties": self.uncertainties[idx1],
            "valid": torch.ones(num_ghosts, dtype=torch.bool, device=self.raw_vitals.device)
        }
        
        # [v21.5 SOTA] Ghost Demographic Preservation
        # Extract static context (Demographics) from the canonical vital stream (Columns 22-27).
        # This prevents "Demographic Amnesia" where ghosts were re-injected with zeros.
        out["static"] = out["vitals"][:, 0, 22:].clone()

        # [v19.0] Path B: Manifold Mixup
        if mixup_alpha > 0:
            # Sample indices for 'Partner Ghosts'
            # [SOTA FIX] Use CPU for index sampling to avoid device mismatch
            idx2 = torch.randint(0, int(self.size), (num_ghosts,), generator=rng, device='cpu').to(self.raw_vitals.device)
            
            # [SOTA FIX]: Use CPU for mixup lambda sampling
            u = torch.rand((num_ghosts, 1), generator=rng, device='cpu')
            dist = torch.distributions.Beta(torch.tensor([mixup_alpha], device='cpu'), 
                                          torch.tensor([mixup_alpha], device='cpu'))
            lam = dist.icdf(u).to(self.raw_vitals.device)
            
            # Mix Latent Anchors and Labels (Clinical Continuity)
            out["anchors"] = lam * self.latent_anchors[idx1] + (1 - lam) * self.latent_anchors[idx2]
            out["labels"] = lam.squeeze(-1) * self.raw_labels[idx1].float() + (1 - lam).squeeze(-1) * self.raw_labels[idx2].float()
            out["uncertainties"] = lam * self.uncertainties[idx1] + (1 - lam) * self.uncertainties[idx2]

        # [v135.0 SOTA FIX] Ghost Latent Adapter (Smoking Gun #135)
        # Rationale: Historical anchors drift as the model trains. 
        # Fix: Soft-Align sampled anchors toward the current prototype EMA to maintain relevance.
        if self.latent_adapter_strength > 0 and self.prototype_ema.abs().sum() > 0:
            out["anchors"] = (1.0 - self.latent_adapter_strength) * out["anchors"] + \
                             self.latent_adapter_strength * self.prototype_ema

        return out

        return out

    def extra_repr(self) -> str:
        return f"capacity={self.capacity}, size={self.size.item()}, is_full={self.is_full.item()}"
