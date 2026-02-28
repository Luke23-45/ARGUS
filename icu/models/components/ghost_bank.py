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
        self.register_buffer("prototype_ema_decay", torch.tensor([prototype_ema_decay]).float())
        self.base_latent_adapter_strength = latent_adapter_strength 
        self.register_buffer("latent_adapter_strength", torch.tensor([latent_adapter_strength]).float())

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
        self.register_buffer("ptr", torch.tensor([0], dtype=torch.long))
        self.register_buffer("size", torch.tensor([0], dtype=torch.long))
        self.register_buffer("is_full", torch.tensor([False], dtype=torch.bool))
        
        # [v12.0 SOTA] CPU Shadows for Zero-Sync
        # Rationale: Prevents hot-path .item() syncs in update() and sample().
        self._shadow_ptr = 0
        self._shadow_size = 0
        self._shadow_is_full = False

    @torch.no_grad()
    def _update_prototype(self, new_latents: torch.Tensor, decay_override: Optional[float] = None):
        """
        [v2026 SOTA] Zero-Sync Prototype Consensus
        """
        device = self.prototype_ema.device
        if torch.distributed.is_initialized():
            # 1. Coalesce local signal
            local_sum = new_latents.sum(dim=0, keepdim=True) if new_latents.shape[0] > 0 else torch.zeros(1, self.latent_dim, device=device)
            local_count = torch.tensor([float(new_latents.shape[0])], device=device)
            
            # 2. Synchronize across cluster
            sync_buffer = torch.cat([local_sum.flatten(), local_count])
            torch.distributed.all_reduce(sync_buffer, op=torch.distributed.ReduceOp.SUM)
            global_sum = sync_buffer[:-1].view(1, -1)
            global_count = sync_buffer[-1:] # [1] 1D Tensor for DDP consensus
            
            # Vectorized gate
            valid_gate = (global_count > 1e-6)
            batch_avg = torch.where(valid_gate, global_sum / (global_count + 1e-8), torch.zeros_like(global_sum))
        else:
            # [v2026 SOTA FIX] valid_gate must be a Tensor for torch.where() 
            # (Python bool causes TypeError: "got (bool, Tensor, Tensor)")
            valid_gate = torch.tensor(new_latents.shape[0] > 0, device=device)
            batch_avg = new_latents.mean(dim=0, keepdim=True) if valid_gate else torch.zeros(1, self.latent_dim, device=device)
            
        # [v161.0 SOTA FIX] Atomic Update (Zero-Sync)
        # Rationale: Replaced if branches with lerp_ and finite-checks.
        # we still use the valid_gate if it's a scalar or tensor to avoid update
        # but to avoid host-sync, we use it inside torch.where or as a weight.
        
        is_new = (self.prototype_ema.abs().sum() == 0) # This is a 0D boolean tensor
        eff_decay = torch.as_tensor(decay_override if decay_override is not None else self.prototype_ema_decay, device=device)
        
        # [v2026 SOTA] Branchless Consensus Update
        # 1. Calculate the potential new EMA (hard init vs lerp)
        new_val = torch.where(is_new, batch_avg, self.prototype_ema.lerp(batch_avg, 1.0 - eff_decay))
        
        # 2. Only apply if the batch_avg and gate is valid (Zero-Sync)
        # valid_gate is a tensor (0D) from line 98
        self.prototype_ema.copy_(torch.where(valid_gate, new_val, self.prototype_ema))
        
        # 3. Rescale to unit hypersphere with epsilon stability
        self.prototype_ema.copy_(F.normalize(self.prototype_ema, dim=1, eps=1e-8))

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies bank capacity and decay across step densities."""
        if n_curr <= 0: return
        
        self.prototype_ema_decay.fill_(ScalingSteward.get_decay(0.99, n_curr))
        # Adapt mix rate: (1 - strength) is the retention factor.
        retention_ref = 1.0 - self.base_latent_adapter_strength
        retention_curr = ScalingSteward.get_decay(retention_ref, n_curr)
        self.latent_adapter_strength.fill_(1.0 - retention_curr)
        
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
            
            # Sync shadows
            self._shadow_size = new_size
            self._shadow_ptr = new_size % new_capacity
            self._shadow_is_full = (new_size == new_capacity)
            
        # [v2026 SOTA FIX] Unconditional Shadow Sync (Smoking Gun #Desync)
        # Rationale: Component-level contract for resumption parity.
        self.sync_shadows()

    def sync_shadows(self):
        """[SOTA 2026] Hard-syncs Python shadows with registered buffer state."""
        # [v2026 SOTA FIX] Bulletproof Clamping (Smoking Gun #IndexError)
        true_capacity = self.raw_vitals.shape[0]
        self.size.fill_(min(int(self.size), true_capacity))
        self.ptr.fill_(int(self.ptr) % true_capacity)
        
        self._shadow_size = int(self.size)
        self._shadow_ptr = int(self.ptr)
        self._shadow_is_full = (self._shadow_size == true_capacity)
        self.is_full.fill_(self._shadow_is_full)

    def load_state_dict(self, state_dict, strict=True):
        """Ensures shadows are synced immediately after loading from checkpoint."""
        out = super().load_state_dict(state_dict, strict=strict)
        self.sync_shadows()
        return out

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
        [v2026 SOTA] Atomic Vectorized Update (Zero-Sync)
        """
        device = vitals.device
        if active_mask is not None:
            vitals = vitals[active_mask]
            masks = masks[active_mask]
            labels = labels[active_mask]
            latents = latents[active_mask]
            if uncertainties is not None: uncertainties = uncertainties[active_mask].detach()
        elif uncertainties is not None:
            uncertainties = uncertainties.detach()
        
        # Guard against NaNs/Infs (Zero-Sync)
        is_finite = torch.isfinite(latents).all(dim=1)
        vitals = vitals[is_finite]
        masks = masks[is_finite]
        labels = labels[is_finite]
        latents = latents[is_finite]
        if uncertainties is not None: uncertainties = uncertainties[is_finite]

        # [v2026 SOTA FIX] Batch Size Hard-Cap (Smoking Gun #Overflow)
        # Rationale: Capacity scales with steps/epoch. If steps/epoch is low,
        # batch_size might exceed bank capacity.
        vitals = vitals[:self.capacity]
        masks = masks[:self.capacity]
        labels = labels[:self.capacity]
        latents = latents[:self.capacity]
        if uncertainties is not None: uncertainties = uncertainties[:self.capacity]
        
        B = vitals.shape[0]
        
        # [v2026 SOTA FIX] DDP-Consensus Prototype Update (Abyssal #3.4)
        # Rationale: Regardless of whether local B == 0, ALL ranks must call
        # _update_prototype as it contains a DDP all_reduce. 
        # Skipping this on rank-varying batch sizes (e.g. epoch tail) causes deadlock.
        eff_decay = 0.1 if prototype_burst else None
        self._update_prototype(latents, decay_override=eff_decay)
        
        if B == 0: return

        # Intra-Batch Filtering
        with torch.amp.autocast('cuda', enabled=False):
            norm_b = F.normalize(latents, dim=1).half()
            b_self_sim = torch.matmul(norm_b, norm_b.T).float()
            b_self_sim.fill_diagonal_(0)
            
        similar_to_prev = (torch.tril(b_self_sim, diagonal=-1).max(dim=1).values > 0.99)
        vitals = vitals[~similar_to_prev]
        masks = masks[~similar_to_prev]
        labels = labels[~similar_to_prev]
        latents = latents[~similar_to_prev]
        if uncertainties is not None: uncertainties = uncertainties[~similar_to_prev]
        else: uncertainties = torch.zeros(vitals.shape[0], 1, device=device)
        B = vitals.shape[0]


        # Vectorized Global Check
        # Rationale: Using the shadow variable for the size check to avoid sync.
        current_size_val = self._shadow_size
        
        if current_size_val > 0:
            with torch.amp.autocast('cuda', enabled=False):
                norm_new = F.normalize(latents, dim=1).half()
                norm_old = F.normalize(self.latent_anchors[:current_size_val], dim=1).half()
                sim_matrix = torch.matmul(norm_new, norm_old.T).float()
                max_sim, twin_idx = sim_matrix.max(dim=1)
                
            is_redundant = (max_sim > self.similarity_threshold)
            target_unc = self.uncertainties[twin_idx].flatten()
            is_harder = (uncertainties.flatten() > (target_unc * 1.1))
            to_replace = (is_redundant & is_harder)
            is_diverse = ~is_redundant
            
            # Informative Replacement (Zero-Sync)
            # Rationale: Slicing with a boolean mask handles empty cases without host-side 'if' sync.
            r_idx = twin_idx[to_replace]
            self.raw_vitals[r_idx] = vitals[to_replace]
            self.raw_masks[r_idx] = masks[to_replace]
            self.raw_labels[r_idx] = labels[to_replace]
            self.latent_anchors[r_idx] = latents[to_replace]
            self.uncertainties[r_idx] = uncertainties[to_replace]
        else:
            to_replace = torch.zeros(B, dtype=torch.bool, device=device)
            is_diverse = torch.ones(B, dtype=torch.bool, device=device)
            twin_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Diverse Expansion
        dv, dm, dl, dlat, dunc = vitals[is_diverse], masks[is_diverse], labels[is_diverse], latents[is_diverse], uncertainties[is_diverse]
        num_div = dv.shape[0]
        available = self.capacity - self._shadow_size
        num_fill = min(num_div, available)
        
        if num_fill > 0:
            indices = (torch.arange(num_fill, device=device) + self._shadow_ptr) % self.capacity
            self.raw_vitals[indices] = dv[:num_fill]
            self.raw_masks[indices] = dm[:num_fill]
            self.raw_labels[indices] = dl[:num_fill]
            self.latent_anchors[indices] = dlat[:num_fill]
            self.uncertainties[indices] = dunc[:num_fill]
            
            # Sync Buffers (Zero-Sync)
            self.ptr.fill_((self._shadow_ptr + num_fill) % self.capacity)
            self.size.fill_(min(self._shadow_size + num_fill, self.capacity))
            
            # Sync Shadows 
            self._shadow_ptr = (self._shadow_ptr + num_fill) % self.capacity
            self._shadow_size = min(self._shadow_size + num_fill, self.capacity)
            if self._shadow_size == self.capacity: 
                self.is_full.fill_(True)
                self._shadow_is_full = True

        # Overflow (LVP Replacement)
        num_lvp = (num_div - num_fill)
        if num_lvp > 0 and self._shadow_is_full:
            with torch.amp.autocast('cuda', enabled=False):
                lat_all = F.normalize(self.latent_anchors, dim=1).half()
                K = torch.matmul(lat_all, lat_all.T).float()
                redundancy = (K ** 2).sum(dim=1) - 1.0
            lvp_scores = redundancy / (self.uncertainties.flatten() + 1e-6)
            _, lvp_indices = torch.topk(lvp_scores, num_lvp)
            
            self.raw_vitals[lvp_indices] = dv[num_fill:]
            self.raw_masks[lvp_indices] = dm[num_fill:]
            self.raw_labels[lvp_indices] = dl[num_fill:]
            self.latent_anchors[lvp_indices] = dlat[num_fill:]
            self.uncertainties[lvp_indices] = dunc[num_fill:]

    @torch.no_grad()
    def refresh_anchors(self, encoder: nn.Module, decay: float = 0.0):
        """
        [v33.1 SOTA FIX] Anchor Refresh mechanism (Fix #356).
        Re-encodes all stored trajectories to prevent latent representation drift.
        
        Args:
            encoder: The encoder module (or callable) to use.
            decay: Momentum decay rate (0.0 = Hard Refresh, 0.9 = Soft Update).
                   Higher values (e.g. 0.9) retain more history, reducing gradient shock.
        """
        if self._shadow_size == 0:
            return
            
        # Batched Refresh for VRAM efficiency
        batch_size = 64
        num_iters = (self._shadow_size + batch_size - 1) // batch_size
        
        # [v33.1] Force Evaluation Mode for deterministic encoding
        # [v42.0 SOTA FIX] Polymorphic Support (Module vs Function)
        was_training = None
        if isinstance(encoder, torch.nn.Module):
             was_training = encoder.training
             encoder.eval()
        
        # [v42.1 SOTA Optimization] Single-Source Truth
        # Rationale: Only Rank 0 performs the refresh. Others wait for broadcast.
        # [v14.2] DDP Barrier: Ensure all ranks are synchronized before Rank 0 starts
        # to prevent reading partially updated buffers.
        if torch.distributed.is_initialized():
             torch.distributed.barrier()
        
        is_rank_zero = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        
        if is_rank_zero:
            try:
                for i in range(num_iters):
                    start = i * batch_size
                    end = min(start + batch_size, self._shadow_size)
                    
                    v_batch = self.raw_vitals[start:end]
                    m_batch = self.raw_masks[start:end]
                    
                    # Re-encode using CURRENT encoder weights
                    new_anchors = encoder(v_batch, m_batch)
                    
                    # [SOTA TITANIUM FIX] NaN-Proof Sanitization
                    # Rationale: torch.clamp is a no-op for NaN. nan_to_num + isfinite
                    # ensures toxicity is physically purged from the bank.
                    new_anchors = torch.nan_to_num(new_anchors, nan=0.0, posinf=2.0, neginf=-2.0)
                    new_anchors = new_anchors.clamp(min=-2.0, max=2.0)
                    
                    new_norm = F.normalize(new_anchors, dim=1)
                    
                    # [SOTA FIX v33.2] Momentum Stabilization (Ghost Drift Patch)
                    if decay > 0:
                        # Soft Update: old = decay * old + (1-decay) * new
                        self.latent_anchors[start:end].mul_(decay).add_(new_norm, alpha=1.0 - decay)
                        # Re-normalize to ensure we stay on the hypersphere
                        self.latent_anchors[start:end].copy_(F.normalize(self.latent_anchors[start:end], dim=1))
                    else:
                        # Hard Refresh (Legacy Behavior)
                        self.latent_anchors[start:end].copy_(new_norm)
                    
                    # [v2026 RAM SPIKE FIX] Per-batch memory cleanup (Smoking Gun #RAM-04)
                    # Rationale: Free activations immediately to prevent accumulation across
                    # the 24+ iterations, which can cause ~500MB RAM spike.
                    del new_anchors, new_norm, v_batch, m_batch
            finally:
                pass
        
        # Restore training state (All Ranks must do this!)
        if was_training is not None:
            encoder.train(was_training)
                
        # Re-initialize prototype to match new latent space
        if self.size > 0:
            self.prototype_ema.fill_(0.0)
            self._update_prototype(self.latent_anchors[:self.size])

        # [v36.2 SOTA FIX] DDP Consensus Broadcast (The "One Bank" Protocol)
        # Rationale: Stochastic accumulations in 'update' function cause Bank Divergence 
        # across ranks (Rank 0 has different ghosts than Rank 1). 
        # This causes 'Gradient Conflict' where ranks pull the model in opposing directions,
        # leading to the 'U-Shape' Regression (GMSE 500->612).
        # Optimization: Only Rank 0 re-encodes, then broadcasts. Saves (N-1)x compute.
        if torch.distributed.is_initialized():
            self._broadcast_bank_state(src_rank=0)

    def _broadcast_bank_state(self, src_rank: int = 0):
        """
        [SOTA 2026] Hard Synchronization of the entire Bank state.
        Ensures all ranks possess bit-exact identical buffers.
        """
        # 1. Metadata
        torch.distributed.broadcast(self.ptr, src_rank)
        torch.distributed.broadcast(self.size, src_rank)
        torch.distributed.broadcast(self.is_full, src_rank)
        
        # 2. Heavy Data (Only strict necessary range if possible, but full buffer is safer)
        # 600KB broadcast is negligible on A100/H100/Consumer Cluster.
        torch.distributed.broadcast(self.raw_vitals, src_rank)
        torch.distributed.broadcast(self.raw_masks, src_rank)
        torch.distributed.broadcast(self.raw_labels, src_rank)
        torch.distributed.broadcast(self.latent_anchors, src_rank)
        torch.distributed.broadcast(self.uncertainties, src_rank)
        torch.distributed.broadcast(self.prototype_ema, src_rank)

    def sample(self, num_ghosts: int, seed: int, mixup_alpha: float = 0.0, uncertainty_weighted: bool = False) -> Dict[str, torch.Tensor]:
        """
        [v2026 SOTA] Zero-Sync Prioritized Sampling
        """
        device = self.raw_vitals.device
        
        # Fallback Logic (DDP-Safe Zero-Sync)
        # Rationale: Uses float masking to return zeros if bank is empty.
        is_empty = (self.size == 0)
        
        # Deterministic RNG
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        
        # [v2026 SOTA] Vectorized Random Indices
        # Rationale: Replaced int(size) and cpu() copies with tensor scaling.
        # Note: We sample from full capacity and clamp/mod to valid range.
        u = torch.rand(num_ghosts, generator=rng, device='cpu').to(device)
        idx1 = (u * self.size.float()).long()
        
        # Prioritized Sampling Path
        if uncertainty_weighted:
            # 5% pure uniform noise to ensure exploration
            # Use current_size to prevent out-of-bounds multinomial
            current_size = self.size.clamp(min=1) 
            logits = self.uncertainties[:current_size].flatten() / 0.1
            probs_prioritized = torch.softmax(logits, dim=0)
            probs_uniform = torch.ones_like(probs_prioritized) / current_size
            probs = 0.95 * probs_prioritized + 0.05 * probs_uniform
            
            # [v2026 SOTA] DDP Parity for Multinomial
            # Multinomial requires CPU tensors to properly use the CPU generator across all PyTorch versions.
            idx1 = torch.multinomial(probs.cpu(), num_ghosts, replacement=True, generator=rng).to(device)
            
        out = {
            "vitals": self.raw_vitals[idx1],
            "masks": self.raw_masks[idx1],
            "labels": self.raw_labels[idx1].float(),
            "anchors": self.latent_anchors[idx1],
            "uncertainties": self.uncertainties[idx1],
            "valid": (~is_empty).expand(num_ghosts).clone()
        }
        
        # Zero out if empty
        if is_empty:
             for k in out: out[k] = torch.zeros_like(out[k])
        
        out["static"] = out["vitals"][:, 0, 22:].clone()

        # Manifold Mixup
        if mixup_alpha > 0 and not is_empty:
            u2 = torch.rand(num_ghosts, generator=rng, device='cpu').to(device)
            idx2 = (u2 * self.size.float()).long()
            
            # [v2026 SOTA] Consistent DDP Beta Sampling
            # Beta distribution does not accept generator. We fork and seed manually to ensure DDP parity.
            lam_seed = int(torch.randint(0, 1000000, (1,), generator=rng).item())
            
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                torch.manual_seed(lam_seed)
                dist = torch.distributions.Beta(torch.tensor([mixup_alpha], device=device), 
                                              torch.tensor([mixup_alpha], device=device))
                lam = dist.sample((num_ghosts, 1))
            
            out["anchors"] = lam * self.latent_anchors[idx1] + (1 - lam) * self.latent_anchors[idx2]
            # [NASA-Tier v1.2 FIX] Avoid [N, 1] x [N] -> [N, N] broadcasting catastrophe (Smoking Gun #5)
            out["labels"] = lam.flatten() * self.raw_labels[idx1].float() + (1 - lam).flatten() * self.raw_labels[idx2].float()
            out["uncertainties"] = lam * self.uncertainties[idx1] + (1 - lam) * self.uncertainties[idx2]

        # Soft-Align sampled anchors toward the current prototype EMA
        if (self.latent_adapter_strength > 0) and not is_empty:
            out["anchors"] = (1.0 - self.latent_adapter_strength) * out["anchors"] + \
                             self.latent_adapter_strength * self.prototype_ema

        return out

    def extra_repr(self) -> str:
        return f"capacity={self.capacity}, size={self.size.item()}, is_full={self.is_full.item()}"
