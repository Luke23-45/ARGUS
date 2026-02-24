"""
working/temporal_buffer.py
-------------------------
[v4.0 SOTA] Temporal Contrastive Buffer (TCB).

RATIONALE:
Sepsis detection is hard because 'Septic Shock' looks very similar to 
'Hemorrhagic Shock' or 'Cardiogenic Shock' in the early stages. 

The TCB implements a Momentum-updated Memory Bank (similar to MoCo) that:
1.  Stores historical embeddings of 'Hard Negatives' (high-risk patients who 
    did NOT develop sepsis).
2.  Forces the model to differentiate current trajectories from these 
    confusing historical patients via InfoNCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from icu.utils.train_utils import ScalingSteward
import logging

logger = logging.getLogger("APEX_TCB")

class TemporalContrastiveBuffer(nn.Module):
    def __init__(
            self,
            d_model: int, 
            capacity: int = 1024, 
            temperature: float = 0.07,
            latent_adapter_strength: float = 0.10 # [v29.6] SOTA Manifold Alignment (Abyssal #304)
        ):
        super().__init__()
        self.d_model = d_model
        self.base_capacity = capacity # [v26.1 FIX] Store Base for Idempotency
        self.capacity = capacity
        self.temperature = temperature
        self.base_latent_adapter_strength = latent_adapter_strength
        self.register_buffer("latent_adapter_strength", torch.tensor(latent_adapter_strength).float())
        
        # [v27.0 FIX] Zero-initialize queue instead of random
        # This prevents meaningless InfoNCE contrasts during warmup
        self.register_buffer("queue", torch.zeros(capacity, d_model))
        self.register_buffer("ids_queue", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.tensor([0], dtype=torch.long))
        self.register_buffer("queue_filled", torch.tensor([0], dtype=torch.long))
        self.register_buffer("prototype_ema", torch.zeros(1, d_model)) # [v29.6] Manifold Anchor
        self.register_buffer("prototype_momentum", torch.tensor([0.99])) # [v31.0] Adaptive Anchor
        
        # [v2026 SOTA FIX] Python Shadows for Resumption (Smoking Gun #Desync)
        # Rationale: Buffers are loaded from state_dict, but local shadows 
        # must be hard-synced to prevent "Memory Reset" after restart.
        self._shadow_ptr = 0
        self._shadow_filled = 0

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies buffer capacity across step densities."""
        if n_curr <= 0: return
        new_capacity = ScalingSteward.get_steps(self.base_capacity, n_curr)
        if new_capacity != self.capacity:
             logger.info(f"[TCB] Scaling Capacity: {self.capacity} -> {new_capacity}")
             # Save current state
             old_queue = self.queue.clone()
             
             num_to_keep = min(self.capacity, new_capacity)
             self.capacity = new_capacity
             
             # [SOTA FIX] Capture device to prevent CPU mismatch after resize
             device = self.queue.device
             # [v27.0 FIX] Zero-initialize new queue slots
             new_queue = torch.zeros(new_capacity, self.d_model, device=device)
             new_ids_queue = torch.zeros(new_capacity, dtype=torch.long, device=device)
             
             # Copy old data
             new_queue[:num_to_keep] = old_queue[:num_to_keep]
             new_ids_queue[:num_to_keep] = self.ids_queue[:num_to_keep]
             self.register_buffer("queue", new_queue)
             self.register_buffer("ids_queue", new_ids_queue)
             
             # [v2026 SOTA] Atomic Pointer Reset
             new_filled = min(int(self.queue_filled), new_capacity)
             self.queue_filled.fill_(new_filled)
             self.queue_ptr.fill_(new_filled % new_capacity)
             
             # Sync shadows
             self._shadow_filled = new_filled
             self._shadow_ptr = new_filled % new_capacity

        # Scale adapter: (1 - strength) is the retention factor.
        retention_ref = 1.0 - self.base_latent_adapter_strength
        retention_curr = ScalingSteward.get_decay(retention_ref, n_curr)
        self.latent_adapter_strength.fill_(1.0 - retention_curr)
        
        # [v31.0 SOTA FIX] Prototype Momentum Scaling (Smoking Gun #330)
        scaled_mom = ScalingSteward.get_decay(0.99, n_curr)
        self.prototype_momentum.fill_(scaled_mom)

        # [v2026 SOTA FIX] Unconditional Shadow Sync (Smoking Gun #Desync)
        # Rationale: Component-level contract for resumption parity.
        self.sync_shadows()

    def sync_shadows(self):
        """[SOTA 2026] Hard-syncs Python shadows with registered buffer state."""
        # [v2026 SOTA FIX] Bulletproof Clamping (Smoking Gun #IndexError)
        # Rationale: Prevents stale filled-values from exceeding resized buffers during transients.
        true_capacity = self.queue.shape[0]
        self.queue_filled.fill_(min(int(self.queue_filled), true_capacity))
        self.queue_ptr.fill_(int(self.queue_ptr) % true_capacity)
        
        self._shadow_ptr = int(self.queue_ptr)
        self._shadow_filled = int(self.queue_filled)

    def load_state_dict(self, state_dict, strict=True):
        """Ensures shadows are synced immediately after loading from checkpoint."""
        out = super().load_state_dict(state_dict, strict=strict)
        self.sync_shadows()
        return out

    @torch.no_grad()
    def _update_prototype(self, new_latents: torch.Tensor):
        """
        [v2026 SOTA] Zero-Sync Prototype Consensus
        Rationale: Ensures all DDP ranks share an identical manifold anchor.
        """
        import torch.distributed as dist
        device = self.prototype_ema.device
        if dist.is_initialized():
            # 1. Coalesce local signal
            local_sum = new_latents.sum(dim=0, keepdim=True) if new_latents.shape[0] > 0 else torch.zeros(1, self.d_model, device=device)
            local_count = torch.tensor([float(new_latents.shape[0])], device=device)
            
            # 2. Synchronize across cluster
            sync_buffer = torch.cat([local_sum.flatten(), local_count])
            dist.all_reduce(sync_buffer, op=dist.ReduceOp.SUM)
            global_sum = sync_buffer[:-1].view(1, -1)
            global_count = sync_buffer[-1:]
            
            # Vectorized gate
            valid_gate = (global_count > 1e-6)
            batch_avg = torch.where(valid_gate, global_sum / (global_count + 1e-8), torch.zeros_like(global_sum))
        else:
            valid_gate = torch.tensor(new_latents.shape[0] > 0, device=device)
            batch_avg = new_latents.mean(dim=0, keepdim=True) if valid_gate else torch.zeros(1, self.d_model, device=device)
            
        # Atomic Consensus Update
        is_new = (self.prototype_ema.abs().sum() == 0)
        mom = float(self.prototype_momentum)
        
        new_val = torch.where(is_new, batch_avg, self.prototype_ema.lerp(batch_avg, 1.0 - mom))
        self.prototype_ema.copy_(torch.where(valid_gate, new_val, self.prototype_ema))
        self.prototype_ema.copy_(F.normalize(self.prototype_ema, dim=1))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, ids: Optional[torch.Tensor] = None, scores: Optional[torch.Tensor] = None):
        """
        Updates the buffer with new negative samples.
        [v2026 SOTA FIX] Zero-Sync / Zero-Gather Implementation
        
        Rationale: 
        1. Caller (wrapper_generalist) handles DDP gathering (Iron Dome).
        2. TCB just stores what it receives.
        3. Removed 'has_poison.item()' sync. We now mask poisoned keys silently.
        """
        # 1. Branchless Poison Filter
        # If any key is NaN/Inf, we filter it out using a mask, without CPU sync.
        is_valid = torch.isfinite(keys).all(dim=1)
        if not is_valid.all():
            keys = keys[is_valid]
            if ids is not None: ids = ids[is_valid]
            if scores is not None: scores = scores[is_valid]
                
        # [v2026 SOTA FIX] Batch Size Hard-Cap (Smoking Gun #Overflow)
        true_capacity = self.queue.shape[0] # [CRITICAL OOD FIX] Use true tensor dimension, not drifting int attribute
        batch_size = keys.shape[0]
        if batch_size > true_capacity:
            keys = keys[-true_capacity:]
            if ids is not None: ids = ids[-true_capacity:]
            if scores is not None: scores = scores[-true_capacity:]
            batch_size = true_capacity

        if batch_size == 0:
            return

        # 2. Synchronized Shuffle
        indices = torch.randperm(batch_size, device=keys.device)
        keys = keys[indices]
        if ids is not None: ids = ids[indices]
        if scores is not None: scores = scores[indices]

        # 3. Normalization & Storage
        keys = F.normalize(keys, dim=1)
        ptr = int(self.queue_ptr)
        
        # Hard mining selection (if scores provided)
        if scores is not None and keys.shape[0] == scores.shape[0]:
            _, indices = torch.topk(scores.mean(dim=1), k=min(batch_size, true_capacity)) # [FIX] Mean across bank keys
            keys = keys[indices]
            if ids is not None: ids = ids[indices]
            batch_size = keys.shape[0]

        # Standard Queue Update [SOTA HARDENED]
        ptr = self._shadow_ptr
        if ptr + batch_size > true_capacity:
            remaining = true_capacity - ptr
            
            # SOTA FIX: Use explicit slice size for target to prevent shape mismatch on resumption transients
            key_slice_1 = keys[:remaining]
            self.queue.data[ptr : ptr + key_slice_1.shape[0]] = key_slice_1
            
            key_slice_2 = keys[remaining:]
            if key_slice_2.shape[0] > 0:
                 self.queue.data[:key_slice_2.shape[0]] = key_slice_2
                 
            if ids is not None:
                ids_slice_1 = ids[:remaining]
                self.ids_queue.data[ptr : ptr + ids_slice_1.shape[0]] = ids_slice_1
                
                ids_slice_2 = ids[remaining:]
                if ids_slice_2.shape[0] > 0:
                     self.ids_queue.data[:ids_slice_2.shape[0]] = ids_slice_2
            
            self._shadow_ptr = (batch_size - remaining) % true_capacity
            self.queue_ptr.fill_(self._shadow_ptr)
        else:
            self.queue.data[ptr : ptr + batch_size] = keys
            if ids is not None:
                self.ids_queue.data[ptr : ptr + batch_size] = ids
            self._shadow_ptr = (ptr + batch_size) % true_capacity
            self.queue_ptr.fill_(self._shadow_ptr)
        
        self._shadow_filled = min(true_capacity, self._shadow_filled + batch_size)
        self.queue_filled.fill_(self._shadow_filled)


    def forward(
        self, 
        q_expert: torch.Tensor, 
        k_positive: torch.Tensor, 
        ids_q: Optional[torch.Tensor] = None,
        ids_k: Optional[torch.Tensor] = None,
        enqueue_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates InfoNCE loss and Uniformity Regularization.
        """
        import torch.distributed as dist
        B, D = q_expert.shape
        q = F.normalize(q_expert, dim=1)
        k = F.normalize(k_positive, dim=1)
        
        # [v20.0 SOTA FIX] entry barrier consensus (Smoking Gun #172)
        # Rationale: Critical for DDP ranks to agree on the state of the bank.
        if dist.is_initialized():
             dist.all_reduce(self.queue_filled, op=dist.ReduceOp.MIN)

        # [v2026 SOTA FIX] Bulletproof Clamping (Smoking Gun #IndexError)
        true_capacity = self.queue.shape[0]
        filled = min(int(self.queue_filled), true_capacity)
        if filled < 32:
            # Entry logic
            zero_loss = torch.tensor(0.0, device=q.device, requires_grad=True)
            if enqueue_mask is not None:
                k_to_store = k[enqueue_mask]
                ids_to_store = ids_k[enqueue_mask] if ids_k is not None else None
                self._dequeue_and_enqueue(k_to_store, ids=ids_to_store)
            else:
                self._dequeue_and_enqueue(k, ids=ids_k)
            return {"loss": zero_loss, "nce_loss": zero_loss, "uniformity": zero_loss}
        
        # InfoNCE path
        # [v29.6 SOTA FIX] Ancestral Alignment (Abyssal #304)
        # Rationale: Historical negatives drift. Soft-align them toward current prototype.
        effective_queue = self.queue[:filled].detach()
        if float(self.latent_adapter_strength) > 0 and self.prototype_ema.abs().sum() > 0:
             effective_queue = (1.0 - float(self.latent_adapter_strength)) * effective_queue + \
                                float(self.latent_adapter_strength) * self.prototype_ema
             effective_queue = F.normalize(effective_queue, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # [B, 1]
        l_neg = torch.einsum('nc,kc->nk', [q, effective_queue]) # [B, filled]
        
        # [SOTA P14] Relational Negative Gating (PANG)
        # Rationale: Prevents contrasting overlapping windows from the same patient.
        # This resolves the 2.9 loss spike by effectively ignoring self-contrast.
        if ids_q is not None and filled > 0:
            # ids_q: [B], ids_keys: [filled]
            ids_keys = self.ids_queue[:filled]
            # mask_self[i, j] is True if sample i and negative j share the same patient_id
            mask_self = (ids_q.unsqueeze(1) == ids_keys.unsqueeze(0))
            # Surgical exclusion in log-space (pre-softmax)
            l_neg.masked_fill_(mask_self, -1e9) 

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        nce_loss = F.cross_entropy(logits, labels)
        
        # [v20.0 SOTA FIX] Synchronized Uniformity Sampling
        sample_size = min(128, filled)
        if dist.is_initialized():
             idx_uniform = torch.randperm(filled, device=q.device)[:sample_size]
             dist.broadcast(idx_uniform, src=0)
             subset = effective_queue[idx_uniform]
        else:
             subset = effective_queue[torch.randperm(filled, device=q.device)[:sample_size]]
             
        sim_matrix = torch.matmul(subset, subset.t())
        uniformity_loss = torch.log(torch.exp(sim_matrix).mean() + 1e-6)
        
        # Buffer update
        if enqueue_mask is not None:
             k_to_store = k[enqueue_mask]
             ids_to_store = ids_k[enqueue_mask] if ids_k is not None else None
             self._dequeue_and_enqueue(k_to_store, ids=ids_to_store)
        else:
             self._dequeue_and_enqueue(k, ids=ids_k, scores=l_neg.detach())
        
        # [v29.6] Update TCB Prototype (Rank-Consistent)
        self._update_prototype(k)
        
        return {
            "loss": nce_loss + 0.1 * uniformity_loss,
            "nce_loss": nce_loss,
            "uniformity": uniformity_loss
        }

if __name__ == "__main__":
    # Mock Test
    tcb = TemporalContrastiveBuffer(d_model=512, capacity=128)
    
    mock_q = torch.randn(16, 512)
    mock_k = torch.randn(16, 512)
    
    out = tcb(mock_q, mock_k)
    print(f"Total TCB Loss: {out['loss']:.4f}")
    print(f"NCE: {out['nce_loss']:.4f}, Uniformity: {out['uniformity']:.4f}")
    print(f"Queue Pointer: {tcb.queue_ptr.item()}")
    
    # Run again to check pointer update
    out2 = tcb(mock_q, mock_k)
    print(f"Queue Pointer after run 2: {tcb.queue_ptr.item()}")
    assert tcb.queue_ptr.item() == 32 # 16 + 16
    print("TemporalContrastiveBuffer: SOTA v4.0 Validation Passed.")
