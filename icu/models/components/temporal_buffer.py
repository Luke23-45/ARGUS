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
        self.latent_adapter_strength = latent_adapter_strength
        
        # [v27.0 FIX] Zero-initialize queue instead of random
        # This prevents meaningless InfoNCE contrasts during warmup
        self.register_buffer("queue", torch.zeros(capacity, d_model))
        self.register_buffer("queue_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("queue_filled", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prototype_ema", torch.zeros(1, d_model)) # [v29.6] Manifold Anchor
        self.register_buffer("prototype_momentum", torch.tensor(0.99)) # [v31.0] Adaptive Anchor

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies buffer capacity across step densities."""
        if n_curr <= 0: return
        
        new_capacity = ScalingSteward.get_steps(self.base_capacity, n_curr)
        if new_capacity != self.capacity:
             logger.info(f"⚡ [TCB] Scaling Capacity: {self.capacity} -> {new_capacity}")
             # Save current state
             old_queue = self.queue.clone()
             
             num_to_keep = min(self.capacity, new_capacity)
             self.capacity = new_capacity
             
             # [SOTA FIX] Capture device to prevent CPU mismatch after resize
             device = self.queue.device
             # [v27.0 FIX] Zero-initialize new queue slots
             new_queue = torch.zeros(new_capacity, self.d_model, device=device)
             
             # Copy old data
             new_queue[:num_to_keep] = old_queue[:num_to_keep]
             self.register_buffer("queue", new_queue)
             self.queue_filled.fill_(min(int(self.queue_filled), num_to_keep))

        # Scale adapter: (1 - strength) is the retention factor.
        retention_ref = 1.0 - self.base_latent_adapter_strength
        retention_curr = ScalingSteward.get_decay(retention_ref, n_curr)
        self.latent_adapter_strength = 1.0 - retention_curr
        
        # [v31.0 SOTA FIX] Prototype Momentum Scaling (Smoking Gun #330)
        # Rationale: Manifold anchoring must adapt at the same epoch-rate.
        scaled_mom = ScalingSteward.get_decay(0.99, n_curr)
        self.prototype_momentum.fill_(scaled_mom)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, scores: Optional[torch.Tensor] = None):
        """
        Updates the buffer with new negative samples.
        [v20.0 SOTA FIX] Global Memory Bank Parity (Smoking Gun #171)
        """
        # [v23.0 SOTA FIX] Synchronized Poison Check (Smoking Gun #241)
        # Rationale: All ranks MUST agree to return early or all ranks HANG.
        has_poison = torch.tensor([float(torch.isnan(keys).any() or torch.isinf(keys).any())], device=keys.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(has_poison, op=torch.distributed.ReduceOp.MAX)
        
        if has_poison.item() > 0:
            return

        import torch.distributed as dist
        
        # 1. Gather keys from all ranks to prevent bank divergence
        if dist.is_initialized():
             # Asymmetric All-Gather: Ranks may have different numbers of negatives
             local_b = torch.tensor([keys.shape[0]], device=keys.device)
             all_b = [torch.zeros(1, dtype=torch.long, device=keys.device) for _ in range(dist.get_world_size())]
             dist.all_gather(all_b, local_b)
             
             max_b = max(b.item() for b in all_b)
             if max_b > 0:
                 padded = torch.zeros(max_b, self.d_model, device=keys.device)
                 padded[:keys.shape[0]] = keys
                 gathered = [torch.zeros(max_b, self.d_model, device=keys.device) for _ in range(dist.get_world_size())]
                 dist.all_gather(gathered, padded)
                 
                 # Reconstruct global pool
                 pool = []
                 for i, b in enumerate(all_b):
                      if b.item() > 0:
                           pool.append(gathered[i][:b.item()])
                 keys = torch.cat(pool, dim=0)
             else:
                 return # Nothing to store on any rank
        
        # 2. Synchronized Shuffle (Smoking Gun #176)
        # Ensures all ranks pick the same subset if hard mining or queue limit is hit
        if keys.shape[0] > 1:
            if dist.is_initialized():
                indices = torch.randperm(keys.shape[0], device=keys.device)
                dist.broadcast(indices, src=0)
                keys = keys[indices]
            else:
                indices = torch.randperm(keys.shape[0], device=keys.device)
                keys = keys[indices]

        # 3. Normalization & Storage
        keys = F.normalize(keys, dim=1)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Hard mining selection (if scores provided)
        if scores is not None and keys.shape[0] == scores.shape[0]:
            # Rationale: Only use scores if they align with keys (local mode mostly)
            hard_scores = scores.mean(dim=1)
            _, indices = torch.topk(hard_scores, k=min(batch_size, self.capacity))
            keys = keys[indices]
            batch_size = keys.shape[0]

        # Standard Queue Update
        if ptr + batch_size > self.capacity:
            remaining = self.capacity - ptr
            self.queue.data[ptr:] = keys[:remaining]
            self.queue.data[:batch_size - remaining] = keys[remaining:]
            self.queue_ptr.fill_((batch_size - remaining) % self.capacity)
        else:
            self.queue.data[ptr : ptr + batch_size] = keys
            self.queue_ptr.fill_((ptr + batch_size) % self.capacity)
        
        new_filled = min(self.capacity, int(self.queue_filled) + batch_size)
        self.queue_filled.fill_(new_filled)

    def forward(
        self, 
        q_expert: torch.Tensor, 
        k_positive: torch.Tensor, 
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

        filled = int(self.queue_filled)
        if filled < 32:
            # Entry logic
            zero_loss = torch.tensor(0.0, device=q.device, requires_grad=True)
            if enqueue_mask is not None:
                k_to_store = k[enqueue_mask]
                self._dequeue_and_enqueue(k_to_store)
            else:
                self._dequeue_and_enqueue(k)
            return {"loss": zero_loss, "nce_loss": zero_loss, "uniformity": zero_loss}
        
        # InfoNCE path
        # [v29.6 SOTA FIX] Ancestral Alignment (Abyssal #304)
        # Rationale: Historical negatives drift. Soft-align them toward current prototype.
        effective_queue = self.queue[:filled].detach()
        if self.latent_adapter_strength > 0 and self.prototype_ema.abs().sum() > 0:
             effective_queue = (1.0 - self.latent_adapter_strength) * effective_queue + \
                               self.latent_adapter_strength * self.prototype_ema
             effective_queue = F.normalize(effective_queue, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # [B, 1]
        l_neg = torch.einsum('nc,kc->nk', [q, effective_queue]) # [B, filled]
        
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
             self._dequeue_and_enqueue(k_to_store)
        else:
             self._dequeue_and_enqueue(k, scores=l_neg.detach())
        
        # [v29.6] Update TCB Prototype
        with torch.no_grad():
             batch_avg = k.mean(dim=0, keepdim=True)
             if self.prototype_ema.abs().sum() == 0:
                  self.prototype_ema.copy_(batch_avg)
             else:
                  # [v31.0 SOTA FIX] Use scaled momentum for density-invariance
                  mom = float(self.prototype_momentum)
                  self.prototype_ema.mul_(mom).add_(batch_avg, alpha=1.0 - mom)
             self.prototype_ema.copy_(F.normalize(self.prototype_ema, dim=1))
        
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
