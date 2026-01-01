### Recommended Patches for Upgrading the Auxiliary Head

Based on your architecture (DiT-based diffusion with a history encoder producing `ctx_seq` [B, T_hist+1, D] and `global_ctx` [B, D]), here are **three progressive patches** to make the aux head more sequence-aware. These are prioritized by impact/ease:

1. **Patch 1: CLS Token Approach** (Recommended starting point — highest expected gain, aligns with SOTA sepsis Transformers using full-sequence attention + CLS for classification).
2. **Patch 2: Dual Heads** (Pooled + Sequence-aware — safe hybrid, preserves your current global_ctx stability).
3. **Patch 3: Deeper Sequence-Aware Head** (Add more residual blocks on top of #1 or #2).

These changes target the bottleneck: transient sepsis signals lost in pooling. Recent sepsis models (2024–2025) consistently show **sequence-aware heads** (full Transformer over history + CLS or hierarchical attention) push AUROC from ~0.6–0.7 to **0.85–0.96** on MIMIC-like data.

#### Patch 1: CLS Token Sequence-Aware Head (Primary Recommendation)

Add a learnable **[CLS]** token prepended to `ctx_seq`. Process the full sequence with a small Transformer stack, then classify on the final CLS hidden state.

**Code Changes** (in `ICUUnifiedPlanner.__init__` and `forward`):

```python
# In __init__ (after existing aux_head if keeping it, or replace)
self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model))  # Learnable CLS

# Small sequence Transformer (reuse your EncoderBlock or DiTBlock1D style)
self.seq_aux_layers = nn.ModuleList([
    EncoderBlock(cfg) for _ in range(2)  # 2-4 layers; start with 2
])
self.seq_aux_norm = RMSNorm(cfg.d_model)

# Replace or add new head
self.aux_head = ClinicalResidualHead(cfg.d_model, cfg.num_phases, dropout=0.1)
# Or deeper: stack 2-3 ClinicalResidualHeads if needed
```

```python
# In forward() — after encoding: ctx_seq, global_ctx, ctx_mask
if self.cfg.use_auxiliary_head and "phase_label" in batch:
    # Prepend CLS token
    B = ctx_seq.shape[0]
    cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
    seq_with_cls = torch.cat([cls_tokens, ctx_seq], dim=1)  # [B, T+2, D]
    
    # Extend mask: CLS is always valid
    cls_valid = torch.zeros(B, 1, dtype=torch.bool, device=ctx_seq.device)
    full_seq_mask = torch.cat([cls_valid, ctx_mask], dim=1)
    
    # RoPE for extended sequence (offset=0 same as history)
    cos, sin = self.model.encoder.rope(seq_with_cls.shape[1], ctx_seq.device, offset=0)
    
    # Process sequence
    seq_repr = seq_with_cls
    for layer in self.seq_aux_layers:
        seq_repr = layer(seq_repr, cos, sin, full_seq_mask)
    seq_repr = self.seq_aux_norm(seq_repr)
    
    # CLS representation
    cls_repr = seq_repr[:, 0]  # [B, D]
    
    # Classification
    aux_logits = self.aux_head(cls_repr)
    
    # Loss (keep existing)
    aux_loss = F.cross_entropy(aux_logits, batch["phase_label"].long(), weight=w)
```

**Why this works best**: CLS learns to aggregate task-specific (sepsis phase) signals from the full history via self-attention. Outperforms pure pooling in time-series classification (especially transient events).

#### Patch 2: Dual Heads (Pooled + Sequence-Aware)

Keep your current global_ctx head + add a sequence-aware one → fuse logits.

```python
# In __init__
self.seq_aux_head = ClinicalResidualHead(cfg.d_model, cfg.num_phases, dropout=0.1)  # Separate or shared

# Optional: small seq transformer as in Patch 1, then mean-pool final seq_repr[:, 1:] (exclude static if needed)
```

```python
# In forward()
# Existing pooled logits
pooled_logits = self.aux_head(global_ctx)

# Sequence-aware (e.g., mean-pool after small transformer or directly on ctx_seq)
seq_pooled = ctx_seq.mean(dim=1)  # Simple start, or use Patch 1's seq_repr[:, 1:].mean(1)
seq_logits = self.seq_aux_head(seq_pooled)

# Fuse (average or learnable weight)
aux_logits = (pooled_logits + seq_logits) / 2
# Or: fused = self.fusion_linear(torch.cat([pooled_logits, seq_logits], dim=-1))
```

**Benefits**: Safe incremental improvement; sequence head captures transients while pooled retains stability.

#### Patch 3: Make Head Deeper (Complementary)

Stack multiple `ClinicalResidualHead` blocks.

```python
# In __init__
self.aux_head = nn.Sequential(
    ClinicalResidualHead(cfg.d_model, cfg.d_model),
    ClinicalResidualHead(cfg.d_model, cfg.d_model),
    ClinicalResidualHead(cfg.d_model, cfg.num_phases)  # Final projection
)
```

Apply after CLS or fused repr from above.

#### Additional Tips
- Increase `aux_loss_scale` to 1.0–5.0 during experiments.
- Add focal loss for imbalance: `F.cross_entropy(..., reduction='none')` + focal weighting.
- Monitor AUPRC (better than AUROC for rare sepsis phases).
- Expected: +0.15–0.30 AUROC gain toward 0.85+.

Start with **Patch 1** — it's the most direct path to modern sepsis benchmark performance while reusing your existing blocks. Let me know results or if you need refinements!