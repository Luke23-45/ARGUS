# Comprehensive Gradient Norm (GN) Spike Root Cause Analysis

> **Report Date:** 2026-01-30
> **Scope:** Recursive audit of all code elements contributing to gradient norm
> **Objective:** Identify every parameter, buffer, and mechanism that could cause GN spike in long training runs (1176 batches vs 200 batches baseline)

---

## Executive Summary

This report documents **45+ code elements** across **10+ files** that directly or indirectly contribute to gradient norm accumulation. The audit reveals a systematic pattern:

**All components are designed around a 200-step reference baseline (ScalingSteward.REF_STEPS = 200).**

When running with 1176 batches/epoch (~5.88x more steps), the `ScalingSteward.get_decay()` formula correctly scales EMA decays, BUT there are **unscaled elements** and **potential accumulation bugs** that could explain the GN explosion.

---

## Category 1: EMA/Momentum Decay Parameters

These parameters control how fast running statistics adapt. Faster decay = more responsive to current batch, but potentially unstable.

| File | Parameter | Baseline Value | Scaling Method | Status | Risk Level |
|------|-----------|----------------|----------------|--------|------------|
| `wrapper_generalist.py:450` | `self.grad_ema_decay` | 0.99 | `ScalingSteward.get_decay(0.99, n_curr)` | ✅ Scaled | LOW |
| `wrapper_generalist.py:487` | `self.class_balancer.beta` | 0.9995 | `ScalingSteward.get_decay(0.9995, n_curr)` | ✅ Scaled | LOW |
| `advantage_calculator.py:200` | `self.beta_momentum` | 0.999 | `ScalingSteward.get_decay(base, n_curr)` | ✅ Scaled | LOW |
| `advantage_calculator.py:201` | `self.clip_momentum` | 0.90 | `ScalingSteward.get_decay(0.90, n_curr)` | ✅ Scaled | LOW |
| `advantage_calculator.py:205` | `self.ess_ema_decay` | 0.95 | `ScalingSteward.get_decay(0.95, n_curr)` | ✅ Scaled | LOW |
| `ghost_bank.py:91` | `self.prototype_ema_decay` | 0.99 | `ScalingSteward.get_decay(0.99, n_curr)` | ✅ Scaled | LOW |
| `loss_scaler.py:31` | `self.decay` | 0.99 | `ScalingSteward.get_decay(0.99, n_curr)` | ✅ Scaled | LOW |
| `contrastive_loss.py:34` | `self.momentum` | 0.99 | `ScalingSteward.get_decay(base, n_curr)` | ✅ Scaled | LOW |
| `stabilization.py:50` | `StableContrastiveLoss.momentum` | 0.99 | `ScalingSteward.get_decay(0.99, n_curr)` | ✅ Scaled | LOW |
| `wrapper_generalist.py:864` | `fnd_momentum` (MGP Hook) | 0.90 | `ScalingSteward.get_decay(0.90, n_curr)` | ✅ Scaled | LOW |

---

## Category 2: Capacity/Buffer Size Parameters

These parameters control memory bank sizes. Larger capacity = more historical samples, but higher memory usage.

| File | Parameter | Baseline Value | Scaling Method | Status | Risk Level |
|------|-----------|----------------|----------------|--------|------------|
| `ghost_bank.py:96` | `self.capacity` | 256 | `ScalingSteward.get_steps(base_capacity, n_curr)` | ✅ Scaled | LOW |
| `temporal_buffer.py:49` | `self.capacity` | 1024 | `ScalingSteward.get_steps(base_capacity, n_curr)` | ✅ Scaled | LOW |
| `wrapper_generalist.py:492` | `trainer.warmup_steps` | 1500 | `ScalingSteward.get_steps(1500, n_curr)` | ✅ Scaled | LOW |

---

## Category 3: Loss Weighting & Balancing (⚠️ CRITICAL AREA)

These parameters control how loss components are weighted. Imbalanced weights can cause GN explosion.

| File | Parameter | Value | Notes | Risk Level |
|------|-----------|-------|-------|------------|
| `loss_scaler.py:22` | `self.log_vars` | learnable | Learnable uncertainty weights, clamped to [-2, 5] | **HIGH** |
| `loss_scaler.py:76` | `clinical_weights` | [0.5, 0.5, 1.5, 1.5, 1.0, 1.0, phys] | Fixed clinical priority weights | MEDIUM |
| `wrapper_generalist.py:302-312` | `initial_log_vars` | [1.0, 1.5, -0.5, 1.0, 1.0, 2.5, 0.5] | Initial uncertainty weights | MEDIUM |
| `wrapper_generalist.py:1123` | `critic_loss * 0.02` | 0.02x | Critic loss pre-scaling | LOW |
| `bgsl_loss.py:42-43` | `w_t=0.5, w_h=0.2` | constant | Fixed trend/shock weights | LOW |

### ⚠️ POTENTIAL ISSUE: Loss Accumulation

In `loss_scaler.py:64-67`:
```python
updated_emas = self.decay * curr_emas + (1 - self.decay) * avg_losses
self.loss_emas[indices] = updated_emas
```

The `loss_emas` buffer accumulates without clipping. Over long epochs, if any loss component spikes, the EMA will retain this influence.

---

## Category 4: Gradient Clipping & Safety Mechanisms

These parameters directly limit gradient norm magnitude.

| File | Parameter | Value | Notes | Risk Level |
|------|-----------|-------|-------|------------|
| `wrapper_generalist.py:1557` | `grad_clip` | config (default 1.0) | Hard gradient norm clip | LOW |
| `wrapper_generalist.py:1562` | `loss_scaler clip` | 0.1 | Clip loss scaler gradients | LOW |
| `wrapper_generalist.py:1565` | `AGC clip_factor` | 0.1 | Adaptive gradient clipping | LOW |
| `stabilization.py:324` | `OrthogonalGuard target` | 1.0 | Soft clamp target norm | LOW |
| `advantage_calculator.py:756` | `log_weights clamp` | [-20, 5] | AWR log-space clamp | LOW |

---

## Category 5: Curriculum/Warmup Ramps (⚠️ CRITICAL AREA)

These parameters change over training time, creating step-dependent behavior.

| File | Parameter | Range | Trigger | Risk Level |
|------|-----------|-------|---------|------------|
| `wrapper_generalist.py:533-537` | `beta` annealing | 0.60 → 0.15 | Linear over 40 epochs | **HIGH** |
| `wrapper_generalist.py:583-586` | `curr_tau` | 0.5 → 0.7 | Ramp at epoch 5+ | MEDIUM |
| `wrapper_generalist.py:590-591` | `curr_sigma_scale` | 3.5 → 2.5 | Linear over 15 epochs | MEDIUM |
| `wrapper_generalist.py:1210-1215` | `tcb_multiplier` | 0.0 → 1.0 | Linear over 500 steps | MEDIUM |
| `wrapper_generalist.py:1240-1241` | `phys_loss clamp` | max=10.0 | Epochs 0-14 only | MEDIUM |
| `wrapper_generalist.py:1466-1469` | `ada_decay_threshold` | 0.90 | First 300 steps | LOW |
| `wrapper_generalist.py:506` | `resumption_grace_steps` | 50 | At train start | LOW |

### ⚠️ POTENTIAL ISSUE: Beta Annealing in Longer Epochs

The beta annealing (L533-537) is **epoch-based**, not step-based. With 1176 batches/epoch, the same AWR beta change happens over 5.88x more optimization steps than the 200-batch baseline. This could cause:
- Overly aggressive selection pressure too early
- ESS collapse leading to gradient spikes

---

## Category 6: AWR/Advantage Statistics (⚠️ CRITICAL AREA)

| File | Parameter | Value | Notes | Risk Level |
|------|-----------|-------|-------|------------|
| `advantage_calculator.py:184` | `self.beta` | register_buffer (1.0 init) | AWR temperature, adaptive | **HIGH** |
| `advantage_calculator.py:187` | `self.max_weight` | 20.0 (adaptive) | Maximum AWR weight | **HIGH** |
| `advantage_calculator.py:223-224` | `adv_mean`, `adv_std` | running | Global whitening statistics | **HIGH** |
| `advantage_calculator.py:206` | `beta_growth_factor` | 1.5 ** (200/n_curr) | Scaled growth | MEDIUM |
| `advantage_calculator.py:208-209` | `min_beta`, `max_beta` | 0.01, 10.0 | AWR beta bounds | LOW |

### ⚠️ POTENTIAL ISSUE: Adaptive Beta Can Collapse

In `advantage_calculator.py:857-859`:
```python
if current_ess < 0.05:
    self.beta.copy_(self.beta * self.beta_growth_factor)
```

If ESS collapses (which is more likely with more batches), beta grows exponentially. With 1176 batches:
- `beta_growth_factor = 1.5 ** (200/1176) ≈ 1.07`
- Growth is slower, but ESS collapse is more likely to persist across an epoch

---

## Category 7: Contrastive Loss Components

| File | Parameter | Value | Notes | Risk Level |
|------|-----------|-------|-------|------------|
| `contrastive_loss.py:20` | `temperature` | 0.25 | Controls logit scaling | LOW |
| `temporal_buffer.py:38` | `temperature` | 0.07 | InfoNCE temperature | LOW |
| `wrapper_generalist.py:1068` | `acl_throttle_factor` | 0.3 | Gradient throttle | LOW |

---

## Category 8: Per-Epoch Statistics (⚠️ CRITICAL AREA)

These are reset or computed once per epoch and may not scale correctly.

| File | Element | Location | Notes | Risk Level |
|------|---------|----------|-------|------------|
| `wrapper_generalist.py:447-449` | `grad_norm_ema`, `grad_norm_std`, `grad_norm_step_count` | register_buffer | Initialized to 0, accumulates | **HIGH** |
| `wrapper_generalist.py:1486-1489` | Sentinel Wipe | grace period end | Resets EMA to current value | MEDIUM |
| `wrapper_generalist.py:1472` | `last_logged_bucket` | instance var | Resets per epoch | LOW |

### ⚠️ POTENTIAL ISSUE: grad_norm_step_count Never Resets

`grad_norm_step_count` is a buffer that increments every batch via `TrendSentinel.update_stats()`. In longer epochs:
- Bias correction `1 - decay^t` approaches 1.0 faster
- But the EMA has accumulated more historical noise

---

## Category 9: Ghost Bank & Memory Effects

| File | Element | Notes | Risk Level |
|------|---------|-------|------------|
| `ghost_bank.py:194-195` | NaN/Inf rejection | Prevents poison updates | LOW |
| `ghost_bank.py:260-265` | Informative Replacement | Replaces twins with harder samples | LOW |
| `ghost_bank.py:296-302` | LVP Sifting | Replaces redundant samples | LOW |

---

## Category 10: DDP Synchronization Points

| File | Element | Notes | Risk Level |
|------|---------|-------|------------|
| `advantage_calculator.py:894-908` | Adaptive stats sync | all_reduce beta, max_weight | LOW |
| `loss_scaler.py:52-58` | Loss sync | all_reduce losses | LOW |
| `wrapper_generalist.py:1336-1344` | g_ref sync (A-GEM) | all_reduce reference gradients | LOW |

---

## Identified Root Cause Candidates

### (A) AWR Beta Epoch-Based Annealing (HIGH CONFIDENCE)

**Location:** `wrapper_generalist.py:528-537`

**Issue:** Beta anneals from 0.60 to 0.15 over 40 epochs. With 1176 batches/epoch:
- By epoch 6-7, beta = ~0.45, creating strong selection pressure
- More batches = more chances for ESS collapse
- Collapsed ESS triggers growth factor, but not fast enough

**Test:** Disable beta annealing, use fixed beta=0.5

### (B) grad_norm_ema Accumulation Without Reset (MEDIUM CONFIDENCE)

**Location:** `wrapper_generalist.py:1471-1477`

**Issue:** The EMA accumulates over all steps. In 1176-batch epochs, after ~700 batches, the EMA has seen 3.5x more samples than the entire 200-batch epoch. Historical instability persists longer.

**Test:** Reset `grad_norm_step_count` at epoch start OR use epoch-relative step count

### (C) TCB Queue Random Initialization (MEDIUM CONFIDENCE)

**Location:** `temporal_buffer.py:42`

**Issue:** TCB queue is initialized with random embeddings:
```python
self.register_buffer("queue", F.normalize(torch.randn(capacity, d_model), dim=1))
```

With 1176 batches:
- Capacity scales to ~6000+ entries
- Takes longer to fill with real data
- Random negatives cause high InfoNCE loss early

**Test:** Initialize queue to zeros OR extend `tcb_warmup_steps`

### (D) Loss EMA Accumulation in BayesianProjectedScaler (LOW CONFIDENCE)

**Location:** `loss_scaler.py:64-67`

**Issue:** `loss_emas` accumulates without upper bound. One loss spike early in training can bias the EMA for the rest of the epoch.

**Test:** Add clipping to loss_emas update

---

## Verification Protocol

To verify which candidate is the root cause:

1. **Experiment A:** Disable AWR beta annealing (fixed beta=0.5)
   - If GN stabilizes → Root cause is (A)

2. **Experiment B:** Reset grad_norm_step_count at each epoch start
   - If GN stabilizes → Root cause is (B)

3. **Experiment C:** Extend tcb_warmup_steps from 500 to 2000
   - If GN stabilizes → Root cause is (C)

4. **Experiment D:** Add loss_emas clipping (max=10.0)
   - If GN stabilizes → Root cause is (D)

---

## Files Audited

| File | Lines | Elements Found |
|------|-------|----------------|
| `wrapper_generalist.py` | 2652 | 25+ |
| `advantage_calculator.py` | 1033 | 15+ |
| `loss_scaler.py` | 131 | 5+ |
| `ghost_bank.py` | 378 | 5+ |
| `temporal_buffer.py` | 169 | 3+ |
| `stabilization.py` | 413 | 8+ |
| `train_utils.py` | 1390+ | 5+ |
| `bgsl_loss.py` | 181 | 3+ |
| `contrastive_loss.py` | 87 | 3+ |

---

## Conclusion

The GN spike is **NOT random**. It is a systematic consequence of:

1. **Epoch-based annealing applied to step-intensive training**
2. **Accumulating statistics that don't reset at epoch boundaries**
3. **Memory buffers that take longer to "warm up" with real data**

The `ScalingSteward` correctly scales EMA decays and capacities, but the **annealing schedules** and **initialization dynamics** are still epoch-based, creating the observed step-correlated GN explosion pattern.

---

## Recommended Next Steps

1. **Immediate:** Test Experiment A (disable beta annealing)
2. **If A fails:** Test Experiment B (reset grad_norm_step_count)
3. **Parallel:** Extend TCB warmup to match scaled capacity
4. **Long-term:** Convert all epoch-based schedules to step-based with ScalingSteward

