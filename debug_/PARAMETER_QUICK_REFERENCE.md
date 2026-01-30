# GN Spike Parameter Quick Reference

> Quick lookup table for all parameters affecting gradient norm.

## Parameters by Risk Level

### 🔴 HIGH RISK (Potential Root Causes)

| Parameter | File:Line | Current Value | Issue |
|-----------|-----------|---------------|-------|
| AWR beta annealing | `wrapper_generalist.py:528-537` | 0.60→0.15 over 40 epochs | Epoch-based, not step-based |
| `grad_norm_step_count` | `wrapper_generalist.py:449` | accumulates forever | Never resets at epoch |
| `self.beta` (AWR) | `advantage_calculator.py:184` | 1.0 init, adaptive | Can collapse ESS |
| TCB queue init | `temporal_buffer.py:42` | random randn | Takes longer to fill with real data |
| `loss_emas` | `loss_scaler.py:25` | ones(7) init | No upper bound clipping |

### 🟡 MEDIUM RISK (Monitor These)

| Parameter | File:Line | Value |
|-----------|-----------|-------|
| `curr_tau` | `wrapper_generalist.py:442` | 0.5→0.7 ramp |
| `curr_sigma_scale` | `wrapper_generalist.py:443` | 3.5→2.5 ramp |
| `tcb_warmup_steps` | `wrapper_generalist.py:1213` | 500 (fixed) |
| `max_weight` (AWR) | `advantage_calculator.py:187` | 20.0, adaptive |
| phys_loss clamp | `wrapper_generalist.py:1240` | max=10.0 (epochs 0-14) |

### 🟢 LOW RISK (Correctly Scaled)

| Parameter | File:Line | Scaling |
|-----------|-----------|---------|
| `grad_ema_decay` | `wrapper_generalist.py:450` | `ScalingSteward.get_decay()` |
| `beta_momentum` | `advantage_calculator.py:200` | `ScalingSteward.get_decay()` |
| ghost_bank capacity | `ghost_bank.py:96` | `ScalingSteward.get_steps()` |
| TCB capacity | `temporal_buffer.py:49` | `ScalingSteward.get_steps()` |
| warmup_steps | `wrapper_generalist.py:492` | `ScalingSteward.get_steps()` |

## Quick Experiments to Run

```bash
# Experiment A: Disable AWR beta annealing
# In wrapper_generalist.py, change line 537 to:
# curr_beta = 0.5  # Fixed, no annealing

# Experiment B: Reset step count each epoch
# Add at end of on_train_epoch_end():
# self.grad_norm_step_count.fill_(0)

# Experiment C: Extend TCB warmup
# In wrapper_generalist.py:1213, change:
# tcb_warmup_steps = 2000  # was 500

# Experiment D: Clip loss_emas
# In loss_scaler.py:66, add:
# updated_emas = updated_emas.clamp(max=10.0)
```

## ScalingSteward Reference

```python
# Located in train_utils.py:88-118
class ScalingSteward:
    REF_STEPS = 200  # Baseline from M_short

    @staticmethod
    def get_decay(ref_decay: float, n_curr: int) -> float:
        # v_curr = v_ref^(200/n_curr)
        return ref_decay ** (200 / n_curr)

    @staticmethod
    def get_steps(ref_steps: int, n_curr: int) -> int:
        # v_curr = v_ref * (n_curr/200)
        return int(ref_steps * (n_curr / 200))
```

**For 1176 batches/epoch:**
- `get_decay(0.99, 1176)` = 0.99^(200/1176) = **0.9983**
- `get_steps(500, 1176)` = 500*(1176/200) = **2940**
