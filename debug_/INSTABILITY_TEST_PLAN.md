# GN Instability Root Cause Isolation: Test Strategy

> **Objective:** Efficiently identify which of the 4 candidates causes gradient norm explosion.
> **Constraint:** Minimize compute time while maximizing diagnostic confidence.

---

## Recursive Analysis: What's the Best Approach?

### Option 1: Full Training Runs ❌
- Run 4 separate training experiments (each ~2-4 hours)
- **Pros:** Most realistic, catches all interactions
- **Cons:** ~8-16 hours GPU time, wasteful if first test fails
- **Verdict:** Too expensive for initial diagnosis

### Option 2: Pure Unit Tests ❌
- Write isolated tests for each component with mock data
- **Pros:** Fast, can run on CPU
- **Cons:** Misses component interactions, may not replicate real accumulation dynamics
- **Verdict:** Too disconnected from reality

### Option 3: Numerical Simulation (No GPU) ✅ **CHOSEN**
- Import the **actual classes** from the codebase
- Feed them **synthetic data** that mimics real training patterns
- Simulate 200 vs 1176 "steps" in seconds
- Compare internal states to detect divergence
- **Pros:** Fast (seconds), uses real code, catches accumulation bugs
- **Cons:** May miss GPU-specific edge cases (acceptable for this diagnosis)
- **Verdict:** Optimal balance of speed and realism

### Option 4: Mathematical Proof Only ❌
- Derive stability conditions analytically
- **Pros:** Instant, no code needed
- **Cons:** Doesn't prove implementation matches theory
- **Verdict:** Useful supplement, not primary method

---

## Chosen Strategy: Numerical Simulation

### Why This Works

The GN explosion happens because **something accumulates incorrectly over more steps**. We don't need GPU or real data to test this — we just need to:

1. Initialize the component
2. Feed it synthetic-but-plausible inputs for N steps
3. Observe if internal state diverges as N increases

### What We're Testing

| Candidate | Component | Observable | Pass Criterion |
|-----------|-----------|------------|----------------|
| (A) AWR Beta Annealing | `ICUAdvantageCalculator` | ESS trajectory | ESS stable above 0.05 |
| (B) grad_norm_step_count | `TrendSentinel` + buffers | EMA bias correction | Bias factor < 2.0 |
| (C) TCB Queue Init | `TemporalContrastiveBuffer` | InfoNCE loss trajectory | Loss decreases monotonically |
| (D) Loss EMA Accumulation | `BayesianProjectedScaler` | `loss_emas` values | No value > 10.0 |

---

## Test File Structure

```
debug_/tests/
├── __init__.py
├── conftest.py           # Shared fixtures (mock data generators)
├── test_a_awr_beta.py    # Candidate A: AWR beta annealing
├── test_b_grad_ema.py    # Candidate B: grad_norm_step_count
├── test_c_tcb_queue.py   # Candidate C: TCB queue initialization
├── test_d_loss_ema.py    # Candidate D: Loss EMA accumulation
└── run_all_tests.py      # Single entry point for all tests
```

---

## Test Specifications

### Test A: AWR Beta Annealing (`test_a_awr_beta.py`)

**Hypothesis:** Epoch-based beta annealing causes ESS collapse with more steps/epoch.

**Method:**
```python
from icu.utils.advantage_calculator import ICUAdvantageCalculator

def test_ess_stability():
    calc = ICUAdvantageCalculator(beta=0.6, adaptive_beta=True)
    
    # Simulate epoch with N_short=200 steps
    ess_short = simulate_epoch(calc, n_steps=200, epoch=5)
    
    # Reset and simulate with N_long=1176 steps BUT same beta schedule
    calc_long = ICUAdvantageCalculator(beta=0.6, adaptive_beta=True)
    ess_long = simulate_epoch(calc_long, n_steps=1176, epoch=5)
    
    # ESS should be similar if scaling is correct
    assert abs(ess_short[-1] - ess_long[-1]) < 0.10, "ESS diverges!"
```

**Key Insight:** If ESS in long run is <0.05 while short run is >0.15, this confirms the bug.

---

### Test B: Gradient EMA Step Count (`test_b_grad_ema.py`)

**Hypothesis:** `grad_norm_step_count` never resets, causing EMA bias to compound across epochs.

**Method:**
```python
from icu.utils.stabilization import TrendSentinel
import torch

def test_bias_correction():
    ema = torch.tensor(0.0)
    std = torch.tensor(0.0)
    step_count = torch.tensor(0)
    decay = 0.99
    
    # Simulate 3 epochs of 1176 steps each
    for epoch in range(3):
        for step in range(1176):
            TrendSentinel.update_stats(5.0, ema, std, decay, step_count)
        
        bias_factor = 1.0 - (decay ** step_count.item())
        # By end of epoch 3, step_count = 3528
        # bias_factor = 1 - 0.99^3528 ≈ 1.0 (fully corrected)
        
    # Key check: Is the EMA value reasonable?
    corrected_ema = ema.item() / bias_factor
    assert 4.0 < corrected_ema < 6.0, f"EMA drifted to {corrected_ema}"
```

**Key Insight:** If EMA value after 3 epochs differs drastically from expected (5.0), the step count is causing issues.

---

### Test C: TCB Queue Initialization (`test_c_tcb_queue.py`)

**Hypothesis:** Random queue initialization with large capacity causes high InfoNCE loss early in training.

**Method:**
```python
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer
import torch

def test_infonce_trajectory():
    tcb = TemporalContrastiveBuffer(d_model=256, capacity=1024)
    tcb.scale_dynamics(1176)  # Scales capacity to ~6000
    
    losses = []
    for step in range(500):
        q = torch.randn(32, 256)  # Fake queries
        k = torch.randn(32, 256)  # Fake keys
        out = tcb(q, k)
        losses.append(out['loss'].item())
    
    # Loss should decrease as queue fills with real data
    assert losses[-1] < losses[0] * 0.5, "InfoNCE not decreasing!"
    
    # Check queue fill rate
    expected_fill = min(500 * 32, tcb.capacity)  # ~6000
    actual_ptr = tcb.queue_ptr.item()
    # With capacity 6000 and 500*32=16000 samples, should wrap
```

**Key Insight:** If early InfoNCE loss is >5.0 and doesn't drop, queue init is a problem.

---

### Test D: Loss EMA Accumulation (`test_d_loss_ema.py`)

**Hypothesis:** `loss_emas` has no upper bound, allowing early spikes to persist.

**Method:**
```python
from icu.models.components.loss_scaler import BayesianProjectedScaler
import torch

def test_loss_ema_bounds():
    scaler = BayesianProjectedScaler(num_tasks=7)
    scaler.scale_dynamics(1176)
    
    # Simulate normal losses for 100 steps
    for _ in range(100):
        losses = torch.tensor([0.5, 0.3, 0.1, 0.2, 0.4, 0.1, 0.3])
        scaler(losses)
    
    # Inject a spike
    spike_losses = torch.tensor([50.0, 0.3, 0.1, 0.2, 0.4, 0.1, 0.3])
    scaler(spike_losses)
    
    # Continue normal training
    for _ in range(200):
        losses = torch.tensor([0.5, 0.3, 0.1, 0.2, 0.4, 0.1, 0.3])
        scaler(losses)
    
    # Check if spike influence persists
    ema_after = scaler.loss_emas[0].item()
    assert ema_after < 5.0, f"Spike persists: EMA={ema_after}"
```

**Key Insight:** If EMA of task 0 is >5.0 after 200 steps of normal losses, the spike persists too long.

---

## Execution Plan

| Phase | Action | Time Est. |
|-------|--------|-----------|
| 1 | Create test infrastructure (`conftest.py`, `run_all_tests.py`) | 5 min |
| 2 | Implement Test A (AWR beta) | 10 min |
| 3 | Run Test A → If FAIL → **ROOT CAUSE FOUND** | 1 min |
| 4 | If PASS → Implement Test B | 10 min |
| 5 | Run Test B → If FAIL → **ROOT CAUSE FOUND** | 1 min |
| 6 | Continue until root cause identified | ... |

**Total Estimated Time:** 30-60 minutes (vs 8-16 hours for full training)

---

## Key Advantage: Early Termination

Unlike full training runs, this approach allows **early termination**:
- If Test A fails → Fix A, skip B/C/D
- If Test A passes → Proceed to Test B
- etc.

This is the **genius optimization**: we don't test everything, we stop at the first failure.

---

## Success Criteria

A test **FAILS** if it detects step-count-dependent divergence. This is GOOD — it means we found the bug.

After identifying the failing test:
1. Implement the fix in the actual codebase
2. Re-run the same test to confirm fix works
3. Run full training to validate end-to-end

---

## Approval Required

Please confirm this plan before I proceed with implementation.

**Questions:**
1. Should I run tests in order (A→B→C→D) or all in parallel?
2. Do you have a preference for test framework (pytest vs raw scripts)?
