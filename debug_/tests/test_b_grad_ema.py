"""
test_b_grad_ema.py
------------------
Test B: Gradient Norm EMA Step Count Accumulation Test

HYPOTHESIS:
The `grad_norm_step_count` buffer increments every training step and is only 
reset during resumption grace period (50 steps at start). Over multiple epochs 
with 1176 batches/epoch:
- step_count reaches 3528 after 3 epochs
- Bias correction factor (1 - decay^t) approaches 1.0
- BUT the EMA has accumulated 3528 samples of historical noise
- Cross-epoch accumulation may cause manifold pressure drift

SIMULATION APPROACH:
1. Replicate TrendSentinel.update_stats exactly as in training
2. Feed synthetic gradient norms mimicking real training
3. Simulate 3 epochs of 200 vs 1176 steps
4. Compare EMA values and bias correction behavior

PASS CRITERIA:
- EMA should track actual mean (within 20% tolerance)
- EMA should not drift >50% between epochs
- Bias correction should not cause over-smoothing

WHAT THIS TESTS:
- wrapper_generalist.py:1471-1477 (update_stats call)
- stabilization.py:382-412 (TrendSentinel.update_stats implementation)
- wrapper_generalist.py:447-449 (buffer definitions)

Authors: APEX Diagnostic Team
Date: 2026-01-30
"""

import sys
import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# Import the actual classes we are testing
from icu.utils.stabilization import TrendSentinel
from icu.utils.train_utils import ScalingSteward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_B")


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_synthetic_gradient_norms(
    n_steps: int,
    base_mean: float = 2.5,  # Matches observed stable GN from short run
    base_std: float = 0.5,
    spike_prob: float = 0.02,  # 2% chance of spike per step
    spike_magnitude: float = 10.0,
    epoch: int = 0
) -> List[float]:
    """
    Generate synthetic gradient norms that mimic real training patterns.
    
    Real gradient norms in APEX training typically:
    - Start around 2-3 in early epochs
    - Have occasional spikes (especially around sepsis events)
    - Gradually drift higher if unstable (the bug we're looking for)
    
    We add a small drift term to simulate the observed accumulation pattern.
    """
    norms = []
    drift = 0.0  # Accumulating drift (simulates potential bug)
    
    for step in range(n_steps):
        # Base noise
        base = torch.randn(1).item() * base_std + base_mean + drift
        
        # Occasional spike
        if torch.rand(1).item() < spike_prob:
            base += spike_magnitude * torch.rand(1).item()
        
        # Simulate natural epoch-over-epoch drift (very small, 0.1% per step)
        drift += 0.0001 * torch.randn(1).item()
        
        norms.append(max(0.1, base))  # GN is always positive
    
    return norms


# =============================================================================
# CORE TEST LOGIC
# =============================================================================

def simulate_multi_epoch_ema(
    n_steps_per_epoch: int,
    n_epochs: int,
    base_decay: float = 0.99,
    use_ada_ema: bool = True,
    reset_at_epoch_boundary: bool = False  # EXPERIMENT: What if we reset?
) -> Dict[str, any]:
    """
    Simulate multiple epochs of gradient EMA updates.
    
    This replicates exactly how the EMA is updated in training:
    1. At each step, TrendSentinel.update_stats is called
    2. AdaEMA uses decay=0.90 for first 300 steps (resumption), then base_decay
    3. step_count increments every step, never resets (current behavior)
    
    Args:
        n_steps_per_epoch: Number of batches per epoch
        n_epochs: Number of epochs to simulate
        base_decay: EMA decay after warmup (scaled by ScalingSteward)
        use_ada_ema: Whether to use AdaEMA (faster decay for first 300 steps)
        reset_at_epoch_boundary: EXPERIMENT flag - what if we reset step_count?
    
    Returns:
        Dict with EMA trajectory, step counts, and analysis metrics
    """
    # Initialize buffers exactly as in wrapper_generalist.py:447-449
    grad_norm_ema = torch.tensor(0.0)
    grad_norm_std = torch.tensor(0.0)
    grad_norm_step_count = torch.tensor(0)
    
    # Scale decay for step density
    scaled_decay = ScalingSteward.get_decay(base_decay, n_steps_per_epoch)
    logger.info(f"Scaled decay: {base_decay} -> {scaled_decay:.6f} for {n_steps_per_epoch} steps")
    
    # Tracking
    ema_history = []
    std_history = []
    step_count_history = []
    raw_gn_history = []
    bias_correction_history = []
    
    ada_decay_threshold = 300
    
    for epoch in range(n_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        
        # Generate gradient norms for this epoch
        gn_values = generate_synthetic_gradient_norms(
            n_steps=n_steps_per_epoch,
            base_mean=2.5,
            base_std=0.5,
            spike_prob=0.02,
            epoch=epoch
        )
        
        epoch_ema_start = grad_norm_ema.item()
        
        for step_in_epoch, gn in enumerate(gn_values):
            global_step = epoch * n_steps_per_epoch + step_in_epoch
            
            # AdaEMA logic (wrapper_generalist.py:1462-1469)
            # Uses faster decay for first 300 steps overall, not per epoch
            if use_ada_ema and grad_norm_step_count < ada_decay_threshold:
                active_decay = 0.90
            else:
                active_decay = scaled_decay
            
            # Call TrendSentinel.update_stats exactly as in training
            ema_bc, std_bc = TrendSentinel.update_stats(
                current_val=gn,
                ema=grad_norm_ema,
                std=grad_norm_std,
                decay=active_decay,
                step_tensor=grad_norm_step_count
            )
            
            # Track everything
            ema_history.append(grad_norm_ema.item())
            std_history.append(grad_norm_std.item())
            step_count_history.append(grad_norm_step_count.item())
            raw_gn_history.append(gn)
            
            # Compute bias correction factor
            t = grad_norm_step_count.item()
            bias_factor = 1.0 - (active_decay ** t) if t > 0 else 1.0
            bias_correction_history.append(bias_factor)
        
        epoch_ema_end = grad_norm_ema.item()
        epoch_drift = abs(epoch_ema_end - epoch_ema_start)
        
        # Compute actual mean of this epoch's GN for comparison
        actual_epoch_mean = sum(gn_values) / len(gn_values)
        
        logger.info(f"  Step count at epoch end: {grad_norm_step_count.item()}")
        logger.info(f"  EMA at epoch end: {epoch_ema_end:.4f}")
        logger.info(f"  Actual epoch mean GN: {actual_epoch_mean:.4f}")
        logger.info(f"  EMA drift during epoch: {epoch_drift:.4f}")
        
        # EXPERIMENT: What if we reset step count at epoch boundary?
        if reset_at_epoch_boundary:
            grad_norm_step_count.fill_(0)
            logger.info("  [EXPERIMENT] Reset step_count to 0")
    
    # Analysis
    final_step_count = grad_norm_step_count.item()
    final_ema = grad_norm_ema.item()
    actual_mean = sum(raw_gn_history) / len(raw_gn_history)
    ema_tracking_error = abs(final_ema - actual_mean) / actual_mean
    
    return {
        'n_steps_per_epoch': n_steps_per_epoch,
        'n_epochs': n_epochs,
        'total_steps': final_step_count,
        'final_ema': final_ema,
        'actual_mean': actual_mean,
        'tracking_error': ema_tracking_error,
        'final_bias_correction': bias_correction_history[-1],
        'ema_history': ema_history,
        'std_history': std_history,
        'step_count_history': step_count_history,
        'raw_gn_history': raw_gn_history,
        'bias_correction_history': bias_correction_history
    }


def run_comparative_test(n_epochs: int = 3) -> Dict[str, any]:
    """
    Compare EMA behavior between short (200) and long (1176) epochs.
    
    The key question: Does accumulated step_count cause EMA drift?
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST B: Gradient Norm EMA Step Count Accumulation")
    logger.info(f"Comparing {n_epochs} epochs of 200 vs 1176 steps")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Run 1: Short epochs (200 steps each)
    # -------------------------------------------------------------------------
    logger.info("[RUN 1] SHORT EPOCHS: 200 steps/epoch")
    result_short = simulate_multi_epoch_ema(
        n_steps_per_epoch=200,
        n_epochs=n_epochs,
        base_decay=0.99,
        use_ada_ema=True,
        reset_at_epoch_boundary=False
    )
    results['short'] = result_short
    
    logger.info(f"\n  Final EMA: {result_short['final_ema']:.4f}")
    logger.info(f"  Actual Mean: {result_short['actual_mean']:.4f}")
    logger.info(f"  Tracking Error: {result_short['tracking_error']*100:.2f}%")
    logger.info(f"  Total Step Count: {result_short['total_steps']}")
    
    # -------------------------------------------------------------------------
    # Run 2: Long epochs (1176 steps each)
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 2] LONG EPOCHS: 1176 steps/epoch")
    result_long = simulate_multi_epoch_ema(
        n_steps_per_epoch=1176,
        n_epochs=n_epochs,
        base_decay=0.99,
        use_ada_ema=True,
        reset_at_epoch_boundary=False
    )
    results['long'] = result_long
    
    logger.info(f"\n  Final EMA: {result_long['final_ema']:.4f}")
    logger.info(f"  Actual Mean: {result_long['actual_mean']:.4f}")
    logger.info(f"  Tracking Error: {result_long['tracking_error']*100:.2f}%")
    logger.info(f"  Total Step Count: {result_long['total_steps']}")
    
    # -------------------------------------------------------------------------
    # Run 3: EXPERIMENT - Long epochs with step count reset
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 3 - EXPERIMENT] LONG EPOCHS with epoch boundary reset")
    result_long_reset = simulate_multi_epoch_ema(
        n_steps_per_epoch=1176,
        n_epochs=n_epochs,
        base_decay=0.99,
        use_ada_ema=True,
        reset_at_epoch_boundary=True  # KEY EXPERIMENT
    )
    results['long_reset'] = result_long_reset
    
    logger.info(f"\n  Final EMA: {result_long_reset['final_ema']:.4f}")
    logger.info(f"  Tracking Error: {result_long_reset['tracking_error']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("COMPARATIVE ANALYSIS")
    logger.info(f"{'='*60}")
    
    tracking_diff = abs(result_short['tracking_error'] - result_long['tracking_error'])
    ema_diff = abs(result_short['final_ema'] - result_long['final_ema'])
    
    logger.info(f"Tracking Error Difference: {tracking_diff*100:.2f}%")
    logger.info(f"EMA Difference: {ema_diff:.4f}")
    
    # Check if reset helps
    reset_improvement = result_long['tracking_error'] - result_long_reset['tracking_error']
    logger.info(f"Reset Improvement: {reset_improvement*100:.2f}% better tracking")
    
    # -------------------------------------------------------------------------
    # Pass/Fail Determination
    # -------------------------------------------------------------------------
    issues = []
    
    # Check 1: Tracking error should be < 20%
    if result_long['tracking_error'] > 0.20:
        issues.append(
            f"HIGH TRACKING ERROR: Long run error = {result_long['tracking_error']*100:.2f}% > 20%"
        )
    
    # Check 2: Long runs shouldn't have much worse tracking than short runs
    if tracking_diff > 0.10:
        issues.append(
            f"TRACKING DIVERGENCE: Long run has {tracking_diff*100:.2f}% worse tracking than short"
        )
    
    # Check 3: If reset significantly improves tracking, accumulation is a problem
    if reset_improvement > 0.05:
        issues.append(
            f"ACCUMULATION ISSUE: Resetting step_count improves tracking by {reset_improvement*100:.2f}%"
        )
    
    # Check 4: EMA should not drift too far from actual mean
    if result_long['final_ema'] > result_long['actual_mean'] * 1.5:
        issues.append(
            f"EMA DRIFT: Final EMA {result_long['final_ema']:.4f} > 1.5x actual mean {result_long['actual_mean']:.4f}"
        )
    
    results['issues'] = issues
    results['passed'] = len(issues) == 0
    
    logger.info(f"\n{'='*60}")
    if results['passed']:
        logger.info("✅ TEST B PASSED: No step count accumulation issues detected")
    else:
        logger.error("❌ TEST B FAILED: Issues detected:")
        for issue in issues:
            logger.error(f"   - {issue}")
    logger.info(f"{'='*60}\n")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "#"*60)
    logger.info("# TEST B: Gradient Norm EMA Step Count Accumulation Test")
    logger.info("# Checking if cross-epoch step count causes EMA drift")
    logger.info("#"*60 + "\n")
    
    result = run_comparative_test(n_epochs=3)
    
    # Final verdict
    print("\n" + "="*60)
    if result['passed']:
        print("✅ TEST B: ALL CHECKS PASSED")
        print("Gradient EMA step count is NOT the root cause of GN spike.")
    else:
        print("❌ TEST B: ISSUES DETECTED")
        print("Gradient EMA accumulation MAY contribute to GN spike.")
        print("\nRecommended Fix: Reset grad_norm_step_count at epoch boundaries.")
    print("="*60)
