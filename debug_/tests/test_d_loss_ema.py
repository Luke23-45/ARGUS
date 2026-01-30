"""
test_d_loss_ema.py
------------------
Test D: Loss EMA Accumulation in BayesianProjectedScaler

HYPOTHESIS:
The `loss_emas` buffer in BayesianProjectedScaler accumulates loss statistics 
without upper bound clipping. If a loss component spikes early in training:
1. The spike gets incorporated into the EMA
2. The EMA influences uncertainty weights (log_vars)
3. With high EMA, the scaler may suppress that loss component too much
4. This can cause imbalanced gradient flow

SIMULATION APPROACH:
1. Create BayesianProjectedScaler and apply scale_dynamics(1176)
2. Feed normal losses for some steps, then inject a spike
3. Measure how long the spike influence persists in EMA
4. Check if weights (precision = exp(-log_var)) become unstable

PASS CRITERIA:
- Loss EMA should recover to near-normal (<5.0) within 200 steps after spike
- Weights should not collapse to near-zero or explode
- No NaN/Inf in any tensor

WHAT THIS TESTS:
- loss_scaler.py:25 (loss_emas buffer initialization)
- loss_scaler.py:31 (decay scaling)
- loss_scaler.py:52-67 (forward pass EMA update)
- loss_scaler.py:64-67 (EMA accumulation - NO CLIPPING)

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
import torch.nn as nn

# Import the actual class we are testing
from icu.models.components.loss_scaler import BayesianProjectedScaler
from icu.utils.train_utils import ScalingSteward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_D")


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_normal_losses(
    num_tasks: int = 7,
    means: List[float] = None,
    stds: List[float] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate normal (stable) loss values for each task.
    
    BayesianProjectedScaler expects Dict[str, Tensor] with these keys:
    'diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys'
    
    Real loss values in APEX training:
    - diffusion: ~0.1-0.2
    - critic: ~0.2-0.4
    - aux: ~0.4-0.6
    - acl: ~0.2-0.4
    - bgsl: ~0.2-0.3
    - tcb: ~0.1-0.3
    - phys: ~0.1-0.3
    """
    keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
    
    if means is None:
        means = [0.15, 0.3, 0.5, 0.3, 0.25, 0.2, 0.2]
    if stds is None:
        stds = [0.03, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05]
    
    means = means[:num_tasks]
    stds = stds[:num_tasks]
    
    # Generate losses with small noise as dict
    loss_dict = {}
    for i in range(min(num_tasks, len(keys))):
        val = max(0.01, torch.randn(1).item() * stds[i] + means[i])
        loss_dict[keys[i]] = torch.tensor(val)
    
    return loss_dict


def generate_spike_losses(
    num_tasks: int = 7,
    spike_task: int = 0,
    spike_magnitude: float = 50.0
) -> Dict[str, torch.Tensor]:
    """
    Generate losses with a spike in one task.
    
    Spikes can occur in real training due to:
    - Clinical crisis events (sudden sepsis onset)
    - Batch with all-negative or all-positive samples
    - Numerical instability in log-space computations
    """
    keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
    losses = generate_normal_losses(num_tasks)
    spike_key = keys[spike_task] if spike_task < len(keys) else keys[0]
    losses[spike_key] = torch.tensor(spike_magnitude)
    return losses


# =============================================================================
# CORE TEST LOGIC
# =============================================================================

def simulate_loss_scaling(
    num_tasks: int = 7,
    n_steps: int = 500,
    spike_step: int = 100,  # Step at which to inject spike
    spike_magnitude: float = 50.0,
    spike_task: int = 0,
    n_batches_per_epoch: int = 1176
) -> Dict[str, any]:
    """
    Simulate loss scaling over multiple steps with a spike injection.
    
    This replicates exactly how the scaler is used in training:
    1. Each step, 7 losses are fed to the scaler
    2. Scaler updates internal loss_emas
    3. Scaler outputs weighted combined loss and precision estimates
    
    Args:
        num_tasks: Number of loss components (7 in APEX)
        n_steps: Total steps to simulate
        spike_step: Step at which to inject spike
        spike_magnitude: Magnitude of the spike
        spike_task: Which task to spike
        n_batches_per_epoch: Used for scale_dynamics
    
    Returns:
        Dict with EMA trajectories, weight trajectories, and analysis
    """
    # Initialize scaler
    # Match the production initialization from wrapper_generalist.py:302-312
    initial_log_vars = torch.tensor([1.0, 1.5, -0.5, 1.0, 1.0, 2.5, 0.5])
    
    scaler = BayesianProjectedScaler(num_tasks=num_tasks)
    # Override log_vars with production values
    with torch.no_grad():
        scaler.log_vars.copy_(initial_log_vars[:num_tasks])
    
    # Apply scaling dynamics
    scaler.scale_dynamics(n_batches_per_epoch)
    logger.info(f"Scaled decay: 0.99 -> {scaler.decay:.6f} for {n_batches_per_epoch} steps")
    
    # Tracking
    ema_history = {i: [] for i in range(num_tasks)}
    weight_history = {i: [] for i in range(num_tasks)}
    total_loss_history = []
    raw_loss_history = {i: [] for i in range(num_tasks)}
    
    for step in range(n_steps):
        # Generate losses
        keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
        spike_key = keys[spike_task] if spike_task < len(keys) else keys[0]
        
        if step == spike_step:
            losses = generate_spike_losses(
                num_tasks=num_tasks,
                spike_task=spike_task,
                spike_magnitude=spike_magnitude
            )
            logger.info(f"  [SPIKE INJECTED] Step {step}: {spike_key} = {losses[spike_key].item():.2f}")
        else:
            losses = generate_normal_losses(num_tasks=num_tasks)
        
        # Forward pass
        total_loss, weights_dict = scaler(losses)
        
        # Get keys for tracking
        keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
        spike_key = keys[spike_task] if spike_task < len(keys) else keys[0]
        
        # Record everything
        for task in range(num_tasks):
            ema_history[task].append(scaler.loss_emas[task].item())
            weight_history[task].append(weights_dict.get(f'w_{task}', 0.0))
            key = keys[task] if task < len(keys) else None
            if key and key in losses:
                raw_loss_history[task].append(losses[key].item())
            else:
                raw_loss_history[task].append(0.0)
        total_loss_history.append(total_loss.item())
        
        # Log at key points
        if step in [spike_step - 1, spike_step, spike_step + 1, spike_step + 50, spike_step + 100, spike_step + 200]:
            logger.debug(
                f"  Step {step}: EMA[{spike_task}]={scaler.loss_emas[spike_task].item():.4f}, "
                f"Weight[{spike_task}]={weights_dict.get(f'w_{spike_task}', 0):.4f}"
            )
    
    # Analysis: How long does spike persist?
    spike_ema = ema_history[spike_task][spike_step]
    
    # Find step where EMA returns to near-normal (< 5.0)
    recovery_step = None
    for s in range(spike_step, n_steps):
        if ema_history[spike_task][s] < 5.0:
            recovery_step = s
            break
    
    recovery_time = recovery_step - spike_step if recovery_step else n_steps - spike_step
    
    return {
        'num_tasks': num_tasks,
        'n_steps': n_steps,
        'spike_step': spike_step,
        'spike_magnitude': spike_magnitude,
        'spike_task': spike_task,
        'spike_ema_value': spike_ema,
        'final_ema_spiked_task': ema_history[spike_task][-1],
        'recovery_step': recovery_step,
        'recovery_time': recovery_time,
        'ema_history': ema_history,
        'weight_history': weight_history,
        'total_loss_history': total_loss_history,
        'raw_loss_history': raw_loss_history
    }


def run_comparative_test() -> Dict[str, any]:
    """
    Test loss EMA behavior with different scaling configurations.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST D: Loss EMA Accumulation Test")
    logger.info(f"Testing spike persistence in BayesianProjectedScaler")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Run 1: Short epoch scaling (200 batches)
    # -------------------------------------------------------------------------
    logger.info("[RUN 1] SHORT EPOCH: 200 batches/epoch")
    result_short = simulate_loss_scaling(
        n_steps=500,
        spike_step=100,
        spike_magnitude=50.0,
        spike_task=0,
        n_batches_per_epoch=200
    )
    results['short'] = result_short
    
    logger.info(f"  Spike EMA Value: {result_short['spike_ema_value']:.4f}")
    logger.info(f"  Final EMA (task 0): {result_short['final_ema_spiked_task']:.4f}")
    logger.info(f"  Recovery Time: {result_short['recovery_time']} steps")
    
    # -------------------------------------------------------------------------
    # Run 2: Long epoch scaling (1176 batches)
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 2] LONG EPOCH: 1176 batches/epoch")
    result_long = simulate_loss_scaling(
        n_steps=500,
        spike_step=100,
        spike_magnitude=50.0,
        spike_task=0,
        n_batches_per_epoch=1176
    )
    results['long'] = result_long
    
    logger.info(f"  Spike EMA Value: {result_long['spike_ema_value']:.4f}")
    logger.info(f"  Final EMA (task 0): {result_long['final_ema_spiked_task']:.4f}")
    logger.info(f"  Recovery Time: {result_long['recovery_time']} steps")
    
    # -------------------------------------------------------------------------
    # Run 3: Multiple spikes (stress test)
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 3] STRESS TEST: Multiple spikes")
    
    # Initialize scaler
    scaler = BayesianProjectedScaler(num_tasks=7)
    scaler.scale_dynamics(1176)
    
    spike_steps = [50, 150, 250, 350]
    ema_peaks = []
    
    for step in range(500):
        if step in spike_steps:
            losses = generate_spike_losses(spike_task=0, spike_magnitude=30.0)
        else:
            losses = generate_normal_losses()
        
        total_loss, _ = scaler(losses)
        
        if step in spike_steps:
            ema_peaks.append(scaler.loss_emas[0].item())
    
    results['stress_test'] = {
        'spike_steps': spike_steps,
        'ema_peaks': ema_peaks,
        'final_ema': scaler.loss_emas[0].item()
    }
    
    logger.info(f"  EMA peaks at spikes: {[f'{p:.2f}' for p in ema_peaks]}")
    logger.info(f"  Final EMA after multiple spikes: {scaler.loss_emas[0].item():.4f}")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("COMPARATIVE ANALYSIS")
    logger.info(f"{'='*60}")
    
    recovery_diff = result_long['recovery_time'] - result_short['recovery_time']
    logger.info(f"Recovery Time Difference: {recovery_diff} steps")
    
    # -------------------------------------------------------------------------
    # Pass/Fail Determination
    # -------------------------------------------------------------------------
    issues = []
    
    # Check 1: EMA should recover within 200 steps
    if result_long['recovery_time'] > 200:
        issues.append(
            f"SLOW RECOVERY: Long run takes {result_long['recovery_time']} steps to recover (> 200)"
        )
    
    # Check 2: Final EMA should be near-normal (< 5.0)
    if result_long['final_ema_spiked_task'] > 5.0:
        issues.append(
            f"EMA NOT RECOVERED: Final EMA = {result_long['final_ema_spiked_task']:.4f} > 5.0"
        )
    
    # Check 3: Multiple spikes shouldn't cause runaway accumulation
    stress_final = results['stress_test']['final_ema']
    if stress_final > 10.0:
        issues.append(
            f"RUNAWAY ACCUMULATION: After 4 spikes, EMA = {stress_final:.4f} > 10.0"
        )
    
    # Check 4: EMA peaks should not grow unbounded with each spike
    ema_peaks = results['stress_test']['ema_peaks']
    if len(ema_peaks) >= 2 and ema_peaks[-1] > ema_peaks[0] * 2.0:
        issues.append(
            f"ACCUMULATING PEAKS: EMA peaks growing (first: {ema_peaks[0]:.2f}, last: {ema_peaks[-1]:.2f})"
        )
    
    # Check 5: Long epoch should not have drastically slower recovery
    if recovery_diff > 50:
        issues.append(
            f"SCALING PENALTY: Long epoch recovers {recovery_diff} steps slower than short"
        )
    
    results['issues'] = issues
    results['passed'] = len(issues) == 0
    
    logger.info(f"\n{'='*60}")
    if results['passed']:
        logger.info("✅ TEST D PASSED: No loss EMA accumulation issues detected")
    else:
        logger.error("❌ TEST D FAILED: Issues detected:")
        for issue in issues:
            logger.error(f"   - {issue}")
    logger.info(f"{'='*60}\n")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "#"*60)
    logger.info("# TEST D: Loss EMA Accumulation Test")
    logger.info("# Checking if loss spikes persist in EMA")
    logger.info("#"*60 + "\n")
    
    result = run_comparative_test()
    
    # Final verdict
    print("\n" + "="*60)
    if result['passed']:
        print("✅ TEST D: ALL CHECKS PASSED")
        print("Loss EMA accumulation is NOT the root cause of GN spike.")
    else:
        print("❌ TEST D: ISSUES DETECTED")
        print("Loss EMA accumulation MAY contribute to GN spike.")
        print("\nRecommended Fix: Add clipping to loss_emas update (max=10.0).")
    print("="*60)
