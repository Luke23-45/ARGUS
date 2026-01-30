"""
test_a_awr_beta.py
------------------
Test A: AWR Beta Annealing ESS Stability Test

HYPOTHESIS:
Epoch-based beta annealing causes ESS (Effective Sample Size) collapse with more 
steps per epoch. With 1176 batches vs 200 batches at the same epoch, the model 
sees the same beta but processes 5.88x more updates, potentially causing:
1. ESS collapse (< 0.05)
2. Runaway weight magnitudes
3. Adaptive beta overcorrection

SIMULATION APPROACH:
1. Create ICUAdvantageCalculator with adaptive_beta=True
2. Generate synthetic advantages mimicking real training distribution
3. Simulate N steps of weight calculation with beta annealing applied
4. Compare ESS trajectory for 200 vs 1176 steps at the same "epoch-equivalent"
5. Detect if ESS diverges or collapses

PASS CRITERIA:
- ESS should remain above 0.05 at all times
- Final ESS difference between 200 and 1176 step runs should be < 0.10

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
import torch.nn.functional as F

# Import the actual class we are testing
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.utils.train_utils import ScalingSteward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_A")


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_synthetic_advantages(
    batch_size: int = 64,
    seq_len: int = 24,
    mean: float = 0.0,
    std: float = 1.0,
    spike_prob: float = 0.05,  # 5% of batches have outliers
    spike_magnitude: float = 5.0
) -> torch.Tensor:
    """
    Generate synthetic advantages that mimic real training distribution.
    
    Real advantages in sepsis prediction are:
    - Mostly normally distributed around 0
    - Occasionally have high-magnitude spikes (clinical crisis events)
    - Have some negative values (poor outcomes)
    """
    base = torch.randn(batch_size, seq_len) * std + mean
    
    # Add occasional spikes (simulates clinical crisis detection)
    if torch.rand(1).item() < spike_prob:
        num_spikes = int(batch_size * 0.1)  # 10% of batch
        spike_idx = torch.randint(0, batch_size, (num_spikes,))
        base[spike_idx] += torch.randn(num_spikes, seq_len) * spike_magnitude
    
    return base


def simulate_beta_annealing(epoch: int, anneal_epochs: int = 40) -> float:
    """
    Replicate the exact beta annealing from wrapper_generalist.py:528-537
    
    Formula:
        if epoch < anneal_epochs:
            frac = epoch / anneal_epochs
            beta = start_beta + (end_beta - start_beta) * frac
        else:
            beta = end_beta
    """
    start_beta = 0.60
    end_beta = 0.15
    
    if epoch < anneal_epochs:
        frac = epoch / anneal_epochs
        return start_beta + (end_beta - start_beta) * frac
    else:
        return end_beta


# =============================================================================
# CORE TEST LOGIC
# =============================================================================

def simulate_epoch_training(
    calc: ICUAdvantageCalculator,
    n_steps: int,
    epoch: int,
    batch_size: int = 64,
    seq_len: int = 24,
    apply_scale_dynamics: bool = True
) -> Dict[str, List[float]]:
    """
    Simulate one epoch of training with the AWR calculator.
    
    This replicates exactly how the calculator is used in training:
    1. At epoch start, beta is set via annealing (if not adaptive)
    2. For each step, compute AWR weights from advantages
    3. Adaptive stats update happens inside calculate_awr_weights
    
    Returns:
        Dict with ESS, beta, max_weight trajectories per step
    """
    # Apply scaling dynamics as done in on_train_start
    if apply_scale_dynamics:
        calc.scale_dynamics(n_steps)
    
    # Set beta based on epoch (mimicking on_train_epoch_start)
    annealed_beta = simulate_beta_annealing(epoch)
    
    # IMPORTANT: In adaptive mode, wrapper does NOT overwrite beta
    # See wrapper_generalist.py:541-542
    if not calc.adaptive_beta:
        calc.beta.fill_(annealed_beta)
    
    # Initialize statistics (pretend we've already computed global stats)
    # This mimics the calibration phase
    calc.set_stats(mean=0.0, std=1.0)
    
    # Tracking
    ess_history = []
    beta_history = []
    max_weight_history = []
    weight_mean_history = []
    
    for step in range(n_steps):
        # Generate synthetic advantages for this batch
        advantages = generate_synthetic_advantages(
            batch_size=batch_size, 
            seq_len=seq_len,
            mean=0.0,
            std=1.0,
            spike_prob=0.05
        )
        
        # Create mask (no padding in this simulation)
        mask = torch.ones_like(advantages, dtype=torch.bool)
        
        # Compute AWR weights exactly as in training
        # This calls calculate_awr_weights which internally calls _update_adaptive_stats
        weights, diagnostics = calc.calculate_awr_weights(advantages, mask=mask)
        
        # Record telemetry
        ess_history.append(diagnostics['ess'])
        beta_history.append(calc.beta.item())
        max_weight_history.append(diagnostics['weights_max'])
        weight_mean_history.append(diagnostics['weights_mean'])
        
        # Log every 100 steps
        if step % 100 == 0 or step == n_steps - 1:
            logger.debug(
                f"Step {step}/{n_steps}: ESS={diagnostics['ess']:.4f}, "
                f"Beta={calc.beta.item():.4f}, MaxW={diagnostics['weights_max']:.2f}"
            )
    
    return {
        'ess': ess_history,
        'beta': beta_history,
        'max_weight': max_weight_history,
        'weight_mean': weight_mean_history
    }


def run_comparative_test(epoch: int = 6) -> Dict[str, any]:
    """
    Run the core comparison: 200 steps vs 1176 steps at the same epoch.
    
    At epoch 6:
    - Annealed beta = 0.60 + (0.15 - 0.60) * (6/40) = 0.5325
    - This is the transition zone where issues typically emerge
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST A: AWR Beta Annealing ESS Stability")
    logger.info(f"Epoch: {epoch}, Annealed Beta: {simulate_beta_annealing(epoch):.4f}")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Run 1: Short epoch (200 steps) - BASELINE
    # -------------------------------------------------------------------------
    logger.info("[RUN 1] SHORT EPOCH: 200 steps")
    calc_short = ICUAdvantageCalculator(
        beta=1.0,  # Will be set by annealing or adaptive
        adaptive_beta=True,  # Matches production config
        adaptive_clipping=True,
        beta_momentum=0.999,  # Matches config
        max_weight=20.0
    )
    
    history_short = simulate_epoch_training(
        calc=calc_short,
        n_steps=200,
        epoch=epoch,
        apply_scale_dynamics=True
    )
    
    results['short'] = {
        'n_steps': 200,
        'final_ess': history_short['ess'][-1],
        'min_ess': min(history_short['ess']),
        'mean_ess': sum(history_short['ess']) / len(history_short['ess']),
        'final_beta': history_short['beta'][-1],
        'max_weight_peak': max(history_short['max_weight']),
        'history': history_short
    }
    
    logger.info(f"  Final ESS: {results['short']['final_ess']:.4f}")
    logger.info(f"  Min ESS: {results['short']['min_ess']:.4f}")
    logger.info(f"  Final Beta: {results['short']['final_beta']:.4f}")
    logger.info(f"  Max Weight Peak: {results['short']['max_weight_peak']:.2f}")
    
    # -------------------------------------------------------------------------
    # Run 2: Long epoch (1176 steps) - TEST SUBJECT
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 2] LONG EPOCH: 1176 steps")
    calc_long = ICUAdvantageCalculator(
        beta=1.0,
        adaptive_beta=True,
        adaptive_clipping=True,
        beta_momentum=0.999,
        max_weight=20.0
    )
    
    history_long = simulate_epoch_training(
        calc=calc_long,
        n_steps=1176,
        epoch=epoch,
        apply_scale_dynamics=True
    )
    
    results['long'] = {
        'n_steps': 1176,
        'final_ess': history_long['ess'][-1],
        'min_ess': min(history_long['ess']),
        'mean_ess': sum(history_long['ess']) / len(history_long['ess']),
        'final_beta': history_long['beta'][-1],
        'max_weight_peak': max(history_long['max_weight']),
        'history': history_long
    }
    
    logger.info(f"  Final ESS: {results['long']['final_ess']:.4f}")
    logger.info(f"  Min ESS: {results['long']['min_ess']:.4f}")
    logger.info(f"  Final Beta: {results['long']['final_beta']:.4f}")
    logger.info(f"  Max Weight Peak: {results['long']['max_weight_peak']:.2f}")
    
    # -------------------------------------------------------------------------
    # Analysis: Compare Results
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("COMPARATIVE ANALYSIS")
    logger.info(f"{'='*60}")
    
    ess_diff = abs(results['short']['final_ess'] - results['long']['final_ess'])
    beta_diff = abs(results['short']['final_beta'] - results['long']['final_beta'])
    
    logger.info(f"ESS Difference (Short vs Long): {ess_diff:.4f}")
    logger.info(f"Beta Difference (Short vs Long): {beta_diff:.4f}")
    
    # -------------------------------------------------------------------------
    # Pass/Fail Determination
    # -------------------------------------------------------------------------
    issues = []
    
    # Check 1: ESS should never collapse below 0.05
    if results['long']['min_ess'] < 0.05:
        issues.append(f"ESS COLLAPSE: Long run min ESS = {results['long']['min_ess']:.4f} < 0.05")
    
    if results['short']['min_ess'] < 0.05:
        issues.append(f"ESS COLLAPSE: Short run min ESS = {results['short']['min_ess']:.4f} < 0.05")
    
    # Check 2: Final ESS should be similar (within 0.10)
    if ess_diff > 0.10:
        issues.append(f"ESS DIVERGENCE: Difference = {ess_diff:.4f} > 0.10 threshold")
    
    # Check 3: Beta should stabilize (not runaway)
    if results['long']['final_beta'] > 5.0:
        issues.append(f"BETA RUNAWAY: Long run final beta = {results['long']['final_beta']:.4f} > 5.0")
    
    if results['long']['final_beta'] < 0.02:
        issues.append(f"BETA COLLAPSE: Long run final beta = {results['long']['final_beta']:.4f} < 0.02")
    
    results['issues'] = issues
    results['passed'] = len(issues) == 0
    
    logger.info(f"\n{'='*60}")
    if results['passed']:
        logger.info("✅ TEST A PASSED: No AWR beta/ESS instability detected")
    else:
        logger.error("❌ TEST A FAILED: Issues detected:")
        for issue in issues:
            logger.error(f"   - {issue}")
    logger.info(f"{'='*60}\n")
    
    return results


def run_multi_epoch_test() -> Dict[str, any]:
    """
    Run the test across multiple epochs to catch late-onset issues.
    
    Tests epochs 0, 5, 10, 20, 35 (covering full annealing range)
    """
    logger.info("\n" + "="*60)
    logger.info("MULTI-EPOCH TEST: Checking stability across annealing range")
    logger.info("="*60 + "\n")
    
    epochs_to_test = [0, 5, 10, 20, 35]
    all_results = {}
    all_issues = []
    
    for epoch in epochs_to_test:
        logger.info(f"\n--- Testing Epoch {epoch} ---")
        result = run_comparative_test(epoch=epoch)
        all_results[f'epoch_{epoch}'] = result
        all_issues.extend(result['issues'])
    
    logger.info("\n" + "="*60)
    logger.info("MULTI-EPOCH SUMMARY")
    logger.info("="*60)
    
    for epoch in epochs_to_test:
        res = all_results[f'epoch_{epoch}']
        status = "✅" if res['passed'] else "❌"
        logger.info(
            f"Epoch {epoch:2d}: {status} | "
            f"ESS(short)={res['short']['final_ess']:.3f} | "
            f"ESS(long)={res['long']['final_ess']:.3f} | "
            f"Beta(long)={res['long']['final_beta']:.4f}"
        )
    
    return {
        'passed': len(all_issues) == 0,
        'issues': all_issues,
        'epoch_results': all_results
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "#"*60)
    logger.info("# TEST A: AWR Beta Annealing ESS Stability Test")
    logger.info("# Checking if epoch-based beta annealing causes ESS collapse")
    logger.info("#"*60 + "\n")
    
    # Run single epoch test first
    single_result = run_comparative_test(epoch=6)
    
    # If single test passes, run multi-epoch
    if single_result['passed']:
        multi_result = run_multi_epoch_test()
        final_passed = multi_result['passed']
    else:
        final_passed = False
        multi_result = None
    
    # Final verdict
    print("\n" + "="*60)
    if final_passed:
        print("✅ TEST A: ALL CHECKS PASSED")
        print("AWR beta annealing is NOT the root cause of GN spike.")
    else:
        print("❌ TEST A: ISSUES DETECTED")
        print("AWR beta annealing MAY contribute to GN spike.")
        print("\nRecommended Fix: Disable beta annealing or switch to step-based schedule.")
    print("="*60)
