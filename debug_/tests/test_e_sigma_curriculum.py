"""
test_e_sigma_curriculum.py
--------------------------
Test E: Sigma Scale Curriculum Step-Density Analysis

HYPOTHESIS:
The `curr_sigma_scale` curriculum (3.50 -> 2.50 over 15 epochs) is epoch-based,
NOT step-density scaled. With 1176 batches/epoch vs 200 batches/epoch:
- Each epoch contains ~6x more optimization steps
- The curriculum progresses at the same epoch rate regardless of batch count
- This means the physics envelope TIGHTENS at the same epoch pace, but sees more samples

SIMULATION APPROACH:
1. Simulate the sigma_scale value at each optimization STEP (not epoch)
2. Compare: How many STEPS does it take to reach each sigma value?
3. Check if this could cause asymmetric gradient pressure

PASS CRITERIA:
- If sigma curriculum is step-density aware: step-count to reach sigma=3.0 should be similar
- If NOT step-density aware: Long runs reach aggressive sigma after MORE steps (potential issue)

NOTE: This test does NOT import any production code. It's a pure mathematical simulation.

Authors: APEX Diagnostic Team
Date: 2026-01-30
"""

import sys
import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Note: We DO NOT import production code here - this is a pure simulation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_E")


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def simulate_sigma_curriculum(
    n_batches_per_epoch: int,
    n_epochs: int = 20,
    start_sigma: float = 3.50,
    end_sigma: float = 2.50,
    ramp_epochs: float = 15.0
) -> Dict[str, any]:
    """
    Simulate the sigma_scale curriculum over training.
    
    This matches the logic in wrapper_generalist.py:588-591:
        sigma_progress = min(1.0, self.current_epoch / sigma_ramp_epochs)
        self.curr_sigma_scale.fill_(3.50 - (3.50 - 2.50) * sigma_progress)
    
    Returns:
        Dict with step-by-step sigma values and analysis
    """
    sigma_history = []
    step_at_sigma = {}  # sigma_threshold -> first step to reach it
    
    total_steps = n_batches_per_epoch * n_epochs
    step = 0
    
    for epoch in range(n_epochs):
        # Compute sigma for this epoch (epoch-based, not step-based)
        sigma_progress = min(1.0, epoch / ramp_epochs)
        curr_sigma = start_sigma - (start_sigma - end_sigma) * sigma_progress
        
        for batch_idx in range(n_batches_per_epoch):
            sigma_history.append({
                'step': step,
                'epoch': epoch,
                'batch_idx': batch_idx,
                'sigma': curr_sigma
            })
            
            # Track first step to reach sigma thresholds
            for threshold in [3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5]:
                if threshold not in step_at_sigma and curr_sigma <= threshold:
                    step_at_sigma[threshold] = step
            
            step += 1
    
    return {
        'n_batches_per_epoch': n_batches_per_epoch,
        'n_epochs': n_epochs,
        'total_steps': total_steps,
        'sigma_history': sigma_history,
        'step_at_sigma': step_at_sigma,
        'final_sigma': sigma_history[-1]['sigma']
    }


def run_comparative_test() -> Dict[str, any]:
    """
    Compare sigma curriculum behavior between short and long epochs.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST E: Sigma Scale Curriculum Analysis")
    logger.info(f"Checking if sigma curriculum is step-density aware")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Run 1: Short epochs (200 batches/epoch)
    # -------------------------------------------------------------------------
    logger.info("[RUN 1] SHORT EPOCHS: 200 batches/epoch")
    result_short = simulate_sigma_curriculum(
        n_batches_per_epoch=200,
        n_epochs=20
    )
    results['short'] = result_short
    
    logger.info(f"  Total Steps: {result_short['total_steps']}")
    logger.info(f"  Final Sigma: {result_short['final_sigma']:.4f}")
    logger.info(f"  Step to reach sigma=3.0: {result_short['step_at_sigma'].get(3.0, 'Not reached')}")
    logger.info(f"  Step to reach sigma=2.5: {result_short['step_at_sigma'].get(2.5, 'Not reached')}")
    
    # -------------------------------------------------------------------------
    # Run 2: Long epochs (1176 batches/epoch)
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 2] LONG EPOCHS: 1176 batches/epoch")
    result_long = simulate_sigma_curriculum(
        n_batches_per_epoch=1176,
        n_epochs=20
    )
    results['long'] = result_long
    
    logger.info(f"  Total Steps: {result_long['total_steps']}")
    logger.info(f"  Final Sigma: {result_long['final_sigma']:.4f}")
    logger.info(f"  Step to reach sigma=3.0: {result_long['step_at_sigma'].get(3.0, 'Not reached')}")
    logger.info(f"  Step to reach sigma=2.5: {result_long['step_at_sigma'].get(2.5, 'Not reached')}")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("COMPARATIVE ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Calculate step ratios
    step_ratio_3_0 = None
    step_ratio_2_5 = None
    
    if 3.0 in result_short['step_at_sigma'] and 3.0 in result_long['step_at_sigma']:
        step_ratio_3_0 = result_long['step_at_sigma'][3.0] / result_short['step_at_sigma'][3.0]
        logger.info(f"Step ratio to reach sigma=3.0: {step_ratio_3_0:.2f}x (long/short)")
    
    if 2.5 in result_short['step_at_sigma'] and 2.5 in result_long['step_at_sigma']:
        step_ratio_2_5 = result_long['step_at_sigma'][2.5] / result_short['step_at_sigma'][2.5]
        logger.info(f"Step ratio to reach sigma=2.5: {step_ratio_2_5:.2f}x (long/short)")
    
    batch_ratio = result_long['n_batches_per_epoch'] / result_short['n_batches_per_epoch']
    logger.info(f"Batch count ratio: {batch_ratio:.2f}x")
    
    # -------------------------------------------------------------------------
    # Pass/Fail Determination
    # -------------------------------------------------------------------------
    issues = []
    
    # The key insight: If curriculum is NOT step-density aware, the step ratios
    # will equal the batch ratio (~5.88x). If it IS aware, ratios would be ~1.0x.
    
    if step_ratio_3_0 is not None:
        if abs(step_ratio_3_0 - batch_ratio) < 0.5:
            # Step ratio matches batch ratio -> NOT step-density aware
            issues.append(
                f"STEP-DENSITY UNAWARE: Long run takes {step_ratio_3_0:.1f}x more steps "
                f"to reach sigma=3.0 (matches batch ratio {batch_ratio:.1f}x)"
            )
    
    if step_ratio_2_5 is not None:
        if abs(step_ratio_2_5 - batch_ratio) < 0.5:
            issues.append(
                f"STEP-DENSITY UNAWARE: Long run takes {step_ratio_2_5:.1f}x more steps "
                f"to reach sigma=2.5 (matches batch ratio {batch_ratio:.1f}x)"
            )
    
    # Check: Does the long run experience MORE steps at aggressive (low) sigma?
    sigma_threshold = 3.0
    steps_below_threshold_short = sum(
        1 for h in result_short['sigma_history'] if h['sigma'] <= sigma_threshold
    )
    steps_below_threshold_long = sum(
        1 for h in result_long['sigma_history'] if h['sigma'] <= sigma_threshold
    )
    
    logger.info(f"\nSteps with sigma <= 3.0:")
    logger.info(f"  Short run: {steps_below_threshold_short} steps")
    logger.info(f"  Long run: {steps_below_threshold_long} steps")
    logger.info(f"  Ratio: {steps_below_threshold_long / max(1, steps_below_threshold_short):.1f}x")
    
    if steps_below_threshold_long > steps_below_threshold_short * 3:
        issues.append(
            f"ASYMMETRIC EXPOSURE: Long run has {steps_below_threshold_long} steps "
            f"at aggressive sigma vs {steps_below_threshold_short} for short run"
        )
    
    results['issues'] = issues
    results['passed'] = len(issues) == 0
    results['step_ratio_3_0'] = step_ratio_3_0
    results['step_ratio_2_5'] = step_ratio_2_5
    
    logger.info(f"\n{'='*60}")
    if results['passed']:
        logger.info("TEST E PASSED: Sigma curriculum appears step-density aware")
    else:
        logger.info("TEST E DETECTED ASYMMETRY:")
        for issue in issues:
            logger.info(f"   - {issue}")
        logger.info("\nNOTE: This is an OBSERVATION, not proof of causation.")
        logger.info("The curriculum being epoch-based may or may not cause GN spikes.")
    logger.info(f"{'='*60}\n")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "#"*60)
    logger.info("# TEST E: Sigma Scale Curriculum Step-Density Analysis")
    logger.info("# Checking if physics envelope curriculum is step-density aware")
    logger.info("#"*60 + "\n")
    
    result = run_comparative_test()
    
    # Final verdict
    print("\n" + "="*60)
    if result['passed']:
        print("TEST E: SIGMA CURRICULUM IS STEP-DENSITY AWARE")
        print("The curriculum adapts to different batch densities.")
    else:
        print("TEST E: SIGMA CURRICULUM IS EPOCH-BASED (NOT STEP-AWARE)")
        print(f"Long runs experience {result.get('step_ratio_3_0', 'N/A')}x more steps at each sigma level.")
        print("\nIMPACT: This is an ASYMMETRY, but may not directly cause GN spikes.")
        print("Physics loss at low sigma (tight bounds) could contribute to gradient pressure.")
    print("="*60)
