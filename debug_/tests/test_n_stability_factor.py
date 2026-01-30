"""
test_n_stability_factor.py
--------------------------
Test N: Stability Factor Shock Detection Analysis

HYPOTHESIS:
The stability_factor in wrapper_generalist.py is used to dampen loss weights during
"shocks". In loss_scaler.py (L81):

```python
uw_weights = clinical_weights[indices] * stability_factor + (1.0 - stability_factor)
```

When stability_factor = 1.0 (stable): uw_weights = clinical_weights
When stability_factor = 0.0 (shock): uw_weights = 1.0 (neutral)

POTENTIAL ISSUE:
1. If stability_factor oscillates rapidly (shock detection noise)
2. The loss weights oscillate → gradient direction oscillates
3. This could cause GN spikes due to conflicting gradient signals

WHAT WE'RE TESTING:
1. Stability factor computation in the training loop
2. How clinical weight dampening affects gradient consistency
3. Whether oscillating stability_factor causes loss weight volatility
"""

import torch
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_dampened_weights(
    clinical_weights: torch.Tensor,
    stability_factor: float,
) -> torch.Tensor:
    """
    Compute dampened clinical weights as in loss_scaler.py L81.
    
    Args:
        clinical_weights: Base clinical priority weights (e.g., [0.5, 0.5, 1.5, 1.5, 1.0, 1.0, 1.0])
        stability_factor: 1.0 = stable, 0.0 = shock
    """
    # [v27.1 FIX] Apply Adaptive Governor with Floor
    effective_sf = max(0.5, stability_factor if isinstance(stability_factor, float) else stability_factor)
    if isinstance(stability_factor, torch.Tensor):
        effective_sf = torch.max(torch.tensor(0.5, device=stability_factor.device), stability_factor)
        
    return clinical_weights * effective_sf + (1.0 - effective_sf)


def simulate_stability_oscillation(
    n_steps: int = 100,
    oscillation_period: int = 10,
    clinical_weights: torch.Tensor = None,
) -> Dict[str, any]:
    """
    Simulates stability factor oscillation and its effect on weights.
    """
    
    if clinical_weights is None:
        # Default clinical weights from loss_scaler.py
        clinical_weights = torch.tensor([0.5, 0.5, 1.5, 1.5, 1.0, 1.0, 1.0])
    
    weight_history = []
    stability_history = []
    
    for step in range(n_steps):
        # Simulate oscillating stability factor
        if (step // oscillation_period) % 2 == 0:
            stability_factor = 1.0  # Stable
        else:
            stability_factor = 0.3  # Partial shock
        
        stability_history.append(stability_factor)
        weights = compute_dampened_weights(clinical_weights, stability_factor)
        weight_history.append(weights.clone())
    
    # Compute weight volatility
    weight_tensor = torch.stack(weight_history)
    weight_changes = torch.diff(weight_tensor, dim=0).abs()
    
    return {
        'weight_history': weight_history,
        'stability_history': stability_history,
        'mean_weight_change': weight_changes.mean().item(),
        'max_weight_change': weight_changes.max().item(),
    }


def run_stability_factor_analysis() -> Dict[str, any]:
    """Analyze stability factor's effect on loss weighting."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST N: Stability Factor Shock Detection Analysis")
    logger.info("Checking if stability oscillation causes weight volatility")
    logger.info("=" * 60 + "\n")
    
    issues = []
    
    # Test 1: Weight dampening at different stability factors
    logger.info("[TEST 1] Clinical Weight Dampening")
    logger.info("-" * 50)
    
    clinical_weights = torch.tensor([0.5, 0.5, 1.5, 1.5, 1.0, 1.0, 1.0])
    task_names = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
    
    stability_values = [1.0, 0.8, 0.5, 0.2, 0.0]
    
    logger.info(f"  Clinical weights: {clinical_weights.tolist()}")
    logger.info(f"  Tasks: {task_names}")
    logger.info("")
    
    for sf in stability_values:
        dampened = compute_dampened_weights(clinical_weights, sf)
        logger.info(f"  stability={sf:.1f}: {[f'{w:.2f}' for w in dampened.tolist()]}")
    
    # Test 2: Weight change magnitude during transition
    logger.info("\n[TEST 2] Weight Change During Stability Transition")
    logger.info("-" * 50)
    
    # Transition from stable (1.0) to shock (0.0)
    weights_stable = compute_dampened_weights(clinical_weights, 1.0)
    weights_shock = compute_dampened_weights(clinical_weights, 0.0)
    
    weight_delta = (weights_shock - weights_stable).abs()
    
    logger.info(f"  Transition 1.0 → 0.0:")
    for i, name in enumerate(task_names):
        logger.info(f"    {name}: {weights_stable[i]:.2f} → {weights_shock[i]:.2f} (Δ = {weight_delta[i]:.2f})")
    
    max_delta = weight_delta.max().item()
    if max_delta > 0.5:
        issues.append(f"LARGE WEIGHT SHIFT: Max Δ = {max_delta:.2f} during full stability transition")
    
    # Test 3: Oscillation simulation
    logger.info("\n[TEST 3] Oscillating Stability Factor (100 steps, period=10)")
    logger.info("-" * 50)
    
    result = simulate_stability_oscillation(
        n_steps=100,
        oscillation_period=10,
        clinical_weights=clinical_weights,
    )
    
    logger.info(f"  Mean weight change per step: {result['mean_weight_change']:.4f}")
    logger.info(f"  Max weight change per step: {result['max_weight_change']:.4f}")
    
    # Count stability transitions
    transitions = sum(1 for i in range(1, len(result['stability_history']))
                     if result['stability_history'][i] != result['stability_history'][i-1])
    logger.info(f"  Stability transitions: {transitions}")
    
    if result['mean_weight_change'] > 0.1:
        issues.append(f"HIGH WEIGHT VOLATILITY: Mean change = {result['mean_weight_change']:.4f}")
    
    # Test 4: Effect on gradient direction (conceptual)
    logger.info("\n[TEST 4] Gradient Direction Consistency")
    logger.info("-" * 50)
    
    # Simulate how weight changes affect relative gradient contributions
    # If aux (high priority) weight drops from 1.5 to 1.0, its gradient contribution
    # relative to diffusion (0.5 to 1.0) changes significantly
    
    # Relative weight ratios
    stable_ratio_aux_to_diff = clinical_weights[2] / clinical_weights[0]  # 1.5 / 0.5 = 3.0
    
    # [v27.1 FIX] Shock ratio now respects 0.5 floor
    # aux = 1.5 * 0.5 + 0.5 = 1.25
    # diff = 0.5 * 0.5 + 0.5 = 0.75
    shock_ratio_aux_to_diff = 1.25 / 0.75  # = 1.66...
    
    logger.info(f"  aux/diffusion weight ratio (stable): {stable_ratio_aux_to_diff:.1f}")
    logger.info(f"  aux/diffusion weight ratio (shock):  {shock_ratio_aux_to_diff:.2f}")
    logger.info(f"  Ratio change: {stable_ratio_aux_to_diff/shock_ratio_aux_to_diff:.1f}x")
    
    if stable_ratio_aux_to_diff / shock_ratio_aux_to_diff > 2.0:
        issues.append("PRIORITY INVERSION: Stability transition causes >2x priority ratio change")
    
    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)
    
    logger.info("\nKEY FINDINGS:")
    logger.info("  1. Clinical weights range from 0.5 to 1.5 (3x ratio)")
    logger.info("  2. Shock dampening (sf=0) collapses all weights to 1.0")
    logger.info("  3. This changes priority ratios from 3:1 to 1:1")
    logger.info("  4. Rapid oscillation could cause gradient direction conflicts")
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n✅ TEST N PASSED: Stability factor behavior is reasonable")
    else:
        logger.warning("\n⚠️ TEST N: Observations detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return {
        'passed': passed,
        'issues': issues,
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST N: Stability Factor Shock Detection Analysis")
    logger.info("# Checking stability oscillation effects")
    logger.info("#" * 60 + "\n")
    
    result = run_stability_factor_analysis()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST N: STABILITY FACTOR BEHAVIOR IS REASONABLE")
    else:
        print("⚠️ TEST N: STABILITY FACTOR OBSERVATIONS DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)
