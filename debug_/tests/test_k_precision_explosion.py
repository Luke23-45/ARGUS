"""
test_k_precision_explosion.py
-----------------------------
Test K: Bayesian Precision Explosion Analysis

HYPOTHESIS:
In BayesianProjectedScaler (loss_scaler.py:L86), the precision is computed as:

    precision = torch.exp(-log_vars)

When log_vars becomes very negative (e.g., -5.0 at the lower bound):
    precision = exp(-(-5.0)) = exp(5.0) = 148.4

This creates a 148x amplification of that task's loss contribution.

ACTUAL CODE (L22, L86, L117):
```python
self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # L22
precision = torch.exp(-log_vars_active)  # L86
self.log_vars.clamp_(min=-2.0, max=5.0)  # L117 - projection bounds
```

POTENTIAL ISSUE:
With log_vars at -2.0 (lower bound): precision = exp(2.0) = 7.4x
With log_vars at 5.0 (upper bound): precision = exp(-5.0) = 0.007x

The asymmetry creates potential for tasks with low log_vars to dominate gradients.

WHAT WE'RE TESTING:
1. Precision range with current bounds [-2, 5]
2. Gradient contribution at boundary values
3. Whether precision asymmetry could cause GN spikes
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_precision_range(min_log_var: float = -2.0, max_log_var: float = 5.0) -> Dict[str, float]:
    """Compute the precision range given log_var bounds."""
    
    # Precision at boundaries
    precision_at_min = torch.exp(torch.tensor(-min_log_var)).item()  # Most certain
    precision_at_max = torch.exp(torch.tensor(-max_log_var)).item()  # Least certain
    precision_at_zero = 1.0  # Default initialization
    
    return {
        'precision_at_min_logvar': precision_at_min,
        'precision_at_max_logvar': precision_at_max,
        'precision_at_zero': precision_at_zero,
        'precision_ratio': precision_at_min / precision_at_max if precision_at_max > 0 else float('inf'),
    }


def simulate_weighted_loss(log_vars: torch.Tensor, losses: torch.Tensor) -> Dict[str, float]:
    """
    Simulates the Bayesian loss weighting as in BayesianProjectedScaler.
    """
    precision = torch.exp(-log_vars)
    regularization = F.softplus(log_vars)
    
    # Weighted loss: 0.5 * precision * loss + regularization
    weighted = 0.5 * precision * losses + regularization
    
    return {
        'log_vars': log_vars.tolist(),
        'precision': precision.tolist(),
        'regularization': regularization.tolist(),
        'weighted_losses': weighted.tolist(),
        'total_loss': weighted.sum().item(),
    }


def run_precision_analysis() -> Dict[str, any]:
    """Analyze precision behavior across log_var bounds."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST K: Bayesian Precision Explosion Analysis")
    logger.info("Checking if exp(-log_vars) can cause loss weight explosion")
    logger.info("=" * 60 + "\n")
    
    issues = []
    
    # Test 1: Precision Range with PATCHED Bounds [-1.5, 3.0]
    logger.info("[TEST 1] Precision Range with PATCHED Bounds [-1.5, 3.0]")
    logger.info("-" * 50)
    
    range_info = compute_precision_range(-1.5, 3.0)
    logger.info(f"  log_var=-1.5 (max certainty): precision = {range_info['precision_at_min_logvar']:.2f}")
    logger.info(f"  log_var= 0.0 (default):       precision = {range_info['precision_at_zero']:.2f}")
    logger.info(f"  log_var= 3.0 (min certainty): precision = {range_info['precision_at_max_logvar']:.6f}")
    logger.info(f"  Precision ratio (max/min): {range_info['precision_ratio']:.0f}x")
    
    if range_info['precision_ratio'] > 100:
        issues.append(f"EXTREME PRECISION RATIO: {range_info['precision_ratio']:.0f}x between log_var bounds")
    
    # Test 2: Simulate realistic loss weighting scenarios
    logger.info("\n[TEST 2] Simulated Loss Weighting Scenarios")
    logger.info("-" * 50)
    
    # Scenario A: All tasks at default (log_var=0)
    log_vars_default = torch.zeros(7)
    losses_equal = torch.ones(7) * 1.0
    result_default = simulate_weighted_loss(log_vars_default, losses_equal)
    
    logger.info(f"  Scenario A (all log_var=0): total_loss = {result_default['total_loss']:.2f}")
    
    # Scenario B: One task at min bound (high certainty)
    log_vars_one_certain = torch.zeros(7)
    log_vars_one_certain[0] = -2.0  # Task 0 is very certain
    result_one_certain = simulate_weighted_loss(log_vars_one_certain, losses_equal)
    
    logger.info(f"  Scenario B (one log_var=-2.0): total_loss = {result_one_certain['total_loss']:.2f}")
    logger.info(f"    Task 0 weighted loss: {result_one_certain['weighted_losses'][0]:.2f}")
    logger.info(f"    Task 1 weighted loss: {result_one_certain['weighted_losses'][1]:.2f}")
    
    # Check for dominance
    ratio = result_one_certain['weighted_losses'][0] / result_one_certain['weighted_losses'][1] if result_one_certain['weighted_losses'][1] > 0 else float('inf')
    if ratio > 5:
        issues.append(f"TASK DOMINANCE: log_var=-2.0 creates {ratio:.1f}x loss weight vs default")
    
    # Scenario C: Extreme case - one task at min, one at max
    log_vars_extreme = torch.zeros(7)
    log_vars_extreme[0] = -2.0  # Very certain
    log_vars_extreme[1] = 5.0   # Very uncertain
    result_extreme = simulate_weighted_loss(log_vars_extreme, losses_equal)
    
    logger.info(f"\n  Scenario C (log_var: -2.0 vs 5.0):")
    logger.info(f"    Task 0 (certain): precision={result_extreme['precision'][0]:.2f}, weighted={result_extreme['weighted_losses'][0]:.2f}")
    logger.info(f"    Task 1 (uncertain): precision={result_extreme['precision'][1]:.6f}, weighted={result_extreme['weighted_losses'][1]:.2f}")
    
    extreme_ratio = result_extreme['weighted_losses'][0] / result_extreme['weighted_losses'][1] if result_extreme['weighted_losses'][1] > 0 else float('inf')
    if extreme_ratio > 100:
        issues.append(f"EXTREME TASK IMBALANCE: {extreme_ratio:.0f}x between certain and uncertain tasks")
    
    # Test 3: Gradient magnitude under extreme certainty
    logger.info("\n[TEST 3] Gradient Magnitude Analysis")
    logger.info("-" * 50)
    
    # Simulate gradients with learnable log_vars
    log_vars_param = torch.tensor([-2.0], requires_grad=True)
    loss_value = torch.tensor([1.0])
    
    precision = torch.exp(-log_vars_param)
    weighted_loss = 0.5 * precision * loss_value
    weighted_loss.backward()
    
    grad_at_min = log_vars_param.grad.item()
    logger.info(f"  d(weighted_loss)/d(log_var) at log_var=-2.0: {grad_at_min:.4f}")
    
    # Compare with default
    log_vars_default = torch.tensor([0.0], requires_grad=True)
    precision = torch.exp(-log_vars_default)
    weighted_loss = 0.5 * precision * loss_value
    weighted_loss.backward()
    
    grad_at_zero = log_vars_default.grad.item()
    logger.info(f"  d(weighted_loss)/d(log_var) at log_var=0.0: {grad_at_zero:.4f}")
    logger.info(f"  Gradient ratio: {abs(grad_at_min / grad_at_zero) if grad_at_zero != 0 else 'inf'}x")
    
    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n✅ TEST K PASSED: Precision behavior is within expected range")
    else:
        logger.warning("\n⚠️ TEST K: Observations detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return {
        'passed': passed,
        'issues': issues,
        'precision_range': range_info
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST K: Bayesian Precision Explosion Analysis")
    logger.info("# Checking loss_scaler precision weighting behavior")
    logger.info("#" * 60 + "\n")
    
    result = run_precision_analysis()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST K: PRECISION BEHAVIOR IS WITHIN EXPECTED RANGE")
    else:
        print("⚠️ TEST K: PRECISION OBSERVATIONS DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)
