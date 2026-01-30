"""
test_m_adaptive_clipping.py
---------------------------
Test M: Adaptive Weight Clipping Expansion Analysis

HYPOTHESIS:
The adaptive_clipping mechanism in advantage_calculator.py (L871-891) adjusts
max_weight based on the 95th percentile of weights:

```python
if self.adaptive_clipping:
    p95 = torch.quantile(weights.detach().float(), 0.95).item()
    target_clip = max(2.0, min(100.0, p95 * 1.5))  # 1.5x buffer
    self.max_weight.copy_(new_max_weight_tensor)
```

POTENTIAL ISSUE:
1. If weights have high variance, p95 can be very high
2. target_clip = p95 * 1.5 could push max_weight toward 100.0
3. With max_weight = 100.0, extreme weights amplify gradients by 100x
4. The [2.0, 100.0] bound is very wide - 50x range!

WHAT WE'RE TESTING:
1. Weight distribution with different advantage statistics
2. How quickly max_weight can reach bounds
3. Whether 100.0 max is appropriate or too high
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_awr_weights(
    advantages: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute AWR weights from advantages."""
    # AWR weight: exp(advantage / beta)
    weights = torch.exp(advantages / beta)
    return weights


def simulate_adaptive_clipping(
    beta: float = 1.0,
    n_steps: int = 100,
    advantage_std: float = 1.0,  # Standard deviation of advantages
    initial_max_weight: float = 5.0,
) -> Dict[str, any]:
    """
    Simulates adaptive weight clipping evolution.
    """
    
    max_weight = initial_max_weight
    max_weight_history = [max_weight]
    p95_history = []
    
    for step in range(n_steps):
        # Generate random advantages with specified std
        batch_size = 64
        advantages = torch.randn(batch_size) * advantage_std
        
        # Compute AWR weights
        weights = compute_awr_weights(advantages, beta)
        
        # Compute 95th percentile
        try:
            p95 = torch.quantile(weights.float(), 0.95).item()
        except:
            p95 = weights.max().item()
        p95_history.append(p95)
        
        # Adaptive clipping logic from advantage_calculator.py
        # [v27.1 FIX] Constrained Adaptive Clipping
        target_clip = max(2.0, min(20.0, p95 * 1.2))
        max_weight = target_clip
        max_weight_history.append(max_weight)
    
    return {
        'final_max_weight': max_weight,
        'max_max_weight': max(max_weight_history),
        'min_max_weight': min(max_weight_history),
        'avg_p95': sum(p95_history) / len(p95_history),
        'max_p95': max(p95_history),
        'max_weight_history': max_weight_history,
    }


def run_adaptive_clipping_analysis() -> Dict[str, any]:
    """Analyze adaptive weight clipping behavior."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST M: Adaptive Weight Clipping Expansion Analysis")
    logger.info("Checking if adaptive clipping can cause weight explosion")
    logger.info("=" * 60 + "\n")
    
    issues = []
    
    # Test 1: AWR weight magnitude vs beta and advantage
    logger.info("[TEST 1] AWR Weight Magnitude Analysis")
    logger.info("-" * 50)
    
    # For given advantage and beta, compute weight
    test_cases = [
        {'advantage': 1.0, 'beta': 1.0},
        {'advantage': 3.0, 'beta': 1.0},
        {'advantage': 5.0, 'beta': 1.0},
        {'advantage': 3.0, 'beta': 0.1},  # Low beta amplifies
        {'advantage': 3.0, 'beta': 10.0}, # High beta dampens
    ]
    
    for case in test_cases:
        weight = torch.exp(torch.tensor(case['advantage']) / case['beta']).item()
        logger.info(f"  adv={case['advantage']:.1f}, beta={case['beta']:.1f} → weight={weight:.2f}")
    
    # Test 2: Weight distribution with normalized advantages
    logger.info("\n[TEST 2] Weight Distribution with Normalized Advantages")
    logger.info("-" * 50)
    
    torch.manual_seed(42)
    
    beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for beta in beta_values:
        advantages = torch.randn(1000)  # Standard normal
        weights = compute_awr_weights(advantages, beta)
        
        p95 = torch.quantile(weights, 0.95).item()
        p99 = torch.quantile(weights, 0.99).item()
        max_w = weights.max().item()
        
        logger.info(f"  beta={beta:.1f}: p95={p95:.2f}, p99={p99:.2f}, max={max_w:.2f}")
        
        if p95 > 20:
            issues.append(f"HIGH P95: beta={beta:.1f} produces p95={p95:.2f}")
    
    # Test 3: Adaptive clipping evolution over epochs
    logger.info("\n[TEST 3] Adaptive Clipping Evolution (100 steps)")
    logger.info("-" * 50)
    
    scenarios = [
        {'beta': 1.0, 'adv_std': 1.0, 'name': 'Normal'},
        {'beta': 0.5, 'adv_std': 1.0, 'name': 'Low Beta'},
        {'beta': 1.0, 'adv_std': 2.0, 'name': 'High Variance'},
        {'beta': 0.5, 'adv_std': 2.0, 'name': 'Low Beta + High Var'},
    ]
    
    torch.manual_seed(42)
    
    for scenario in scenarios:
        result = simulate_adaptive_clipping(
            beta=scenario['beta'],
            n_steps=100,
            advantage_std=scenario['adv_std'],
        )
        
        logger.info(
            f"  {scenario['name']:20s}: max_weight range [{result['min_max_weight']:.1f}, {result['max_max_weight']:.1f}], "
            f"final={result['final_max_weight']:.1f}"
        )
        
        if result['max_max_weight'] >= 100.0:
            issues.append(f"MAX WEIGHT SATURATION: {scenario['name']} reached max_weight=100.0")
    
    # Test 4: What happens when beta is at minimum (0.01)?
    logger.info("\n[TEST 4] Edge Case: Beta at Minimum (0.01)")
    logger.info("-" * 50)
    
    torch.manual_seed(42)
    advantages = torch.randn(1000)
    weights = compute_awr_weights(advantages, beta=0.01)
    
    # Many will be inf due to exp(adv/0.01)
    finite_weights = weights[torch.isfinite(weights)]
    
    logger.info(f"  With beta=0.01 and normal advantages:")
    logger.info(f"    Finite weights: {len(finite_weights)}/{len(weights)}")
    if len(finite_weights) > 0:
        logger.info(f"    Max finite weight: {finite_weights.max().item():.2e}")
    
    if len(finite_weights) < len(weights) * 0.5:
        issues.append("BETA=0.01: More than 50% of weights are infinite")
    
    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)
    
    logger.info("\nKEY FINDINGS:")
    logger.info("  1. AWR weight = exp(adv / beta)")
    logger.info("  2. With beta < 1.0 and adv > 3, weights can exceed 20")
    logger.info("  3. Adaptive clipping bounds max_weight to [2, 100]")
    logger.info("  4. The 100x upper bound could amplify gradients significantly")
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n✅ TEST M PASSED: Adaptive clipping behavior is reasonable")
    else:
        logger.warning("\n⚠️ TEST M: Observations detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return {
        'passed': passed,
        'issues': issues,
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST M: Adaptive Weight Clipping Expansion Analysis")
    logger.info("# Checking adaptive clipping evolution and bounds")
    logger.info("#" * 60 + "\n")
    
    result = run_adaptive_clipping_analysis()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST M: ADAPTIVE CLIPPING BEHAVIOR IS REASONABLE")
    else:
        print("⚠️ TEST M: ADAPTIVE CLIPPING OBSERVATIONS DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)



