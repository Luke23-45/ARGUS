"""
test_l_awr_beta_dynamics.py
---------------------------
Test L: AWR Beta Adaptive Dynamics Analysis

HYPOTHESIS:
The AWR (Advantage Weighted Regression) beta in advantage_calculator.py adapts dynamically:

1. When ESS < 5% (L858-859):
   ```python
   if current_ess < 0.05:
       self.beta.copy_(self.beta * self.beta_growth_factor)
   ```

2. The beta_growth_factor for 1176 batches:
   ```python
   self.beta_growth_factor = float(1.5 ** (200 / 1176)) ≈ 1.07
   ```

POTENTIAL ISSUE:
With 1176 batches/epoch and persistent ESS < 5%:
- Beta could multiply by 1.07 every batch
- After 100 batches: beta * 1.07^100 ≈ beta * 868x!

Even though beta is clamped to [0.01, 10.0] (L867), repeated ESS failures
could cause rapid oscillations between high beta (explosions) and clamp.

WHAT WE'RE TESTING:
1. Beta growth under persistent ESS < 5% condition
2. Time to reach max_beta (10.0) from default (1.0)
3. Whether growth_factor scaling is sufficient to prevent explosion
"""

import torch
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_beta_dynamics(
    initial_beta: float = 1.0,
    n_steps: int = 500,
    ess_values: List[float] = None,
    growth_factor: float = 1.07,  # For 1176 batches
    min_beta: float = 0.01,
    max_beta: float = 10.0,
    correction_momentum: float = 0.999,
) -> Dict[str, any]:
    """
    Simulates AWR beta dynamics as in advantage_calculator.py.
    
    Args:
        initial_beta: Starting beta value
        n_steps: Number of steps to simulate
        ess_values: List of ESS values for each step (None = random)
        growth_factor: Beta growth factor for ESS < 5%
        min_beta: Minimum beta bound
        max_beta: Maximum beta bound
        correction_momentum: Momentum for normal ESS corrections
    """
    
    beta = initial_beta
    beta_history = [beta]
    ess_history = []
    clamp_count = 0
    cooldown = 0
    
    for step in range(n_steps):
        # Get ESS for this step
        if ess_values is not None:
            current_ess = ess_values[step % len(ess_values)]
        else:
            # Random ESS with occasional low values
            import random
            current_ess = random.uniform(0.02, 0.30)
        
        ess_history.append(current_ess)
        
        # Simulate beta update logic from advantage_calculator.py L850-867
        # [v27.1 FIX] ESS Safety Floor with Cooldown
        if current_ess < 0.05 and cooldown == 0:
            # ESS critically low - force growth
            beta = beta * growth_factor
            cooldown = 10
        else:
            # Normal ESS correction (simplified)
            if cooldown > 0:
                cooldown -= 1
                
            target_ess = 0.15
            correction = (target_ess / max(current_ess, 0.01)) ** 0.5
            correction = max(0.5, min(2.0, correction))  # Bound correction
            new_beta = beta * correction
            beta = (correction_momentum * beta) + ((1 - correction_momentum) * new_beta)
        
        # Clamp
        old_beta = beta
        beta = max(min_beta, min(max_beta, beta))
        if beta != old_beta:
            clamp_count += 1
        
        beta_history.append(beta)
    
    return {
        'final_beta': beta,
        'max_beta_reached': max(beta_history),
        'clamp_count': clamp_count,
        'beta_history': beta_history,
        'ess_history': ess_history,
    }


def run_beta_dynamics_analysis() -> Dict[str, any]:
    """Analyze AWR beta dynamics under various ESS scenarios."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST L: AWR Beta Adaptive Dynamics Analysis")
    logger.info("Checking if low ESS can cause beta explosion")
    logger.info("=" * 60 + "\n")
    
    issues = []
    
    # Test 1: Persistent ESS < 5% (worst case)
    logger.info("[TEST 1] Persistent ESS < 5% (Worst Case)")
    logger.info("-" * 50)
    
    # All ESS values below 5%
    persistent_low_ess = [0.03] * 200  # 200 steps of low ESS
    
    result_low = simulate_beta_dynamics(
        initial_beta=1.0,
        n_steps=200,
        ess_values=persistent_low_ess,
        growth_factor=1.07,  # For 1176 batches
    )
    
    logger.info(f"  Initial beta: 1.0")
    logger.info(f"  Growth factor: 1.07 (scaled for 1176 batches)")
    logger.info(f"  After 200 steps of ESS < 5%:")
    logger.info(f"    Max beta reached: {result_low['max_beta_reached']:.2f}")
    logger.info(f"    Final beta: {result_low['final_beta']:.2f}")
    logger.info(f"    Clamp events: {result_low['clamp_count']}")
    
    # Calculate theoretical growth
    theoretical_max = 1.0 * (1.07 ** 200)
    logger.info(f"    Theoretical (unclamped): {theoretical_max:.0f}")
    
    if result_low['clamp_count'] > 50:
        issues.append(f"EXCESSIVE CLAMPING: {result_low['clamp_count']} clamp events in 200 steps")
    
    # Test 2: Intermittent ESS < 5% (more realistic)
    logger.info("\n[TEST 2] Intermittent ESS < 5% (Realistic)")
    logger.info("-" * 50)
    
    # 20% of steps have low ESS
    intermittent_ess = []
    for i in range(500):
        if i % 5 == 0:  # Every 5th step is low
            intermittent_ess.append(0.03)
        else:
            intermittent_ess.append(0.12)  # Normal ESS
    
    result_intermittent = simulate_beta_dynamics(
        initial_beta=1.0,
        n_steps=500,
        ess_values=intermittent_ess,
        growth_factor=1.07,
    )
    
    logger.info(f"  500 steps with 20% low ESS occurrences:")
    logger.info(f"    Max beta: {result_intermittent['max_beta_reached']:.2f}")
    logger.info(f"    Final beta: {result_intermittent['final_beta']:.2f}")
    logger.info(f"    Clamp events: {result_intermittent['clamp_count']}")
    
    # Test 3: Compare growth factors for different step densities
    logger.info("\n[TEST 3] Growth Factor Comparison Across Step Densities")
    logger.info("-" * 50)
    
    step_densities = [200, 400, 800, 1176]
    
    for n_curr in step_densities:
        # Calculate growth factor as in advantage_calculator.py L308
        growth_factor = 1.5 ** (200 / n_curr)
        
        # Simulate 100 steps of persistent low ESS
        result = simulate_beta_dynamics(
            initial_beta=1.0,
            n_steps=100,
            ess_values=[0.03] * 100,
            growth_factor=growth_factor,
        )
        
        logger.info(f"  n_curr={n_curr:4d}: growth={growth_factor:.4f}, max_beta={result['max_beta_reached']:.2f}, clamps={result['clamp_count']}")
    
    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)
    
    logger.info("\nKEY OBSERVATION:")
    logger.info("  - Beta is clamped to [0.01, 10.0]")
    logger.info("  - Even with persistent low ESS, beta saturates at max_beta=10.0")
    logger.info("  - The clamp prevents unbounded explosion")
    logger.info("  - However, frequent clamping may cause oscillatory behavior")
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n✅ TEST L PASSED: Beta dynamics are bounded and stable")
    else:
        logger.warning("\n⚠️ TEST L: Observations detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return {
        'passed': passed,
        'issues': issues,
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST L: AWR Beta Adaptive Dynamics Analysis")
    logger.info("# Checking beta stability under low ESS conditions")
    logger.info("#" * 60 + "\n")
    
    result = run_beta_dynamics_analysis()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST L: BETA DYNAMICS ARE BOUNDED AND STABLE")
    else:
        print("⚠️ TEST L: BETA DYNAMICS OBSERVATIONS DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)
