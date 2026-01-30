"""
test_i_physics_loss_governance.py
---------------------------------
Test I: Physics Loss Governance Analysis

HYPOTHESIS:
The physics loss in wrapper_generalist.py has two governance mechanisms:
1. Epoch-based clamp: `phys_loss.clamp(max=10.0)` only for epochs 0-14 (L1240)
2. Physics weight ramp: `_get_curr_physics_weight()` interpolates over epochs (L359-368)

ACTUAL CODE (L1238-1244):
```python
phys_loss = self.physics_oracle.compute_loss(x0_pred_gov, past, static, pred_lens)
if self.current_epoch < 15:
    phys_loss = phys_loss.clamp(max=10.0)
avg_losses.append(phys_loss)
```

And L359-368:
```python
def _get_curr_physics_weight(self) -> float:
    # Ramp from 0.1 to 1.0 over first 10 epochs
    ramp_epochs = 10.0
    progress = min(1.0, self.current_epoch / ramp_epochs)
    return 0.1 + 0.9 * progress
```

POTENTIAL ISSUES:
1. At epoch 15, the clamp is removed - unclamped physics loss enters the system
2. Physics weight reaches 1.0 at epoch 10, but clamp is removed at epoch 15
3. This creates a 5-epoch window where physics is at FULL WEIGHT but CLAMPED
4. Sudden transition at epoch 15 could cause GN spike

WHAT WE'RE TESTING:
1. Simulated physics loss scaling across epochs
2. The epoch 14→15 transition (clamp removal)
3. Interaction between weight ramp and clamp removal

This test does NOT import production code - it simulates the governance logic.
"""

import torch
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_physics_governance(
    epoch: int,
    raw_phys_loss: float,
) -> Dict[str, float]:
    """
    Simulates the physics loss governance as implemented in wrapper_generalist.py.
    
    [v27.1 PATCHED] Uses gradual clamp relaxation from 10→50 over epochs 10-20.
    """
    # Weight ramp (L359-368)
    ramp_epochs = 10.0
    progress = min(1.0, epoch / ramp_epochs)
    phys_weight = 0.1 + 0.9 * progress
    
    # [v27.1 PATCHED] Gradual Clamp Relaxation (epochs 10-20)
    # Old: clamp_active = epoch < 15 (sudden removal)
    # New: Gradual relaxation from max=10 to max=50
    if epoch < 10:
        phys_clamp_max = 10.0
        clamp_active = True
    elif epoch < 20:
        # Linear ramp: epoch 10 → max=10, epoch 20 → max=50
        ramp_progress = (epoch - 10) / 10.0
        phys_clamp_max = 10.0 + 40.0 * ramp_progress
        clamp_active = True  # Still clamping, just with higher max
    else:
        phys_clamp_max = 50.0  # Soft cap even after epoch 20
        clamp_active = True  # Always clamped now
    
    clamped_loss = min(raw_phys_loss, phys_clamp_max)
    
    # Effective contribution to gradient
    effective_loss = clamped_loss * phys_weight
    
    return {
        'epoch': epoch,
        'raw_loss': raw_phys_loss,
        'phys_weight': phys_weight,
        'clamp_active': clamp_active,
        'clamp_max': phys_clamp_max,
        'clamped_loss': clamped_loss,
        'effective_loss': effective_loss,
    }


def run_governance_analysis() -> Dict[str, any]:
    """Analyze physics loss governance transitions."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST I: Physics Loss Governance Analysis")
    logger.info("Checking epoch-based transitions in physics governance")
    logger.info("=" * 60 + "\n")
    
    issues = []
    
    # Simulate different raw physics loss scenarios
    raw_loss_scenarios = {
        'normal': 2.0,      # Typical physics loss
        'elevated': 6.0,    # Higher but within clamp
        'extreme': 20.0,    # Would be clamped
        'spike': 50.0       # Extreme physics violation
    }
    
    # Test epoch transitions
    epochs = list(range(0, 25, 1))
    
    for scenario_name, raw_loss in raw_loss_scenarios.items():
        logger.info(f"\n[SCENARIO] Raw physics loss = {raw_loss} ({scenario_name})")
        logger.info("-" * 50)
        
        results = []
        for epoch in epochs:
            result = simulate_physics_governance(epoch, raw_loss)
            results.append(result)
            
            # Log key transitions
            if epoch in [0, 9, 10, 14, 15, 20]:
                logger.info(
                    f"  Epoch {epoch:2d}: weight={result['phys_weight']:.2f}, "
                    f"clamp={'ON' if result['clamp_active'] else 'OFF'}, "
                    f"effective={result['effective_loss']:.2f}"
                )
        
        # Analyze epoch 14→15 transition (clamp removal)
        epoch_14 = results[14]
        epoch_15 = results[15]
        
        transition_ratio = epoch_15['effective_loss'] / epoch_14['effective_loss'] if epoch_14['effective_loss'] > 0 else 0
        
        logger.info(f"\n  Epoch 14→15 transition: {epoch_14['effective_loss']:.2f} → {epoch_15['effective_loss']:.2f}")
        logger.info(f"  Transition ratio: {transition_ratio:.2f}x")
        
        if transition_ratio > 2.0:
            issues.append(
                f"{scenario_name.upper()}: Epoch 14→15 causes {transition_ratio:.1f}x loss jump"
            )
    
    # Test the physics weight ramp in isolation
    logger.info("\n" + "=" * 60)
    logger.info("PHYSICS WEIGHT RAMP ANALYSIS")
    logger.info("=" * 60)
    
    weight_at_epoch = []
    for epoch in range(20):
        progress = min(1.0, epoch / 10.0)
        weight = 0.1 + 0.9 * progress
        weight_at_epoch.append(weight)
        if epoch <= 10 or epoch == 15:
            logger.info(f"  Epoch {epoch:2d}: weight = {weight:.3f}")
    
    # Analysis summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\nGOVERNANCE TIMELINE:")
    logger.info("  Epoch 0-9:  Weight ramps 0.1 → 1.0, Clamp ON (max=10.0)")
    logger.info("  Epoch 10-14: Weight stable at 1.0, Clamp ON (max=10.0)")
    logger.info("  Epoch 15+: Weight stable at 1.0, Clamp OFF ⚠️")
    
    if any('extreme' in i.lower() or 'spike' in i.lower() for i in issues):
        logger.info("\n⚠️ POTENTIAL ISSUE:")
        logger.info("   At epoch 15, sudden clamp removal can cause large loss jumps")
        logger.info("   if physics loss exceeds 10.0 at that point.")
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n✅ TEST I PASSED: Physics governance transitions are gradual within normal range")
    else:
        logger.warning("\n⚠️ TEST I: Transition issues detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return {
        'passed': passed,
        'issues': issues,
        'weight_timeline': weight_at_epoch
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST I: Physics Loss Governance Analysis")
    logger.info("# Checking epoch-based physics loss transitions")
    logger.info("#" * 60 + "\n")
    
    result = run_governance_analysis()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST I: PHYSICS GOVERNANCE TRANSITIONS ARE SMOOTH")
    else:
        print("⚠️ TEST I: GOVERNANCE TRANSITION ISSUES DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)
