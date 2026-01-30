"""
test_h_agem_projection.py
-------------------------
Test H: A-GEM Gradient Projection Stability Analysis

HYPOTHESIS:
The A-GEM (Averaged Gradient Episodic Memory) implementation in wrapper_generalist.py
uses gradient projection to prevent catastrophic forgetting. The projection formula is:

  g' = g - (dot(g, g_ref) / dot(g_ref, g_ref)) * g_ref   [when dot(g, g_ref) < 0]

ACTUAL CODE (L1380-1385):
```python
if dot_val < 0:
    norm_val = ref_sq_tensors[i].sum() + 1e-8
    alphas.append(-1.0 * (dot_val / norm_val).item())
```

POTENTIAL ISSUES:
1. When g_ref has very small magnitude (norm → 0), the projection alpha explodes
2. The `+ 1e-8` guard may be insufficient for numerical stability
3. DDP synchronization (L1326-1349) averages g_ref, which could cause rank divergence

WHAT WE'RE TESTING:
1. Projection stability with various g_ref magnitudes
2. Alpha explosion when g_ref → 0
3. Impact of projection on gradient norm

This test does NOT import production code - it simulates the A-GEM math.
"""

import torch
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def project_gradients_agem(
    g: torch.Tensor,
    g_ref: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, float]:
    """
    Simulates A-GEM projection as implemented in wrapper_generalist.py L1380-1385.
    
    Returns:
        projected_g: The projected gradient
        alpha: The projection coefficient (0 if no projection needed)
    """
    dot_val = torch.dot(g.flatten(), g_ref.flatten())
    
    if dot_val < 0:
        # Projection needed - g conflicts with reference direction
        norm_val = torch.dot(g_ref.flatten(), g_ref.flatten()) + eps
        alpha = -1.0 * (dot_val / norm_val).item()
        projected_g = g + alpha * g_ref  # g' = g - (dot/norm)*(-g_ref) = g + alpha*g_ref
        return projected_g, alpha
    else:
        # No projection needed
        return g, 0.0


def test_projection_stability() -> Dict[str, any]:
    """Test A-GEM projection stability across various conditions."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST H: A-GEM Projection Stability Analysis")
    logger.info("Checking if projection is stable with small reference gradients")
    logger.info("=" * 60 + "\n")
    
    issues = []
    d = 512  # Typical gradient dimension
    
    # Test 1: Normal magnitude reference gradients
    logger.info("[TEST 1] Normal Reference Gradient Magnitudes")
    normal_results = []
    for magnitude in [1.0, 0.1, 0.01, 0.001]:
        g = torch.randn(d)
        g_ref = torch.randn(d) * magnitude
        
        projected, alpha = project_gradients_agem(g, g_ref)
        
        original_norm = g.norm().item()
        projected_norm = projected.norm().item()
        ratio = projected_norm / original_norm if original_norm > 0 else 0
        
        normal_results.append({
            'ref_magnitude': magnitude,
            'alpha': alpha,
            'original_norm': original_norm,
            'projected_norm': projected_norm,
            'norm_ratio': ratio
        })
        logger.info(f"  ref_mag={magnitude:.4f}: alpha={alpha:.4f}, norm_ratio={ratio:.4f}")
    
    # Test 2: Very small reference gradients (potential explosion point)
    logger.info("\n[TEST 2] Very Small Reference Gradient Magnitudes (Edge Case)")
    edge_results = []
    for magnitude in [1e-3, 1e-5, 1e-7, 1e-9]:
        g = torch.randn(d)
        g_ref = torch.randn(d) * magnitude
        
        projected, alpha = project_gradients_agem(g, g_ref)
        
        original_norm = g.norm().item()
        projected_norm = projected.norm().item()
        ratio = projected_norm / original_norm if original_norm > 0 else 0
        
        edge_results.append({
            'ref_magnitude': magnitude,
            'alpha': alpha,
            'original_norm': original_norm,
            'projected_norm': projected_norm,
            'norm_ratio': ratio
        })
        logger.info(f"  ref_mag={magnitude:.2e}: alpha={alpha:.4f}, norm_ratio={ratio:.4f}")
        
        # Check for explosion
        if abs(alpha) > 1000:
            issues.append(f"ALPHA EXPLOSION: alpha={alpha:.1f} when ref_mag={magnitude:.2e}")
        if ratio > 10:
            issues.append(f"NORM EXPLOSION: ratio={ratio:.1f}x when ref_mag={magnitude:.2e}")
    
    # Test 3: Conflicting gradients (worst case for projection)
    logger.info("\n[TEST 3] Maximally Conflicting Gradients")
    conflict_results = []
    for magnitude in [1.0, 0.1, 0.01]:
        g = torch.randn(d)
        g_ref = -g * magnitude  # Maximally opposed (180 degrees)
        
        projected, alpha = project_gradients_agem(g, g_ref)
        
        original_norm = g.norm().item()
        projected_norm = projected.norm().item()
        ratio = projected_norm / original_norm if original_norm > 0 else 0
        
        # In maximally opposed case, projection should significantly reduce gradient
        conflict_results.append({
            'ref_magnitude': magnitude,
            'alpha': alpha,
            'projected_norm': projected_norm,
            'norm_ratio': ratio
        })
        logger.info(f"  180° conflict, ref_mag={magnitude:.2f}: alpha={alpha:.4f}, norm_ratio={ratio:.4f}")
    
    # Test 4: Statistical stability (many trials)
    logger.info("\n[TEST 4] Statistical Stability (100 trials)")
    alphas_collected = []
    ratios_collected = []
    for _ in range(100):
        g = torch.randn(d)
        g_ref = torch.randn(d) * 0.1  # Modest reference magnitude
        projected, alpha = project_gradients_agem(g, g_ref)
        if alpha != 0:
            alphas_collected.append(alpha)
            ratios_collected.append(projected.norm().item() / g.norm().item())
    
    if alphas_collected:
        mean_alpha = sum(alphas_collected) / len(alphas_collected)
        max_alpha = max(alphas_collected)
        mean_ratio = sum(ratios_collected) / len(ratios_collected)
        max_ratio = max(ratios_collected)
        
        logger.info(f"  Projections triggered: {len(alphas_collected)}/100")
        logger.info(f"  Alpha - mean: {mean_alpha:.4f}, max: {max_alpha:.4f}")
        logger.info(f"  Norm ratio - mean: {mean_ratio:.4f}, max: {max_ratio:.4f}")
        
        if max_ratio > 5:
            issues.append(f"HIGH VARIANCE: max_ratio={max_ratio:.1f}x in statistical test")
    
    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n✅ TEST H PASSED: A-GEM projection is numerically stable")
        logger.info("   The eps=1e-8 guard appears sufficient for tested cases")
    else:
        logger.warning("\n⚠️ TEST H: Issues detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return {
        'passed': passed,
        'issues': issues,
        'normal_results': normal_results,
        'edge_results': edge_results,
        'conflict_results': conflict_results
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST H: A-GEM Gradient Projection Stability")
    logger.info("# Checking projection math for numerical edge cases")
    logger.info("#" * 60 + "\n")
    
    result = test_projection_stability()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST H: A-GEM PROJECTION IS STABLE")
    else:
        print("⚠️ TEST H: PROJECTION STABILITY ISSUES DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)
