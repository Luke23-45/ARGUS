"""
test_f_dynamic_thresholding.py
------------------------------
Test F: DynamicThresholding (Governance) Layer Analysis

HYPOTHESIS:
The `DynamicThresholding` layer (used at L1100 for x0_approx governance) 
performs percentile-based rescaling. If x0 predictions have high variance:
1. The 99.5th percentile could be very different between batches
2. This causes variable scale factors per batch
3. Variable scaling could create gradient variance across batches

SIMULATION APPROACH:
1. Import actual DynamicThresholding class
2. Generate synthetic x0 predictions with different distributions
3. Measure how the scale factor varies
4. Check if outlier batches produce extreme rescaling

PASS CRITERIA:
- Scale factor should be relatively stable (std/mean < 0.2)
- No extreme scale factors (< 0.1 or > 10.0)

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_F")

# Try to import the actual class
try:
    from icu.utils.stability import DynamicThresholding
    PRODUCTION_AVAILABLE = True
except ImportError:
    logger.warning("Could not import DynamicThresholding - using local implementation")
    PRODUCTION_AVAILABLE = False
    
    # Fallback implementation matching stability.py:5-33
    class DynamicThresholding(nn.Module):
        """Local copy of production code for testing if import fails."""
        def __init__(self, percentile: float = 0.995, threshold: float = 3.0):
            super().__init__()
            self.percentile = percentile
            self.threshold = threshold

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            abs_x = torch.abs(x)
            flat_abs = abs_x.view(B, -1)
            
            s = torch.quantile(flat_abs.detach().float(), self.percentile, dim=1).view(B, 1, 1)
            s = torch.clamp(s, min=self.threshold)
            scale = self.threshold / s
            
            if x.dim() == 2:
                scale = scale.squeeze(-1)
                
            return x * scale


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_normal_x0(batch_size: int, seq_len: int, dim: int, sigma: float = 1.0) -> torch.Tensor:
    """Generate normal x0 predictions (normalized space, ~N(0, sigma))."""
    return torch.randn(batch_size, seq_len, dim) * sigma


def generate_outlier_x0(
    batch_size: int, 
    seq_len: int, 
    dim: int, 
    base_sigma: float = 1.0,
    outlier_fraction: float = 0.01,
    outlier_magnitude: float = 10.0
) -> torch.Tensor:
    """Generate x0 with outliers (some values far from mean)."""
    x = torch.randn(batch_size, seq_len, dim) * base_sigma
    
    # Add outliers
    n_outliers = int(x.numel() * outlier_fraction)
    if n_outliers > 0:
        flat = x.view(-1)
        outlier_indices = torch.randperm(flat.numel())[:n_outliers]
        flat[outlier_indices] = torch.randn(n_outliers) * outlier_magnitude
    
    return x


def generate_mixed_x0(
    batch_size: int, 
    seq_len: int, 
    dim: int,
    mix_ratio: float = 0.5
) -> torch.Tensor:
    """
    Generate mixed x0: some samples are stable, some have high variance.
    This simulates batches with mixed clinical states (stable + crisis).
    """
    x = torch.zeros(batch_size, seq_len, dim)
    
    n_stable = int(batch_size * mix_ratio)
    n_volatile = batch_size - n_stable
    
    # Stable samples: low variance
    x[:n_stable] = torch.randn(n_stable, seq_len, dim) * 0.5
    
    # Volatile samples: high variance (simulating crisis predictions)
    x[n_stable:] = torch.randn(n_volatile, seq_len, dim) * 3.0
    
    return x


# =============================================================================
# CORE TEST LOGIC
# =============================================================================

def analyze_thresholding_behavior(
    n_batches: int = 100,
    batch_size: int = 32,
    seq_len: int = 6,
    dim: int = 28,
    percentile: float = 0.995,
    threshold: float = 3.0
) -> Dict[str, any]:
    """
    Analyze how DynamicThresholding behaves across multiple batches.
    
    Returns:
        Dict with scale factor statistics and analysis
    """
    governance = DynamicThresholding(percentile=percentile, threshold=threshold)
    
    scale_factors_normal = []
    scale_factors_outlier = []
    scale_factors_mixed = []
    
    for _ in range(n_batches):
        # Test 1: Normal distribution
        x_normal = generate_normal_x0(batch_size, seq_len, dim, sigma=1.0)
        y_normal = governance(x_normal)
        
        # Calculate per-batch scale factor (output / input for non-zero elements)
        with torch.no_grad():
            mask = x_normal.abs() > 1e-6
            if mask.any():
                scales = (y_normal[mask] / x_normal[mask]).abs()
                scale_factors_normal.append(scales.mean().item())
        
        # Test 2: With outliers
        x_outlier = generate_outlier_x0(batch_size, seq_len, dim, outlier_fraction=0.02)
        y_outlier = governance(x_outlier)
        
        with torch.no_grad():
            mask = x_outlier.abs() > 1e-6
            if mask.any():
                scales = (y_outlier[mask] / x_outlier[mask]).abs()
                scale_factors_outlier.append(scales.mean().item())
        
        # Test 3: Mixed batch
        x_mixed = generate_mixed_x0(batch_size, seq_len, dim)
        y_mixed = governance(x_mixed)
        
        with torch.no_grad():
            mask = x_mixed.abs() > 1e-6
            if mask.any():
                scales = (y_mixed[mask] / x_mixed[mask]).abs()
                scale_factors_mixed.append(scales.mean().item())
    
    return {
        'normal': {
            'mean': sum(scale_factors_normal) / len(scale_factors_normal),
            'std': torch.tensor(scale_factors_normal).std().item(),
            'min': min(scale_factors_normal),
            'max': max(scale_factors_normal),
            'values': scale_factors_normal
        },
        'outlier': {
            'mean': sum(scale_factors_outlier) / len(scale_factors_outlier),
            'std': torch.tensor(scale_factors_outlier).std().item(),
            'min': min(scale_factors_outlier),
            'max': max(scale_factors_outlier),
            'values': scale_factors_outlier
        },
        'mixed': {
            'mean': sum(scale_factors_mixed) / len(scale_factors_mixed),
            'std': torch.tensor(scale_factors_mixed).std().item(),
            'min': min(scale_factors_mixed),
            'max': max(scale_factors_mixed),
            'values': scale_factors_mixed
        }
    }


def run_comparative_test() -> Dict[str, any]:
    """
    Test DynamicThresholding with various input distributions.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST F: DynamicThresholding Governance Analysis")
    logger.info(f"Checking if governance layer causes scale factor variance")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Production class available: {PRODUCTION_AVAILABLE}")
    
    results = analyze_thresholding_behavior(n_batches=100)
    
    # Log results
    logger.info("\n[RESULTS] Scale Factor Statistics:")
    for dist_type in ['normal', 'outlier', 'mixed']:
        stats = results[dist_type]
        logger.info(f"\n  {dist_type.upper()} distribution:")
        logger.info(f"    Mean scale: {stats['mean']:.4f}")
        logger.info(f"    Std scale: {stats['std']:.4f}")
        logger.info(f"    Min scale: {stats['min']:.4f}")
        logger.info(f"    Max scale: {stats['max']:.4f}")
        logger.info(f"    CV (std/mean): {stats['std']/stats['mean']:.4f}")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS")
    logger.info(f"{'='*60}")
    
    issues = []
    
    # Check 1: High variance in scale factors
    for dist_type in ['normal', 'outlier', 'mixed']:
        stats = results[dist_type]
        cv = stats['std'] / stats['mean']
        if cv > 0.2:
            issues.append(
                f"HIGH VARIANCE ({dist_type}): CV = {cv:.4f} > 0.2"
            )
    
    # Check 2: Extreme scale factors
    for dist_type in ['normal', 'outlier', 'mixed']:
        stats = results[dist_type]
        if stats['min'] < 0.1:
            issues.append(
                f"EXTREME COMPRESSION ({dist_type}): min scale = {stats['min']:.4f}"
            )
        if stats['max'] > 5.0:
            issues.append(
                f"EXTREME AMPLIFICATION ({dist_type}): max scale = {stats['max']:.4f}"
            )
    
    # Check 3: Large difference between outlier and normal
    normal_mean = results['normal']['mean']
    outlier_mean = results['outlier']['mean']
    if outlier_mean < normal_mean * 0.5:
        issues.append(
            f"OUTLIER SENSITIVITY: Outlier batches compressed {normal_mean/outlier_mean:.1f}x more"
        )
    
    results['issues'] = issues
    results['passed'] = len(issues) == 0
    
    logger.info(f"\n{'='*60}")
    if results['passed']:
        logger.info("TEST F PASSED: DynamicThresholding behavior is stable")
    else:
        logger.info("TEST F DETECTED ISSUES:")
        for issue in issues:
            logger.info(f"   - {issue}")
        logger.info("\nNOTE: This indicates POTENTIAL gradient variance, not proof of causation.")
    logger.info(f"{'='*60}\n")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "#"*60)
    logger.info("# TEST F: DynamicThresholding Governance Analysis")
    logger.info("# Checking if governance layer creates gradient variance")
    logger.info("#"*60 + "\n")
    
    result = run_comparative_test()
    
    # Final verdict
    print("\n" + "="*60)
    if result['passed']:
        print("TEST F: DYNAMIC THRESHOLDING BEHAVIOR IS STABLE")
        print("Scale factors are consistent across different input distributions.")
    else:
        print("TEST F: DYNAMIC THRESHOLDING SHOWS VARIANCE")
        print("Different input distributions produce different scale factors.")
        print("\nThis could contribute to gradient variance between batches.")
    print("="*60)
