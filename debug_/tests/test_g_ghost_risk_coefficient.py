"""
test_g_ghost_risk_coefficient.py
---------------------------------
Test G: Ghost Sample Risk Coefficient Analysis

HYPOTHESIS:
Ghost samples are assigned a CONSTANT risk_coef = 2.0 (L657 in wrapper_generalist.py):
```python
risk_coef_ghost = torch.ones(risk_shape, device=self.device) * 2.0
```

This is 2x the typical risk coefficient for normal samples. With 4 ghosts per batch:
- Ghost samples receive amplified gradient signal
- This amplification is CONSTANT regardless of epoch or batch composition
- Could cause gradient instability if ghosts have high-variance embeddings

WHAT WE'RE TESTING:
1. Gradient magnitude with different ghost risk_coef values (1.0, 2.0, 4.0)
2. Whether ghost-induced gradients scale linearly with risk_coef
3. Impact of num_ghosts * risk_coef on total gradient norm

VERIFICATION:
This test does NOT import production code. It simulates the mathematical impact
of different risk_coef values on gradient magnitude.

Expected: risk_coef=2.0 should produce 2x gradient contribution vs risk_coef=1.0
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_ghost_gradients(
    batch_size: int = 32,
    num_ghosts: int = 4,
    risk_coef_ghosts: float = 2.0,
    risk_coef_samples: float = 1.0,
    d_model: int = 128,
) -> Dict[str, float]:
    """
    Simulates the gradient contribution of ghost samples vs regular samples.
    
    Based on the actual loss computation pattern in wrapper_generalist.py:
    - Loss for each sample is scaled by its risk_coef
    - Ghosts have constant risk_coef=2.0
    - Regular samples have variable risk_coef (1.0 is baseline)
    """
    
    # Create a simple linear layer to measure gradients
    linear = torch.nn.Linear(d_model, d_model)
    
    # Simulate embeddings
    regular_emb = torch.randn(batch_size, d_model, requires_grad=True)
    ghost_emb = torch.randn(num_ghosts, d_model, requires_grad=True)
    
    # Combine like the production code does
    all_emb = torch.cat([regular_emb, ghost_emb], dim=0)
    
    # Create risk coefficients (matching L657-660)
    regular_risk = torch.ones(batch_size) * risk_coef_samples
    ghost_risk = torch.ones(num_ghosts) * risk_coef_ghosts
    all_risk = torch.cat([regular_risk, ghost_risk], dim=0)
    
    # Simulate a weighted loss computation
    outputs = linear(all_emb)  # [B+G, D]
    targets = torch.randn_like(outputs)
    
    # Per-sample loss
    per_sample_loss = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1)  # [B+G]
    
    # Risk-weighted loss (how the production code effectively works)
    weighted_loss = (per_sample_loss * all_risk).mean()
    
    # Compute gradients
    weighted_loss.backward()
    
    # Measure gradient norms
    total_grad_norm = 0.0
    for p in linear.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    # Calculate theoretical contribution
    # Ghost contribution = num_ghosts * risk_coef_ghosts * avg_ghost_loss
    # Regular contribution = batch_size * risk_coef_samples * avg_regular_loss
    ghost_fraction = (num_ghosts * risk_coef_ghosts) / (batch_size * risk_coef_samples + num_ghosts * risk_coef_ghosts)
    
    return {
        'total_grad_norm': total_grad_norm,
        'ghost_fraction_of_weight': ghost_fraction,
        'num_ghosts': num_ghosts,
        'risk_coef_ghosts': risk_coef_ghosts,
    }


def run_comparative_test() -> Dict[str, any]:
    """Run comparative analysis of different ghost risk coefficients."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST G: Ghost Risk Coefficient Analysis")
    logger.info("Checking if ghost risk_coef=2.0 causes gradient amplification")
    logger.info("=" * 60 + "\n")
    
    results = []
    issues = []
    
    # Test different risk_coef values for ghosts
    risk_coefs = [1.0, 2.0, 3.0, 4.0]
    batch_size = 32
    num_ghosts = 4
    
    # Run multiple trials to reduce variance
    n_trials = 50
    
    for risk_coef in risk_coefs:
        trial_grads = []
        trial_fractions = []
        
        for _ in range(n_trials):
            result = simulate_ghost_gradients(
                batch_size=batch_size,
                num_ghosts=num_ghosts,
                risk_coef_ghosts=risk_coef,
                risk_coef_samples=1.0,
            )
            trial_grads.append(result['total_grad_norm'])
            trial_fractions.append(result['ghost_fraction_of_weight'])
        
        avg_grad = sum(trial_grads) / len(trial_grads)
        avg_fraction = sum(trial_fractions) / len(trial_fractions)
        
        results.append({
            'risk_coef': risk_coef,
            'avg_grad_norm': avg_grad,
            'ghost_weight_fraction': avg_fraction,
        })
        
        logger.info(f"risk_coef={risk_coef:.1f}: avg_grad_norm={avg_grad:.4f}, ghost_weight_fraction={avg_fraction:.3f}")
    
    # Analyze scaling behavior
    baseline_grad = results[0]['avg_grad_norm']  # risk_coef=1.0
    production_grad = results[1]['avg_grad_norm']  # risk_coef=2.0 (production)
    
    grad_amplification = production_grad / baseline_grad if baseline_grad > 0 else 0
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Production (risk_coef=2.0) vs Baseline (risk_coef=1.0):")
    logger.info(f"  Gradient amplification: {grad_amplification:.2f}x")
    logger.info(f"  Ghost weight fraction at 2.0: {results[1]['ghost_weight_fraction']:.3f}")
    
    # Check if amplification is significant
    if grad_amplification > 1.3:
        issues.append(f"SIGNIFICANT AMPLIFICATION: Ghost risk_coef=2.0 causes {grad_amplification:.2f}x gradient increase")
        logger.info(f"\n  ⚠️ Ghost samples with risk_coef=2.0 contribute disproportionately to gradients")
    else:
        logger.info(f"\n  ✓ Ghost contribution is within expected range")
    
    # Check consistency of scaling
    for i in range(1, len(results)):
        expected_fraction = (num_ghosts * risk_coefs[i]) / (batch_size * 1.0 + num_ghosts * risk_coefs[i])
        actual_fraction = results[i]['ghost_weight_fraction']
        if abs(expected_fraction - actual_fraction) > 0.01:
            issues.append(f"SCALING MISMATCH at risk_coef={risk_coefs[i]}")
    
    passed = len(issues) == 0
    
    if passed:
        logger.info("\n" + "=" * 60)
        logger.info("✅ TEST G PASSED: Ghost risk coefficient behavior is as expected")
        logger.info("=" * 60)
    else:
        logger.info("\n" + "=" * 60)
        logger.warning("⚠️ TEST G: Observations detected:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        logger.info("=" * 60)
    
    return {
        'passed': passed,
        'issues': issues,
        'grad_amplification': grad_amplification,
        'results': results
    }


if __name__ == "__main__":
    logger.info("\n" + "#" * 60)
    logger.info("# TEST G: Ghost Sample Risk Coefficient Analysis")
    logger.info("# Checking if constant risk_coef=2.0 causes gradient amplification")
    logger.info("#" * 60 + "\n")
    
    result = run_comparative_test()
    
    print("\n" + "=" * 60)
    if result['passed']:
        print("TEST G: GHOST RISK COEFFICIENT BEHAVIOR NORMAL")
    else:
        print("⚠️ TEST G: OBSERVATIONS DETECTED")
        for issue in result['issues']:
            print(f"  - {issue}")
    print("=" * 60)
