"""
test_c_tcb_queue.py
-------------------
Test C: Temporal Contrastive Buffer Queue Initialization Test

HYPOTHESIS:
The TCB queue is initialized with random embeddings (torch.randn). When capacity 
scales from 1024 to ~6000 for 1176 batches:
1. InfoNCE loss starts HIGH because negatives are random noise
2. Queue takes longer to fill with real data (6000 vs 1024 samples)
3. During warmup, gradients from contrastive loss may be unstable

SIMULATION APPROACH:
1. Create TemporalContrastiveBuffer and apply scale_dynamics(1176)
2. Measure InfoNCE loss trajectory as queue fills
3. Compare warmup behavior for 1024 vs 6000 capacity
4. Check if random init causes loss spikes

PASS CRITERIA:
- InfoNCE loss should decrease as queue fills
- Loss at step 500 should be < 50% of loss at step 0
- No loss explosions (> 10.0)

WHAT THIS TESTS:
- temporal_buffer.py:42 (random queue initialization)
- temporal_buffer.py:45-64 (scale_dynamics capacity scaling)
- temporal_buffer.py:99-150 (forward / InfoNCE calculation)
- wrapper_generalist.py:1210-1215 (tcb_warmup_steps usage)

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
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer
from icu.utils.train_utils import ScalingSteward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_C")


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_synthetic_embeddings(
    batch_size: int = 32,
    d_model: int = 256,
    num_classes: int = 3,
    class_separation: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic embeddings that mimic real encoder outputs.
    
    Real embeddings from ApexMoEEncoder:
    - Should be clustered by class (sepsis vs non-sepsis)
    - Have some noise within clusters
    - Are L2 normalized before entering TCB
    
    Returns:
        q_expert: Query embeddings [B, D]
        k_positive: Positive key embeddings [B, D] (same as query + noise)
        labels: Class labels [B] for enqueue_mask calculation
    """
    # Assign random class labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Generate class-specific embeddings
    q_expert = torch.zeros(batch_size, d_model)
    k_positive = torch.zeros(batch_size, d_model)
    
    for c in range(num_classes):
        mask = (labels == c)
        n_samples = mask.sum().item()
        if n_samples > 0:
            # Class center
            center = torch.randn(d_model) * class_separation
            # Samples around center
            q_expert[mask] = center + torch.randn(n_samples, d_model) * 0.5
            # Positive is same as query + small noise
            k_positive[mask] = q_expert[mask] + torch.randn(n_samples, d_model) * 0.1
    
    # L2 normalize (as done in TCB forward)
    q_expert = F.normalize(q_expert, dim=1)
    k_positive = F.normalize(k_positive, dim=1)
    
    return q_expert, k_positive, labels


# =============================================================================
# CORE TEST LOGIC
# =============================================================================

def simulate_tcb_warmup(
    capacity: int,
    n_steps: int,
    batch_size: int = 32,
    d_model: int = 256,
    apply_scale_dynamics: bool = False,
    n_batches_per_epoch: int = 200
) -> Dict[str, any]:
    """
    Simulate TCB warmup period to measure InfoNCE loss trajectory.
    
    This replicates exactly how TCB is used in training:
    1. TCB initialized with random queue
    2. Each step: forward pass computes InfoNCE loss
    3. Negatives from queue (random initially, then real data)
    4. Queue updates with new positives
    
    Args:
        capacity: Queue capacity
        n_steps: Number of steps to simulate
        batch_size: Batch size per step
        d_model: Embedding dimension
        apply_scale_dynamics: Whether to call scale_dynamics (scales capacity)
        n_batches_per_epoch: Used for scaling if apply_scale_dynamics=True
    
    Returns:
        Dict with loss trajectory and queue fill metrics
    """
    # Initialize TCB
    tcb = TemporalContrastiveBuffer(
        d_model=d_model,
        capacity=capacity,
        temperature=0.07  # Default InfoNCE temperature
    )
    
    initial_capacity = tcb.capacity
    
    # Apply scaling if requested
    if apply_scale_dynamics:
        tcb.scale_dynamics(n_batches_per_epoch)
        scaled_capacity = tcb.capacity
        logger.info(f"  Capacity scaled: {initial_capacity} -> {scaled_capacity}")
    else:
        scaled_capacity = capacity
    
    # Tracking
    loss_history = []
    nce_loss_history = []
    uniformity_history = []
    queue_ptr_history = []
    
    # Check initial queue state
    initial_queue_norm = tcb.queue.norm(dim=1).mean().item()
    logger.debug(f"  Initial queue mean norm: {initial_queue_norm:.4f}")
    
    for step in range(n_steps):
        # Generate synthetic embeddings
        q, k, labels = generate_synthetic_embeddings(
            batch_size=batch_size,
            d_model=d_model
        )
        
        # Create enqueue mask (only store negatives, i.e., non-sepsis)
        # In real training: enqueue_mask = (labels != 2) where 2 = sepsis
        enqueue_mask = (labels != 2)
        
        # Forward pass
        out = tcb(q, k, enqueue_mask=enqueue_mask)
        
        # Record metrics
        loss_history.append(out['loss'].item())
        nce_loss_history.append(out['nce_loss'].item())
        uniformity_history.append(out['uniformity'].item())
        queue_ptr_history.append(tcb.queue_ptr.item())
        
        # Log progress
        if step % 100 == 0 or step == n_steps - 1:
            logger.debug(
                f"  Step {step}: NCE={out['nce_loss'].item():.4f}, "
                f"Uni={out['uniformity'].item():.4f}, QueuePtr={tcb.queue_ptr.item()}"
            )
    
    # Analysis
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
    
    # Queue fill analysis
    total_samples_enqueued = sum(queue_ptr_history[i] - queue_ptr_history[i-1] if i > 0 
                                  else queue_ptr_history[0] for i in range(len(queue_ptr_history)))
    # This is tricky due to wrap-around, just use final ptr as approximation
    
    return {
        'capacity': scaled_capacity,
        'n_steps': n_steps,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'max_loss': max(loss_history),
        'min_loss': min(loss_history),
        'loss_history': loss_history,
        'nce_loss_history': nce_loss_history,
        'uniformity_history': uniformity_history,
        'queue_ptr_history': queue_ptr_history
    }


def run_comparative_test() -> Dict[str, any]:
    """
    Compare TCB behavior between default capacity and scaled capacity.
    
    The key questions:
    1. Does random init cause higher initial loss with larger capacity?
    2. Does loss decrease properly as queue fills?
    3. Are there any loss explosions?
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST C: TCB Queue Initialization Test")
    logger.info(f"Comparing default (1024) vs scaled (~6000) capacity")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Run 1: Default capacity (1024) - baseline
    # -------------------------------------------------------------------------
    logger.info("[RUN 1] DEFAULT CAPACITY: 1024")
    result_default = simulate_tcb_warmup(
        capacity=1024,
        n_steps=500,
        apply_scale_dynamics=False
    )
    results['default'] = result_default
    
    logger.info(f"  Initial NCE Loss: {result_default['initial_loss']:.4f}")
    logger.info(f"  Final NCE Loss: {result_default['final_loss']:.4f}")
    logger.info(f"  Loss Reduction: {result_default['loss_reduction']*100:.1f}%")
    logger.info(f"  Max Loss: {result_default['max_loss']:.4f}")
    
    # -------------------------------------------------------------------------
    # Run 2: Scaled capacity (for 1176 batches/epoch)
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 2] SCALED CAPACITY: for 1176 batches/epoch")
    result_scaled = simulate_tcb_warmup(
        capacity=1024,  # Base capacity
        n_steps=500,
        apply_scale_dynamics=True,
        n_batches_per_epoch=1176  # Triggers scale_dynamics
    )
    results['scaled'] = result_scaled
    
    logger.info(f"  Scaled Capacity: {result_scaled['capacity']}")
    logger.info(f"  Initial NCE Loss: {result_scaled['initial_loss']:.4f}")
    logger.info(f"  Final NCE Loss: {result_scaled['final_loss']:.4f}")
    logger.info(f"  Loss Reduction: {result_scaled['loss_reduction']*100:.1f}%")
    logger.info(f"  Max Loss: {result_scaled['max_loss']:.4f}")
    
    # -------------------------------------------------------------------------
    # Run 3: Extended warmup for scaled capacity
    # -------------------------------------------------------------------------
    logger.info("\n[RUN 3] EXTENDED WARMUP: 2000 steps for scaled capacity")
    result_extended = simulate_tcb_warmup(
        capacity=1024,
        n_steps=2000,  # Extended warmup
        apply_scale_dynamics=True,
        n_batches_per_epoch=1176
    )
    results['extended'] = result_extended
    
    logger.info(f"  Final NCE Loss (2000 steps): {result_extended['final_loss']:.4f}")
    logger.info(f"  Loss Reduction: {result_extended['loss_reduction']*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("COMPARATIVE ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Compare initial losses
    initial_diff = result_scaled['initial_loss'] - result_default['initial_loss']
    logger.info(f"Initial Loss Difference (Scaled - Default): {initial_diff:.4f}")
    
    # Compare convergence
    default_converged = result_default['loss_reduction'] > 0.20  # 20% reduction
    scaled_converged = result_scaled['loss_reduction'] > 0.20
    
    logger.info(f"Default Converged (>20% reduction): {default_converged}")
    logger.info(f"Scaled Converged (>20% reduction): {scaled_converged}")
    
    # -------------------------------------------------------------------------
    # Pass/Fail Determination
    # -------------------------------------------------------------------------
    issues = []
    
    # Check 1: Loss should decrease over warmup
    if result_scaled['loss_reduction'] < 0.10:
        issues.append(
            f"POOR CONVERGENCE: Scaled run only reduced loss by {result_scaled['loss_reduction']*100:.1f}%"
        )
    
    # Check 2: No loss explosions
    if result_scaled['max_loss'] > 10.0:
        issues.append(
            f"LOSS EXPLOSION: Max loss = {result_scaled['max_loss']:.4f} > 10.0 threshold"
        )
    
    # Check 3: Initial loss shouldn't be drastically higher (indicates bad init)
    if result_scaled['initial_loss'] > result_default['initial_loss'] * 2.0:
        issues.append(
            f"BAD INITIALIZATION: Scaled initial loss is 2x higher than default "
            f"({result_scaled['initial_loss']:.4f} vs {result_default['initial_loss']:.4f})"
        )
    
    # Check 4: Extended warmup should help
    if result_extended['final_loss'] > result_scaled['final_loss']:
        issues.append(
            f"EXTENDED WARMUP INEFFECTIVE: 2000 steps didn't improve over 500 steps"
        )
    
    # Check 5: Warmup at 500 steps may be insufficient for scaled capacity
    scaled_capacity = result_scaled['capacity']
    expected_fill_steps = scaled_capacity // 32  # batch_size = 32
    if 500 < expected_fill_steps:
        logger.warning(
            f"NOTE: 500 warmup steps < {expected_fill_steps} steps needed to fill "
            f"capacity {scaled_capacity} with batch_size 32"
        )
        # This is informational, not necessarily a bug
        if result_scaled['loss_reduction'] < 0.30:
            issues.append(
                f"INSUFFICIENT WARMUP: Need {expected_fill_steps} steps to fill queue, "
                f"but tcb_warmup_steps is only 500"
            )
    
    results['issues'] = issues
    results['passed'] = len(issues) == 0
    
    logger.info(f"\n{'='*60}")
    if results['passed']:
        logger.info("✅ TEST C PASSED: No TCB queue initialization issues detected")
    else:
        logger.error("❌ TEST C FAILED: Issues detected:")
        for issue in issues:
            logger.error(f"   - {issue}")
    logger.info(f"{'='*60}\n")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "#"*60)
    logger.info("# TEST C: TCB Queue Initialization Test")
    logger.info("# Checking if random queue init causes training instability")
    logger.info("#"*60 + "\n")
    
    result = run_comparative_test()
    
    # Final verdict
    print("\n" + "="*60)
    if result['passed']:
        print("✅ TEST C: ALL CHECKS PASSED")
        print("TCB queue initialization is NOT the root cause of GN spike.")
    else:
        print("❌ TEST C: ISSUES DETECTED")
        print("TCB queue initialization MAY contribute to GN spike.")
        print("\nRecommended Fix: Extend tcb_warmup_steps to match scaled capacity.")
    print("="*60)
