import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BP")

def test_tail_batch_scaling():
    logger.info("Simulating Tail Batch Gradient Scaling (#91)...")
    
    # 1. Standard Acc=4 step
    w = torch.tensor([1.0], requires_grad=True)
    lr = 0.1
    acc = 4
    
    # Sum 4 gradients of 1.0 each
    for _ in range(acc):
        loss = w * 1.0
        loss.backward()
    
    # Gradient Sum = 4.0
    # Step = -0.1 * 4.0 = -0.4
    grad_sum4 = w.grad.clone()
    logger.info(f"Full Acc Gradient Sum: {grad_sum4.item():.2f}")
    w.grad.zero_()
    
    # 2. Tail Acc=1 step (last batch of epoch)
    for _ in range(1):
        loss = w * 1.0
        loss.backward()
    
    # Gradient Sum = 1.0
    # Step = -0.1 * 1.0 = -0.1
    grad_sum1 = w.grad.clone()
    logger.info(f"Tail Acc Gradient Sum: {grad_sum1.item():.2f}")
    
    # DISCONTINUITY!
    # The tail step is 4x smaller than the standard steps.
    # In clinical trajectories, this "Half-Step" at the end of an epoch
    # leaves the model in an intermediate state, causing a "Manifold Shock"
    # when the next epoch starts with a full 16-acc step.
    
    ratio = grad_sum4.item() / grad_sum1.item()
    logger.info(f"Scaling Imbalance: {ratio:.1f}x")
    
    if abs(ratio - acc) < 0.1:
        logger.error(f"❌ Smoking Gun #91 CONFIRMED! Tail batches have {ratio:.1f}x smaller effective step sizes.")
        logger.warning("⚠️ Rationale: Manual optimization must divide gradients by 'actual_accum' to maintain step consistency.")

if __name__ == "__main__":
    test_tail_batch_scaling()
