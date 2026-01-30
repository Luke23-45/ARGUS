import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BQ")

class MockGradNorm:
    def __init__(self, num_tasks=2):
        self.weights = torch.ones(num_tasks)
        self.step_count = 0

    def get_weights(self):
        return self.weights

    def update(self, losses):
        # SIMULATE: GradNorm logic (simplified)
        # It calculates task weights based on current loss magnitudes
        # If one loss is huge, it decreases its weight.
        # [SOTA FIX]: This should use the AVERAGE loss across the cycle.
        # [BUG]: It only sees the loss of the last batch.
        
        # Simplified: weight = 1.0 / (loss + 1e-8)
        new_weights = 1.0 / (losses + 1e-8)
        new_weights = new_weights / new_weights.sum() * len(losses)
        
        self.weights = 0.9 * self.weights + 0.1 * new_weights
        return torch.tensor(0.0), {}

def test_gradnorm_accumulation_bias():
    logger.info("Simulating GradNorm Accumulation Bias (#107)...")
    
    gn = MockGradNorm(num_tasks=2)
    acc_batches = 4
    
    # SCENARIO: 
    # Batches 1-3: Loss A is high (Needs priority), Loss B is low.
    # Batch 4: Loss B spikes (Outlier), Loss A is low.
    
    losses_history = [
        torch.tensor([10.0, 1.0]), # B1
        torch.tensor([11.0, 1.1]), # B2
        torch.tensor([10.5, 0.9]), # B3
        torch.tensor([1.0, 50.0]), # B4 (Outlier spike)
    ]
    
    # 1. BUGGY BEHAVIOR: Only update on batch 4
    for i in range(acc_batches):
        current_losses = losses_history[i]
        if (i + 1) % acc_batches == 0:
            gn.update(current_losses)
            
    weights_buggy = gn.get_weights()
    logger.info(f"Buggy Weights (Rank 4 bias): {weights_buggy}")
    
    # 2. IDEAL BEHAVIOR: Update on Average
    gn_ideal = MockGradNorm(num_tasks=2)
    avg_losses = torch.stack(losses_history).mean(dim=0)
    gn_ideal.update(avg_losses)
    
    weights_ideal = gn_ideal.get_weights()
    logger.info(f"Ideal Weights (Global average): {weights_ideal}")
    
    # ANALYSIS:
    # In the buggy version, Task B (which normally is low priority)
    # gets a massive weight decrease (priority suppression) 
    # while Task A (which was the bottleneck for 75% of the data) 
    # gets a massive weight increase.
    # This causes a "Task Weight Lurch" at the epoch boundary or every 16 steps.
    
    diff = torch.norm(weights_buggy - weights_ideal).item()
    if diff > 0.1:
        logger.error(f"❌ Smoking Gun #107 CONFIRMED! GradNorm weights are biased by tail batch (Diff: {diff:.4f})")
        logger.warning("⚠️ Rationale: GradNorm must be fed the average losses of the entire accumulation cycle.")

if __name__ == "__main__":
    test_gradnorm_accumulation_bias()
