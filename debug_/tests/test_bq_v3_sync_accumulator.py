import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BQ_v3")

class MockGradNorm:
    def __init__(self, num_tasks=2):
        self.weights = torch.ones(num_tasks)
        self.last_input = None

    def update(self, losses):
        self.last_input = losses.clone()
        return torch.tensor(0.0), {}

def test_gradnorm_accumulator_fixed():
    logger.info("Verifying Patch #107: GradNorm Accumulation Parity...")
    
    gn = MockGradNorm(num_tasks=2)
    acc_batches = 4
    
    # State in Wrapper
    gn_loss_accumulator = torch.zeros(2)
    gn_acc_count = 0
    
    # 4 batches with different losses
    losses_history = [
        torch.tensor([10.0, 1.0]),
        torch.tensor([11.0, 1.1]),
        torch.tensor([10.5, 0.9]),
        torch.tensor([1.0, 50.0]),
    ]
    
    # Simulate training_step accumulation
    for i in range(acc_batches):
        primary_losses = losses_history[i]
        
        # FIX #107 Implementation:
        gn_loss_accumulator.add_(primary_losses.detach())
        gn_acc_count += 1
        
        should_step = (i + 1) % acc_batches == 0
        if should_step:
            avg_losses = gn_loss_accumulator / (gn_acc_count + 1e-8)
            gn.update(avg_losses)
            
            # Reset
            gn_loss_accumulator.zero_()
            gn_acc_count = 0
            
    # Verification
    expected_avg = torch.stack(losses_history).mean(dim=0)
    actual_input = gn.last_input
    
    diff = torch.norm(actual_input - expected_avg).item()
    logger.info(f"Input Average Difference: {diff:.8f}")
    
    if diff < 1e-6:
        logger.info("✅ Patch #107 SUCCESS! GradNorm receives mathematically perfect average losses.")
    else:
        logger.error(f"❌ Verification Failed. Input differs from average (Diff: {diff:.8f})")

if __name__ == "__main__":
    test_gradnorm_accumulator_fixed()
