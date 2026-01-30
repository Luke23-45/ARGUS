import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BL")

class MockTCB:
    def __init__(self, capacity=10, d_model=128):
        self.capacity = capacity
        self.queue = torch.zeros(capacity, d_model)
        self.ptr = 0
        self.temperature = 0.07

    def update_local(self, keys):
        # SIMULATE THE BUG: Local randperm
        indices = torch.randperm(keys.shape[0])
        keys = keys[indices]
        
        batch_size = keys.shape[0]
        num_fill = min(batch_size, self.capacity)
        self.queue[:num_fill] = keys[:num_fill]

    def forward(self, queries):
        # InfoNCE
        queries = F.normalize(queries, dim=1)
        queue = F.normalize(self.queue, dim=0) # Simple mock
        logits = torch.matmul(queries, queue.T) / self.temperature
        return logits.mean()

def test_tcb_rank_divergence():
    logger.info("Simulating TCB Rank Divergence (#87)...")
    
    # 1. Setup two identical TCBs (simulating Rank 0 and Rank 1)
    tcb0 = MockTCB()
    tcb1 = MockTCB()
    
    # 2. Provide IDENTICAL keys (gathered from DDP)
    keys_global = torch.randn(20, 128)
    
    # 3. Local Updates with local randomness
    torch.manual_seed(0) # Rank 0 seed
    tcb0.update_local(keys_global)
    
    torch.manual_seed(1) # Rank 1 seed (Different!)
    tcb1.update_local(keys_global)
    
    # 4. Check Queues
    diff_queue = torch.norm(tcb0.queue - tcb1.queue).item()
    logger.info(f"TCB Queue L2 Difference between Ranks: {diff_queue:.4f}")
    
    # 5. Compute Loss for same query
    query = torch.randn(1, 128)
    loss0 = tcb0.forward(query)
    loss1 = tcb1.forward(query)
    
    loss_diff = abs(loss0 - loss1).item()
    logger.info(f"Loss Difference between Ranks: {loss_diff:.4f}")
    
    if loss_diff > 1e-4:
        logger.error(f"❌ Smoking Gun #87 CONFIRMED! TCB losses diverge by {loss_diff:.4f} despite same global inputs.")
        logger.warning("⚠️ Rationale: Local shuffling in DDP ranks causes memory decoherence. All GPU units must store identical negatives for consistent InfoNCE gradients.")

if __name__ == "__main__":
    test_tcb_rank_divergence()
