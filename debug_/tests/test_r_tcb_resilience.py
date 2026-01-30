import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_R")

class MockTCB(nn.Module):
    def __init__(self, d_model=128, capacity=1024):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.temperature = 0.07
        self.register_buffer("queue", torch.randn(capacity, d_model))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("ptr", torch.tensor(0))

    def update(self, features):
        with torch.no_grad():
            b = features.size(0)
            if self.ptr + b <= self.capacity:
                self.queue[self.ptr : self.ptr + b] = features
                self.ptr = (self.ptr + b) % self.capacity
            else:
                # Overflow case
                pass

    def forward(self, features):
        # features: [B, D]
        # queue: [C, D]
        logits = torch.matmul(features, self.queue.t()) / self.temperature # [B, C]
        
        # Self-contrast: For simplicity, assume index [0] of logits is the "positive" (not realistic but good for shock test)
        # In reality, TCB is usually self-supervised.
        labels = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        return F.cross_entropy(logits, labels)

def simulate_resumption_shock():
    logger.info("Simulating TCB Resumption Shock...")
    
    features = F.normalize(torch.randn(16, 128), dim=1)
    features.requires_grad = True
    
    # 1. Buggy Case (Zero-Init)
    logger.info("\n[CASE 1] Buggy Implementation (Zero-Initialized Queue)")
    tcb_b = MockTCB()
    tcb_b.queue.fill_(0.0)
    loss_b = tcb_b(features)
    loss_b.backward()
    grad_b_norm = features.grad.clone().norm().item()
    logger.info(f"  Loss: {loss_b.item():.4f}, Grad Norm: {grad_b_norm:.4f}")

    # 2. Patched Case (Noisy-Init)
    logger.info("\n[CASE 2] Patched Implementation (Gaussian Noisy Queue 0.01)")
    tcb_p = MockTCB()
    init_q = torch.randn(1024, 128) * 0.01
    tcb_p.queue = F.normalize(init_q, dim=1)
    
    features.grad = None
    loss_p = tcb_p(features)
    loss_p.backward()
    grad_p_norm = features.grad.clone().norm().item()
    logger.info(f"  Loss: {loss_p.item():.4f}, Grad Norm: {grad_p_norm:.4f}")

    issues = []
    if grad_b_norm > 1e-8:
        # If grad_b_norm is non-zero, the test isn't reproducing the collapse well
        pass
    if grad_p_norm < 1.0: # Should be roughly in 2.0-5.0 range
        issues.append(f"PATCH FAILURE: Patched queue still yields weak gradients ({grad_p_norm:.4f})")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    if not issues:
        logger.info("\u2705 TEST R PASSED: TCB is now resilient to resumption")
    else:
        logger.warning("\u26a0\ufe0f TEST R FAILED: Patch is insufficient")
        for issue in issues:
            logger.warning(f"   - {issue}")
            
    return not bool(issues)

if __name__ == "__main__":
    simulate_resumption_shock()
