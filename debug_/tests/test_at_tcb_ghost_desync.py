import torch
import torch.nn.functional as F
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AT")

def test_tcb_ghost_desync():
    logger.info("Simulating TCB-Ghost Desync (Smoking Gun #29)...")
    
    d_model = 512
    tcb = TemporalContrastiveBuffer(d_model=d_model, capacity=1024)
    
    # 1. Warmup: Fill the buffer with 'Modern' samples
    # We'll put some samples in so it's not empty
    for _ in range(32):
        q = torch.randn(1, d_model)
        k = torch.randn(1, d_model)
        tcb(q, k)
    
    # 2. Modern Contrast: Similar views within the same manifold
    q_modern = F.normalize(torch.randn(1, d_model, requires_grad=True), dim=1)
    q_modern.retain_grad()
    k_modern = F.normalize(q_modern.detach() + 0.1 * torch.randn(1, d_model), dim=1)
    
    out_modern = tcb(q_modern, k_modern)
    l_modern = out_modern['loss']
    l_modern.backward()
    grad_modern = q_modern.grad.norm().item()
    logger.info(f"Modern Loss: {l_modern.item():.4f} | Grad Norm: {grad_modern:.4f}")
    
    # 3. Adversarial Case: Ancient Anchor Desync
    # q_new is a modern student embedding
    # k_ancient is a ghost anchor from a completely different region of space (desync)
    q_new = F.normalize(torch.randn(1, d_model, requires_grad=True), dim=1)
    q_new.retain_grad()
    # k_ancient is orthogonal to q_new
    k_ancient = F.normalize(torch.randn(1, d_model), dim=1)
    
    # Zero out the gradient from the previous backward
    q_new.grad = None
    
    out_ancient = tcb(q_new, k_ancient)
    l_ancient = out_ancient['loss']
    l_ancient.backward()
    grad_ancient = q_new.grad.norm().item()
    
    logger.info(f"Ancient Desync Loss: {l_ancient.item():.4f} | Grad Norm: {grad_ancient:.4f}")
    
    ratio = grad_ancient / (grad_modern + 1e-8)
    logger.info(f"Desync Shock Ratio: {ratio:.2f}x")
    
    # In InfoNCE, large distance (low similarity) leads to high loss but bounded gradients (softmax).
    # However, if the entire buffer is 'far away' and the positive is also 'far away', 
    # the model might get 'lost' in the manifold.
    if ratio > 10.0:
        logger.error("❌ TCB Desync Gradient is significantly higher than Modern gradients!")
    else:
        logger.info("✅ TCB handles anchor desync gracefully (within 10x range).")

if __name__ == "__main__":
    test_tcb_ghost_desync()
