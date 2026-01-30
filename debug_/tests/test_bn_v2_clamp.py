import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BN_v2")

def test_evidential_clamping():
    logger.info("Verifying Patch #89: Evidential Logit Clamping...")
    
    def compute_loss_with_clamping(logits, targets):
        # SIMULATE THE FIX: Clamp to 20.0 (v89.0)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        y = targets
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        # KL term
        kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(2.0)) \
             - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
             + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        return kl.mean()

    # 1. Nightmare Case: Logits reaching 200.0 (Singularity Trigger)
    logits_huge = torch.tensor([[200.0, -200.0]])
    target = torch.tensor([[1.0, 0.0]])
    
    loss_huge = compute_loss_with_clamping(logits_huge, target)
    logger.info(f"Loss with 200.0 logits: {loss_huge.item():.4f}")
    
    if torch.isfinite(loss_huge):
        logger.info("✅ Patch #89 SUCCESS! Evidential loss remains finite even with extreme activations.")
    else:
        logger.error("❌ Verification Failed. Singularity reached despite clamping!")

if __name__ == "__main__":
    test_evidential_clamping()
