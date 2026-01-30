import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BN")

def test_evidential_singularity():
    logger.info("Simulating Evidential Singularity (#89)...")
    
    # Logic from sequence_aux_head.py
    def compute_loss(logits, targets):
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # y: one-hot
        y = targets
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        # KL term from class EvidentialLoss
        # Correct Formula: log(Gamma(S_tilde)/Gamma(K)) - sum(log(Gamma(alpha_tilde))) + sum((alpha_tilde - 1) * (digamma(alpha_tilde) - digamma(S_tilde)))
        kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(2.0)) \
             - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
             + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        return kl.mean()

    # 1. Normal Case
    logits_ok = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    loss_ok = compute_loss(logits_ok, target)
    logger.info(f"Loss (Normal): {loss_ok.item():.4f}")
    
    # 2. Positive Explosion (Smoking Gun #89)
    # If the model is extremely confident (evidence=100)
    logits_big = torch.tensor([[50.0, -50.0]])
    loss_big = compute_loss(logits_big, target)
    logger.info(f"Loss (Logits=50): {loss_big.item():.4f}")
    
    # If logits reach 100...
    logits_huge = torch.tensor([[200.0, -200.0]])
    loss_huge = compute_loss(logits_huge, target)
    logger.info(f"Loss (Logits=200): {loss_huge.item()}")
    
    if torch.isinf(loss_huge) or torch.isnan(loss_huge):
        logger.error("❌ Smoking Gun #89 CONFIRMED! Evidential Loss explodes to Inf with large logits.")
        logger.warning("⚠️ Rationale: lgamma(S_tilde) overflows for large alpha. We MUST soft-clamp logits to [min, max] before softplus.")
    else:
        logger.info(f"Loss (Logits=200): {loss_huge.item():.4f}")

if __name__ == "__main__":
    test_evidential_singularity()
