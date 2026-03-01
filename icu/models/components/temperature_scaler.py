import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("Temperature_Scaler")

def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor):
    """
    [Iteration 10 SOTA] Post-Training Temperature Scaling (Guo et al., 2017)
    Rationale: Minimizes ECE by scaling the logits of the auxiliary head.
    
    This component is now a permanent part of the icu.models.components infrastructure.
    """
    T = nn.Parameter(torch.tensor(2.5, device=logits.device))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    
    def closure():
        optimizer.zero_grad()
        # Binary Cross Entropy with Temperature Scaling
        loss = F.binary_cross_entropy_with_logits(logits / T, labels.float())
        loss.backward()
        return loss
        
    logger.info(f"🚀 Starting Temperature Scaling. Initial T: {T.item():.4f}")
    optimizer.step(closure)
    logger.info(f"✅ Calibration Complete. Optimal T: {T.item():.4f}")
    
    return T.detach()

if __name__ == "__main__":
    # Internal Unit Test
    l = torch.randn(100, 1)
    y = (torch.rand(100, 1) > 0.5).long()
    t = calibrate_temperature(l, y)
    print(f"Final T: {t}")
