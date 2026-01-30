import torch
import torch.nn as nn
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AH")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 10)
        self.head1 = nn.Linear(10, 1)
        self.head2 = nn.Linear(10, 1)

class MockGradNorm:
    def __init__(self, params):
        self.weights = nn.Parameter(torch.ones(2))
        self.optimizer = torch.optim.Adam([self.weights], lr=0.1)
        self.shared_params = list(params)

def simulate_gradnorm_fp16_and_resumption():
    logger.info("Simulating GradNorm Scaler Underflow & Resumption...")
    
    model = SimpleModel()
    gn = MockGradNorm(model.backbone.parameters())
    scaler = torch.amp.GradScaler(init_scale=65536.0) # Standard FP16 scale
    
    # 1. SCALER UNDERFLOW SMOKING GUN (Test for G6)
    # Simulate a backward pass with a scale
    loss = torch.tensor(1.0, requires_grad=True)
    scaled_loss = scaler.scale(loss)
    
    # GradNorm update (Buggy: no scaler used for gradnorm.optimizer)
    gn.optimizer.zero_grad()
    # In reality, gn_loss is computed from gradients of model.
    # We simulate a "Scaled" gradnorm_loss
    gradnorm_loss = torch.tensor(0.5, requires_grad=True) * scaler.get_scale() 
    
    gradnorm_loss.backward()
    logger.info(f"GradNorm Weight Grad (Scaled): {gn.weights.grad}")
    
    # Buggy step:
    gn.optimizer.step()
    logger.info(f"GradNorm Weights after Scaled Step: {gn.weights.detach()}")
    
    if (gn.weights.abs() > 2.0).any():
        logger.error("\u274c SMOKING GUN #6: GradNorm weights exploded due to missing scaler unscaling!")
    
    # 2. RESUMPTION AMNESIA (Test for G11)
    # Simulate Adam momentum
    for _ in range(5):
        gn.optimizer.zero_grad()
        (gn.weights.sum() * 0.1).backward()
        gn.optimizer.step()
    
    state_before = copy.deepcopy(gn.optimizer.state_dict())
    
    # Re-init (Resumption)
    gn_resumed = MockGradNorm(model.backbone.parameters())
    # Buggy resumption: only weights are restored, not optimizer state
    gn_resumed.weights.data.copy_(gn.weights.data)
    
    state_after = gn_resumed.optimizer.state_dict()
    
    if len(state_after['state']) == 0:
        logger.error("\u274c SMOKING GUN #11: GradNorm Resumption Amnesia! Optimizer state lost.")
    else:
        logger.info("\u2705 Optimizer state preserved (Unexpected in current buggy codebase).")

if __name__ == "__main__":
    simulate_gradnorm_fp16_and_resumption()
