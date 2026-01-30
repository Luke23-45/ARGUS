import torch
import torch.nn as nn
import contextlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BR_v2")

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(self.dropout(x))

class MockWrapper:
    def __init__(self):
        self.model = MockModel()
        self.ema = None # Simulating Student-Teacher pass

    @contextlib.contextmanager
    def ema_teacher_context(self):
        # THE FIX: Enforce eval() during context
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

def test_teacher_determinism_fixed():
    logger.info("Verifying Patch #106: Teacher Determinism...")
    
    wrapper = MockWrapper()
    x = torch.randn(1, 10)
    
    # Ensure model is in train mode initially
    wrapper.model.train()
    
    with wrapper.ema_teacher_context():
        # Inside context, model should be in eval mode
        if not wrapper.model.training:
            logger.info("✅ Model correctly switched to EVAL mode.")
        else:
            logger.error("❌ Model failed to switch to EVAL mode.")
            
        t1 = wrapper.model(x)
        t2 = wrapper.model(x)
        
        diff = torch.abs(t1 - t2).item()
        logger.info(f"Target Drift within context: {diff:.8f}")
        
        if diff == 0:
            logger.info("✅ Patch #106 SUCCESS! Targets are fully deterministic.")
        else:
            logger.error(f"❌ Verification Failed. Targets are still jittery (Diff: {diff:.8f})")

    if wrapper.model.training:
        logger.info("✅ Model correctly restored to TRAIN mode.")
    else:
        logger.error("❌ Model failed to restore to TRAIN mode.")

if __name__ == "__main__":
    test_teacher_determinism_fixed()
