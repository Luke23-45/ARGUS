import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BR")

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(self.dropout(x))

def test_teacher_stochasticity_bias():
    logger.info("Simulating Teacher Stochastic Decoherence (#106)...")
    
    torch.manual_seed(42)
    model = MockModel()
    x = torch.randn(1, 10)
    
    # 1. BUGGY BEHAVIOR: Teacher pass in TRAIN mode (Stochastic)
    model.train()
    # Student pass
    student_out = model(x)
    # Teacher pass (Same model, but dropout will pick a different mask)
    teacher_out_stochastic = model(x)
    
    diff_stochastic = torch.abs(student_out - teacher_out_stochastic).item()
    logger.info(f"Target Jitter (Stochastic Teacher): {diff_stochastic:.4f}")
    
    # 2. FIXED BEHAVIOR: Teacher pass in EVAL mode (Deterministic)
    model.eval()
    teacher_out_eval = model(x)
    
    # Student was in train, so we compare train-student to eval-teacher
    model.train()
    student_out_v2 = model(x)
    diff_eval = torch.abs(student_out_v2 - teacher_out_eval).item()
    # Note: There will still be diff because student has dropout, 
    # but the teacher's target is now the MEAN representation (scaled by 1-p).
    
    # Real test: Does Stochastic Teacher change the target every pass?
    model.train()
    t1 = model(x)
    t2 = model(x)
    target_drift = torch.abs(t1 - t2).item()
    logger.info(f"Intrinsic Target Drift (Stochastic Teacher): {target_drift:.4f}")
    
    if target_drift > 0:
        logger.error("❌ Smoking Gun #106 CONFIRMED! Teacher targets are jittery due to active Dropout.")
        logger.warning("⚠️ Rationale: ema_teacher_context MUST set the model to eval() to provide a stable, mean-field target.")

if __name__ == "__main__":
    test_teacher_stochasticity_bias()
