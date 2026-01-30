import torch
import logging
import math
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test_W")

def simulate_ema_lag(accum_steps=16, total_steps=320, decay=0.9999):
    logger.info(f"Simulating EMA Teacher Lag: Accumulation={accum_steps}, Steps={total_steps}, Decay={decay}")
    
    # 1. Standard Case (Accumulation=1, Per-Step Update)
    student_1 = torch.tensor(1.0)
    teacher_1 = torch.tensor(1.0)
    
    # 2. Accumulation Case (Buggy: N-step jump with 1-step decay)
    student_acc = torch.tensor(1.0)
    teacher_acc = torch.tensor(1.0)

    # 3. Patched Case (SOTA: N-step jump with Power-Law decay)
    student_patched = torch.tensor(1.0)
    teacher_patched = torch.tensor(1.0)
    
    # Step sizes for simulated learning
    learning_rate = 0.01
    
    for i in range(total_steps):
        # Student moves (Constant drift)
        move = 0.5 + torch.randn(1).item() * 0.1
        student_1.add_(move * learning_rate)
        student_acc.add_(move * learning_rate)
        student_patched.add_(move * learning_rate)
        
        # Teacher 1 updates every single step
        teacher_1.copy_(decay * teacher_1 + (1 - decay) * student_1)
        
        # Every N steps (end of accumulation cycle)
        if (i + 1) % accum_steps == 0:
            # BUGGY: Uses 1-step decay for 16-step jump
            teacher_acc.copy_(decay * teacher_acc + (1 - decay) * student_acc)
            
            # PATCHED: Uses Power-Law Scaled Decay (v' = v^N)
            decay_patched = decay ** accum_steps
            teacher_patched.copy_(decay_patched * teacher_patched + (1 - decay_patched) * student_patched)

    lag_buggy = (teacher_1 - teacher_acc).item()
    lag_patched = (teacher_1 - teacher_patched).item()
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    logger.info(f"Ideal Teacher (Per-Step) Final: {teacher_1.item():.6f}")
    logger.info(f"Buggy Teacher (Every N)  Final: {teacher_acc.item():.6f} (Lag: {lag_buggy:.6f})")
    logger.info(f"Patched Teacher (Power)  Final: {teacher_patched.item():.6f} (Lag: {lag_patched:.6f})")

    improvement = (abs(lag_buggy) - abs(lag_patched)) / (abs(lag_buggy) + 1e-8)
    logger.info(f"Error Reduction: {improvement*100:.1f}%")

    if abs(lag_patched) < abs(lag_buggy) * 0.1:
        logger.info("\u2705 TEST W PASSED: Power-Law decay restored synchronization.")
    else:
        logger.warning("\u26a0\ufe0f TEST W: Power-Law fix insufficient.")
            
    return improvement

if __name__ == "__main__":
    simulate_ema_lag()
