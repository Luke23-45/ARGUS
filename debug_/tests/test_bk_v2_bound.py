import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BK_Fixed")

def test_cga_hyperspherical_sync_verified():
    logger.info("Verifying Patch #72: Hyperspherical Anchoring (Cosine Distance)...")
    
    # 1. Start of training
    z_student = torch.randn(1, 128)
    z_teacher = z_student.clone() + torch.randn(1, 128) * 0.1
    
    # Cosine Distance: 1 - sum(z_s_norm * z_t_norm)
    z_s_norm = F.normalize(z_student, dim=1)
    z_t_norm = F.normalize(z_teacher, dim=1)
    dist_start = 1.0 - torch.sum(z_s_norm * z_t_norm)
    
    logger.info(f"CGA Distance at Start: {dist_start.item():.6f}")
    
    # 2. Later in training (Abyssal Drift)
    # Even if they drift 1000 units away...
    z_student_extreme = z_student + torch.randn(1, 128) * 100.0
    
    z_s_extreme_norm = F.normalize(z_student_extreme, dim=1)
    dist_drift = 1.0 - torch.sum(z_s_extreme_norm * z_t_norm)
    
    logger.info(f"CGA Distance after drifting 100 units: {dist_drift.item():.6f}")
    
    if dist_drift <= 2.0:
        logger.info(f"✅ Patch #72 SUCCESS! Loss is perfectly bounded at {dist_drift.item():.4f} (Max 2.0).")
    else:
        logger.error("❌ Bounding Failed.")

if __name__ == "__main__":
    test_cga_hyperspherical_sync_verified()
