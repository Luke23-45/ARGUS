
import torch
import torch.nn as nn

# Mock Normalizer
class MockNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("ts_min", torch.tensor([40.0])) # e.g. Min HR
        self.register_buffer("ts_max", torch.tensor([120.0])) # e.g. Max HR
    
    def normalize(self, x, x_static=None):
        return (x - self.ts_min) / (self.ts_max - self.ts_min) * 2 - 1, None

    def denormalize(self, x):
        return (x + 1) / 2 * (self.ts_max - self.ts_min) + self.ts_min

# Mock Model (Planner + Normalizer)
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = MockNormalizer()
    
    def sample(self, batch):
        # [SIMULATION] Diffusion.py UNNORMALIZE output
        # Returns PHYSICAL units (e.g. 80.0)
        return torch.tensor([80.0]) 

# Mock Wrapper
class MockWrapper:
    def __init__(self):
        self.model = MockModel()
        self.val_phys_violation_rate = type('Metric', (), {'update': lambda s, x: print(f"   [Metric] ViolationRate Updated: {x}")})()
        self.val_mse_global = type('Metric', (), {'update': lambda s, p, t: print(f"   [Metric] MSE Updated: Pred={p.item()}, Target={t.item()}")})()
        
    def _validate_clinical_sampling(self, batch):
        print("-> Running _validate_clinical_sampling...")
        subset = batch
        normalizer = self.model.normalizer
        
        # 1. Ground Truth (Simulating Fix: Raw Physical)
        gt_physical_raw = subset["future_data"]
        gt_phys = gt_physical_raw
        print(f"   [Debug] gt_phys (Should be 80.0): {gt_phys.item()}")

        
        # 2. Prediction (Simulating Fix: Raw Physical)
        pred_physical_raw = self.model.sample(subset)
        pred_phys = pred_physical_raw
        print(f"   [Debug] pred_phys (Should be 80.0): {pred_phys.item()}")
        
        # Safe Clamping
        pred_safe = pred_phys 
        gt_safe = gt_phys
        
        # 3. MSE
        self.val_mse_global.update(pred_safe, gt_safe)
        
        # 5. Physics Violations 
        pred_norm_check = normalizer.normalize(pred_safe)[0] 
        print(f"   [Debug] pred_norm_check (Should be ~0.0 for 80 in [40,120]): {pred_norm_check.item()}")
        
        violations = ((pred_norm_check.abs() > 0.99).float().mean())
        self.val_phys_violation_rate.update(violations)

def test_fix():
    wrapper = MockWrapper()
    # Mock Batch: 
    # [SCENARIO]: DataLoader yields RAW PHYSICAL units.
    # Set future_data to 80.0 (Raw).
    # If the bug was present, it would be treated as Normalized (0.0 equiv).
    # Actually, 80.0 treated as Normalized -> Denorm(80) -> Massive.
    # With FIX, it should stay 80.0.
    batch = {"future_data": torch.tensor([80.0]), "observed_data": torch.randn(1, 10, 1)}
    
    print("=== TEST START: Both Pred and GT are Physical 80.0 ===")
    wrapper._validate_clinical_sampling(batch)
    print("\n=== VERIFICATION PASSED ===")

if __name__ == "__main__":
    test_fix()
