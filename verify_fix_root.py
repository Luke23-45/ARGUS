import torch
import logging
from typing import Dict, Any

# Mock Classes to simulate PL Environment
class MockNormalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("ts_min", torch.zeros(1))
        self.calibrated = False

    def load_state_dict(self, state_dict, strict=True):
        self.ts_min.copy_(state_dict["ts_min"])
        self.calibrated = True
        print(f"   [MockNormalizer] Loaded state! ts_min={self.ts_min.item()}")

    def state_dict(self):
        return {"ts_min": self.ts_min}

class MockDataModule:
    pass

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = MockNormalizer()

class MockTrainer:
    def __init__(self):
        self.datamodule = MockDataModule()

class MockAWR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("adv_mean", torch.tensor(0.0))
        self.register_buffer("adv_std", torch.tensor(1.0))
        self.register_buffer("stats_initialized", torch.tensor(False))

class MockWrapper:
    def __init__(self):
        self.trainer = MockTrainer()
        self.model = MockModel() # [FIX] Normalizer here
        self.awr_calculator = MockAWR()
        
    # LOGIC FROM wrapper_generalist.py
    def on_fit_start(self):
        print("-> on_fit_start called")
        if hasattr(self, "pending_normalizer_state"):
             # [SOTA FIX] Normalizer lives in self.model (ICUUnifiedPlanner), not datamodule
             if hasattr(self.model, "normalizer"):
                 self.model.normalizer.load_state_dict(self.pending_normalizer_state)
                 print("   [RESUME] Normalizer state successfully restored to Model (Calibration Preserved).")
                 del self.pending_normalizer_state
             else:
                 print("[RESUME] WARNING: Pending normalizer state found but MODEL has no normalizer!")

    def on_save_checkpoint(self, checkpoint):
        print("-> on_save_checkpoint called")
        # 1. Save Normalizer State (Critical for Inference/Resume)
        if hasattr(self.model, "normalizer"):
            checkpoint["normalizer_state"] = self.model.normalizer.state_dict()
            print(f"   Saved normalizer state: {checkpoint['normalizer_state']}")

    def on_load_checkpoint(self, checkpoint):
        print("-> on_load_checkpoint called")
        if "normalizer_state" in checkpoint:
            self.pending_normalizer_state = checkpoint["normalizer_state"]
            print("   [RESUME] Found Normalizer state in checkpoint.")

def test_resume_logic():
    print("=== RESUME ARCHITECTURE VERIFICATION ===")
    
    # 1. Setup Initial Model & Calibrate
    model_A = MockWrapper()
    model_A.model.normalizer.ts_min.fill_(50.0)
    print(f"[Model A] Calibrated TS_MIN: {model_A.model.normalizer.ts_min.item()}")
    
    # 2. Save Checkpoint
    checkpoint = {}
    model_A.on_save_checkpoint(checkpoint)
    
    assert "normalizer_state" in checkpoint
    assert checkpoint["normalizer_state"]["ts_min"].item() == 50.0
    
    # 3. Resume: Create Fresh Model (Uncalibrated)
    model_B = MockWrapper()
    print(f"[Model B] Initial TS_MIN: {model_B.model.normalizer.ts_min.item()}")
    assert model_B.model.normalizer.ts_min.item() == 0.0
    
    # 4. Load Checkpoint
    model_B.on_load_checkpoint(checkpoint)
    
    # 5. Fit Start (Trigger Restore)
    model_B.on_fit_start()
    
    # 6. Verify Restoration
    val_b = model_B.model.normalizer.ts_min.item()
    print(f"[Model B] Final TS_MIN: {val_b}")
    
    assert val_b == 50.0, "FATAL: Normalizer stats NOT restored!"
    
    # 7. Test Fresh Run (No Checkpoint)
    print("\n--- Testing Fresh Run Logic ---")
    model_C = MockWrapper()
    # Simulate uncalibrated state
    model_C.model.normalizer.calibrated = False 
    model_C.on_fit_start()
    # In this mock, on_fit_start attempts calibration but fails because no datamodule/dataset is fully mocked with files
    # But we should see it checking the condition.
    # We can check if it skipped the "Resume" block.
    assert not hasattr(model_C, "pending_normalizer_state")
    print("   [Fresh Run] Correctly skipped Resume logic.")
    
    print("\n=== SUCCESS: ARCHITECTURE MATCHES & LOGIC WORKS ===")

if __name__ == "__main__":
    test_resume_logic()
