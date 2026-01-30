import torch
import unittest
from icu.utils.stability import DynamicThresholding

class TestCQDynamicThresholding(unittest.TestCase):
    def test_accumulation_awareness(self):
        print("\n" + "="*60)
        print("AUDIT: Verifying DynamicThresholding Accumulation Awareness")
        print("="*60)

        dt = DynamicThresholding(percentile=0.9, threshold=1.0, ema_decay=0.9)
        dt.train() # Enable training mode
        
        initial_ema = dt.ema_s.item()
        x = torch.randn(10, 100) * 5.0 # High variance to trigger update
        
        # 1. Test: update_ema=False (Accumulation phase)
        dt(x, update_ema=False)
        after_acc_ema = dt.ema_s.item()
        
        if initial_ema == after_acc_ema:
            print("✅ PASS: EMA did not update during accumulation (update_ema=False).")
        else:
            print(f"❌ FAIL: EMA updated during accumulation! Diff: {after_acc_ema - initial_ema}")
            
        # 2. Test: update_ema=True (Stepping batch)
        dt(x, update_ema=True)
        after_step_ema = dt.ema_s.item()
        
        if initial_ema != after_step_ema:
            print(f"✅ PASS: EMA updated during stepping (update_ema=True). New: {after_step_ema:.4f}")
        else:
            print("❌ FAIL: EMA failed to update during stepping!")

    def test_mode_awareness(self):
        print("\n" + "="*60)
        print("AUDIT: Verifying DynamicThresholding Mode Awareness")
        print("="*60)

        dt = DynamicThresholding(percentile=0.9, threshold=1.0, ema_decay=0.9)
        dt.eval() # Enable eval mode (Validation)
        
        initial_ema = dt.ema_s.item()
        x = torch.randn(10, 100) * 5.0
        
        # Test: update_ema=True but in EVAL mode
        dt(x, update_ema=True)
        after_eval_ema = dt.ema_s.item()
        
        if initial_ema == after_eval_ema:
            print("✅ PASS: EMA did not update during evaluation (dt.eval()).")
        else:
            print(f"❌ FAIL: EMA updated during evaluation! Diff: {after_eval_ema - initial_ema}")

if __name__ == "__main__":
    unittest.main()
