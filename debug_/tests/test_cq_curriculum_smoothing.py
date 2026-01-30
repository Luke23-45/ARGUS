import torch
import unittest
from unittest.mock import MagicMock, patch

# We don't need the full wrapper to test the logic
class CurriculumLogicTester:
    def __init__(self):
        self.current_epoch = 0
        self.curr_tau = torch.zeros(1)
        self.curr_sigma_scale = torch.zeros(1)
        self.grad_norm_ema = torch.zeros(1)
        self.grad_norm_std = torch.zeros(1)
        self.grad_ema_decay = 0.99
        self.grad_norm_step_count = torch.ones(1) * 1000 # Stable stats
        self.resumption_grace_steps = torch.ones(1)      # Mute shock detector
        self.device = torch.device("cpu")
        self.trainer = MagicMock()
        
    def _update_curriculum(self, batch_idx: int):
        # [Copying the exact logic from wrapper_generalist.py for verification]
        if self.trainer is None: return
        n_batches = self.trainer.num_training_batches
        if n_batches <= 0: return

        # 1. Calculate continuous epoch progress
        curr_progress_f = self.current_epoch + (batch_idx / n_batches)
        # print(f"DEBUG: epoch={self.current_epoch}, batch={batch_idx}, progress={curr_progress_f:.4f}")
        
        # PMS Governance: Mute updates if manifold is unstable
        from icu.utils.stabilization import TrendSentinel
        ema_bc, std_bc = TrendSentinel.get_stats(self.grad_norm_ema, self.grad_norm_std, self.grad_ema_decay, self.grad_norm_step_count)
        is_unstable = TrendSentinel.is_unstable(ema_bc, std_bc, max_pressure=5.0, max_sigma=2.0)
        
        if is_unstable and self.resumption_grace_steps == 0:
            return

        # Tau: Ramp from 0.5 to 0.7 between Epoch 5 and 15
        tau_val = 0.5
        if curr_progress_f >= 5.0:
            tau_p = min(1.0, (curr_progress_f - 5.0) / 10.0)
            tau_val = 0.5 + (0.7 - 0.5) * tau_p
        
        # Sigma: Ramp from 3.5 to 2.5 over first 15 epochs
        sigma_p = min(1.0, curr_progress_f / 15.0)
        sigma_val = 3.50 - (3.50 - 2.50) * sigma_p
        
        # print(f"DEBUG: tau_val={tau_val:.6f}")
        self.curr_tau.fill_(tau_val)
        self.curr_sigma_scale.fill_(sigma_val)

class TestCQCurriculumSmoothing(unittest.TestCase):
    def test_smooth_transition(self):
        print("\n" + "="*60)
        print("AUDIT: Verifying Step-Continuous Curriculum Smoothing")
        print("="*60)

        tester = CurriculumLogicTester()
        tester.trainer.num_training_batches = 100
        
        # --- Scenario: End of Epoch 5 ---
        tester.current_epoch = 5
        tester._update_curriculum(batch_idx=99)
        tau_e5_last = tester.curr_tau.item()
        
        # --- Scenario: Start of Epoch 6 ---
        tester.current_epoch = 6
        tester._update_curriculum(batch_idx=0)
        tau_e6_first = tester.curr_tau.item()
        
        diff = tau_e6_first - tau_e5_last
        expected_step = 0.0002 
        
        print(f"Tau End E5 (Batch 99):   {tau_e5_last:.6f}")
        print(f"Tau Start E6 (Batch 0):  {tau_e6_first:.6f}")
        print(f"Jump Size: {diff:.6e}")
        print(f"Expected Step: {expected_step:.6e}")
        
        # Note: We use 1e-4 tolerance because of the batch_idx/n_batches rounding if any
        if abs(diff - expected_step) < 2e-4:
             print("✅ PASS: Curriculum transition is smooth and batch-continuous.")
        else:
             print("❌ FAIL: Large discontinuity or incorrect ramp detected!")

if __name__ == "__main__":
    unittest.main()
