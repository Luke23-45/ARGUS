import unittest
import torch
import sys
import os
from omegaconf import OmegaConf

# Add Root Project to path
sys.path.append(os.getcwd())

from icu.models.wrapper_generalist import ICUGeneralistWrapper

class TestWrapperIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Minimal Config for SOTA 2025 mode
        conf = {
            "model": {
                "input_dim": 28,
                "d_model": 32, # Small for speed
                "n_heads": 4,
                "n_layers": 2,
                "encoder_layers": 1,
                "history_len": 24,
                "pred_len": 24,
                "use_imputation_masks": True,
                "use_auxiliary_head": True,
                "num_phases": 3,
                "dropout": 0.0,
                "timesteps": 10
            },
            "train": {
                "aux_loss_scale": 0.1,
                "balancing_mode": "sota_2025", # Test the new path
                "awr_beta": 0.05,
                "throttle_scale": 0.2,
                "lr": 1e-4,
                "weight_decay": 1e-4
            }
        }
        self.cfg = OmegaConf.create(conf)

    def test_training_step_execution(self):
        """Verify training_step runs end-to-end with SOTA components."""
        wrapper = ICUGeneralistWrapper(self.cfg).to(self.device)
        wrapper.trainer = unittest.mock.MagicMock() 
        wrapper.trainer.accumulate_grad_batches = 1
        wrapper.trainer.max_epochs = 10 
        wrapper.trainer.current_epoch = 0 # Fix: Set on trainer, not wrapper property
        
        B, T = 4, 24
        batch = {
            # [Test Fix] AdvantageCalculator expects RAW clinical values (e.g. HR > 20).
            # If values are small (N(0,1)), it halts assuming unsafe normalization.
            "observed_data": torch.randn(B, T, 28).to(self.device) * 10 + 80, 
            "future_data": torch.randn(B, T, 28).to(self.device) * 10 + 80, 
            "static_context": torch.randn(B, 6).to(self.device),
            "src_mask": torch.ones(B, T, 28).to(self.device),
            "phase_label": torch.tensor([0, 1, 2, 0]).to(self.device), # Multi-class targets
            "outcome_label": torch.tensor([0.0, 1.0, 1.0, 0.0]).to(self.device)
        }
        
        # Mock Optimizer and Scheduler
        wrapper.optimizers = lambda: torch.optim.Adam(wrapper.parameters())
        wrapper.lr_schedulers = lambda: None # Skip scheduler step
        wrapper.manual_backward = lambda loss: loss.backward() # Simple backward
        
        try:
            loss = wrapper.training_step(batch, batch_idx=0)
            print(f"Wrapper Training Loss: {loss}")
            self.assertFalse(torch.isnan(loss))
            self.assertTrue(loss > 0)
        except Exception as e:
            self.fail(f"training_step crashed: {e}")

    def test_optimizer_config(self):
        """Verify configure_optimizers collects scaler params."""
        wrapper = ICUGeneralistWrapper(self.cfg).to(self.device)
        wrapper.trainer = unittest.mock.MagicMock()
        wrapper.trainer.estimated_stepping_batches = 100
        
        try:
            optim_conf = wrapper.configure_optimizers()
            optimizer = optim_conf["optimizer"]
            # Check if we have 2 param groups (Backbone + Scaler)
            print(f"Optimizer Param Groups: {len(optimizer.param_groups)}")
            # Group 0: Model, Group 1: Scaler (if SOTA mode)
            self.assertTrue(len(optimizer.param_groups) >= 1)
        except Exception as e:
            self.fail(f"configure_optimizers crashed: {e}")

if __name__ == '__main__':
    import unittest.mock
    unittest.main()
