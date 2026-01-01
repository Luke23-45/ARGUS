
import unittest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
import os

# Add Root Project to path
sys.path.append(os.getcwd())

from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
from icu.models.apex_moe_planner import APEX_MoE_Planner
from icu.models.wrapper_apex import ICUSpecialistWrapper

class MockAuxHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        # SOTA Signature: (x, mask) -> (logits, loss)
        # x: [B, T, D] or [B, D] depending on input. 
        # But wait, SequenceAuxHead expects [B, T, D].
        # APEX passes [B, T, D] (ctx_seq).
        if x.dim() == 3:
            # Pool or take last? SequenceAuxHead uses CLS token or pooling.
            # Mock simple mean pooling
            logits = self.linear(x.mean(dim=1))
        else:
            logits = self.linear(x)
        return logits, torch.tensor(0.0)

class TestApexMoE(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Config
        conf = {
            "model": {
                "input_dim": 28,
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 2,
                "encoder_layers": 1,
                "history_len": 24,
                "pred_len": 24,
                "use_imputation_masks": True,
                "use_auxiliary_head": True,
                "num_phases": 3, # 3 Experts
                "dropout": 0.0,
                "timesteps": 10
            },
            "train": {
                "aux_loss_scale": 0.1,
                "balancing_mode": "sota_2025",
                "awr_beta": 0.05,
                "uw_lr": 0.02,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "pretrained_path": "dummy.ckpt" # Mocked later
            }
        }
        self.cfg = OmegaConf.create(conf)
        
        # 2. Mock Generalist
        base_cfg = ICUConfig(**self.cfg.model)
        self.generalist = ICUUnifiedPlanner(base_cfg).to(self.device)
        # Replace AuxHead with Mock that supports SOTA signature
        self.generalist.aux_head = MockAuxHead(32, 3).to(self.device)

    def test_planner_forward(self):
        """Verify APEX Planner forward pass with SOTA patch."""
        # Initialize APEX from Generalist
        apex = APEX_MoE_Planner(self.generalist).to(self.device)
        
        B, T = 4, 24
        batch = {
            "observed_data": torch.randn(B, T, 28).to(self.device),
            "future_data": torch.randn(B, T, 28).to(self.device),
            "static_context": torch.randn(B, 6).to(self.device),
            "src_mask": torch.ones(B, T, 28).to(self.device),
            "phase_label": torch.tensor([0, 1, 2, 0]).to(self.device),
            "outcome_label": torch.tensor([0.0, 1.0, 1.0, 0.0]).to(self.device)
        }
        
        # Forward
        out = apex(batch)
        print(f"APEX Loss: {out['loss']}")
        self.assertFalse(torch.isnan(out['loss']))
        self.assertIn("router_ce_loss", out)
        
    def test_wrapper_training_step(self):
        """Verify APEX Wrapper training logic with SOTA patches."""
        # Mock load_model to avoid file requirement
        with unittest.mock.patch('icu.models.wrapper_apex.SurgicalCheckpointLoader') as loader:
            loader.load_model.return_value = None
            
            wrapper = ICUSpecialistWrapper(self.cfg).to(self.device)
            # Inject mocked generalist components into wrapper.model
            # Note: Wrapper init creates a FRESH generalist. We need to swap the aux head
            wrapper.model.router = MockAuxHead(32, 3).to(self.device)
            
            wrapper.trainer = unittest.mock.MagicMock()
            wrapper.trainer.accumulate_grad_batches = 1
            wrapper.trainer.current_epoch = 0
            
            # Setup manual optimization mocks
            opt = unittest.mock.MagicMock()
            # Mock param_groups for LR logging
            opt.param_groups = [{"lr": 1e-4}]
            wrapper.optimizers = lambda: opt
            wrapper.manual_backward = lambda loss: loss.backward()
            
            B, T = 4, 24
            batch = {
                # Scale to realistic clinical values (e.g. 80 +/- 10) to pass Safety Check
                "observed_data": torch.randn(B, T, 28).to(self.device) * 10 + 80,
                "future_data": torch.randn(B, T, 28).to(self.device) * 10 + 80,
                "static_context": torch.randn(B, 6).to(self.device),
                "src_mask": torch.ones(B, T, 28).to(self.device),
                "phase_label": torch.tensor([0, 1, 2, 0]).to(self.device),
                "outcome_label": torch.tensor([0.0, 1.0, 1.0, 0.0]).to(self.device)
            }
            
            # Run Step
            try:
                wrapper.training_step(batch, batch_idx=0)
                print("Wrapper training_step successful")
            except Exception as e:
                self.fail(f"APEX training_step failed: {e}")

if __name__ == '__main__':
    unittest.main()
