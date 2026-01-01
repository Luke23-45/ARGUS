import torch
import torch.nn as nn
import unittest
import sys
import os

# Add Root Project to path
sys.path.append(os.getcwd())

from icu.models.components.geometric_projector import GeometricProjector
from icu.models.components.temporal_sampler import TemporalSampler
from icu.models.components.nth_encoder import NTHEncoderBlock
from icu.models.components.sequence_aux_head import SequenceAuxHead
from icu.models.components.loss_scaler import UncertaintyLossScaler

# Integration Test
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig

class TestSOTAComponents(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.D = 4, 24, 64
        self.input_dim = 28 # From config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_geometric_projector(self):
        print("\n--- Testing Geometric Projector ---")
        # Input dim is 22 dynamic features (as sliced inside)
        # Actually input to projector is usually raw data size.
        # The projector expects input_channels=22 or 44.
        # Imputation masks = True -> 44 channels
        input_channels = 22 * 2 
        model = GeometricProjector(d_model=self.D, use_imputation_masks=True).to(self.device)
        
        x = torch.randn(self.B, self.T, input_channels).to(self.device)
        out = model(x)
        
        print(f"Input: {x.shape}, Output: {out.shape}")
        self.assertEqual(out.shape, (self.B, self.T, self.D))
        
        # Test Gradients
        loss = out.mean()
        loss.backward()
        print("Backward pass successful")
        
    def test_temporal_sampler(self):
        print("\n--- Testing Temporal Sampler ---")
        model = TemporalSampler(d_model=self.D, top_k=None).to(self.device)
        x = torch.randn(self.B, self.T, self.D).to(self.device)
        
        out, weights = model(x)
        print(f"Soft Sampling Output: {out.shape}, Weights: {weights.shape}")
        self.assertEqual(out.shape, (self.B, self.T, self.D))
        
        # Test Top-K
        k = 10
        model_k = TemporalSampler(d_model=self.D, top_k=k).to(self.device)
        out_k, weights_k = model_k(x)
        print(f"Top-K ({k}) Output: {out_k.shape}")
        self.assertEqual(out_k.shape, (self.B, k, self.D))
        
        # Verify Temporal Order Preserved
        # We can't easily check indices here without internal access, 
        # but if it runs without error sorting, that's a good sign.
        
    def test_nth_encoder(self):
        print("\n--- Testing NTH Encoder ---")
        model = NTHEncoderBlock(d_model=self.D, n_heads=4, hidden_dim=self.D*2).to(self.device)
        x = torch.randn(self.B, self.T, self.D).to(self.device)
        
        out = model(x)
        print(f"Output: {out.shape}")
        self.assertEqual(out.shape, (self.B, self.T, self.D))
        
    def test_sequence_aux_head(self):
        print("\n--- Testing Sequence Aux Head ---")
        model = SequenceAuxHead(d_model=self.D, num_classes=1).to(self.device)
        x = torch.randn(self.B, self.T, self.D).to(self.device)
        targets = torch.randint(0, 2, (self.B, 1)).float().to(self.device)
        
        logits, loss = model(x, targets=targets)
        print(f"Logits: {logits.shape}, Loss: {loss}")
        self.assertEqual(logits.shape, (self.B, 1))
        self.assertIsNotNone(loss)
        
        # Test Asymmetric Loss Logic
        # If target 1, logits should be pushed up.
        loss.backward()
        print("Backward pass successful")
        
    def test_loss_scaler(self):
        print("\n--- Testing Loss Scaler ---")
        model = UncertaintyLossScaler(num_tasks=2).to(self.device)
        losses = {
            'diffusion': torch.tensor(1.5, requires_grad=True),
            'aux': torch.tensor(0.5, requires_grad=True)
        }
        
        total_loss, logs = model(losses)
        print(f"Total Loss: {total_loss}, Logs: {logs}")
        
        total_loss.backward()
        print("Backward pass successful. Gradients on scales:", model.log_vars.grad)
        
    def test_integrated_planner(self):
        print("\n--- Testing Integrated ICUUnifiedPlanner ---")
        cfg = ICUConfig(
            input_dim=28, static_dim=6, history_len=24, pred_len=6,
            d_model=64, n_heads=4, n_layers=2, encoder_layers=2,
            use_auxiliary_head=True
        )
        model = ICUUnifiedPlanner(cfg).to(self.device)
        
        # Mock Batch
        B = 2
        batch = {
            "observed_data": torch.randn(B, 24, 28).to(self.device),
            "future_data": torch.randn(B, 6, 28).to(self.device),
            "static_context": torch.randn(B, 6).to(self.device),
            "phase_label": torch.randint(0, 3, (B,)).to(self.device),
            "padding_mask": torch.zeros(B, 24, dtype=torch.bool).to(self.device) # All valid
        }
        
        out = model(batch)
        print("Forward pass successful. Keys:", out.keys())
        print("Loss:", out["loss"].item())
        
        out["loss"].backward()
        print("Backward pass successful")
        
if __name__ == '__main__':
    unittest.main()
