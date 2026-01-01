import unittest
import torch
import sys
import os

# Add Root Project to path
sys.path.append(os.getcwd())

from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
from icu.models.components.sequence_aux_head import SequenceAuxHead

class TestPhase2Audit(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = ICUConfig(
            input_dim=28,
            d_model=64,
            use_imputation_masks=True,
            use_auxiliary_head=True,
            num_phases=3 # Multi-class
        )

    def test_sample_padding_inference(self):
        """Verify sample() handles implicitly padded batches via src_mask."""
        model = ICUUnifiedPlanner(self.cfg).to(self.device)
        model.eval()
        
        B, T_obs, D = 2, 24, 28
        
        # Batch with padding: User 2 has only 10 valid steps, rest 0
        observed = torch.randn(B, T_obs, D).to(self.device)
        static = torch.randn(B, 6).to(self.device)
        
        # src_mask: [B, T, D]
        src_mask = torch.ones(B, T_obs, D).to(self.device)
        # Pad second user after step 10
        src_mask[1, 10:, :] = 0.0 
        
        batch = {
            "observed_data": observed,
            "static_context": static,
            "src_mask": src_mask,
            # INTENTIONALLY OMIT 'padding_mask' to test inference logic
        }
        
        # Should not crash
        try:
            sample_out = model.sample(batch, num_steps=2) # Short steps for speed
            print(f"Sample Output Shape: {sample_out.shape}")
            self.assertEqual(sample_out.shape, (B, self.cfg.pred_len, D))
        except Exception as e:
            self.fail(f"sample() crashed on padded batch: {e}")

    def test_aux_head_one_hot_conversion(self):
        """Verify SequenceAuxHead handles integer targets for multi-class."""
        d_model = 64
        num_classes = 3
        head = SequenceAuxHead(d_model, num_classes, n_heads=4).to(self.device)
        
        B = 4
        T_seq = 10
        ctx_seq = torch.randn(B, T_seq, d_model).to(self.device)
        
        # Integer targets: [0, 2, 1, 0]
        targets_int = torch.tensor([0, 2, 1, 0], dtype=torch.long).to(self.device)
        
        logits, loss = head(ctx_seq, targets=targets_int)
        
        print(f"Aux Head Loss (Int Targets): {loss}")
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))
        # Logits should be [B, 3]
        self.assertEqual(logits.shape, (B, num_classes))

if __name__ == '__main__':
    unittest.main()
