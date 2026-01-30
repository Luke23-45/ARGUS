
import os
import torch
import torch.nn as nn
from icu.models.components.loss_scaler import BayesianProjectedScaler
import unittest

class TestCQLossScalerAccumulation(unittest.TestCase):
    def test_accumulation_ema_logic(self):
        """
        Verifies if BayesianProjectedScaler correctly accounts for all sub-batches in its EMA
        on a single GPU/CPU (Simulating accumulation).
        """
        torch.manual_seed(42)
        
        num_tasks = 7
        # Set decay=0.5 for easy math: EMA_new = 0.5*EMA_old + 0.5*NewValue
        scaler = BayesianProjectedScaler(num_tasks=num_tasks, decay=0.5)
        
        # Initial EMA state is 1.0 (buffer)
        
        # Simulate accumulation: 4 sub-batches
        acc_batches = 4
        
        # Sub-batches losses for 'diffusion'
        # Batch 0: 1.0
        # Batch 1: 2.0
        # Batch 2: 3.0
        # Batch 3: 4.0
        # Average: (1+2+3+4)/4 = 2.5
        
        for b in range(acc_batches):
            is_acc = (b < acc_batches - 1)
            losses = {
                'diffusion': torch.tensor(1.0 + b, requires_grad=True),
                'critic': torch.tensor(2.0 + b, requires_grad=True)
            }
            
            ema_before = scaler.loss_emas.clone()
            scaled_loss, logs = scaler(losses, is_accumulating=is_acc, batch_size=1)
            ema_after = scaler.loss_emas.clone()
            
            ema_changed = not torch.allclose(ema_before, ema_after)
            
            if is_acc:
                self.assertFalse(ema_changed, f"EMA updated during accumulation at sub-batch {b}")
            else:
                self.assertTrue(ema_changed, "EMA failed to update on stepping batch")
                
                # Verify value
                # EMA_old = 1.0
                # New Global Avg = 2.5
                # EMA_new = 0.5 * 1.0 + 0.5 * 2.5 = 1.75
                
                ema_val = ema_after[0].item() # diffusion
                print(f"Final EMA Diffusion: {ema_val:.4f}")
                
                # If the bug was present (blind to 0-2):
                # EMA_new = 0.5 * 1.0 + 0.5 * 4.0 = 2.5
                
                if abs(ema_val - 1.75) < 0.01:
                    print("SUCCESS: Scaler is accumulation-aware (Average of all sub-batches used).")
                elif abs(ema_val - 2.5) < 0.01:
                    self.fail("FAILURE: Scaler is still BLIND to accumulation sub-batches! (Used only last batch)")
                else:
                    self.fail(f"UNKNOWN EMA Behavior: {ema_val}")

if __name__ == "__main__":
    unittest.main()
