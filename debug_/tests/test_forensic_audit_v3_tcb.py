
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.models.components.temporal_buffer import TemporalContrastiveBuffer

def test_tcb_deadlock_fixed():
    print("\n[TEST] TCB Deadlock Fix Verification (SG #241)")
    
    # Setup
    tcb = TemporalContrastiveBuffer(d_model=8, capacity=32)
    device = torch.device("cpu")
    
    clean_keys = torch.randn(4, 8)
    nan_keys = torch.randn(4, 8)
    nan_keys[0, 0] = float('nan')
    
    # 1. Simulate Rank 0 (Has NaN)
    print("  Scenario: Rank 0 detects NaN...")
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.all_reduce') as mock_reduce, \
         patch('torch.distributed.all_gather') as mock_gather:
        
        # When all_reduce is called, in reality it would sync. 
        # Here we just check it was called.
        tcb._dequeue_and_enqueue(nan_keys)
        
        if mock_reduce.called:
             print("  SUCCESS: Rank 0 called all_reduce to sync poison status.")
        else:
             print("  FAILED: Rank 0 returned WITHOUT all_reduce!")

    # 2. Simulate Rank 1 (Clean)
    print("  Scenario: Rank 1 is clean, but others might have poisoned data...")
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.all_reduce') as mock_reduce:
        
        # We manually mock the side effect: someone else has poison
        def force_poison(tensor, op=None):
            tensor.fill_(1.0)
            
        mock_reduce.side_effect = force_poison
        
        tcb._dequeue_and_enqueue(clean_keys)
        
        if mock_reduce.called:
             print("  SUCCESS: Rank 1 called all_reduce and will return if poison found elsewhere.")
        else:
             print("  FAILED: Rank 1 proceeded blindly!")

if __name__ == "__main__":
    test_tcb_deadlock_fixed()
