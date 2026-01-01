import torch
import logging
from icu.utils.advantage_calculator import ICUAdvantageCalculator

# Set up logging to see the output
logging.basicConfig(level=logging.INFO)

def reproduce_fix():
    print("Testing GAE Fix for Dimensional Mismatch...")
    
    # Initialize calculator
    calc = ICUAdvantageCalculator(gamma=0.99, lambda_gae=0.95)
    
    # Mock data: B=256, T=6
    B, T = 256, 6
    device = torch.device("cpu")
    
    rewards = torch.randn(B, T, device=device)
    values = torch.randn(B, T, device=device)
    bootstrap_value = torch.randn(B, 1, device=device)
    
    # Case 1: dones is (B,) - This caused the crash
    print(f"\n[Case 1] dones has shape (B,): {B}")
    dones_1d = (torch.rand(B, device=device) > 0.8).bool()
    
    try:
        advantages = calc.compute_gae(
            rewards, 
            values, 
            dones=dones_1d, 
            bootstrap_value=bootstrap_value
        )
        print(f"SUCCESS: Advantages computed. Shape: {advantages.shape}")
        assert advantages.shape == (B, T)
    except Exception as e:
        print(f"FAILURE: Case 1 failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Case 2: dones is (B, T) - Standard case
    print(f"\n[Case 2] dones has shape (B, T): ({B}, {T})")
    dones_2d = (torch.rand(B, T, device=device) > 0.9).bool()
    
    try:
        advantages = calc.compute_gae(
            rewards, 
            values, 
            dones=dones_2d, 
            bootstrap_value=bootstrap_value
        )
        print(f"SUCCESS: Advantages computed. Shape: {advantages.shape}")
        assert advantages.shape == (B, T)
    except Exception as e:
        print(f"FAILURE: Case 2 failed with error: {e}")

    # Case 3: dones is None
    print(f"\n[Case 3] dones is None")
    try:
        advantages = calc.compute_gae(
            rewards, 
            values, 
            dones=None, 
            bootstrap_value=bootstrap_value
        )
        print(f"SUCCESS: Advantages computed. Shape: {advantages.shape}")
        assert advantages.shape == (B, T)
    except Exception as e:
        print(f"FAILURE: Case 3 failed with error: {e}")

    print("\nAll Smoke Tests Passed!")

if __name__ == "__main__":
    reproduce_fix()
