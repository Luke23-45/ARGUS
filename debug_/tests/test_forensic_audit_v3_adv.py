
import torch
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.utils.advantage_calculator import ICUAdvantageCalculator

def test_adv_ema_activation():
    print("\n[TEST] Advantage EMA Activation (SG #245)")
    
    # Initialize with default stats_initialized=False
    calc = ICUAdvantageCalculator(beta=1.0)
    
    # Verify initial state
    print(f"  Initial stats_initialized: {calc.stats_initialized.item()}")
    print(f"  Initial adv_mean: {calc.adv_mean.item()}")
    
    # Simulate a few steps
    adv = torch.randn(4, 24) + 5.0 # Positive bias advantages
    
    print("  Feeding batch 1...")
    calc.calculate_awr_weights(adv)
    print(f"  After Batch 1 - stats_initialized: {calc.stats_initialized.item()}")
    print(f"  After Batch 1 - adv_mean: {calc.adv_mean.item()}")
    
    print("  Feeding batch 2...")
    calc.calculate_awr_weights(adv)
    print(f"  After Batch 2 - adv_mean: {calc.adv_mean.item()}")
    
    if not calc.stats_initialized.item() and calc.adv_mean.item() == 0.0:
        print("  CONFIRMED: EMA remains dead. Advantage statistics are NOT being tracked.")
    else:
        print("  SUCCESS: EMA branch is active.")

def test_adv_resume_bias_correction():
    print("\n[TEST] Advantage Resume Shock (Bias Correction Div/Zero)")
    calc = ICUAdvantageCalculator(beta=1.0)
    
    # Simulate resumption: count=0, mu=5.0
    calc.adv_mean.fill_(5.0)
    calc.stats_count.fill_(0)
    calc.stats_initialized.fill_(True)
    
    # mom = 0.999
    # t = 0
    # bias_correction = 1 - 0.999^0 = 0
    # mu = 5.0 / 0.0 -> INF/NaN
    
    adv = torch.randn(4, 24)
    weights, diag = calc.calculate_awr_weights(adv)
    
    mu_val = diag['adv_mean']
    print(f"  Resumption mu (with count=0): {mu_val}")
    
    if torch.isinf(torch.as_tensor(mu_val)) or torch.isnan(torch.as_tensor(mu_val)):
        print("  CONFIRMED: Bias correction causes NaN/Inf shock on resumption!")
    else:
        print("  SUCCESS: Mu is finite.")

if __name__ == "__main__":
    test_adv_ema_activation()
    test_adv_resume_bias_correction()
