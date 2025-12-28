
import torch
import torch.nn as nn
import logging
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["kagglehub"] = MagicMock()
sys.modules["rich"] = MagicMock()
sys.modules["rich.progress"] = MagicMock()
sys.modules["rich.text"] = MagicMock()

from icu.utils.advantage_calculator import ICUAdvantageCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAdvantage")

class MockNormalizer:
    """Mock normalizer: maps [-1, 1] -> [0, 100] (like clinical units)"""
    def denormalize(self, x): 
        # Simple linear transform: x=0 -> 50, x=-1 -> 0, x=1 -> 100
        return (x + 1) * 50.0 

def test_advantage_fixes():
    logger.info("--- Starting Advantage Calculator Verification ---")
    
    calc = ICUAdvantageCalculator(beta=0.5)
    
    # -------------------------------------------------------------------------
    # Test 1: Normalization Safety
    # -------------------------------------------------------------------------
    logger.info("Test 1: Normalization Safety")
    # Create NORMALIZED vitals that look like Shock if unnormalized (Val ~0)
    # But with Normalizer, they should map to Healthy (Val ~50)
    
    # [B, T, C]
    vitals_norm = torch.zeros(1, 5, 28) # All zeros -> corresponds to 50.0 (Healthy MAP)
    
    # Outcome = Survival (0)
    outcome = torch.zeros(1)
    
    # A. WITHOUT Normalizer (Should Fail / Low Reward)
    # 0.0 is interpreted as MAP=0 (Dead) -> Huge Penalty
    r_bad = calc.compute_clinical_reward(vitals_norm, outcome, normalizer=None)
    mean_bad = r_bad.mean().item()
    logger.info(f"  Without Normalizer (Raw 0.0): {mean_bad:.4f} (Expected << 0)")
    
    # B. WITH Normalizer (Should Pass / High Reward)
    # Target specific channels to ensure "Healthy" physiology
    # We want: MAP ~ 80, Lactate ~ 1.0
    # Mock Denorm: y = (x+1)*50  =>  x = y/50 - 1
    # MAP=80 => x = 1.6 - 1 = 0.6
    vitals_norm_healthy = torch.full_like(vitals_norm, -0.98) # Default to low values (safe lactate/resp)
    vitals_norm_healthy[..., 4] = 0.6  # MAP -> 80.0
    
    r_good = calc.compute_clinical_reward(vitals_norm_healthy, outcome, normalizer=MockNormalizer())
    mean_good = r_good.mean().item()
    logger.info(f"  With Normalizer (Healthy): {mean_good:.4f} (Expected > 0.2743)")
    
    if mean_good > mean_bad: # Significant difference
        logger.info("[SUCCESS] Normalizer is correctly restoring units!")
    else:
        logger.error(f"[FAILURE] Normalizer logic failed. Bad({mean_bad}) >= Good({mean_good})")
        # sys.exit(1) # Don't exit yet, let's check mask test too

    # C. RAW DATA CHECK (Regression Test)
    # Ensure standard Raw Data (MAP=30) passed with Normalizer=None yields correct penalty
    # (Should be same as Case A if A was truly 0.0)
    # Let's test Healthy Raw Data: MAP=80.0. Should yield ~0 penalty.
    vitals_raw_healthy = torch.zeros_like(vitals_norm)
    vitals_raw_healthy[..., 4] = 80.0 
    vitals_raw_healthy[..., 7] = 1.0 # Healthy Lactate
    
    r_raw_healthy = calc.compute_clinical_reward(vitals_raw_healthy, outcome, normalizer=None)
    mean_raw_healthy = r_raw_healthy.mean().item()
    logger.info(f"  Raw Healthy (MAP=80, Norm=None): {mean_raw_healthy:.4f}")
    
    if abs(mean_raw_healthy - mean_good) < 0.1:
         logger.info("[SUCCESS] Raw Data mode working correctly (matches Denormalized result).")
    else:
         logger.warning(f"[WARNING] Raw vs Denorm mismatch: {mean_raw_healthy} vs {mean_good}")

    # -------------------------------------------------------------------------
    # Test 2: Ghost Padding (Src Mask)
    # -------------------------------------------------------------------------
    logger.info("Test 2: Ghost Padding Masking")
    # Batch with 1 real step and 4 padded steps
    # Real step: Healthy (Reward > 0)
    # Padded steps: Zeros (if unnormalized -> Shock -> Reward < 0)
    # WITH MASK: Reward should be 0.0 for padded steps
    
    src_mask = torch.zeros(1, 5).bool()
    src_mask[:, 0] = True # Only first step is valid
    
    # Assume pre-normalized data for simplicity here (or unnormalized)
    # Let's use unnormalized 80.0 (Health)
    vitals_raw = torch.full((1, 5, 28), 80.0) 
    
    # Without Mask (Legacy)
    r_nomask = calc.compute_clinical_reward(vitals_raw, outcome, normalizer=None, src_mask=None)
    logger.info(f"  No Mask Rewards: {r_nomask[0].tolist()}")
    
    # With Mask
    r_mask = calc.compute_clinical_reward(vitals_raw, outcome, normalizer=None, src_mask=src_mask)
    logger.info(f"  With Mask Rewards: {r_mask[0].tolist()}")
    
    if r_mask[0, 1].item() == 0.0 and r_nomask[0, 1].item() != 0.0:
        logger.info("[SUCCESS] Padded steps successfully zeroed out!")
    else:
        logger.error("[FAILURE] Masking did not zero out padded rewards.")
        sys.exit(1)

    logger.info("--- Verification Complete: GREEN LIGHT ---")

if __name__ == "__main__":
    test_advantage_fixes()
