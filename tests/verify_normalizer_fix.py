
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from icu.datasets.normalizer import ClinicalNormalizer, CANONICAL_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NormalizerTest")

def test_alias_handling():
    """
    Verifies that 'Bilirubin_total' is correctly mapped to 'Bilirubin'
    BEFORE the strict canonical validation check.
    """
    print("="*60)
    print("TEST: 'Phantom Normalizer' Fix Verification")
    print("="*60)

    # 1. Create Dummy Stats File
    stats_data = {
        "metadata": {
            "stats": {
                "ts_count": 2048,
                "ts_min": [-1.0] * 28,
                "ts_max": [1.0] * 28
            }
        }
    }
    stats_path = Path("temp_test_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f)
    
    # 2. Simulate "Dirty" Columns (PhysioNet Style)
    # Replace 'Bilirubin' with 'Bilirubin_total'
    dirty_columns = list(CANONICAL_COLUMNS)
    b_idx = dirty_columns.index("Bilirubin")
    dirty_columns[b_idx] = "Bilirubin_total"
    
    print(f"[INPUT] Dirty Columns provided (Index {b_idx}): {dirty_columns[b_idx]}")
    
    # 3. Initialize Normalizer
    norm = ClinicalNormalizer(ts_channels=28, static_channels=6)
    
    try:
        # 4. Attempt Calibration
        print("[ACTION] Calibrating...")
        norm.calibrate_from_stats(stats_path, dirty_columns)
        
        # 5. Verify Success
        if norm.is_calibrated:
            print("[SUCCESS] Normalizer successfully handled 'Bilirubin_total'!")
            print(f"[STATE] calibrated={norm.is_calibrated.item()}")
        else:
            print("[FAILURE] Calibration function returned but is_calibrated is False.")
            
    except ValueError as e:
        print(f"[FAILURE] ValueError raised: {e}")
        print("The fix is NOT working correctly.")
        exit(1)
    except Exception as e:
        print(f"[FAILURE] Unexpected exception: {e}")
        exit(1)
    finally:
        # Cleanup
        if stats_path.exists():
            stats_path.unlink()

if __name__ == "__main__":
    test_alias_handling()
