"""
Test BP: Tail Batch Scaling Verification (Static Analysis)
----------------------------------------------------------
Verifies that wrapper_generalist.py contains the fix for Smoking Gun #91.
Fix: Scaling gradients by '1.0 / actual_accum' instead of static accumulation.
"""
import unittest
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BP_Verification")

class TestTailScalingVerification(unittest.TestCase):
    def test_scaling_logic_presence(self):
        logger.info("Verifying Patch #91 (Tail Scaling) in wrapper_generalist.py...")
        
        target_file = r"C:\Users\Hellx\Documents\Programming\python\Project\iron\icu_research\icu\models\wrapper_generalist.py"
        
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Key patterns indicating the fix
        pattern1 = "scale = 1.0 / float(actual_accum)"
        pattern2 = "torch._foreach_mul_"
        
        if pattern1 in content and pattern2 in content:
            logger.info("✅ SUCCESS: Tail scaling logic found in codebase.")
            logger.info(f"Found: '{pattern1}'")
        else:
            logger.error("❌ FAILURE: Tail scaling logic NOT found.")
            self.fail("Fix for Smoking Gun #91 missing in wrapper_generalist.py")

if __name__ == "__main__":
    unittest.main()
