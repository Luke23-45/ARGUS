"""
run_all_tests.py
----------------
Master runner script for all GN instability diagnostic tests.

This script runs each test in sequence and generates a summary report.
Tests are run in order of hypothesis confidence:
1. Test A: AWR Beta Annealing (HIGH confidence)
2. Test B: Gradient EMA Step Count (MEDIUM confidence)
3. Test C: TCB Queue Initialization (MEDIUM confidence)
4. Test D: Loss EMA Accumulation (LOW confidence)

Usage:
    python run_all_tests.py [--stop-on-fail]

    --stop-on-fail: Stop after first failing test (default: run all)

Authors: APEX Diagnostic Team
Date: 2026-01-30
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / 'test_results.log')
    ]
)
logger = logging.getLogger("TEST_RUNNER")


def run_test_a() -> Dict[str, any]:
    """Run Test A: AWR Beta Annealing."""
    from test_a_awr_beta import run_comparative_test, run_multi_epoch_test
    
    logger.info("\n" + "="*70)
    logger.info(" TEST A: AWR Beta Annealing ESS Stability")
    logger.info("="*70)
    
    start = time.time()
    
    # Run single epoch test
    single_result = run_comparative_test(epoch=6)
    
    # If passed, run multi-epoch
    if single_result['passed']:
        multi_result = run_multi_epoch_test()
        passed = multi_result['passed']
        issues = multi_result['issues']
    else:
        passed = False
        issues = single_result['issues']
    
    elapsed = time.time() - start
    
    return {
        'name': 'Test A: AWR Beta Annealing',
        'passed': passed,
        'issues': issues,
        'elapsed': elapsed
    }


def run_test_b() -> Dict[str, any]:
    """Run Test B: Gradient EMA Step Count."""
    from test_b_grad_ema import run_comparative_test
    
    logger.info("\n" + "="*70)
    logger.info(" TEST B: Gradient EMA Step Count Accumulation")
    logger.info("="*70)
    
    start = time.time()
    result = run_comparative_test(n_epochs=3)
    elapsed = time.time() - start
    
    return {
        'name': 'Test B: Gradient EMA Step Count',
        'passed': result['passed'],
        'issues': result['issues'],
        'elapsed': elapsed
    }


def run_test_c() -> Dict[str, any]:
    """Run Test C: TCB Queue Initialization."""
    from test_c_tcb_queue import run_comparative_test
    
    logger.info("\n" + "="*70)
    logger.info(" TEST C: TCB Queue Initialization")
    logger.info("="*70)
    
    start = time.time()
    result = run_comparative_test()
    elapsed = time.time() - start
    
    return {
        'name': 'Test C: TCB Queue Init',
        'passed': result['passed'],
        'issues': result['issues'],
        'elapsed': elapsed
    }


def run_test_d() -> Dict[str, any]:
    """Run Test D: Loss EMA Accumulation."""
    from test_d_loss_ema import run_comparative_test
    
    logger.info("\n" + "="*70)
    logger.info(" TEST D: Loss EMA Accumulation")
    logger.info("="*70)
    
    start = time.time()
    result = run_comparative_test()
    elapsed = time.time() - start
    
    return {
        'name': 'Test D: Loss EMA Accumulation',
        'passed': result['passed'],
        'issues': result['issues'],
        'elapsed': elapsed
    }


def run_test_e() -> Dict[str, any]:
    """Run Test E: Sigma Curriculum Step-Density Analysis."""
    from test_e_sigma_curriculum import run_comparative_test
    
    logger.info("\n" + "="*70)
    logger.info(" TEST E: Sigma Curriculum Step-Density Analysis")
    logger.info("="*70)
    
    start = time.time()
    result = run_comparative_test()
    elapsed = time.time() - start
    
    return {
        'name': 'Test E: Sigma Curriculum',
        'passed': result['passed'],
        'issues': result['issues'],
        'elapsed': elapsed
    }


def run_test_f() -> Dict[str, any]:
    """Run Test F: DynamicThresholding Variance."""
    from test_f_dynamic_thresholding import run_comparative_test
    
    logger.info("\n" + "="*70)
    logger.info(" TEST F: DynamicThresholding Variance")
    logger.info("="*70)
    
    start = time.time()
    result = run_comparative_test()
    elapsed = time.time() - start
    
    return {
        'name': 'Test F: DynamicThresholding',
        'passed': result['passed'],
        'issues': result['issues'],
        'elapsed': elapsed
    }


def generate_summary_report(results: List[Dict[str, any]], output_path: Path):

    """Generate a markdown summary report."""
    
    with open(output_path, 'w') as f:
        f.write("# GN Instability Diagnostic Test Results\n\n")
        f.write(f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Test | Status | Issues | Time |\n")
        f.write("|------|--------|--------|------|\n")
        
        for r in results:
            status = "✅ PASS" if r['passed'] else "❌ FAIL"
            issues = len(r['issues'])
            f.write(f"| {r['name']} | {status} | {issues} | {r['elapsed']:.1f}s |\n")
        
        f.write("\n---\n\n")
        
        # Detailed results
        f.write("## Detailed Results\n\n")
        
        for r in results:
            status = "✅ PASSED" if r['passed'] else "❌ FAILED"
            f.write(f"### {r['name']}\n\n")
            f.write(f"**Status:** {status}\n\n")
            
            if r['issues']:
                f.write("**Issues Detected:**\n\n")
                for issue in r['issues']:
                    f.write(f"- {issue}\n")
            else:
                f.write("No issues detected.\n")
            
            f.write("\n---\n\n")
        
        # Recommendations
        failing_tests = [r for r in results if not r['passed']]
        
        f.write("## Recommendations\n\n")
        
        if not failing_tests:
            f.write("All tests passed. The 4 primary hypotheses are not the root cause.\n\n")
            f.write("**Next Steps:**\n")
            f.write("1. Review additional code paths not covered by these tests\n")
            f.write("2. Enable verbose logging during full training run\n")
            f.write("3. Check for interactions between components\n")
        else:
            f.write(f"**{len(failing_tests)} test(s) failed.** These are likely root causes:\n\n")
            
            for r in failing_tests:
                f.write(f"### {r['name']}\n\n")
                for issue in r['issues']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            f.write("**Recommended Fixes:**\n\n")
            
            for r in failing_tests:
                if 'AWR' in r['name']:
                    f.write("- **AWR Beta:** Disable beta annealing or use step-based schedule\n")
                elif 'Gradient EMA' in r['name']:
                    f.write("- **Grad EMA:** Reset `grad_norm_step_count` at epoch boundaries\n")
                elif 'TCB' in r['name']:
                    f.write("- **TCB Queue:** Extend `tcb_warmup_steps` to match scaled capacity\n")
                elif 'Loss EMA' in r['name']:
                    f.write("- **Loss EMA:** Add clipping to `loss_emas` update (max=10.0)\n")
    
    logger.info(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run GN Instability Diagnostic Tests')
    parser.add_argument('--stop-on-fail', action='store_true', 
                        help='Stop after first failing test')
    args = parser.parse_args()
    
    logger.info("\n" + "#"*70)
    logger.info("# GN INSTABILITY DIAGNOSTIC TEST SUITE")
    logger.info("# Running all 4 root cause hypothesis tests")
    logger.info("#"*70)
    
    total_start = time.time()
    results = []
    
    # Test functions in order of confidence
    tests = [
        ('A', run_test_a),
        ('B', run_test_b),
        ('C', run_test_c),
        ('D', run_test_d),
        ('E', run_test_e),
        ('F', run_test_f)
    ]
    
    for test_id, test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
            
            if not result['passed'] and args.stop_on_fail:
                logger.warning(f"\n[STOP ON FAIL] Test {test_id} failed. Stopping.")
                break
                
        except Exception as e:
            logger.error(f"\n[ERROR] Test {test_id} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': f'Test {test_id}',
                'passed': False,
                'issues': [f'CRASH: {str(e)}'],
                'elapsed': 0
            })
            
            if args.stop_on_fail:
                break
    
    total_elapsed = time.time() - total_start
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info(" FINAL SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    
    logger.info(f"\nTotal Tests Run: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total Time: {total_elapsed:.1f}s")
    
    for r in results:
        status = "✅" if r['passed'] else "❌"
        logger.info(f"  {status} {r['name']}: {len(r['issues'])} issues ({r['elapsed']:.1f}s)")
    
    # Generate report
    report_path = Path(__file__).parent.parent / 'TEST_RESULTS_REPORT.md'
    generate_summary_report(results, report_path)
    
    # Exit code
    if failed > 0:
        logger.info("\n❌ SOME TESTS FAILED - Root causes identified!")
        sys.exit(1)
    else:
        logger.info("\n✅ ALL TESTS PASSED - Investigate other hypotheses")
        sys.exit(0)


if __name__ == "__main__":
    main()
