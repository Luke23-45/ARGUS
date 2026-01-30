import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_BY")

def test_sampler_coverage_gap():
    logger.info("Verifying Smoking Gun #129: Sampler Coverage Gap...")
    
    # Simulate a large dataset: 5,000,000 samples
    n_samples = 5_000_000
    
    # Simulate 5% sepsis rate (250,000 sepsis samples)
    sepsis_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    sepsis_mask = np.zeros(n_samples, dtype=bool)
    sepsis_mask[sepsis_indices] = True
    
    # CURRENT IMPLEMENTATION LOGIC:
    max_samples_to_scan = 100_000
    scan_count = min(n_samples, max_samples_to_scan)
    indices_to_scan = np.linspace(0, n_samples - 1, scan_count, dtype=int)
    
    # Tracking weights
    weights = torch.ones(n_samples)
    boost_factor = 10.0
    
    # Scanned Sepsis Boosted
    sepsis_scan_count = 0
    for idx in indices_to_scan:
        if sepsis_mask[idx]:
            weights[idx] = boost_factor
            sepsis_scan_count +=1
            
    # RESULTS ANALYSIS
    total_sepsis = sepsis_mask.sum()
    boosted_sepsis = (weights[sepsis_mask] == boost_factor).sum().item()
    ignored_sepsis = total_sepsis - boosted_sepsis
    
    coverage_pct = (boosted_sepsis / total_sepsis) * 100
    
    logger.info(f"Total Sepsis Samples in Dataset: {total_sepsis:,}")
    logger.info(f"Sepsis Samples Boosted (Scanned): {int(boosted_sepsis):,}")
    logger.info(f"Sepsis Samples Ignored (Unscanned): {int(ignored_sepsis):,}")
    logger.info(f"Coverage: {coverage_pct:.2f}%")
    
    # THE SMOKING GUN:
    # If coverage is low (e.g. 2%), then 98% of sepsis samples are treated as "Negative" or "Rare" outliers.
    if coverage_pct < 5.0:
        logger.error(f"❌ Smoking Gun #129 CONFIRMED! Sampler ignores {100-coverage_pct:.1f}% of sepsis population.")
        logger.warning("⚠️ Rationale: Regional scanning creates an 'Elite Class' of 2% sepsis samples that the model masters, while the other 98% cause massive gradient spikes when encountered randomly.")

if __name__ == "__main__":
    test_sampler_coverage_gap()
