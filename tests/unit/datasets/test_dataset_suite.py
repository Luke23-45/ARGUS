import pytest
import torch
import numpy as np
from icu.datasets.normalizer import ClinicalNormalizer
from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn

# ==============================================================================
# 1. NORMALIZER TESTS
# ==============================================================================

def test_normalizer_clinical_defaults():
    """Verify normalizer works with hardcoded medical bounds."""
    norm = ClinicalNormalizer(ts_channels=28, static_channels=6)
    # Note: Normalizer needs calibration before use
    # For this test, we just verify construction
    assert norm.ts_channels == 28
    assert norm.static_channels == 6
    assert not norm.is_calibrated.item()

def test_normalizer_empirical_fitting(mock_dataset_stats, tmp_path):
    """Verify normalizer fits to empirical data stats."""
    from icu.datasets.dataset import CANONICAL_COLUMNS
    
    norm = ClinicalNormalizer(ts_channels=28, static_channels=6)
    
    # Create a mock stats file
    stats_file = tmp_path / "test_index.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump({"metadata": {"stats": mock_dataset_stats}}, f)
    
    # Calibrate from file
    norm.calibrate_from_stats(stats_file, CANONICAL_COLUMNS)
    
    # Verify calibration succeeded
    assert norm.is_calibrated.item()

# ==============================================================================
# 2. DATASET TESTS
# ==============================================================================

def test_dataset_loading(dummy_lmdb):
    """Verify ICUSotaDataset correctly loads and slices LMDB data."""
    ds = ICUSotaDataset(
        dataset_dir=str(dummy_lmdb),
        split="train",
        history_len=24,
        pred_len=6,
        augment_noise=0.0  # Disable noise for pure load test
    )
    
    assert len(ds) == 1  # 30 - (24+6) + 1 = 1 window
    
    sample = ds[0]
    assert sample is not None
    assert sample["observed_data"].shape == (24, 28)  # Clinical 28
    assert sample["future_data"].shape == (6, 28)     # Clinical 28
    assert sample["static_context"].shape == (6,)     # Group D
    assert isinstance(sample["outcome_label"], torch.Tensor)

def test_dataset_augmentation(dummy_lmdb):
    """Verify noise injection changes values but preserves shape."""
    ds_no_noise = ICUSotaDataset(dataset_dir=str(dummy_lmdb), augment_noise=0.0)
    ds_with_noise = ICUSotaDataset(dataset_dir=str(dummy_lmdb), augment_noise=0.1)
    
    sample_clean = ds_no_noise[0]
    sample_noisy = ds_with_noise[0]
    
    assert not torch.equal(sample_clean["observed_data"], sample_noisy["observed_data"])
    assert sample_clean["observed_data"].shape == sample_noisy["observed_data"].shape

# ==============================================================================
# 3. ROBUSTNESS TESTS
# ==============================================================================

def test_robust_collate():
    """Verify batching works correctly even with 'None' (failed) samples."""
    batch = [
        {"data": torch.randn(1)},
        None,
        {"data": torch.randn(1)}
    ]
    
    collated = robust_collate_fn(batch)
    assert collated["data"].shape[0] == 2 # Filtered down to 2 samples
    
    # Test empty batch
    assert robust_collate_fn([None, None]) == {}
