import torch
import sys
import os

# Add project root to path
# Project root is c:\Users\Hellx\Documents\Programming\python\Project\iron\icu_research
ROOT_DIR = r"c:\Users\Hellx\Documents\Programming\python\Project\iron\icu_research"
sys.path.append(ROOT_DIR)

from icu.datasets.dataset import CANONICAL_COLUMNS
from icu.models.wrapper_generalist import ICUGeneralistWrapper
from icu.utils.advantage_calculator import ICUAdvantageCalculator, DEFAULT_FEATURE_INDICES
from omegaconf import OmegaConf

def verify_indices():
    print("--- Verifying Feature Indices (Exhaustive 28-Channel Check) ---")
    
    # 1. Source of Truth from phase1_ingest.py
    from icu.datasets.phase1_ingest import CLINICAL_SPECS
    print(f"Ingest Spec Loaded: {len(CLINICAL_SPECS)} channels.")
    
    # 2. Check Dataset Canonical Columns
    assert len(CANONICAL_COLUMNS) == 28, f"Canonical Columns length mismatch: {len(CANONICAL_COLUMNS)}"
    
    for i in range(28):
        ingest_name = CLINICAL_SPECS[i]['name']
        canon_name = CANONICAL_COLUMNS[i]
        
        # Normalized check (handle Bilirubin vs Bilirubin_total if necessary, though ingest.py uses 'Bilirubin' in 'name')
        assert ingest_name == canon_name, f"Index {i} mismatch: Ingest='{ingest_name}' vs Canon='{canon_name}'"
        
    print("✓ Phase 1 Ingest and Dataset Canonical Columns are perfectly aligned.")

    # 3. Check Advantage Calculator Defaults
    for feat, expected_idx in DEFAULT_FEATURE_INDICES.items():
        ingest_feat_name = CLINICAL_SPECS[expected_idx]['name'].lower()
        # Handle aliases like o2sat/spo2
        if feat == 'o2sat' or feat == 'spo2':
            assert ingest_feat_name in ['o2sat', 'spo2'], f"AdvCalc Mismatch for {feat}: Expected {ingest_feat_name} at index {expected_idx}"
        else:
            assert feat == ingest_feat_name, f"AdvCalc Mismatch: {feat} vs {ingest_feat_name} at index {expected_idx}"
    
    print(f"✓ Advantage Calculator Defaults ({len(DEFAULT_FEATURE_INDICES)} tracked) are correct.")

    # 4. Check Wrapper Initialization
    cfg = OmegaConf.create({
        'model': {
            'history_len': 24, 'pred_len': 6, 'd_model': 128, 'n_heads': 4,
            'n_layers': 2, 'encoder_layers': 2, 'dropout': 0.1, 'use_auxiliary_head': True
        },
        'train': {'lr': 1e-4, 'weight_decay': 1e-2, 'warmup_steps': 100, 'total_steps': 1000},
        'seed': 42, 'output_dir': 'outputs'
    })
    
    wrapper = ICUGeneralistWrapper(cfg)
    wrapper_idx = wrapper.clinical_feat_idx
    
    required_keys = ['map', 'lactate', 'hr', 'spo2', 'o2sat', 'sbp', 'resp']
    for key in required_keys:
        idx = wrapper_idx[key]
        expected_name = CLINICAL_SPECS[idx]['name'].lower()
        if key in ['spo2', 'o2sat']:
            assert expected_name in ['o2sat', 'spo2']
        else:
            assert key == expected_name
            
    print(f"✓ Wrapper Clinical Indices ({len(wrapper_idx)} mapped) are perfectly aligned.")

def verify_safety_components():
    print("\n--- Verifying Safety Components (Quantile & DType) ---")
    from icu.models.components.clinical_governor import ConfidenceAwareGovernor
    from icu.utils.stability import DynamicThresholding
    
    governor = ConfidenceAwareGovernor()
    threstholder = DynamicThresholding()
    
    # Simulate float16 input (common for torch.compile/ mixed precision)
    dtype = torch.float16
    x = torch.randn(2, 10, 64).to(dtype)
    p = torch.tensor([0.9, 0.95]).to(dtype)
    
    print(f"Testing components with dtype={dtype}...")
    
    try:
        # Test Governor
        out_gov = governor.apply_governance(x, p)
        print("✓ ConfidenceAwareGovernor: Quantile safety check passed.")
        
        # Test DynamicThresholding
        out_thresh = threstholder(x)
        print("✓ DynamicThresholding: Quantile safety check passed.")
    except RuntimeError as e:
        print(f"✗ Verification Failed: {e}")
        sys.exit(1)

def verify_ddp_buffers():
    print("\n--- Verifying DDP Buffers in AdvCalc ---")
    calc = ICUAdvantageCalculator()
    state_dict = calc.state_dict()
    
    required_buffers = ['beta', 'max_weight', 'ess_buffer', 'clip_rate_buffer']
    for buf in required_buffers:
        assert buf in state_dict, f"Buffer {buf} missing from state_dict!"
        print(f"✓ Buffer {buf} registered for DDP synchronization.")

if __name__ == "__main__":
    try:
        verify_indices()
        verify_safety_components()
        verify_ddp_buffers()
        print("\n" + "="*40)
        print("ALL SOTA SAFETY CHECKS PASSED")
        print("="*40)
    except AssertionError as e:
        print(f"\n✗ ASSERTION ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
