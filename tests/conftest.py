import pytest
import torch
import numpy as np
import lmdb
import json
import shutil
from pathlib import Path
from icu.datasets.normalizer import ClinicalNormalizer

@pytest.fixture
def mock_dataset_stats():
    """Mock statistics for normalizer fitting (Clinical 28 Spec)."""
    # 28 channels matching Clinical 28 spec
    return {
        "ts_min": [40.0] * 28,  # Simplified for testing
        "ts_max": [180.0] * 28,
        "safe_min": [33.0] * 28,
        "safe_max": [187.0] * 28
    }

@pytest.fixture
def temp_data_dir(tmp_path):
    """Creates a temporary directory for LMDB tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def dummy_lmdb(temp_data_dir, mock_dataset_stats):
    """Creates small dummy LMDBs for both train and val splits (Clinical 28 Spec)."""
    for split in ["train", "val"]:
        split_dir = temp_data_dir / split
        split_dir.mkdir()
        
        lmdb_path = split_dir / "data.lmdb"  # Fixed: was f"{split}.lmdb"
        index_path = temp_data_dir / f"{split}_index.json"  # Fixed: parent dir
        
        # Create LMDB
        env = lmdb.open(str(lmdb_path), map_size=10**6, subdir=False)
        
        # 1 Episode per split - Clinical 28 channels
        vitals = np.random.randn(30, 28).astype(np.float32)  # 28 channels
        static = np.random.randn(6).astype(np.float32)       # 6 static features
        labels = np.random.randint(0, 2, size=(30,)).astype(np.float32)  # float32 for consistency
        
        with env.begin(write=True) as txn:
            txn.put(b"ep_0_vitals", vitals.tobytes())
            txn.put(b"ep_0_static", static.tobytes())
            txn.put(b"ep_0_labels", labels.tobytes())
        env.close()
        
        # Create Index (aligned with new schema)
        index_data = {
            "metadata": {
                "ts_columns": ["HR", "O2Sat", "SBP", "DBP", "MAP", "Resp", "Temp",
                              "Lactate", "Creatinine", "Bilirubin", "Platelets", "WBC",
                              "pH", "HCO3", "BUN", "Glucose", "Hgb", "Potassium",
                              "Magnesium", "Calcium", "Chloride", "FiO2",
                              "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"],
                "stats": mock_dataset_stats
            },
            "episodes": [{
                "episode_id": "ep_0",
                "patient_id": f"P_{split}",
                "length": 30,
                "modalities": {
                    "vitals": {"key": "ep_0_vitals", "dtype": "float32", "shape": [30, 28]},
                    "static": {"key": "ep_0_static", "dtype": "float32", "shape": [6]},
                    "labels": {"key": "ep_0_labels", "dtype": "float32", "shape": [30]}
                }
            }]
        }
        
        with open(index_path, "w") as f:
            json.dump(index_data, f)
            
    return temp_data_dir
