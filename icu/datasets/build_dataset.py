"""
datasets/build_dataset.py
--------------------------------------------------------------------------------
APEX-MoE: The Frontier Data Engineering Pipeline (v1.0 Production Grade).
Strictly adheres to "Clinical 28" Synchronous Channel Specification.
"""

import os
import sys
import json
import random
import lmdb
import pandas as pd
import numpy as np
import logging
import hydra
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from typing import List, Dict, Optional, Any

try:
    import kagglehub
except ImportError:
    kagglehub = None

# Logger Configuration
logger = logging.getLogger("ICU_Build_Dataset")

# ==============================================================================
# 1. THE FRONTIER SPECIFICATION (STRICT INDEXING)
# ==============================================================================

# THE "CLINICAL 28" - EXACT ORDER (Indices 0-27)
# Group A: Hemodynamic (0-6)
# Group B: Sepsis Drivers/Labs (7-17)
# Group C: Electrolytes/Support (18-21)
# Group D: Static/Context (22-27)
CLINICAL_SPECS = {
    0:  {'col': 'HR',       'default': 75.0,  'name': 'HR'},
    1:  {'col': 'O2Sat',    'default': 98.0,  'name': 'O2Sat'},
    2:  {'col': 'SBP',      'default': 120.0, 'name': 'SBP'},
    3:  {'col': 'DBP',      'default': 80.0,  'name': 'DBP'},
    4:  {'col': 'MAP',      'default': 93.0,  'name': 'MAP'},  # Index 4 per Doc
    5:  {'col': 'Resp',     'default': 16.0,  'name': 'Resp'}, # Index 5 per Doc
    6:  {'col': 'Temp',     'default': 37.0,  'name': 'Temp'},
    7:  {'col': 'Lactate',          'default': 1.0,  'name': 'Lactate'},
    8:  {'col': 'Creatinine',       'default': 1.0,  'name': 'Creatinine'},
    9:  {'col': 'Bilirubin_total',  'default': 0.6,  'name': 'Bilirubin'},
    10: {'col': 'Platelets',        'default': 250.0,'name': 'Platelets'},
    11: {'col': 'WBC',              'default': 9.0,  'name': 'WBC'}, # Fixed Case
    12: {'col': 'pH',               'default': 7.4,  'name': 'pH'},
    13: {'col': 'HCO3',             'default': 24.0, 'name': 'HCO3'},
    14: {'col': 'BUN',              'default': 15.0, 'name': 'BUN'},
    15: {'col': 'Glucose',          'default': 100.0,'name': 'Glucose'},
    16: {'col': 'Hgb',              'default': 14.0, 'name': 'Hgb'},
    17: {'col': 'Potassium',        'default': 4.0,  'name': 'Potassium'},
    18: {'col': 'Magnesium', 'default': 2.0,  'name': 'Magnesium'},
    19: {'col': 'Calcium',   'default': 9.5,  'name': 'Calcium'},
    20: {'col': 'Chloride',  'default': 102.0,'name': 'Chloride'},
    21: {'col': 'FiO2',      'default': 0.21, 'name': 'FiO2'},
    22: {'col': 'Age',          'default': 60.0, 'name': 'Age',      'type': 'static'},
    23: {'col': 'Gender',       'default': 1.0,  'name': 'Gender',   'type': 'static'},
    24: {'col': 'Unit1',        'default': 0.0,  'name': 'Unit1',    'type': 'static'},
    25: {'col': 'Unit2',        'default': 0.0,  'name': 'Unit2',    'type': 'static'},
    26: {'col': 'HospAdmTime',  'default': -10.0,'name': 'HospAdmTime', 'type': 'static'},
    27: {'col': 'ICULOS',       'default': 1.0,  'name': 'ICULOS',   'type': 'dynamic'} # Incremental
}

# SOTA PHYSICS BOUNDS (Outlier Mitigation)
PHYSICS_BOUNDS = {
    'HR': (20.0, 300.0), 'O2Sat': (20.0, 100.0), 'SBP': (20.0, 300.0), 
    'DBP': (10.0, 200.0), 'Resp': (4.0, 80.0), 'MAP': (20.0, 250.0), 
    'Temp': (24.0, 45.0), 'Lactate': (0.1, 30.0), 'WBC': (0.1, 200.0), 
    'Creatinine': (0.1, 25.0), 'Bilirubin': (0.1, 80.0), 'Platelets': (1.0, 2000.0), 
    'Glucose': (10.0, 1200.0), 'BUN': (1.0, 250.0), 'HCO3': (5.0, 60.0), 
    'pH': (6.5, 7.8), 'Hgb': (2.0, 25.0), 'Potassium': (1.0, 12.0), 
    'Magnesium': (0.5, 10.0), 'Calcium': (2.0, 20.0), 'Chloride': (50.0, 150.0), 
    'FiO2': (0.21, 1.0)
}

FEATURE_ORDER = [CLINICAL_SPECS[i]['name'] for i in range(28)]
LABEL_COL = 'SepsisLabel'
KAGGLE_DATASET_ID = "farjanayesmin/the-physionet-challenge-2019-dataset"

# ==============================================================================
# 2. THE ENGINE
# ==============================================================================

class ICUExpertWriter:
    def __init__(self, out_dir: Path, split_name: str, calc_stats: bool = False):
        self.split_dir = out_dir / split_name
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.calc_stats = calc_stats
        
        self.mins = np.full(28, np.inf)
        self.maxs = np.full(28, -np.inf)
        
        self.lmdb_path = self.split_dir / "data.lmdb"
        self.env = lmdb.open(str(self.lmdb_path), map_size=12 * 1024**3, subdir=False)
        self.index = []
        self.ep_cnt = 0
        self.stats_reservoir = [] # For Quantile Hardening

    def _impute_and_clamp(self, df: pd.DataFrame) -> np.ndarray:
        L = len(df)
        data = np.zeros((L, 28), dtype=np.float32)
        
        for i in range(28):
            spec = CLINICAL_SPECS[i]
            col, default, name = spec['col'], spec['default'], spec['name']
            
            # 1. Extraction
            raw = df[col].values if col in df.columns else np.full(L, np.nan)
            
            # 2. Physics Clamping (Outlier Removal)
            if name in PHYSICS_BOUNDS:
                low, high = PHYSICS_BOUNDS[name]
                raw = np.clip(raw, low, high)
            
            # 3. Imputation (Forward -> Backward -> Default)
            if spec.get('type') == 'static':
                # Statics are constant; handle separately to avoid temporal leakage
                valid = df[col].dropna() if col in df.columns else []
                data[:, i] = valid.iloc[0] if len(valid) > 0 else default
            else:
                # Dynamic Sample-and-Hold
                data[:, i] = pd.Series(raw).ffill().bfill().fillna(default).values
        
        return data

    def process_files(self, files: List[str]):
        with self.env.begin(write=True) as txn:
            for fpath in tqdm(files, desc=f"Ingesting {self.split_name}"):
                try:
                    df = pd.read_csv(fpath, sep='|')
                    if len(df) < 6: continue
                    
                    data_matrix = self._impute_and_clamp(df)
                    
                    if self.calc_stats:
                        self.mins = np.minimum(self.mins, np.nanmin(data_matrix, axis=0))
                        self.maxs = np.maximum(self.maxs, np.nanmax(data_matrix, axis=0))
                        
                        # Store subset for Quantiles (Reservoir Sampling)
                        if len(data_matrix) > 0:
                            indices = np.random.choice(len(data_matrix), max(1, len(data_matrix)//20), replace=False)
                            self.stats_reservoir.append(data_matrix[indices])
                    
                    labels = df[LABEL_COL].values.astype(np.float32) if LABEL_COL in df.columns else np.zeros(len(df))
                    
                    ep_id = f"ep_{self.ep_cnt:06d}"
                    txn.put(f"{ep_id}_vitals".encode(), data_matrix.tobytes())
                    txn.put(f"{ep_id}_static".encode(), data_matrix[0, 22:28].tobytes())
                    txn.put(f"{ep_id}_labels".encode(), labels.tobytes())
                    
                    self.index.append({
                        "episode_id": ep_id, "patient_id": Path(fpath).stem, "length": len(df),
                        "modalities": {
                            "vitals": {"key": f"{ep_id}_vitals", "dtype": "float32", "shape": [len(df), 28]},
                            "static": {"key": f"{ep_id}_static", "dtype": "float32", "shape": [6]},
                            "labels": {"key": f"{ep_id}_labels", "dtype": "float32", "shape": [len(df)]}
                        }
                    })
                    self.ep_cnt += 1
                except Exception as e:
                    continue

        self._finalize()

    def _finalize(self):
        metadata = {
            "source": "PhysioNet2019 (Frontier 28)",
            "ts_columns": FEATURE_ORDER,  # Aligned with dataset.py expectations
            "stats": {}
        }
        if self.calc_stats:
            ranges = np.where((self.maxs - self.mins) == 0, 1.0, self.maxs - self.mins)
            
            # Robust Quantiles Calculation
            res_data = np.concatenate(self.stats_reservoir) if self.stats_reservoir else np.zeros((1, 28))
            p01 = np.percentile(res_data, 1, axis=0).tolist()
            p99 = np.percentile(res_data, 99, axis=0).tolist()
            
            metadata["stats"] = {
                "ts_min": self.mins.tolist(),
                "ts_max": self.maxs.tolist(),
                "ts_p01": p01,
                "ts_p99": p99,
                "safe_min": (self.mins - 0.05 * ranges).tolist(),
                "safe_max": (self.maxs + 0.05 * ranges).tolist()
            }
        
        with open(self.split_dir.parent / f"{self.split_name}_index.json", "w") as f:
            json.dump({"episodes": self.index, "metadata": metadata}, f, indent=2)
        self.env.close()

# ==============================================================================
# 3. PIPELINE ORCHESTRATOR
# ==============================================================================

def run_build_pipeline(output_dir: str):
    output_path = Path(output_dir)
    logger.info("Acquiring Dataset...")
    raw_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    all_files = sorted([str(f) for f in raw_path.rglob("*.psv")])
    
    random.seed(2025)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    
    logger.info(f"Processing Train ({split_idx} files)...")
    train_writer = ICUExpertWriter(output_path, "train", calc_stats=True)
    train_writer.process_files(all_files[:split_idx])
    
    logger.info(f"Processing Val ({len(all_files)-split_idx} files)...")
    val_writer = ICUExpertWriter(output_path, "val", calc_stats=False)
    val_writer.process_files(all_files[split_idx:])

@hydra.main(version_base=None, config_path="../../conf", config_name="train/generalist")
def main(cfg: DictConfig):
    run_build_pipeline(cfg.dataset.get("dataset_dir", "data/ready"))

if __name__ == "__main__":
    main()