# ==============================================================================
# PHASE 1: THE "CLINICAL 28" FINAL PIPELINE (SOTA v12.5)
# ==============================================================================
# Standards:
# 1. Feature Set: "Clinical 28" (Aligned with v1.0 Specification)
# 2. Order: [Hemodynamic 0-6] -> [Labs 7-17] -> [Electrolytes 18-21] -> [Static 22-27]
# 3. Imputation: Sample-and-Hold (Forward -> Backward -> Clinical Default)
# 4. Storage: Raw Float32 in LMDB (Normalization deferred to external script)
# ==============================================================================

import os
import json
import lmdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import kagglehub
import logging

# --- CONFIGURATION ---
OUTPUT_DIR = "sepsis_clinical_28"
LMDB_MAP_SIZE = 5 * 1024 ** 3  # 12GB allocation
RANDOM_SEED = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Clinical28")

# ==============================================================================
# PART 1: THE "CLINICAL 28" SPECIFICATION (STRICT INDEXING)
# ==============================================================================
# Indices 0-27 mapped exactly to documentation requirements.
CLINICAL_SPECS = {
    # --- Group A: Hemodynamic (0-6) ---
    0:  {'col': 'HR',       'default': 75.0,  'name': 'HR'},
    1:  {'col': 'O2Sat',    'default': 98.0,  'name': 'O2Sat'},
    2:  {'col': 'SBP',      'default': 120.0, 'name': 'SBP'},
    3:  {'col': 'DBP',      'default': 80.0,  'name': 'DBP'},
    4:  {'col': 'MAP',      'default': 93.0,  'name': 'MAP'},  # FIXED: Index 4
    5:  {'col': 'Resp',     'default': 16.0,  'name': 'Resp'}, # FIXED: Index 5
    6:  {'col': 'Temp',     'default': 37.0,  'name': 'Temp'},
    
    # --- Group B: Sepsis Drivers (Labs) (7-17) ---
    7:  {'col': 'Lactate',          'default': 1.0,  'name': 'Lactate'},
    8:  {'col': 'Creatinine',       'default': 1.0,  'name': 'Creatinine'},
    9:  {'col': 'Bilirubin_total',  'default': 0.6,  'name': 'Bilirubin'},
    10: {'col': 'Platelets',        'default': 250.0,'name': 'Platelets'},
    11: {'col': 'WBC',              'default': 9.0,  'name': 'WBC'},
    12: {'col': 'pH',               'default': 7.4,  'name': 'pH'},
    13: {'col': 'HCO3',             'default': 24.0, 'name': 'HCO3'},
    14: {'col': 'BUN',              'default': 15.0, 'name': 'BUN'},
    15: {'col': 'Glucose',          'default': 100.0,'name': 'Glucose'},
    16: {'col': 'Hgb',              'default': 14.0, 'name': 'Hgb'},
    17: {'col': 'Potassium',        'default': 4.0,  'name': 'Potassium'},

    # --- Group C: Electrolytes & Support (18-21) ---
    18: {'col': 'Magnesium', 'default': 2.0,  'name': 'Magnesium'},
    19: {'col': 'Calcium',   'default': 9.5,  'name': 'Calcium'},
    20: {'col': 'Chloride',  'default': 102.0,'name': 'Chloride'},
    21: {'col': 'FiO2',      'default': 0.21, 'name': 'FiO2'},

    # --- Group D: Context (22-27) ---
    # Note: These are broadcasted to every time-step in the [T, 28] matrix
    22: {'col': 'Age',          'default': 60.0, 'name': 'Age'},
    23: {'col': 'Gender',       'default': 1.0,  'name': 'Gender'},
    24: {'col': 'Unit1',        'default': 0.0,  'name': 'Unit1'},
    25: {'col': 'Unit2',        'default': 0.0,  'name': 'Unit2'},
    26: {'col': 'HospAdmTime',  'default': -10.0,'name': 'HospAdmTime'},
    27: {'col': 'ICULOS',       'default': 1.0,  'name': 'ICULOS'} # Dynamic progression
}

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

# ==============================================================================
# PART 2: THE DATA ENGINE
# ==============================================================================
class Clinical28Ingestor:
    def __init__(self, out_dir, split_name):
        self.split_dir = Path(out_dir) / split_name
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        
        # Stats accumulation (for the external normalization script)
        self.mins = np.full(28, np.inf)
        self.maxs = np.full(28, -np.inf)
        
        # LMDB Setup
        self.lmdb_path = self.split_dir / "data.lmdb"
        self.env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False)
        
        self.index = []
        self.ep_cnt = 0
        self.stats_reservoir = [] # For Robust Quantile Calculation

    def process_files(self, files):
        with self.env.begin(write=True) as txn:
            for fpath in tqdm(files, desc=f"Ingesting {self.split_name}"):
                try:
                    df = pd.read_csv(fpath, sep='|')
                    L = len(df)
                    if L < 6: continue # Minimum window requirement
                    
                    data_matrix = np.zeros((L, 28), dtype=np.float32)
                    
                    for i in range(28):
                        spec = CLINICAL_SPECS[i]
                        col, default, name = spec['col'], spec['default'], spec['name']
                        
                        # Extract and clean
                        raw = df[col].values if col in df.columns else np.full(L, np.nan)
                        
                        # Apply Physics Bounds (Outlier Removal)
                        if name in PHYSICS_BOUNDS:
                            low, high = PHYSICS_BOUNDS[name]
                            raw = np.clip(raw, low, high)
                        
                        # Sample-and-Hold Imputation
                        # Ffill (forward), then Bfill (start of stay), then Default (never measured)
                        clean = pd.Series(raw).ffill().bfill().fillna(default).values
                        data_matrix[:, i] = clean
                    
                    # Update global stats for metadata
                    self.mins = np.minimum(self.mins, np.nanmin(data_matrix, axis=0))
                    self.maxs = np.maximum(self.maxs, np.nanmax(data_matrix, axis=0))
                    
                    # Reservoir Sampling for Robust Quantiles (Collect ~5% of rows)
                    if len(data_matrix) > 0:
                        indices = np.random.choice(len(data_matrix), max(1, len(data_matrix)//20), replace=False)
                        self.stats_reservoir.append(data_matrix[indices])
                        
                    # Memory Safety: Cap the reservoir to prevent OOM
                    if len(self.stats_reservoir) > 10000:
                        combined = np.concatenate(self.stats_reservoir)
                        # Downsample back to a manageable size
                        keep_idx = np.random.choice(len(combined), 100000, replace=False) if len(combined) > 100000 else np.arange(len(combined))
                        self.stats_reservoir = [combined[keep_idx]]
                    
                    # Labels (with defensive check for missing column)
                    if 'SepsisLabel' in df.columns:
                        labels = df['SepsisLabel'].values.astype(np.float32)
                    else:
                        logger.warning(f"Missing SepsisLabel in {Path(fpath).name}, defaulting to zeros")
                        labels = np.zeros(L, dtype=np.float32)
                    
                    # Write to LMDB
                    ep_id = f"ep_{self.ep_cnt:06d}"
                    txn.put(f"{ep_id}_vitals".encode(), data_matrix.tobytes())
                    
                    # Static Context: First row of Group D (indices 22-27)
                    static_context = data_matrix[0, 22:28].astype(np.float32)
                    txn.put(f"{ep_id}_static".encode(), static_context.tobytes())
                    
                    txn.put(f"{ep_id}_labels".encode(), labels.tobytes())
                    
                    # Index metadata (with dtype for robust deserialization)
                    self.index.append({
                        "episode_id": ep_id,
                        "patient_id": Path(fpath).stem,
                        "length": L,
                        "modalities": {
                            "vitals": {"key": f"{ep_id}_vitals", "dtype": "float32", "shape": [L, 28]},
                            "static": {"key": f"{ep_id}_static", "dtype": "float32", "shape": [6]},
                            "labels": {"key": f"{ep_id}_labels", "dtype": "float32", "shape": [L]}
                        }
                    })
                    self.ep_cnt += 1
                    
                except Exception as e:
                    continue

        self.save_index()
        self.env.close()

    def save_index(self):
        # Apply 5% safety margin to stats for future normalization logic
        ranges = self.maxs - self.mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        
        metadata = {
            "version": "1.0-Frontier",
            "feature_set": "Clinical 28",
            "ts_columns": FEATURE_ORDER,  # Aligned with ICUTrajectoryDataset
            "stats": {
                # 1. Primary Min/Max (Legacy Fallback)
                "ts_min": self.mins.tolist(),
                "ts_max": self.maxs.tolist(),
                # 2. Robust Quantiles (New SOTA Standard)
                "ts_p01": np.percentile(np.concatenate(self.stats_reservoir), 1, axis=0).tolist() if self.stats_reservoir else (self.mins if np.isfinite(self.mins).all() else [0.0]*28).tolist(),
                "ts_p99": np.percentile(np.concatenate(self.stats_reservoir), 99, axis=0).tolist() if self.stats_reservoir else (self.maxs if np.isfinite(self.maxs).all() else [1.0]*28).tolist(),
                # 3. Safe Margins (Legacy)
                "safe_min": (self.mins - (0.05 * ranges)).tolist(),
                "safe_max": (self.maxs + (0.05 * ranges)).tolist()
            }
        }
        
        output_path = self.split_dir.parent / f"{self.split_name}_index.json"
        with open(output_path, 'w') as f:
            json.dump({"episodes": self.index, "metadata": metadata}, f, indent=2)
        
        logger.info(f"Successfully serialized {self.ep_cnt} episodes to {self.split_name}")

# ==============================================================================
# PART 3: EXECUTION
# ==============================================================================
def main():
    logger.info("Starting APEX-MoE Data Pipeline...")
    
    # 1. Download
    dataset_path = kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")
    
    # 2. Collect files
    all_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.psv'):
                all_files.append(os.path.join(root, file))
    
    # 3. Train/Val Split
    import random
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # 4. Process
    logger.info(f"Processing {len(train_files)} training and {len(val_files)} validation samples.")
    
    train_proc = Clinical28Ingestor(OUTPUT_DIR, "train")
    train_proc.process_files(train_files)
    
    val_proc = Clinical28Ingestor(OUTPUT_DIR, "val")
    val_proc.process_files(val_files)
    
    logger.info("Pipeline Complete. Data is ready for the normalization script.")

if __name__ == "__main__":
    main()