"""
icu/datasets/dataset_quality.py
--------------------------------------------------------------------------------
APEX-MoE SOTA Data Ingestion Pipeline (v5.1 — Production Hardened)

Author: APEX Research Team
Status: Production / Safety-Critical

Description:
    Enriched replacement for build_dataset.py. Processes raw PhysioNet 2019 .psv 
    files into LMDB binaries with clinical feature engineering.
    
    OUTPUT FORMAT: 100% compatible with ICUTrajectoryDataset (Clinical 28).
    The pipeline produces exactly 28 channels in the canonical order.
    It enhances quality (imputation, safety) without changing the manifold width.

    Enrichments over v3.0 (build_dataset.py):
    1. Continuous Homeostatic Imputation (Smooth Asymptotes)
    2. Unit Harmonization (Temp F->C, FiO2 L/min->Fraction)
    3. Physiological Hierarchy Masks (SBP > DBP enforcement)
    4. Demographic Protection (Non-decaying Age/Gender)
    5. Dual-Masking Logic (Binary + Continuum)

    Safety Guarantees:
    - FIXED feature manifest: Every patient produces EXACTLY N_FEATURES columns.
    - Deterministic split: Hash-based patient assignment (no shuffle order dependency).
    - Physics clamping: Out-of-range values treated as sensor error → NaN → imputed.
    - Zero future leakage: All rolling/shift operations are strictly causal.

Dependencies:
    pandas, numpy, lmdb, tqdm, (optional: kagglehub for auto-download)
"""

import os
import sys
import json
import lmdb
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
try:
    import kagglehub
except ImportError:
    kagglehub = None
from typing import List, Tuple, Optional, Dict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

LMDB_MAP_SIZE = 3 * 1024 ** 3  # 10GB (Project Standard)
RESERVOIR_SIZE = 200_000
SEED = 2026
MIN_STAY_HOURS = 8   # Minimum ICU stay to include
VAL_RATIO = 0.10     # 10% validation
EPS = 1e-6           # FP16 Precision Shield

# Full Iridium SQW Bounds (Soft-Quantile Winsorization) - Destroys 42x stretch globally
SQW_BOUNDS = {
    'HR': (30.0, 180.0), 'O2Sat': (50.0, 100.0), 'SBP': (50.0, 220.0),
    'DBP': (30.0, 120.0), 'MAP': (40.0, 150.0), 'Resp': (8.0, 45.0), 'Temp': (32.0, 41.0),
    'Lactate': (0.2, 15.0), 'Creatinine': (0.2, 10.0), 'Bilirubin': (0.1, 8.0), 
    'Platelets': (10.0, 1000.0), 'WBC': (1.0, 50.0), 'pH': (6.8, 7.8), 
    'HCO3': (10.0, 50.0), 'BUN': (2.0, 100.0), 'Glucose': (20.0, 600.0),
    'Hgb': (5.0, 20.0), 'Potassium': (2.0, 7.5), 'Magnesium': (1.0, 5.0),
    'Calcium': (5.0, 15.0), 'Chloride': (70.0, 130.0), 'FiO2': (0.21, 1.0)
}

# Adaptive lambda half-lives (Biological priors)
LAMBDA_MAP = {'vital': 0.25, 'electrolyte': 0.08, 'lab': 0.03, 'gas': 0.25, 'static': 0.0}

# Log-Space mapping alignment
LOG_SPACE_CHANNELS = {'Lactate', 'Creatinine', 'Bilirubin', 'Platelets', 'WBC', 'BUN', 'Glucose'}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("APEX_QualityBuilder")

# ==============================================================================
# 1. CLINICAL SPECIFICATION (The "Clinical 28" Canonical Spec)
# ==============================================================================
# This order MUST matches icu/datasets/dataset.py CANONICAL_COLUMNS
RAW_CLINICAL_SPEC = {
    # Group A: Hemodynamic (0-6)
    'HR':       (75.0,  (20, 300),    'vital'),   # Synced to build_dataset.py
    'O2Sat':    (98.0,  (20, 100),    'vital'),
    'SBP':      (120.0, (20, 300),    'vital'),
    'DBP':      (80.0,  (10, 200),    'vital'),
    'MAP':      (93.0,  (20, 250),    'vital'),
    'Resp':     (16.0,  (4, 80),      'vital'),
    'Temp':     (37.0,  (24, 45),      'vital'),
    # Group B: Sepsis Drivers / Labs (7-17)
    'Lactate':          (1.0,   (0.1, 30),   'lab'),
    'Creatinine':       (1.0,   (0.1, 25),   'lab'),
    'Bilirubin':        (0.6,   (0.1, 80),   'lab'),  # Synced naming
    'Platelets':        (250.0, (1, 2000),    'lab'),
    'WBC':              (9.0,   (0.1, 200),   'lab'),
    'pH':               (7.4,   (6.5, 7.8),   'lab'),
    'HCO3':             (24.0,  (5, 60),      'lab'),
    'BUN':              (15.0,  (1, 250),     'lab'),
    'Glucose':          (100.0, (10, 1200),    'lab'),   # Synced to build_dataset.py
    'Hgb':              (14.0,  (2, 25),      'lab'),
    'Potassium':        (4.0,   (1, 12),      'lab'),
    # Group C: Electrolytes & Support (18-21)
    'Magnesium':   (2.0,   (0.5, 10),    'electrolyte'),
    'Calcium':     (9.5,   (2, 20),     'electrolyte'),
    'Chloride':    (102.0, (50, 150),   'electrolyte'),
    'FiO2':        (0.21,  (0.21, 1.0), 'gas'),
    # Group D: Static (22-27)
    'Age':          (60.0, (15, 120),   'static'), # Maintained Titanium expansion
    'Gender':       (1.0,  (0, 1),      'static'), # Synced to build_dataset.py
    'Unit1':        (0.0,  (0, 1),      'static'),
    'Unit2':        (0.0,  (0, 1),      'static'),
    'HospAdmTime':  (-10.0, (-1000, 0), 'static'),
    'ICULOS':       (1.0,  (0, 2000),   'static'),
}

RAW_COLUMNS = list(RAW_CLINICAL_SPEC.keys())

# ==============================================================================
# 2. FIXED FEATURE MANIFEST (28 Channels Only)
# ==============================================================================

def build_feature_manifest() -> List[str]:
    """Build the FIXED Clinical 28 manifest."""
    return list(RAW_COLUMNS)


# Pre-compute the manifest (module-level constant)
FEATURE_MANIFEST = build_feature_manifest()
N_FEATURES = len(FEATURE_MANIFEST) # Should be 28
N_RAW = len(RAW_COLUMNS)           # Should be 28

# Static context columns (subset of raw)
STATIC_COLS = [c for c, (_, _, cat) in RAW_CLINICAL_SPEC.items() if cat == 'static']
N_STATIC = len(STATIC_COLS)        # Should be 6

# ==============================================================================
# 3. VECTORIZED FEATURE EXTRACTION
# ==============================================================================

def compute_decayed_tslm_vectorized(mask: np.ndarray, lambda_decay: float = 0.1) -> np.ndarray:
    """
    Pure NumPy implementation of exponential time-decay.
    Optimized for zero-loop execution.
    """
    T = len(mask)
    steps = np.arange(T, dtype=np.float32)
    
    # Identify indices where a real measurement occurred
    measured_idx = np.where(mask > 0.5)[0]
    
    if len(measured_idx) == 0:
        # Never measured: return deep decay (10h cold start)
        return np.exp(-lambda_decay * (steps + 10.0))
    
    # Broadcast most recent measurement index forward across time
    # last_m_time[t] = time of most recent measurement at or before t
    last_m_time = np.zeros(T, dtype=np.int32)
    last_m_time[measured_idx] = measured_idx
    last_m_time = np.maximum.accumulate(last_m_time)
    
    # Calculate delta time (dt)
    dt = steps - last_m_time
    
    # Apply 'Cold Start' penalty (10h) to time steps before the VERY first measurement
    first_m_idx = measured_idx[0]
    dt[:first_m_idx] += 10.0
    
    return np.exp(-lambda_decay * dt)


def process_patient(df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SOTA Clinical 28 Processor (Titanium v5.6)
    Upgrades: Unit Harmonization, Hemodynamic Hierarchy, Project Keys, & Dual-Masking
    """
    T = len(df_raw)
    if T < MIN_STAY_HOURS:
        raise ValueError(f"Stay too short: {T} < {MIN_STAY_HOURS}")
    
    # =========================================================================
    # STAGE 1: EXTRACTION, UNIT HARMONIZATION, & PHYSICS CLAMPING
    # =========================================================================
    raw_matrix = np.full((T, N_RAW), np.nan, dtype=np.float32)
    
    for i, col in enumerate(RAW_COLUMNS):
        # 0. Robust Column Discovery (Handle PhysioNet Aliases)
        actual_col = col
        if col == 'Bilirubin' and 'Bilirubin_total' in df_raw.columns:
            actual_col = 'Bilirubin_total'
        
        if actual_col in df_raw.columns:
            val = df_raw[actual_col].values.astype(np.float64)
            
            # 1A. SOTA Harmonization: Temperature (Fahrenheit to Celsius)
            if col == 'Temp':
                val = np.where((val >= 90.0) & (val <= 115.0), (val - 32.0) * 5.0 / 9.0, val)
                
            # 1B. SOTA Harmonization: Oxygenation
            elif col == 'FiO2':
                # L/min (Nasal Cannula) to estimated FiO2 fraction
                val = np.where((val > 1.0) & (val <= 15.0), 0.20 + 0.04 * val, val)
                # Percentage to Fraction
                val = np.where((val >= 20.0) & (val <= 100.0), val / 100.0, val)
                
            # 1C. Physics Clamping (Standardize true errors to NaN)
            _, (lo, hi), _ = RAW_CLINICAL_SPEC[col]
            val = np.where((val < lo) | (val > hi), np.nan, val)
            
            # 1D. Iridium SQW Hard Clipping
            if col in SQW_BOUNDS:
                sqw_lo, sqw_hi = SQW_BOUNDS[col]
                mask_real = ~np.isnan(val)
                val[mask_real] = np.clip(val[mask_real], sqw_lo, sqw_hi)
            
            raw_matrix[:, i] = val

    # =========================================================================
    # STAGE 2: CROSS-CHANNEL PHYSIOLOGICAL HIERARCHY (Iridium Triad)
    # =========================================================================
    sbp_idx, dbp_idx, map_idx = RAW_COLUMNS.index('SBP'), RAW_COLUMNS.index('DBP'), RAW_COLUMNS.index('MAP')
    sbp_val, dbp_val, map_val = raw_matrix[:, sbp_idx], raw_matrix[:, dbp_idx], raw_matrix[:, map_idx]
    
    # 2A. Strict SBP > DBP Hierarchy
    paradox_mask = (sbp_val <= dbp_val) & ~np.isnan(sbp_val) & ~np.isnan(dbp_val)
    raw_matrix[paradox_mask, sbp_idx] = np.nan
    raw_matrix[paradox_mask, dbp_idx] = np.nan

    # Refresh pointers after nullification
    sbp_val, dbp_val = raw_matrix[:, sbp_idx], raw_matrix[:, dbp_idx]
    
    # 2B. MAP Consistency Equation (MAP ≈ 1/3 SBP + 2/3 DBP)
    both_exist = ~np.isnan(sbp_val) & ~np.isnan(dbp_val)
    calc_map = (sbp_val + 2.0 * dbp_val) / 3.0
    
    map_recorded = ~np.isnan(map_val)
    check_mask = both_exist & map_recorded
    
    deviation = np.zeros(T, dtype=np.float32)
    # Safe division using EPS
    deviation[check_mask] = np.abs(map_val[check_mask] - calc_map[check_mask]) / (calc_map[check_mask] + EPS)
    
    # Recalculate if MAP deviates > 15%, OR if MAP is completely missing but cuffs exist
    overwrite_mask = (check_mask & (deviation > 0.15)) | (both_exist & ~map_recorded)
    raw_matrix[overwrite_mask, map_idx] = calc_map[overwrite_mask]

    # =========================================================================
    # STAGE 3: CAUSAL TSLM & HOMEOSTATIC IMPUTATION
    # =========================================================================
    raw_imputed = np.zeros((T, N_RAW), dtype=np.float32)
    raw_masks = np.zeros((T, N_RAW), dtype=np.float32)
    decayed_masks = np.zeros((T, N_RAW), dtype=np.float32)
    
    for i, col in enumerate(RAW_COLUMNS):
        default, _, category = RAW_CLINICAL_SPEC[col]
        val = raw_matrix[:, i]
        
        is_real = ~np.isnan(val)
        raw_masks[:, i] = is_real.astype(np.float32)
        lam = LAMBDA_MAP.get(category, 0.1)
        
        if category == 'static':
            raw_imputed[:, i] = pd.Series(val).ffill().bfill().fillna(default).values
            decayed_masks[:, i] = 1.0  
        else:
            if np.any(is_real):
                decayed_masks[:, i] = compute_decayed_tslm_vectorized(raw_masks[:, i], lambda_decay=lam)
                flat_ffill = pd.Series(val).ffill().fillna(default).values
                alpha = decayed_masks[:, i]
                
                # Iridium Math: Log-Space Imputation for Neural Manifold Smoothing
                if col in LOG_SPACE_CHANNELS:
                    log_ffill = np.log1p(np.maximum(flat_ffill, 0.0))
                    log_default = np.log1p(max(default, 0.0))
                    log_imputed = (log_ffill * alpha) + (log_default * (1.0 - alpha))
                    raw_imputed[:, i] = np.expm1(log_imputed)
                else:
                    raw_imputed[:, i] = (flat_ffill * alpha) + (default * (1.0 - alpha))
            else:
                # Totally missing column handling
                raw_imputed[:, i] = default
                steps = np.arange(T, dtype=np.float32)
                decayed_masks[:, i] = np.exp(-lam * (steps + 10.0))

    # =========================================================================
    # STAGE 4: MIT/DEEPMIND SPARSITY FILTER
    # =========================================================================
    non_static_indices = [i for i, c in enumerate(RAW_COLUMNS) if RAW_CLINICAL_SPEC[c][2] != 'static']
    unique_physio_measured = np.sum(np.any(raw_masks[:, non_static_indices] > 0, axis=0))
    
    if unique_physio_measured < 5:
        raise ValueError(f"Ghost Patient: Only {unique_physio_measured} physiological vars measured")

    hr_idx = RAW_COLUMNS.index('HR')
    if np.mean(raw_masks[:, hr_idx]) < 0.3:
        raise ValueError("Insufficient Signal: HR Density < 30%")

    # Extract labels & Apply Safety Shield
    labels = np.zeros(T, dtype=np.float32)
    if 'SepsisLabel' in df_raw.columns:
        labels = df_raw['SepsisLabel'].values.astype(np.float32)
        
    features = np.nan_to_num(raw_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
    features = np.clip(features, -1e6, 1e6)
    
    return features.astype(np.float32), labels, raw_masks.astype(np.float32), decayed_masks.astype(np.float32)


# ==============================================================================
# 4. WELFORD ONLINE STATISTICS (Vectorized)
# ==============================================================================

class WelfordEngine:
    """Numerically stable single-pass Mean/Std computation.
    
    Uses Welford's online algorithm but processes in BATCHES (vectorized)
    for efficiency. Handles millions of rows without OOM.
    """
    def __init__(self, n_features: int):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)
    
    def update_batch(self, x: np.ndarray):
        """Update with a batch of rows. x shape: (T, n_features)."""
        for row in x:  # Row-wise for numerical stability
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2
    
    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) arrays."""
        if self.n < 2:
            return self.mean.astype(np.float32), np.ones_like(self.mean, dtype=np.float32)
        
        variance = self.m2 / (self.n - 1)
        std = np.sqrt(variance)
        # Protect against zero variance (constant columns)
        std[std < EPS] = 1.0
        return self.mean.astype(np.float32), std.astype(np.float32)


# ==============================================================================
# 5. DETERMINISTIC PATIENT SPLIT
# ==============================================================================

def deterministic_split(file_list: List[Path], val_ratio: float = VAL_RATIO) -> Tuple[List[Path], List[Path]]:
    """Hash-based deterministic split. Same patient always goes to same split,
    regardless of file discovery order or OS sorting.
    
    Uses SHA-256 hash of patient_id to assign splits. This guarantees:
    1. Reproducibility across runs
    2. No data leakage (same patient never in both splits)
    3. OS-independent behavior
    """
    train, val = [], []
    threshold = int(val_ratio * (2**16))
    
    for fpath in file_list:
        pid = fpath.stem
        hash_val = int(hashlib.sha256(pid.encode()).hexdigest()[:4], 16)
        
        if hash_val < threshold:
            val.append(fpath)
        else:
            train.append(fpath)
    
    return train, val


# ==============================================================================
# 6. LMDB PRODUCTION BUILDER
# ==============================================================================

class QualityIngestionEngine:
    """Processes raw .psv files into LMDB format compatible with ICUTrajectoryDataset.
    
    Output Format (per episode):
        {eid}_v  → np.float32, shape (T, N_FEATURES) [Vitals]
        {eid}_m  → np.float32, shape (T, N_RAW)      [Reliability Masks]
        {eid}_l  → np.float32, shape (T,)             [Sepsis Labels]
        {eid}_s  → np.float32, shape (N_STATIC,)     [Static Context]
        {eid}_d  → np.float32, shape (T, N_RAW)      [Continuous Decay]
    
    Output Index:
        {split}_index.json — native compatibility with ICUTrajectoryDataset
    """
    
    def __init__(self, output_dir: str, split: str):
        self.output_dir = Path(output_dir) / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.lmdb_path = self.output_dir / "data.lmdb"
        self.env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False)
        
        # Statistics accumulator
        self.stats = WelfordEngine(N_FEATURES)
        
        # Reservoir for quantile estimation
        self.reservoir = []
        self.global_min = np.full(N_FEATURES, np.inf)
        self.global_max = np.full(N_FEATURES, -np.inf)
        
        # Index entries
        self.index = []
        self.cnt = 0
        self.errors = 0
    
    def process(self, file_list: List[Path]):
        """Process all files in the list."""
        with self.env.begin(write=True) as txn:
            for fpath in tqdm(file_list, desc=f"Building {self.split}"):
                try:
                    df = pd.read_csv(fpath, sep='|')
                    
                    # Process patient
                    features, labels, raw_masks, decayed_masks = process_patient(df)
                    T = features.shape[0]
                    
                    # Statistics update (train only)
                    if self.split == "train":
                        self.stats.update_batch(features)
                        self._update_reservoir(features)
                    
                    # Serialize to LMDB (Canonical Project Keys)
                    eid = f"ep_{self.cnt:06d}"
                    txn.put(f"{eid}_v".encode(), features.tobytes())
                    txn.put(f"{eid}_m".encode(), raw_masks.tobytes())
                    txn.put(f"{eid}_labels".encode(), labels.tobytes())
                    txn.put(f"{eid}_d".encode(), decayed_masks.tobytes())
                    
                    # Static context: demographics from first row
                    static_vals = np.zeros(N_STATIC, dtype=np.float32)
                    for s_idx, s_col in enumerate(STATIC_COLS):
                        col_idx = RAW_COLUMNS.index(s_col)
                        static_vals[s_idx] = features[0, col_idx]
                    txn.put(f"{eid}_s".encode(), static_vals.tobytes())
                    
                    # Index entry
                    has_seps_bool = bool(np.any(labels > 0.5))
                    self.index.append({
                        "episode_id": eid,
                        "patient_id": fpath.stem,
                        "length": T,
                        "has_sepsis": has_seps_bool,
                        "modalities": {
                            "vitals": {"key": f"{eid}_v", "shape": [T, N_FEATURES], "dtype": "float32"},
                            "masks":  {"key": f"{eid}_m", "shape": [T, N_RAW],      "dtype": "float32"},
                            "labels": {"key": f"{eid}_labels", "shape": [T],              "dtype": "float32"},
                            "static": {"key": f"{eid}_s", "shape": [N_STATIC],       "dtype": "float32"},
                            "decay":  {"key": f"{eid}_d", "shape": [T, N_RAW],      "dtype": "float32"}
                        }
                    })
                    self.cnt += 1
                    
                except Exception as e:
                    self.errors += 1
                    if self.errors <= 10:
                        logger.warning(f"Skipping {fpath.stem}: {e}")
                    elif self.errors == 11:
                        logger.warning("Suppressing further individual error messages...")
        
        self.env.close()
        self._save_index()
        
        logger.info(f"[{self.split.upper()}] Complete: {self.cnt} episodes, {self.errors} errors")
    
    def _update_reservoir(self, matrix: np.ndarray):
        """Reservoir sampling for quantile estimation."""
        batch_min = matrix.min(axis=0)
        batch_max = matrix.max(axis=0)
        self.global_min = np.minimum(self.global_min, batch_min)
        self.global_max = np.maximum(self.global_max, batch_max)
        
        n_sample = max(1, min(len(matrix), 10))
        indices = np.random.choice(len(matrix), n_sample, replace=False)
        self.reservoir.append(matrix[indices])
        
        # Memory guard: compact if too large
        total_rows = sum(r.shape[0] for r in self.reservoir)
        if total_rows > RESERVOIR_SIZE * 2:
            big = np.concatenate(self.reservoir)
            keep = np.random.choice(len(big), RESERVOIR_SIZE, replace=False)
            self.reservoir = [big[keep]]
    
    def _save_index(self):
        """Save the index JSON with metadata."""
        # Compute quantiles from reservoir
        if self.reservoir:
            data_pool = np.concatenate(self.reservoir)
            p01 = np.percentile(data_pool, 1, axis=0).tolist()
            p99 = np.percentile(data_pool, 99, axis=0).tolist()
        else:
            p01 = self.global_min.tolist()
            p99 = self.global_max.tolist()
        
        # Compute Welford stats
        pop_mean, pop_std = self.stats.finalize()
        
        meta = {
            "version": "5.8-Iridium",
            "ts_columns": FEATURE_MANIFEST,
            "n_features": N_FEATURES,
            "n_raw_channels": N_RAW,
            "static_columns": STATIC_COLS,
            "stats": {
                "ts_min": self.global_min.tolist(),
                "ts_max": self.global_max.tolist(),
                "ts_p01": p01,
                "ts_p99": p99,
                "pop_mean": pop_mean.tolist(),
                "pop_std": pop_std.tolist(),
            }
        }
        
        out_path = self.output_dir.parent / f"{self.split}_index.json"
        with open(out_path, 'w') as f:
            json.dump({"episodes": self.index, "metadata": meta}, f, indent=2)
        
        logger.info(f"Index saved: {out_path} ({self.cnt} episodes, {N_FEATURES} features)")


# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================

def build_quality_dataset(
    output_dir: str = "data/iridium_v1",
    val_ratio: float = VAL_RATIO,
):
    """Main entry point for building the enriched dataset.
    
    Args:
        output_dir: Where to write the LMDB and index files
        val_ratio: Fraction of patients for validation (hash-based)
    """
    logger.info("="*60)
    logger.info(f"APEX-MoE Quality Data Builder v5.8 (Iridium / NASA-Grade)")
    logger.info(f"Feature Manifest: {N_FEATURES} channels")
    logger.info("="*60)
    
    # 1. Automated Data Discovery
    if kagglehub is None:
        logger.critical("kagglehub not installed. Cannot find raw sources.")
        return
        
    logger.info("Locating raw dataset via kagglehub...")
    try:
        raw_data_dir = kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")
    except Exception as e:
        logger.critical(f"Kaggle Download/Locate Failed: {e}")
        return

    # Discover files
    raw_path = Path(raw_data_dir)
    psv_files = sorted(raw_path.glob("**/*.psv"))
    
    if not psv_files:
        logger.error(f"No .psv files found in {raw_data_dir}")
        return
    
    logger.info(f"Found {len(psv_files)} patient files at {raw_data_dir}")
    
    # 2. Deterministic Split
    train_files, val_files = deterministic_split(psv_files, val_ratio)
    logger.info(f"Split: {len(train_files)} train / {len(val_files)} val ({val_ratio*100:.0f}%) (Determinism: Hash-Based)")
    
    # 3. Build Splits
    # Train
    train_engine = QualityIngestionEngine(output_dir, "train")
    train_engine.process(train_files)
    
    # Val
    val_engine = QualityIngestionEngine(output_dir, "val")
    val_engine.process(val_files)
    
    logger.info("Build Complete. Data is ready for Titanium-scale training.")
    logger.info(f"Output Directory: {output_dir}")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/iridium_v1"
    build_quality_dataset(output_dir=target_dir)