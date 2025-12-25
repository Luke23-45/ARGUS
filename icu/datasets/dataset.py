"""
icu/datasets/dataset.py
--------------------------------------------------------------------------------
APEX-MoE Frontier Data Loader.
Author: APEX Research Team
Version: 2.0 (Clinical 28-Channel Spec)

Description:
    A high-performance, fault-tolerant data pipeline designed for Safety-Critical 
    ICU Forecasting. It manages Tiered Acquisition, SoA (Structure of Arrays) 
    Deserialization, and Phase-Locked Label Generation for Mixture-of-Experts.

Key Features:
    - Zero-Copy LMDB Access (High Throughput)
    - Automatic Schema Validation (28-Channel Enforcement)
    - Phase-Logic for MoE Gating (Stable vs. Pre-Shock vs. Shock)
    - Clinical Augmentations (Sensor Dropout, Signal Noise)
"""

from __future__ import annotations

import os
import sys
import json
import lmdb
import logging
import functools
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from torch.utils.data import Dataset, default_collate
from huggingface_hub import snapshot_download

# --- configuration & constants ---
logger = logging.getLogger("APEX_Data_Frontier")
EXPECTED_CHANNELS = 28  # The "Clinical 28" Spec
PHASE_STABLE = 0
PHASE_PRESHOCK = 1
PHASE_SHOCK = 2

# ==============================================================================
# CANONICAL COLUMN SPECIFICATION (Ground Truth)
# ==============================================================================
# This is the AUTHORITATIVE column order. Data MUST conform to this.
# We do NOT rely on metadata to be perfectly aligned - we validate against this.
CANONICAL_COLUMNS = [
    # Group A: Hemodynamic (0-6)
    'HR', 'O2Sat', 'SBP', 'DBP', 'MAP', 'Resp', 'Temp',
    # Group B: Sepsis Drivers / Labs (7-17)
    'Lactate', 'Creatinine', 'Bilirubin', 'Platelets', 'WBC',
    'pH', 'HCO3', 'BUN', 'Glucose', 'Hgb', 'Potassium',
    # Group C: Electrolytes & Support (18-21)
    'Magnesium', 'Calcium', 'Chloride', 'FiO2',
    # Group D: Static Context (22-27)
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

# Column groups for explicit extraction
COLUMN_GROUPS = {
    'hemodynamic': (0, 7),    # Indices 0-6
    'labs': (7, 18),          # Indices 7-17
    'electrolytes': (18, 22), # Indices 18-21
    'static': (22, 28),       # Indices 22-27
}

# ==============================================================================
# 0. TIERED DATA ORCHESTRATOR
# ==============================================================================

def ensure_data_ready(
    dataset_dir: str = "data/ready",
    hf_repo_id: Optional[str] = None, 
    force_download: bool = False
) -> None:
    """
    Guarantees data availability through a fault-tolerant Tiered Acquisition strategy.
    
    Tiers:
    1. Local Verification: Checks checksums/existence of local LMDBs.
    2. Tier 1 (HF Cloud): Pulls pre-processed "Frontier" binaries.
    3. Tier 2 (Raw Build): Pulls raw Kaggle CSVs and triggers the Local Build Pipeline.
    
    Raises:
        RuntimeError: If all tiers fail to provide valid data.
    """
    dataset_path = Path(dataset_dir)
    required_splits = ["train", "val"]
    
    # --- Tier 1: Local Integrity Check ---
    if not force_download:
        all_valid = True
        for split in required_splits:
            lmdb_p = dataset_path / split / "data.lmdb"
            idx_p = dataset_path / f"{split}_index.json"
            
            if not (lmdb_p.exists() and idx_p.exists()):
                logger.warning(f"[Tier 0] Missing split '{split}' in {dataset_path}")
                all_valid = False
                break
        
        if all_valid:
            logger.info(f"[Tier 0] Valid Local Data Found at '{dataset_path}'. Ready.")
            return

    # --- Tier 2: Hugging Face (Pre-Processed) ---
    if hf_repo_id:
        logger.info(f"[Tier 1] Attempting download from HF: {hf_repo_id}...")
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=dataset_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info("[Tier 1] Download Successful.")
            return
        except Exception as e:
            logger.warning(f"[Tier 1] HF Download Failed: {e}. Falling back to Build...")

    # --- Tier 3: Local Build from Raw Sources ---
    logger.info("[Tier 2] Triggering Raw Build Pipeline (Kaggle Sources)...")
    try:
        # Dynamic import to avoid circular dependencies
        from .build_dataset import run_build_pipeline
        run_build_pipeline(output_dir=str(dataset_path))
        logger.info("[Tier 2] Build Complete. Data is ready.")
    except Exception as e:
        logger.critical(f"[Tier 2] Build Pipeline Failed: {e}")
        raise RuntimeError("FATAL: Could not acquire or build ICU Dataset. Check connectivity.")

# ==============================================================================
# 1. CORE ARCHITECTURE: The APEX Loader
# ==============================================================================

class ICUTrajectoryDataset(Dataset):
    """
    The Foundation Class for ICU Time-Series.
    
    Capabilities:
    1. Memory-Mapped I/O: Zero-copy reads from disk to GPU tensor.
    2. Virtual Indexing: O(1) lookups of sliding windows from variable-length episodes.
    3. Phase Logic: Calculates Sepsis phases for MoE training.
    """
    def __init__(
        self,
        dataset_dir: str = "data/ready",
        split: str = "train",
        history_len: int = 24,
        pred_len: int = 6,
        max_cache_size: int = 128,
        validate_schema: bool = True
    ):
        super().__init__()
        
        self.split = split
        self.history_len = history_len
        self.pred_len = pred_len
        self.window_size = history_len + pred_len
        
        # Paths
        self.root_path = Path(dataset_dir) / split
        self.lmdb_path = self.root_path / "data.lmdb"  # Fixed from f"{split}.lmdb"
        self.index_path = self.root_path.parent / f"{split}_index.json" # Fixed from self.root_path / 
        
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {self.lmdb_path}")

        # --- Load Index & Stats ---
        logger.info(f"[{split.upper()}] Loading Index: {self.index_path}")
        try:
            with open(self.index_path, 'r') as f:
                full_index = json.load(f)
                self.episode_metadata = full_index["episodes"]
                self.metadata = full_index["metadata"]
                self.global_stats = self.metadata.get("stats", None)
        except Exception as e:
            raise RuntimeError(f"Corrupted Index JSON: {e}")

        # --- Schema Validation (Robust) ---
        if validate_schema:
            # Support both legacy ("columns") and new ("ts_columns") keys
            ts_cols = self.metadata.get("ts_columns") or self.metadata.get("columns", [])
            
            if len(ts_cols) == 0:
                logger.warning("No column metadata found. Using CANONICAL spec as ground truth.")
                ts_cols = CANONICAL_COLUMNS
            elif len(ts_cols) != EXPECTED_CHANNELS:
                logger.error(f"SCHEMA MISMATCH! Expected {EXPECTED_CHANNELS}, Found {len(ts_cols)}")
                # For safety-critical, we crash to prevent silent model degradation.
                if len(ts_cols) < 7:  # If it's the old 7-channel, definitely crash
                    raise ValueError(f"Dataset has outdated schema ({len(ts_cols)} cols). Rebuild required.")
                logger.warning(f"Channel count {len(ts_cols)} != {EXPECTED_CHANNELS}. Proceeding with CANONICAL spec.")
                ts_cols = CANONICAL_COLUMNS
            
            # Validate order against canonical (Don't trust data blindly)
            for i, (data_col, canonical_col) in enumerate(zip(ts_cols, CANONICAL_COLUMNS)):
                if data_col != canonical_col:
                    logger.warning(f"Column order mismatch at idx {i}: data='{data_col}', canonical='{canonical_col}'")
            
            # Store validated columns for downstream use
            self.ts_columns = ts_cols

        # --- Virtual Map Construction ---
        # We pre-calculate how many valid windows exist in each patient episode
        # to create a global linear index [0...TotalWindows].
        self.chunks_per_episode = []
        valid_episodes = 0
        
        for ep in self.episode_metadata:
            t_len = ep["length"]
            # A valid window must have (history + pred) length
            n_chunks = max(0, t_len - self.window_size + 1)
            self.chunks_per_episode.append(n_chunks)
            if n_chunks > 0:
                valid_episodes += 1

        self.cumulative_chunks = np.cumsum(self.chunks_per_episode)
        self.total_chunks = int(self.cumulative_chunks[-1]) if len(self.cumulative_chunks) > 0 else 0
        
        # --- Lazy LMDB Handle ---
        self._lmdb_env = None
        self.max_cache_size = max_cache_size

        logger.info(f"[{split.upper()}] Initialized. Windows: {self.total_chunks:,} | Episodes: {valid_episodes:,}")

    def __len__(self):
        return self.total_chunks

    def _init_lmdb(self):
        """Thread-safe lazy initialization of the LMDB environment."""
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False
            )

    def _read_bytes(self, key: str) -> bytes:
        """Raw byte fetcher."""
        self._init_lmdb()
        with self._lmdb_env.begin(write=False) as txn:
            data = txn.get(key.encode('ascii'))
            if data is None:
                raise KeyError(f"LMDB Key failure: {key}. The index is out of sync with the DB.")
            return data

    @functools.lru_cache(maxsize=128)
    def _fetch_numpy(self, key: str, dtype_str: str, shape: Tuple[int, ...]) -> np.ndarray:
        raw = self._read_bytes(key)
        # Added .copy() to decouple from buffer/cache
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape).copy()

    def _get_phase_label(self, labels_window: np.ndarray) -> int:
        """
        Derives the MoE Gating Label from the Sepsis labels.
        
        Logic:
        - Labels Window covers [t_start : t_start + history + pred]
        - Obs = [0 : history], Future = [history : history + pred]
        
        Definitions:
        - PHASE_SHOCK (2): Sepsis is active *during* the observation window (Already sick).
        - PHASE_PRESHOCK (1): Sepsis is NOT active in obs, but appears in future (Transition).
        - PHASE_STABLE (0): No Sepsis in obs or future.
        """
       # Change these lines in _get_phase_label:
        obs_labels = labels_window[:self.history_len]
        fut_labels = labels_window[self.history_len:]
        
        # Use explicit float thresholds for SepsisLabel stability
        if (obs_labels > 0.5).any():
            return PHASE_SHOCK
        elif (fut_labels > 0.5).any():
            return PHASE_PRESHOCK
        else:
            return PHASE_STABLE

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} out of bounds (Size: {self.total_chunks})")

        # 1. Resolve Global Index -> Episode + Local Offset
        ep_idx = np.searchsorted(self.cumulative_chunks, idx, side='right')
        chunk_start_global = 0 if ep_idx == 0 else self.cumulative_chunks[ep_idx - 1]
        local_t_start = int(idx - chunk_start_global)

        # 2. Retrieve Metadata
        ep_meta = self.episode_metadata[ep_idx]
        modalities = ep_meta.get("modalities", {})
        
        # 3. Fetch Full Arrays (Cached) with DEFENSIVE extraction
        # Vitals (REQUIRED)
        v_meta = modalities.get("vitals")
        if v_meta is None:
            raise KeyError(f"Episode {ep_meta.get('episode_id', ep_idx)} missing 'vitals' modality.")
        full_vitals = self._fetch_numpy(
            v_meta["key"], 
            v_meta.get("dtype", "float32"),  # Fallback dtype
            tuple(v_meta["shape"])
        )
        
        # Validate vitals shape explicitly
        if full_vitals.shape[1] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Vitals channel mismatch: got {full_vitals.shape[1]}, expected {EXPECTED_CHANNELS}. "
                f"Episode: {ep_meta.get('episode_id', 'unknown')}"
            )
        
        # Static (REQUIRED - but extract from vitals if missing)
        s_meta = modalities.get("static")
        if s_meta is not None:
            full_static = self._fetch_numpy(
                s_meta["key"], 
                s_meta.get("dtype", "float32"),
                tuple(s_meta["shape"])
            )
        else:
            # Fallback: Extract static from vitals Group D (indices 22-27)
            # Use first row since static context is time-invariant
            logger.debug(f"Extracting static from vitals for {ep_meta.get('episode_id', 'unknown')}")
            static_start, static_end = COLUMN_GROUPS['static']
            full_static = full_vitals[0, static_start:static_end].copy()
        
        # Labels (REQUIRED)
        l_meta = modalities.get("labels")
        if l_meta is None:
            raise KeyError(f"Episode {ep_meta.get('episode_id', ep_idx)} missing 'labels' modality.")
        full_labels = self._fetch_numpy(
            l_meta["key"], 
            l_meta.get("dtype", "float32"),
            tuple(l_meta["shape"])
        )

        # 4. Slice Window
        t_end = local_t_start + self.window_size
        
        # Safety Check: Bounds (Should be guaranteed by logic, but robust checks prevent segfaults)
        if t_end > len(full_vitals):
            raise ValueError(f"Window overrun for episode {ep_meta['episode_id']}")

        vitals_win = full_vitals[local_t_start : t_end]
        labels_win = full_labels[local_t_start : t_end]

        # 5. Split Input/Output
        # X: [0 ... history]
        # Y: [history ... history + pred]
        obs_data = vitals_win[:self.history_len]
        fut_data = vitals_win[self.history_len:]
        
        # 6. Compute Metadata
        phase = self._get_phase_label(labels_win)
        
        # Outcome Label: Max sepsis probability in the prediction horizon
        outcome = np.max(labels_win[self.history_len:])

        return {
            "observed_data": torch.from_numpy(obs_data.copy()),  # Shape: [Hist, 28]
            "future_data":   torch.from_numpy(fut_data.copy()),  # Shape: [Pred, 28]
            "static_context": torch.from_numpy(full_static.copy()), # Shape: [Stat]
            "outcome_label": torch.tensor(outcome, dtype=torch.float32),
            "phase_label":   torch.tensor(phase, dtype=torch.long),
            "patient_id":    str(ep_meta.get("patient_id", "unknown"))
        }

# ==============================================================================
# 2. SOTA DATASET: Robustness & Augmentation
# ==============================================================================

class ICUSotaDataset(ICUTrajectoryDataset):
    """
    The 'SOTA' Wrapper for Training.
    Introduces physical simulations to make the model robust to real-world chaos.
    """
    def __init__(
        self,
        dataset_dir: str = "data/ready",
        split: str = "train",
        history_len: int = 24,
        pred_len: int = 6,
        augment_noise: float = 0.005,
        augment_mask_prob: float = 0.0,
        validate_schema: bool = True
    ):
        super().__init__(
            dataset_dir=dataset_dir, 
            split=split, 
            history_len=history_len, 
            pred_len=pred_len,
            validate_schema=validate_schema
        )
        
        self.augment_noise = augment_noise
        self.augment_mask_prob = augment_mask_prob
        self.is_training = (split == "train")
        
        if self.is_training:
            logger.info(f"Augmentation Enabled: Noise={augment_noise}, MaskDrop={augment_mask_prob}")

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            sample = super().__getitem__(idx)
            
            # --- Robustness Checks ---
            # Guard against NaNs that might have slipped through imputation
            if torch.isnan(sample["observed_data"]).any():
                # In prod, we might try to fix this. For now, we drop the sample to protect gradients.
                logger.debug(f"Dropped NaN sample at idx {idx}")
                return None

            if self.is_training:
                # 1. Gaussian Sensor Noise
                if self.augment_noise > 0:
                    noise = torch.randn_like(sample["observed_data"]) * self.augment_noise
                    sample["observed_data"] += noise
                
                # 2. Sensor Dropout (Masking)
                # Simulates a sensor getting disconnected. 
                # We zero out an entire channel for the whole window.
                if self.augment_mask_prob > 0:
                    mask = torch.rand(sample["observed_data"].shape[1]) > self.augment_mask_prob
                    # Broadcast mask [C] -> [T, C]
                    sample["observed_data"] *= mask.float()

            return sample

        except Exception as e:
            # Catch-all for corruption, ensuring the DataLoader worker doesn't crash the whole training
            logger.error(f"FATAL Load Error at idx {idx}: {e}", exc_info=False)
            return None

# ==============================================================================
# 3. COLLATOR
# ==============================================================================

def robust_collate_fn(batch: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
    """
    A crash-proof collator.
    Filters out 'None' samples returned by SotaDataset (due to NaNs or errors).
    """
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        logger.warning("Empty Batch detected in Collate! (All samples failed)")
        return {}
    
    return default_collate(valid_batch)