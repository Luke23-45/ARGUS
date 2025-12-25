"""
icu/models/normalizer.py
--------------------------------------------------------------------------------
APEX-MoE: Physics-Aware Clinical Normalization Engine (SOTA Edition).
Version: 3.1 (Robust Winsorization & Quantile Hardening)

Author: APEX Research Team
Context: Life-Critical Sepsis Prediction

Description:
    This is the "Air Traffic Control" for data entering the neural network.
    It transforms raw physiological signals into a high-fidelity latent representation.
    
    CRITICAL IMPROVEMENTS (v3.1):
    1. Winsorization: Implements Robust Quantile Normalization to fix "Signal Squashing" 
       in heavy-tailed features (e.g., Bilirubin, Lactate).
    2. Physics Gating: Hard-clamps inputs to biological possibility BEFORE statistical processing.
    3. Floating-Point Safety: Enhanced epsilon protection against flat-signal NaN propagation.
    4. State Integrity: Strict shape validation to prevent silent tensor broadcasting errors.
"""

import torch
import torch.nn as nn
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

# Logger Configuration
logger = logging.getLogger("APEX_Normalizer")
logger.setLevel(logging.INFO)

# ==============================================================================
# 1. BIOLOGICAL PHYSICS CONSTANTS (GROUND TRUTH)
# ==============================================================================
# Unified Physics Bounds for all 28 Channels.
# These represent the absolute "Humanly Possible" limits. 
# Values outside are strictly sensor errors.
PHYSICS_BOUNDS_TS = {
    # --- Group A: Hemodynamics (0-6) ---
    'HR': (20.0, 300.0), 'O2Sat': (20.0, 100.0), 'SBP': (20.0, 300.0), 
    'DBP': (10.0, 200.0), 'MAP': (20.0, 250.0), 'Resp': (4.0, 80.0), 'Temp': (24.0, 45.0),
    
    # --- Group B: Labs & Sepsis Drivers (7-17) ---
    # NOTE: Labs often have heavy tails. Physics bounds are wide to catch errors,
    # but statistical bounds (calculated later) will handle the resolution.
    'Lactate': (0.1, 30.0), 'Creatinine': (0.1, 25.0), 'Bilirubin': (0.1, 80.0), 
    'Platelets': (1.0, 2000.0), 'WBC': (0.1, 200.0), 'pH': (6.5, 7.8), 
    'HCO3': (5.0, 60.0), 'BUN': (1.0, 250.0), 'Glucose': (10.0, 1200.0), 
    'Hgb': (2.0, 25.0), 'Potassium': (1.0, 12.0),
    
    # --- Group C: Electrolytes & Support (18-21) ---
    'Magnesium': (0.5, 10.0), 'Calcium': (2.0, 20.0), 'Chloride': (50.0, 150.0), 'FiO2': (0.21, 1.0),

    # --- Group D: Context (22-27) ---
    'Age': (10.0, 100.0), 
    'Gender': (0.0, 1.0), 
    'Unit1': (0.0, 1.0), 
    'Unit2': (0.0, 1.0), 
    'HospAdmTime': (-10000.0, 0.0), # Expanded for long-term history
    'ICULOS': (0.0, 2000.0)
}

# ==============================================================================
# 2. THE ROBUST NORMALIZATION ENGINE
# ==============================================================================

class ClinicalNormalizer(nn.Module):
    """
    State-Safe Normalizer with Robust Quantile Clipping (Winsorization).
    
    Mathematical Formulation:
        1. x_phy = clamp(x, physics_min, physics_max)
        2. x_win = clamp(x_phy, stat_p01, stat_p99)  <-- Winsorization Step
        3. x_norm = 2 * (x_win - stat_p01) / (stat_p99 - stat_p01) - 1
        4. x_out = clamp(x_norm, -1, 1)
        
    This ensures that 98% of the data utilizes 100% of the [-1, 1] latent space,
    solving the 'Vanishing Gradient' problem in heavy-tailed distributions.
    """
    
    def __init__(
        self, 
        ts_channels: int = 28, 
        static_channels: int = 6,
        safety_margin: float = 0.05  # Reduced margin (5%) because we use Quantiles now
    ):
        """
        Args:
            ts_channels: Number of Time-Series features.
            static_channels: Number of Static features.
            safety_margin: Padding added to the computed statistical bounds.
        """
        super().__init__()
        self.ts_channels = ts_channels
        self.static_channels = static_channels
        self.safety_margin = safety_margin
        
        # --- Register State Buffers ---
        # "buffers" are persistent state, saved in checkpoints, but not updated by optimizers.
        
        # 1. Statistical Bounds (Effective Range for Winsorization)
        # These will hold P01 (min) and P99 (max) after calibration.
        self.register_buffer('ts_stat_min', torch.zeros(ts_channels))
        self.register_buffer('ts_stat_max', torch.ones(ts_channels))
        
        # 2. Physics Bounds (Hard Biological Limits)
        self.register_buffer('ts_physics_min', torch.zeros(ts_channels))
        self.register_buffer('ts_physics_max', torch.ones(ts_channels))
        
        # 3. Static Stats (Context)
        self.register_buffer('static_min', torch.zeros(static_channels))
        self.register_buffer('static_max', torch.ones(static_channels))
        
        # 4. Status Flags
        self.register_buffer('is_calibrated', torch.tensor(0, dtype=torch.bool))

    def calibrate_from_stats(self, stats_path: Union[str, Path], channel_names_ts: List[str]):
        """
        Hydrates the normalizer using external statistics.
        
        CRITICAL UPGRADE: 
        Prioritizes loading 'quantile_01' and 'quantile_99' from the stats file.
        Falls back to 'min'/'max' only if quantiles are missing, but warns explicitly.
        """
        path = Path(stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file not found: {path}")
            
        logger.info(f"Calibrating Normalizer from {path}...")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Handle nested metadata structure common in APEX datasets
            stats = data.get("metadata", {}).get("stats", {})
            if not stats:
                # Fallback for legacy format
                stats = data.get("stats", {})
                
            if not stats:
                raise ValueError(f"JSON at {path} contains no 'stats' or 'metadata.stats' key.")

            # --- 1. Load Physics Limits (Ground Truth) ---
            p_min_list = []
            p_max_list = []
            
            if len(channel_names_ts) != self.ts_channels:
                 raise ValueError(f"Channel Mismatch! Config expects {self.ts_channels}, provided names count {len(channel_names_ts)}")

            for name in channel_names_ts:
                # Default to wide open if unknown, but log warning
                if name not in PHYSICS_BOUNDS_TS:
                    logger.warning(f"Channel '{name}' has no defined Physics Bounds. Using defaults.")
                bounds = PHYSICS_BOUNDS_TS.get(name, (-10000.0, 10000.0))
                p_min_list.append(bounds[0])
                p_max_list.append(bounds[1])
                
            self.ts_physics_min.copy_(torch.tensor(p_min_list, dtype=torch.float32))
            self.ts_physics_max.copy_(torch.tensor(p_max_list, dtype=torch.float32))

            # --- 2. Load Statistical Bounds (Winsorization Targets) ---
            # Try to load robust quantiles first
            
            # Check for keys in the JSON. Expecting list of floats.
            keys_p01 = ["ts_p01", "p01", "quantile_01", "quantile_1"]
            keys_p99 = ["ts_p99", "p99", "quantile_99", "quantile_99"]
            
            raw_min_data = None
            raw_max_data = None
            mode = "minmax" # default
            
            # Search for P01
            for k in keys_p01:
                if k in stats:
                    raw_min_data = stats[k]
                    mode = "robust_quantile"
                    break
            
            # Search for P99
            for k in keys_p99:
                if k in stats:
                    raw_max_data = stats[k]
                    break
            
            # Fallback to Min/Max if quantiles missing
            if raw_min_data is None or raw_max_data is None:
                logger.warning("Robust Quantiles (P01/P99) NOT found in stats. Falling back to absolute Min/Max. (Susceptible to outliers!)")
                raw_min_data = stats.get("ts_min") or stats.get("empirical_min")
                raw_max_data = stats.get("ts_max") or stats.get("empirical_max")
                mode = "fallback_minmax"
            
            if raw_min_data is None or raw_max_data is None:
                raise ValueError("JSON missing critical statistics. Need 'ts_p01'/'ts_p99' or 'ts_min'/'ts_max'.")
            
            t_min = torch.tensor(raw_min_data, dtype=torch.float32)
            t_max = torch.tensor(raw_max_data, dtype=torch.float32)
            
            if len(t_min) != self.ts_channels:
                raise ValueError(f"Stat Dimension Mismatch: Config={self.ts_channels}, File={len(t_min)}")
            
            # --- 3. Compute Final Normalization Bounds with Margins ---
            # Range = Max - Min
            # Lower = Min - (Range * Margin)
            # Upper = Max + (Range * Margin)
            
            data_range = (t_max - t_min).clamp(min=1e-6) # Prevent zero-range
            margin_val = data_range * self.safety_margin
            
            final_min = t_min - margin_val
            final_max = t_max + margin_val
            
            self.ts_stat_min.copy_(final_min)
            self.ts_stat_max.copy_(final_max)
            
            logger.info(f"Time-Series Statistics Loaded. Mode: {mode.upper()}. Range expanded by {self.safety_margin*100}%.")

            # --- 4. Static Setup (Context) ---
            # If static channels exist, we assume they are the LAST N channels of the TS spec 
            # (as per APEX-MoE dataset structure: Age, Gender, etc.)
            if self.static_channels > 0:
                if self.ts_channels >= self.static_channels:
                    # Copy the stats from the last N channels
                    self.static_min.copy_(self.ts_stat_min[-self.static_channels:])
                    self.static_max.copy_(self.ts_stat_max[-self.static_channels:])
                else:
                    logger.warning("TS channels < Static channels. Defaulting Static stats to 0-1 (Dangerous).")
                    self.static_min.fill_(0.0)
                    self.static_max.fill_(1.0)
            
            self.is_calibrated.fill_(1)
            logger.info("Normalizer Calibration Complete. System Ready.")
            
        except Exception as e:
            logger.critical(f"Calibration Failed: {e}")
            raise

    def _safe_normalize(self, x: torch.Tensor, min_b: torch.Tensor, max_b: torch.Tensor) -> torch.Tensor:
        """
        Internal: Maps [min, max] -> [-1, 1] safely.
        """
        # 1. Map to [0, 1]
        # x_01 = (x - min) / (max - min)
        
        # Robust denominator with larger epsilon to prevent explosion on flat signals
        denominator = (max_b - min_b)
        # If denominator is too small, treating it as 1.0 prevents NaN, though signal is flat.
        denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator), denominator)
        
        x_01 = (x - min_b) / denominator
        
        # 2. Map to [-1, 1]
        x_norm = x_01 * 2.0 - 1.0
        
        return x_norm

    def normalize(self, x_ts: torch.Tensor, x_static: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward Pass: Raw -> Normalized
        
        Steps:
        1. Physics Clamp (Hard Limit)
        2. Winsorization (Soft Clamp to Statistical Bounds)
        3. Linear Scaling to [-1, 1]
        4. Latent Clamp (Strict [-1, 1])
        """
        if not self.is_calibrated:
            logger.warning("Normalizer used without calibration! Returning raw input.")
            return x_ts, x_static

        # --- 1. Dynamic Reshaping for Broadcasting ---
        # Supports (B, T, C) and (B, C) inputs
        rank = len(x_ts.shape)
        # Create view shape: (1, 1, C) or (1, C)
        view_shape = [1] * (rank - 1) + [-1]
        
        # Move buffers to correct device/shape
        p_min = self.ts_physics_min.to(x_ts.device).view(view_shape)
        p_max = self.ts_physics_max.to(x_ts.device).view(view_shape)
        
        s_min = self.ts_stat_min.to(x_ts.device).view(view_shape)
        s_max = self.ts_stat_max.to(x_ts.device).view(view_shape)

        # --- 2. Physics Clamp (Reject Sensor Noise) ---
        # Values like 9999 or -1 are hard-clamped immediately.
        x_phy = torch.clamp(x_ts, p_min, p_max)
        
        # --- 3. Winsorization (The "SOTA" Fix) ---
        # We clamp the PHYSICS-clean data to the STATISTICAL bounds.
        # This handles the "Bilirubin Problem" where P99=3.3 but Max=50.
        # Everything > s_max is capped at s_max.
        x_win = torch.clamp(x_phy, s_min, s_max)
        
        # --- 4. Scale to [-1, 1] ---
        x_ts_norm = self._safe_normalize(x_win, s_min, s_max)
        
        # --- 5. Strict Latent Clamp ---
        # Floating point errors might push values to 1.000001. 
        # Diffusion models explode if inputs are outside [-1, 1].
        x_ts_norm = torch.clamp(x_ts_norm, -1.0, 1.0)
        
        # --- 6. Static Handling ---
        x_static_norm = None
        if x_static is not None:
            # Static is usually (B, C_stat)
            st_min = self.static_min.to(x_static.device).view(1, -1)
            st_max = self.static_max.to(x_static.device).view(1, -1)
            
            # Clamp static context similarly
            x_st_clamped = torch.clamp(x_static, st_min, st_max)
            x_static_norm = self._safe_normalize(x_st_clamped, st_min, st_max)
            x_static_norm = torch.clamp(x_static_norm, -1.0, 1.0)

        return x_ts_norm, x_static_norm

    def denormalize(self, x_ts_norm: torch.Tensor) -> torch.Tensor:
        """
        Inverts normalization. Maps [-1, 1] -> [Clinical Units].
        Used for Interpretability and Metric Calculation.
        """
        if not self.is_calibrated:
            return x_ts_norm
            
        rank = len(x_ts_norm.shape)
        view_shape = [1] * (rank - 1) + [-1]
        
        s_min = self.ts_stat_min.to(x_ts_norm.device).view(view_shape)
        s_max = self.ts_stat_max.to(x_ts_norm.device).view(view_shape)
        
        # x_01 = (x_norm + 1) / 2
        x_01 = (x_ts_norm + 1.0) / 2.0
        
        # x = x_01 * range + min
        x = x_01 * (s_max - s_min) + s_min
        
        return x

    def __repr__(self):
        status = "Calibrated" if self.is_calibrated else "Uncalibrated"
        return (
            f"ClinicalNormalizer("
            f"TS={self.ts_channels}, "
            f"Static={self.static_channels}, "
            f"Status={status}, "
            f"Margin={self.safety_margin})"
        )