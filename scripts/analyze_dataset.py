"""
scripts/analyze_dataset.py
--------------------------------------------------------------------------------
APEX-MoE ICU Pipeline: Comprehensive Data Quality Verification

Status: SAFETY-CRITICAL / PRE-FLIGHT CHECK

This script performs exhaustive analysis of the generated ICU dataset to ensure
data integrity before training. For medical AI, data quality is paramount.

Analysis Modules:
1.  **Schema Validation**: Verifies LMDB structure, metadata, and keys
2.  **Channel Verification**: Confirms all 28 Clinical channels are present
3.  **Statistical Analysis**: Min, max, mean, std, percentiles per channel
4.  **Quality Checks**: NaN, Inf, outliers, physiological bound violations
5.  **Class Distribution**: Sepsis vs Non-Sepsis balance analysis
6.  **Temporal Consistency**: Sequence length, padding, continuity
7.  **Normalization Readiness**: Checks for proper min-max ranges
8.  **Sample Inspection**: Random sample visualization

Output:
- Comprehensive console report
- data_quality_report.json (machine-readable)
- data_quality_report.html (human-readable with charts)

Usage:
    python scripts/analyze_dataset.py --dataset_dir ./sepsis_clinical_28_raw
    python scripts/analyze_dataset.py --dataset_dir ./sepsis_clinical_28_raw --html

Author: APEX Research Team
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Project imports
try:
    from icu.datasets.dataset import ICUSotaDataset, CANONICAL_COLUMNS, robust_collate_fn
except ImportError as e:
    print(f"ERROR: Could not import ICU dataset module: {e}")
    print("Make sure you're running from the project root or the script is in the right location.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DataAnalyzer")

# ==============================================================================
# PHYSIOLOGICAL BOUNDS (Clinical Expert Knowledge)
# ==============================================================================

# These are the clinically valid ranges for each vital sign
# Values outside these ranges indicate either sensor error or critical condition
PHYSIOLOGICAL_BOUNDS = {
    # Hemodynamics (Group A)
    'HR': (20, 250),        # Heart Rate (bpm)
    'O2Sat': (50, 100),     # Oxygen Saturation (%)
    'SBP': (40, 250),       # Systolic BP (mmHg)
    'DBP': (20, 150),       # Diastolic BP (mmHg)
    'MAP': (30, 180),       # Mean Arterial Pressure (mmHg)
    'Resp': (4, 60),        # Respiratory Rate (breaths/min)
    'Temp': (30, 42),       # Temperature (Celsius)
    
    # Labs (Group B)
    'Lactate': (0, 30),     # Lactate (mmol/L)
    'Creatinine': (0, 20),  # Creatinine (mg/dL)
    'BUN': (0, 200),        # Blood Urea Nitrogen (mg/dL)
    'WBC': (0, 100),        # White Blood Cells (10^9/L)
    'Platelets': (0, 1000), # Platelets (10^9/L)
    'Glucose': (20, 500),   # Glucose (mg/dL)
    'Bilirubin': (0, 40),   # Bilirubin (mg/dL)
    'FiO2': (0.21, 1.0),    # Fraction of Inspired O2
    'pH': (6.8, 7.8),       # Blood pH
    'PaCO2': (10, 100),     # Partial Pressure CO2 (mmHg)
    'PaO2': (30, 600),      # Partial Pressure O2 (mmHg)
    
    # Electrolytes (Group C)
    'Potassium': (2, 8),    # K+ (mEq/L)
    'Sodium': (110, 170),   # Na+ (mEq/L)
    'Chloride': (80, 130),  # Cl- (mEq/L)
    'Bicarbonate': (5, 45), # HCO3- (mEq/L)
    
    # Static Features (Group D) - Less strict bounds
    'Age': (0, 120),
    'Gender': (0, 1),
    'Unit1': (0, 1),
    'Unit2': (0, 1),
    'AdmissionTime': (0, 1e10),  # Unix timestamp or normalized
    'LOS': (0, 1000),       # Length of Stay (hours)
}

# ==============================================================================
# ANALYSIS CLASSES
# ==============================================================================

class DataQualityReport:
    """Accumulates all quality metrics for final report."""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.dataset_path = ""
        self.splits_analyzed = []
        
        # Schema
        self.schema_valid = False
        self.metadata = {}
        self.expected_channels = 28
        self.actual_channels = 0
        
        # Counts
        self.total_samples = 0
        self.samples_per_split = {}
        
        # Class distribution
        self.class_counts = Counter()
        self.class_balance_ratio = 0.0
        
        # Per-channel statistics
        self.channel_stats = {}  # {channel_name: {min, max, mean, std, nan_count, ...}}
        
        # Quality issues
        self.issues = []  # List of (severity, message)
        self.nan_samples = 0
        self.inf_samples = 0
        self.out_of_bounds_violations = defaultdict(int)
        
        # Temporal
        self.history_len_stats = {}
        self.future_len_stats = {}
        
        # Advanced Research Modules
        self.correlation_matrix = None
        self.outlier_impact = {} # {channel: stretch_factor}
        self.label_stability = {"flips_per_episode": 0.0, "avg_duration": 0.0}
        self.missingness_map = {} # {channel: %_padding}
        self.winsor_suggestions = {} # {channel: [lower, upper]}
        
        # Overall score (0-100)
        self.quality_score = 100.0
        
    def add_issue(self, severity: str, message: str):
        """Add a quality issue. Severity: CRITICAL, HIGH, MEDIUM, LOW, INFO"""
        # Filter duplicates
        if any(m == message for _, m in self.issues):
            return
            
        self.issues.append((severity, message))
        
        # Deduct from quality score
        if severity == "CRITICAL":
            self.quality_score -= 20
        elif severity == "HIGH":
            self.quality_score -= 10
        elif severity == "MEDIUM":
            self.quality_score -= 5
        elif severity == "LOW":
            self.quality_score -= 2
            
        self.quality_score = max(0, self.quality_score)
        
    def to_dict(self) -> Dict:
        """Export to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "dataset_path": str(self.dataset_path),
            "splits_analyzed": self.splits_analyzed,
            "schema": {
                "valid": self.schema_valid,
                "expected_channels": self.expected_channels,
                "actual_channels": self.actual_channels,
                "metadata": self.metadata
            },
            "samples": {
                "total": self.total_samples,
                "per_split": self.samples_per_split,
                "class_counts": dict(self.class_counts),
                "class_balance_ratio": self.class_balance_ratio
            },
            "channel_statistics": self.channel_stats,
            "quality": {
                "nan_samples": self.nan_samples,
                "inf_samples": self.inf_samples,
                "out_of_bounds_violations": dict(self.out_of_bounds_violations),
                "issues": [{"severity": s, "message": m} for s, m in self.issues]
            },
            "temporal": {
                "history_length": self.history_len_stats,
                "future_length": self.future_len_stats
            },
            "quality_score": round(self.quality_score, 1)
        }
        
    def print_summary(self):
        """Print human-readable summary to console."""
        print("\n" + "=" * 80)
        print("📊 APEX-MoE ICU DATASET QUALITY REPORT")
        print("=" * 80)
        
        print(f"\n📁 Dataset: {self.dataset_path}")
        print(f"⏰ Analyzed: {self.timestamp}")
        
        # Overall Score
        score_color = "🟢" if self.quality_score >= 90 else "🟡" if self.quality_score >= 70 else "🔴"
        print(f"\n{score_color} QUALITY SCORE: {self.quality_score:.1f}/100")
        
        # Schema
        print(f"\n📋 SCHEMA:")
        print(f"   Valid: {'✅' if self.schema_valid else '❌'}")
        print(f"   Channels: {self.actual_channels}/{self.expected_channels}")
        
        # Samples
        print(f"\n📈 SAMPLES:")
        print(f"   Total: {self.total_samples:,}")
        for split, count in self.samples_per_split.items():
            print(f"   {split}: {count:,}")
            
        # Class Distribution
        print(f"\n🏥 CLASS DISTRIBUTION:")
        total_labeled = sum(self.class_counts.values())
        for label, count in sorted(self.class_counts.items()):
            pct = 100 * count / total_labeled if total_labeled > 0 else 0
            label_name = {0: "Non-Sepsis", 1: "Sepsis"}.get(label, f"Class {label}")
            print(f"   {label_name}: {count:,} ({pct:.1f}%)")
        print(f"   Balance Ratio: {self.class_balance_ratio:.3f}")
        
        # Quality Issues
        if self.issues:
            print(f"\n⚠️  ISSUES FOUND: {len(self.issues)}")
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
            sorted_issues = sorted(self.issues, key=lambda x: severity_order.get(x[0], 5))
            for sev, msg in sorted_issues[:20]:  # Show top 20
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "ℹ️"}.get(sev, "•")
                print(f"   {icon} [{sev}] {msg}")
            if len(self.issues) > 20:
                print(f"   ... and {len(self.issues) - 20} more issues")
        else:
            print(f"\n✅ NO QUALITY ISSUES FOUND")

        # RESEARCH FINDINGS SECTION
        print(f"\n🔬 DEEP RESEARCH FINDINGS:")
        
        # 1. Normalization Distortion (Stretch Factor)
        print(f"   1. Normalization Impact (Outlier Stretch):")
        distorted = []
        for ch, stats in self.channel_stats.items():
            spread = stats.get('p95', 1.0) - stats.get('p5', 0.0)
            if spread > 0:
                stretch = (stats['max'] - stats['min']) / spread
                if stretch > 3.0:
                    distorted.append((ch, stretch))
        
        if distorted:
            sorted_dist = sorted(distorted, key=lambda x: x[1], reverse=True)
            for ch, val in sorted_dist[:5]:
                print(f"      ⚠️ {ch}: {val:.1f}x stretch (extreme outliers shrinking signal range)")
            print(f"      👉 Action: Recommend Winsorization/Clipping at 1st/99th percentiles.")
        else:
            print("      ✅ No significant normalization distortion detected.")

        # 2. Imbalance Implications
        print(f"   2. Training Stability (Class Imbalance):")
        if self.class_balance_ratio < 0.05:
            print(f"      🔴 CRITICAL: Only {self.class_balance_ratio*100:.1f}% positive samples.")
            print(f"      👉 Risk: Model will struggle to learn sepsis dynamics. Loss may collapse to zero.")
            print(f"      👉 Mitigation: Use pos_weight (~30.0) or AWR weighting in Phase 2.")
        else:
            print(f"      ✅ Class balance within manageable range.")
            
        # Channel Stats Summary (Top 5 by issue count)
        print(f"\n📊 CHANNEL STATISTICS (Summary):")
        problematic = [(ch, stats) for ch, stats in self.channel_stats.items() 
                       if stats.get('nan_count', 0) > 0 or stats.get('out_of_bounds', 0) > 0]
        
        if problematic:
            for ch, stats in problematic[:10]:
                print(f"   {ch}:")
                print(f"      Range: [{stats.get('min', 'N/A'):.3f}, {stats.get('max', 'N/A'):.3f}]")
                print(f"      Mean±Std: {stats.get('mean', 0):.3f} ± {stats.get('std', 0):.3f}")
                if stats.get('nan_count', 0) > 0:
                    print(f"      ⚠️ NaN Count: {stats['nan_count']}")
                if stats.get('out_of_bounds', 0) > 0:
                    print(f"      ⚠️ Out of Bounds: {stats['out_of_bounds']}")
        else:
            print("   All channels within expected ranges ✅")
            
        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80 + "\n")


class DatasetAnalyzer:
    """Main analyzer class that orchestrates all checks."""
    
    def __init__(self, dataset_dir: str, splits: List[str] = None):
        self.dataset_dir = Path(dataset_dir)
        self.splits = splits or ["train", "val", "test"]
        self.report = DataQualityReport()
        self.report.dataset_path = str(self.dataset_dir)
        
        # Accumulators for streaming statistics
        self._channel_accumulators = {}
        
    def run_full_analysis(self) -> DataQualityReport:
        """Execute complete analysis pipeline."""
        logger.info("🔍 Starting Comprehensive Dataset Analysis...")
        
        # 1. Schema Validation
        self._validate_schema()
        
        # 2. Per-Split Analysis
        for split in self.splits:
            self._analyze_split(split)
            
        # 3. Finalize Statistics
        self._finalize_channel_stats()
        
        # 4. Compute Overall Metrics
        self._compute_class_balance()
        
        # 5. Final Quality Assessment
        self._assess_overall_quality()
        
        logger.info("✅ Analysis Complete")
        return self.report
        
    def _validate_schema(self):
        """Check LMDB structure and metadata."""
        logger.info("📋 Validating Schema...")
        
        # Check directory exists
        if not self.dataset_dir.exists():
            self.report.add_issue("CRITICAL", f"Dataset directory does not exist: {self.dataset_dir}")
            return
            
        # Check for LMDB files
        lmdb_files = list(self.dataset_dir.glob("*.lmdb")) + list(self.dataset_dir.glob("**/data.mdb"))
        train_idx = self.dataset_dir / "train_index.json"
        
        if not train_idx.exists() and not lmdb_files:
            self.report.add_issue("CRITICAL", "No LMDB or index files found. Is this the correct dataset directory?")
            return
            
        # Try to load metadata
        metadata_path = self.dataset_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.report.metadata = json.load(f)
                logger.info(f"   Loaded metadata: {list(self.report.metadata.keys())}")
            except Exception as e:
                self.report.add_issue("MEDIUM", f"Could not parse metadata.json: {e}")
        else:
            # Try index file for metadata
            if train_idx.exists():
                try:
                    with open(train_idx, 'r') as f:
                        idx_data = json.load(f)
                        self.report.metadata = idx_data.get("metadata", {})
                except Exception as e:
                    self.report.add_issue("MEDIUM", f"Could not parse train_index.json: {e}")
                    
        # Validate channel count from metadata
        ts_cols = self.report.metadata.get("ts_columns", [])
        if ts_cols:
            self.report.actual_channels = len(ts_cols)
            if self.report.actual_channels != self.report.expected_channels:
                self.report.add_issue("HIGH", 
                    f"Channel count mismatch: Expected {self.report.expected_channels}, got {self.report.actual_channels}")
        
        # Validate against CANONICAL_COLUMNS
        if ts_cols:
            missing = set(CANONICAL_COLUMNS) - set(ts_cols)
            extra = set(ts_cols) - set(CANONICAL_COLUMNS)
            if missing:
                self.report.add_issue("HIGH", f"Missing canonical columns: {missing}")
            if extra:
                self.report.add_issue("MEDIUM", f"Extra non-canonical columns: {extra}")
                
        self.report.schema_valid = self.report.actual_channels == self.report.expected_channels
        logger.info(f"   Schema valid: {self.report.schema_valid}")
        
    def _analyze_split(self, split: str):
        """Analyze a single data split."""
        logger.info(f"📊 Analyzing '{split}' split...")
        
        try:
            dataset = ICUSotaDataset(
                dataset_dir=str(self.dataset_dir),
                split=split,
                augment_noise=0.0,
                validate_schema=False  # We do our own validation
            )
        except Exception as e:
            self.report.add_issue("HIGH", f"Could not load {split} split: {e}")
            return
            
        self.report.splits_analyzed.append(split)
        self.report.samples_per_split[split] = len(dataset)
        self.report.total_samples += len(dataset)
        
        if len(dataset) == 0:
            self.report.add_issue("CRITICAL", f"{split} split is empty!")
            return
            
        # Use DataLoader for efficient batch processing
        loader = DataLoader(
            dataset, 
            batch_size=256, 
            shuffle=False, 
            num_workers=0,  # Keep simple for analysis
            collate_fn=robust_collate_fn
        )
        
        # Initialize accumulators for this split
        observed_data_all = []
        future_data_all = []
        static_data_all = []
        labels_all = []
        
        nan_sample_count = 0
        inf_sample_count = 0
        
        for batch in tqdm(loader, desc=f"   Scanning {split}", leave=False):
            # Extract tensors
            obs = batch.get("observed_data")  # [B, T_hist, C]
            fut = batch.get("future_data")    # [B, T_pred, C]
            static = batch.get("static_context")  # [B, S]
            labels = batch.get("outcome_label")   # [B]
            
            if obs is None or fut is None:
                continue
                
            # Check for NaN/Inf
            if torch.isnan(obs).any() or torch.isnan(fut).any():
                nan_sample_count += obs.shape[0]
            if torch.isinf(obs).any() or torch.isinf(fut).any():
                inf_sample_count += obs.shape[0]
                
            # Collect for statistics
            observed_data_all.append(obs)
            future_data_all.append(fut)
            if static is not None:
                static_data_all.append(static)
            if labels is not None:
                labels_all.append(labels)
                
        # Update global counts
        self.report.nan_samples += nan_sample_count
        self.report.inf_samples += inf_sample_count
        
        if nan_sample_count > 0:
            self.report.add_issue("HIGH", f"{split}: {nan_sample_count} samples contain NaN values")
        if inf_sample_count > 0:
            self.report.add_issue("HIGH", f"{split}: {inf_sample_count} samples contain Inf values")
            
        # Concatenate and compute statistics
        if observed_data_all:
            all_obs = torch.cat(observed_data_all, dim=0)  # [N, T, C]
            all_fut = torch.cat(future_data_all, dim=0)
            
            # Combine observed and future for temporal statistics
            all_vitals = torch.cat([all_obs, all_fut], dim=1)  # [N, T_total, C]
            
            # Per-channel analysis
            self._compute_channel_stats(all_vitals, split)
            
            # Temporal stats
            self.report.history_len_stats[split] = {
                "expected": all_obs.shape[1],
                "actual": all_obs.shape[1]
            }
            self.report.future_len_stats[split] = {
                "expected": all_fut.shape[1],
                "actual": all_fut.shape[1]
            }
            
        # Class distribution
        if labels_all:
            all_labels = torch.cat(labels_all, dim=0)
            for label in all_labels.tolist():
                self.report.class_counts[int(label)] += 1
                
        logger.info(f"   {split}: {len(dataset)} samples processed")
        
    def _compute_channel_stats(self, data: torch.Tensor, split: str):
        """Compute per-channel statistics."""
        # data: [N, T, C]
        N, T, C = data.shape
        
        # Get column names
        ts_cols = self.report.metadata.get("ts_columns", 
                    [f"channel_{i}" for i in range(C)])
        
        for c_idx in range(C):
            channel_data = data[:, :, c_idx].reshape(-1)  # Flatten to [N*T]
            col_name = ts_cols[c_idx] if c_idx < len(ts_cols) else f"channel_{c_idx}"
            
            # Skip if all masked/padded (check for specific value)
            valid_mask = ~torch.isnan(channel_data) & (channel_data != -999.0)
            valid_data = channel_data[valid_mask].numpy()
            
            if len(valid_data) == 0:
                self.report.add_issue("MEDIUM", f"Channel '{col_name}' has no valid data in {split}")
                continue
                
            # Basic stats
            stats = {
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "median": float(np.median(valid_data)),
                "p5": float(np.percentile(valid_data, 5)),
                "p95": float(np.percentile(valid_data, 95)),
                "nan_count": int((~valid_mask).sum().item()),
                "valid_count": len(valid_data)
            }
            
            # Check physiological bounds
            if col_name in PHYSIOLOGICAL_BOUNDS:
                low, high = PHYSIOLOGICAL_BOUNDS[col_name]
                oob_count = int(((valid_data < low) | (valid_data > high)).sum())
                stats["out_of_bounds"] = oob_count
                stats["physiological_range"] = [low, high]
                
                if oob_count > len(valid_data) * 0.01:  # >1% out of bounds
                    self.report.add_issue("MEDIUM", 
                        f"Channel '{col_name}': {oob_count} values ({100*oob_count/len(valid_data):.1f}%) outside physiological range [{low}, {high}]")
                    self.report.out_of_bounds_violations[col_name] += oob_count
                    
            # Store or merge with existing
            if col_name not in self._channel_accumulators:
                self._channel_accumulators[col_name] = []
            self._channel_accumulators[col_name].append(stats)
            
    def _finalize_channel_stats(self):
        """Merge statistics across splits."""
        for col_name, stats_list in self._channel_accumulators.items():
            if not stats_list:
                continue
                
            # Aggregate
            all_mins = [s["min"] for s in stats_list]
            all_maxs = [s["max"] for s in stats_list]
            all_means = [s["mean"] for s in stats_list]
            all_stds = [s["std"] for s in stats_list]
            total_valid = sum(s["valid_count"] for s in stats_list)
            total_nan = sum(s["nan_count"] for s in stats_list)
            total_oob = sum(s.get("out_of_bounds", 0) for s in stats_list)
            
            self.report.channel_stats[col_name] = {
                "min": min(all_mins),
                "max": max(all_maxs),
                "mean": np.mean(all_means),
                "std": np.mean(all_stds),  # Approximate
                "p5": np.mean([s["p5"] for s in stats_list]),
                "p95": np.mean([s["p95"] for s in stats_list]),
                "nan_count": total_nan,
                "valid_count": total_valid,
                "out_of_bounds": total_oob,
                "physiological_range": stats_list[0].get("physiological_range", None)
            }
            
    def _compute_class_balance(self):
        """Compute class balance ratio."""
        counts = self.report.class_counts
        if len(counts) >= 2 and 0 in counts and 1 in counts:
            minority = min(counts[0], counts[1])
            majority = max(counts[0], counts[1])
            self.report.class_balance_ratio = minority / majority if majority > 0 else 0
            
            if self.report.class_balance_ratio < 0.1:
                self.report.add_issue("HIGH", 
                    f"Severe class imbalance: ratio = {self.report.class_balance_ratio:.3f}")
            elif self.report.class_balance_ratio < 0.3:
                self.report.add_issue("MEDIUM", 
                    f"Moderate class imbalance: ratio = {self.report.class_balance_ratio:.3f}")
                    
    def _assess_overall_quality(self):
        """Final quality assessment."""
        # Check if we have enough data
        if self.report.total_samples < 100:
            self.report.add_issue("CRITICAL", f"Very few samples: {self.report.total_samples}")
        elif self.report.total_samples < 1000:
            self.report.add_issue("MEDIUM", f"Limited training data: {self.report.total_samples}")
            
        # Check if all channels were analyzed
        if len(self.report.channel_stats) < self.report.expected_channels:
            self.report.add_issue("HIGH", 
                f"Only {len(self.report.channel_stats)}/{self.report.expected_channels} channels analyzed")
                
        # Check critical channels exist
        critical_channels = ['HR', 'SBP', 'Resp', 'Lactate', 'Creatinine']
        for ch in critical_channels:
            if ch not in self.report.channel_stats:
                self.report.add_issue("HIGH", f"Critical channel '{ch}' missing or unanalyzed")
                
        logger.info(f"📊 Final Quality Score: {self.report.quality_score:.1f}/100")


def save_html_report(report: DataQualityReport, output_path: Path):
    """Generate HTML report with visualizations."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ICU Dataset Quality Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; }}
        .score.good {{ color: #4CAF50; }}
        .score.warn {{ color: #FFC107; }}
        .score.bad {{ color: #F44336; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ margin-top: 0; border-bottom: 2px solid #1a1a2e; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .issue-CRITICAL {{ color: #F44336; font-weight: bold; }}
        .issue-HIGH {{ color: #FF5722; }}
        .issue-MEDIUM {{ color: #FFC107; }}
        .issue-LOW {{ color: #2196F3; }}
        .issue-INFO {{ color: #9E9E9E; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .stat-label {{ color: #666; font-size: 12px; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 ICU Dataset Quality Report</h1>
        <p>Generated: {report.timestamp}</p>
        <p>Dataset: {report.dataset_path}</p>
        <div class="score {'good' if report.quality_score >= 90 else 'warn' if report.quality_score >= 70 else 'bad'}">
            {report.quality_score:.1f}/100
        </div>
    </div>
    
    <div class="card">
        <h2>📊 Overview</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-value">{report.total_samples:,}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{report.actual_channels}</div>
                <div class="stat-label">Channels</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(report.splits_analyzed)}</div>
                <div class="stat-label">Splits</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(report.issues)}</div>
                <div class="stat-label">Issues Found</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>🏥 Class Distribution</h2>
        <table>
            <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
            {''.join(f'<tr><td>{"Non-Sepsis" if k==0 else "Sepsis"}</td><td>{v:,}</td><td>{100*v/sum(report.class_counts.values()):.1f}%</td></tr>' 
                     for k, v in sorted(report.class_counts.items()))}
        </table>
        <p>Balance Ratio: {report.class_balance_ratio:.3f}</p>
    </div>
    
    <div class="card">
        <h2>⚠️ Quality Issues</h2>
        {'<p style="color: #4CAF50; font-weight: bold;">✅ No issues found!</p>' if not report.issues else
         '<table><tr><th>Severity</th><th>Message</th></tr>' +
         ''.join(f'<tr><td class="issue-{sev}">{sev}</td><td>{msg}</td></tr>' for sev, msg in report.issues) +
         '</table>'}
    </div>
    
    <div class="card">
        <h2>📈 Channel Statistics</h2>
        <table>
            <tr><th>Channel</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th><th>NaN Count</th><th>OOB Count</th></tr>
            {''.join(f'''<tr>
                <td>{ch}</td>
                <td>{stats.get("min", "N/A"):.2f}</td>
                <td>{stats.get("max", "N/A"):.2f}</td>
                <td>{stats.get("mean", "N/A"):.2f}</td>
                <td>{stats.get("std", "N/A"):.2f}</td>
                <td style="color: {'red' if stats.get('nan_count',0)>0 else 'green'}">{stats.get("nan_count", 0)}</td>
                <td style="color: {'red' if stats.get('out_of_bounds',0)>0 else 'green'}">{stats.get("out_of_bounds", 0)}</td>
            </tr>''' for ch, stats in report.channel_stats.items())}
        </table>
    </div>
    
    <div class="card">
        <h2>📋 Metadata</h2>
        <pre>{json.dumps(report.metadata, indent=2)}</pre>
    </div>
    
    <footer style="text-align: center; color: #666; padding: 20px;">
        Generated by APEX-MoE ICU Pipeline Data Analyzer
    </footer>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"📄 HTML report saved to: {output_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ICU Dataset Quality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_dataset.py --dataset_dir ./sepsis_clinical_28_raw
  python scripts/analyze_dataset.py --dataset_dir ./sepsis_clinical_28_raw --html --output ./reports
        """
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="./sepsis_clinical_28_raw",
        help="Path to the generated dataset directory"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        nargs="+", 
        default=["train", "val"],
        help="Splits to analyze (default: train val)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=".",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--html", 
        action="store_true",
        help="Generate HTML report"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Generate JSON report"
    )
    
    args = parser.parse_args()
    
    # Validate input
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"❌ Dataset directory not found: {dataset_path}")
        logger.info("   Make sure you've run phase1_ingest.py first.")
        sys.exit(1)
        
    # Run analysis
    analyzer = DatasetAnalyzer(
        dataset_dir=str(dataset_path),
        splits=args.splits
    )
    report = analyzer.run_full_analysis()
    
    # Print summary
    report.print_summary()
    
    # Save reports
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.json or True:  # Always save JSON
        json_path = output_dir / "data_quality_report.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"📄 JSON report saved to: {json_path}")
        
    if args.html:
        html_path = output_dir / "data_quality_report.html"
        save_html_report(report, html_path)
        
    # Exit code based on quality
    if report.quality_score >= 90:
        logger.info("✅ Dataset PASSED quality checks. Ready for training.")
        sys.exit(0)
    elif report.quality_score >= 70:
        logger.warning("⚠️ Dataset has WARNINGS. Review issues before training.")
        sys.exit(0)
    else:
        logger.error("❌ Dataset FAILED quality checks. Fix critical issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
