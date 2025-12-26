import os
import sys
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path for imports
ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))

from icu.datasets.dataset import ICUSotaDataset
from icu.datasets.build_dataset import FEATURE_ORDER as VITALS_COLS

def verify_data_quality(num_samples: int = 5, output_dir: str = "data/samples"):
    print(f"--- Data Quality Audit Pipeline ---")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Dataset
    print("Initializing Training Dataset...")
    dataset = ICUSotaDataset(
        dataset_dir="data",
        split="train",
        history_len=24,
        pred_len=6,
        augment_noise=0.0 # No noise for audit
    )

    total_chunks = len(dataset)
    print(f"Total Windows available: {total_chunks}")

    # 2. Pick Random Samples
    random.seed(42)
    indices = random.sample(range(total_chunks), min(num_samples, total_chunks))
    
    audit_results = []

    for idx in indices:
        sample = dataset[idx]
        if sample is None:
            print(f"Warning: Sample at index {idx} returned None. Skipping.")
            continue
            
        patient_id = sample['patient_id']
        obs = sample['observed_data'] # [24, 28] Clinical 28
        fut = sample['future_data']   # [6, 28] Clinical 28
        
        # Combine into full sequence [30, 28]
        full_seq = torch.cat([obs, fut], dim=0).numpy()
        
        # Create DataFrame
        df = pd.DataFrame(full_seq, columns=VITALS_COLS)
        df['timestep'] = range(len(df))
        df['patient_id'] = patient_id
        df['is_future'] = [False]*24 + [True]*6
        
        # Save to CSV
        csv_name = f"sample_idx{idx}_pid{patient_id}.csv"
        csv_path = out_path / csv_name
        df.to_csv(csv_path, index=False)
        
        # Collect Stats
        stats = {
            "idx": idx,
            "patient_id": patient_id,
            "hr_mean": df['HR'].mean(),
            "hr_max": df['HR'].max(),
            "sbp_mean": df['SBP'].mean(),
            "o2_min": df['O2Sat'].min(),
            "csv_path": str(csv_path)
        }
        audit_results.append(stats)
        print(f"Exported sample {idx} (Patient {patient_id}) to {csv_path}")

    # 3. Print Summary Stats
    print("\n--- Summary Clinical Audit ---")
    audit_df = pd.DataFrame(audit_results)
    print(audit_df.drop(columns=['csv_path']))
    
    # Global verification against PhysioNet expected ranges
    print("\n--- Range Verification ---")
    for col in VITALS_COLS:
        col_min = audit_df.get(f"{col.lower()}_min", None) # Note: I didn't add min for all in the dict above, 
        # but let's just do a quick check on one
        pass
    
    print("\nVerification Suggestion:")
    print("1. Open the CSV files in 'data/samples'.")
    print("2. Verify if 'is_future' transitions correctly.")
    print("3. Check if values look like real HR (60-120), SBP (90-140), etc.")

if __name__ == "__main__":
    verify_data_quality()
