
import torch
import numpy as np
from icu.datasets.dataset import ICUSotaDataset
from icu.utils.safety import OODGuardian
from torch.utils.data import DataLoader
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DIAG_SIM")

def run_diagnostic():
    logger.info("Starting Diagnostic Simulation for OOD & Sepsis Sensitivity...")
    
    # 1. Initialize Dataset
    # We use validation set to check 'True' clinical reality
    dataset_dir = "sepsis_clinical_28"
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} not found!")
        return

    logger.info(f"Loading dataset from {dataset_dir}...")
    ds = ICUSotaDataset(
        dataset_dir=dataset_dir,
        split="val",
        history_len=24,
        pred_len=6,
        augment_noise=0.0 # No noise for diagnostic truth
    )
    
    # Collate function required because SotaDataset can return None for NaNs
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    
    batch = next(iter(dl))
    if batch is None:
        logger.error("Failed to load first batch.")
        return

    past = batch["observed_data"]    # [B, 24, 28]
    future = batch["future_data"]  # [B, 6, 28]
    stable_total, preshock_total, shock_total = 0, 0, 0
    max_batches = 40  # Scan ~10k samples
    
    logger.info(f"--- Global Prevalence Audit (Scanning {max_batches} batches) ---")
    for i, batch in enumerate(dl):
        if i >= max_batches or batch is None: break
        labels = batch["phase_label"]
        stable_total += (labels == 0).sum().item()
        preshock_total += (labels == 1).sum().item()
        shock_total += (labels == 2).sum().item()
    
    B_total = stable_total + preshock_total + shock_total
    logger.info(f"Total Scanned: {B_total}")
    logger.info(f"Stable (0):    {stable_total:>5} ({100*stable_total/B_total:>5.2f}%)")
    logger.info(f"Pre-Shock (1): {preshock_total:>5} ({100*preshock_total/B_total:>5.2f}%)")
    logger.info(f"Shock (2):     {shock_total:>5} ({100*shock_total/B_total:>5.2f}%)")
    
    # 3. Audit OOD Guardian (Truth Check)
    # Goal: Does REAL CLINICAL DATA pass the OOD Guardian?
    # If YES, then the OOD=1.000 in logs is correctly identifying model noise.
    # If NO, then our OOD thresholds are too tight for real data.
    
    guardian = OODGuardian(verbose=True)
    
    logger.info("--- OOD Guardian Audit (Real Ground Truth) ---")
    # In wrapper_generalist, we call check_trajectories with force_clinical=True
    # because dataset data is supposed to be in clinical units.
    results = guardian.check_trajectories(past, future, force_clinical=True)
    
    logger.info(f"GT OOD Rate:   {results['ood_rate'].item():.4f}")
    logger.info(f"GT Safe Count: {results['safe_count'].item():.0f} / {past.shape[0]}")
    logger.info(f"Mean Stitching error: {results['stitching_error'].item():.4f}")
    
    # Analyze OOD breakdown
    ood_mask = results["ood_mask"]
    if ood_mask.any():
        logger.info(f"Analyzing {ood_mask.sum().item()} OOD failures in REAL data:")
        idx = torch.where(ood_mask)[0][0].item()
        p_v = past[idx]
        f_v = future[idx]
        logger.info(f"Example Fail (Idx {idx}):")
        # Check specific channels (HR=0, SBP=2, MAP=4, Lac=7)
        logger.info(f"  Last Obs:  HR={p_v[-1, 0]:.1f}, SBP={p_v[-1, 2]:.1f}, MAP={p_v[-1, 4]:.1f}, Lac={p_v[-1, 7]:.1f}")
        logger.info(f"  First Pred: HR={f_v[0, 0]:.1f}, SBP={f_v[0, 2]:.1f}, MAP={f_v[0, 4]:.1f}, Lac={f_v[0, 7]:.1f}")

    if results['ood_rate'] < 0.1:
        logger.info("CONCLUSION: OOD GAURDIAN VERIFIED. Real data is ~90%+ Safe.")
        logger.info("The 100% OOD in training logs confirms the model is currently generating physically impossible vitals.")
    else:
        logger.warning("CONCLUSION: OOD THRESHOLDS MIGHT BE TOO TIGHT.")
        logger.warning(f"Even real data has {results['ood_rate'].item()*100:.1f}% OOD rate.")

if __name__ == "__main__":
    run_diagnostic()
