
import torch
import numpy as np
from icu.utils.advantage_calculator import ICUAdvantageCalculator
from icu.models.wrapper_generalist import compute_explained_variance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EV_DIAG")


from icu.datasets.dataset import ICUSotaDataset, robust_collate_fn
from torch.utils.data import DataLoader

def audit_real_returns():
    logger.info("Auditing Real Returns Variance...")
    
    # Setup
    dataset_dir = "C:/Users/Hellx/Documents/Programming/python/Project/iron/icu_research/sepsis_clinical_28"
    ds = ICUSotaDataset(
        dataset_dir=dataset_dir,
        split="val",
        history_len=30,
        pred_len=30
    )
    loader = DataLoader(ds, batch_size=64, collate_fn=robust_collate_fn)
    
    awr = ICUAdvantageCalculator(beta=0.05)
    
    batch = next(iter(loader))
    vitals = batch["future_data"]
    outcome_label = batch["outcome_label"]
    dones = batch["is_terminal"]
    mask = batch["future_mask"]
    
    # Mock Normalizer (assuming standard scale for now)
    class MockNormalizer:
        def denormalize(self, x): return x 
    norm = MockNormalizer()
    
    rewards = awr.compute_clinical_reward(
        vitals, outcome_label, dones=dones, normalizer=norm, src_mask=mask
    )
    
    var_r = torch.var(rewards).item()
    mean_r = torch.mean(rewards).item()
    
    logger.info(f"Real Reward Stats: Mean={mean_r:.4f}, Var={var_r:.6e}")
    
    if var_r < 1e-9:
        logger.error("!!! REWARD COLLAPSE DETECTED !!! All clinical rewards are constant.")
        # Check why
        logger.info(f"Outcome Label Sum: {outcome_label.sum().item()}")
        logger.info(f"Vitals Max/Min: {vitals.max().item():.2f}/{vitals.min().item():.2f}")
    else:
        logger.info("Rewards have healthy variance.")

if __name__ == "__main__":
    # simulate_ev_collapse() # Skip basic sim
    audit_real_returns()
