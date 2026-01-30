import torch
import numpy as np
import logging
import os
from pathlib import Path
from icu.datasets.dataset import create_sepsis_aware_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify_BY_v2")

class MockDataset:
    def __init__(self, n_samples=1000000):
        self.split = "test"
        self.root_path = Path("./mock_data")
        self.root_path.mkdir(exist_ok=True)
        self.window_size = 30
        
        # Simulate 100 episodes of 10,000 steps each = 1,000,000 total steps
        # Each episode has (10,000 - 30 + 1) = 9971 chunks
        self.episode_metadata = []
        self.chunks_per_episode = []
        for i in range(100):
            self.episode_metadata.append({
                "length": 10000,
                "modalities": {"labels": {"key": f"ep_{i}_labels", "shape": [10000]}}
            })
            self.chunks_per_episode.append(10000 - 30 + 1)
            
        self.total_chunks = sum(self.chunks_per_episode)
        
        # Randomly assign a few "sepsis" episodes
        self.sepsis_episodes = {5, 50, 95} 

    def __len__(self):
        return self.total_chunks

    def _init_lmdb(self):
        pass

    def _fetch_numpy(self, key, dtype, shape):
        # Determine which episode this key belongs to
        ep_idx = int(key.split("_")[1])
        # Return all sepsis (1.0) if it's one of our sepsis episodes, else 0.0
        if ep_idx in self.sepsis_episodes:
            return np.ones(shape)
        return np.zeros(shape)

    def _get_phase_label(self, window):
        # If any label > 0.5, return PHASE_SHOCK (2)
        if (window > 0.5).any():
            return 2
        return 0

def test_sampler_expansion_verification():
    logger.info("Verifying Sampler Coverage Expansion (#129)...")
    
    n_samples = 1_000_000
    dataset = MockDataset(n_samples=n_samples)
    
    # Run sampler creation
    # It should scan ALL 100 episodes and find sepsis in 3 of them.
    sampler = create_sepsis_aware_sampler(dataset, sepsis_boost_factor=10.0)
    
    weights = sampler.weights
    n_boosted = (weights > 1.0).sum().item()
    
    # Expected Boosted: 3 episodes * 9971 chunks/episode = 29913
    expected_boosted = 3 * (10000 - 30 + 1)
    
    logger.info(f"Boosted Samples Count: {n_boosted} (Expected: {expected_boosted})")
    
    # Cleanup mock data
    if (dataset.root_path / "test_sepsis_index.npy").exists():
        (dataset.root_path / "test_sepsis_index.npy").unlink()
    dataset.root_path.rmdir()

    if abs(n_boosted - expected_boosted) < 10:
        logger.info("✅ Verification SUCCESS! 100% population coverage confirmed.")
    else:
        logger.error(f"❌ Verification FAILED! Expected {expected_boosted} boosted, got {n_boosted}.")

if __name__ == "__main__":
    test_sampler_expansion_verification()
