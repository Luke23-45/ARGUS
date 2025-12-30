"""
icu/utils/samplers.py
--------------------------------------------------------------------------------
SOTA Episode-Aware Sampler for Time-Series.
Author: APEX Research Team
Version: 3.0 (Cache-Optimized / DDP-Strict / Life-Critical)

Problem:
    Standard RandomSampler jumps between episodes randomly.
    For datasets with LRU Caching (like ICUSotaDataset/LMDB), this causes
    "Cache Thrashing" - loading an episode, reading 1 frame, and dumping it.
    
    Additionally, standard DistributedSampler partitions by index, breaking
    contiguity within episodes.

Solution:
    This sampler shuffles the *order of episodes*, but reads all frames
    within an episode *sequentially*.
    
    Safety Guarantees:
    1. DDP Sync: Enforces exact frame count equality across ranks (prevents hangs).
    2. Cache Locality: sequential reads maximize LMDB/OS page cache hits.
    3. Determinism: Seeded generator ensures reproducible experiments.
"""

import math
import logging
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import Sampler, Subset
from typing import Iterator, Sized, Optional, List, Dict, Union

logger = logging.getLogger("APEX_Samplers")

class EpisodeAwareSampler(Sampler[int]):
    """
    Performance-Critical Sampler for ICUSotaDataset.
    
    Logic:
    1. Analysis: Maps every global frame index to its parent Episode ID.
    2. Grouping: Buckets indices by Episode.
    3. Shuffling: Shuffles the *Episodes* (not the frames).
    4. Partitioning: Distributes Episodes to DDP ranks.
    5. Balancing: Pads/Truncates the final stream to ensure strict DDP length equality.
    
    Args:
        dataset: The dataset (must be compatible with chunk logic).
        shuffle: Whether to shuffle episode order (True for Train, False for Val).
        seed: Random seed for reproducibility across ranks.
        drop_last: Whether to drop the tail of data if not divisible by replicas.
    """
    
    def __init__(self, 
                 dataset: Sized, 
                 shuffle: bool = True, 
                 seed: int = 42, 
                 drop_last: bool = False):
        super().__init__(dataset)
        
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # --- 1. DDP Setup ---
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
            
        if self.rank == 0:
            logger.info(f"[Sampler] Initializing EpisodeAwareSampler (Ranks: {self.num_replicas}, Shuffle: {shuffle})")

        # --- 2. Unwrap & Inspect Dataset ---
        # Handle Subset wrapping to find the underlying structure
        if isinstance(dataset, Subset):
            subset_indices = np.array(dataset.indices)
            root_ds = dataset.dataset
        else:
            subset_indices = np.arange(len(dataset))
            root_ds = dataset

        # Validate Compatibility
        # We need `cumulative_chunks` to map global frame index -> episode ID
        if not hasattr(root_ds, "cumulative_chunks"):
             # Fallback or Error? For safety critical, we Error.
             # We cannot guess the structure of an unknown dataset.
             raise ValueError(
                 "EpisodeAwareSampler requires a dataset with 'cumulative_chunks' "
                 "(e.g., ICUSotaDataset/ICUTrajectoryDataset)."
             )

        # --- 3. Vectorized Index Mapping (The "Smart" Part) ---
        # We need to group [0, 1, 2, 100, 101] into {Ep1: [0, 1, 2], Ep2: [100, 101]}
        
        # O(log N) lookup to find which episode each frame belongs to.
        # cumulative_chunks stores END indices.
        try:
            # map subset_indices to episode IDs
            episode_ids = np.searchsorted(root_ds.cumulative_chunks, subset_indices, side='right')
        except Exception as e:
            logger.error(f"[Sampler] CRITICAL: Failed to map indices to episodes. Error: {e}")
            raise e
        
        # Optimize grouping: Sort by Episode ID first
        # This brings all frames of Ep X together in the list
        sort_order = np.argsort(episode_ids)
        sorted_indices = subset_indices[sort_order]
        sorted_ep_ids = episode_ids[sort_order]
        
        # Find boundaries where episode ID changes
        unique_eps, split_pos = np.unique(sorted_ep_ids, return_index=True)
        
        # Split into list of arrays: [ [idx_A1, idx_A2], [idx_B1...], ... ]
        grouped_indices = np.split(sorted_indices, split_pos[1:])
        
        # Store metadata
        self.available_episodes: List[int] = []
        self.episode_map: Dict[int, np.ndarray] = {}
        
        total_frames_mapped = 0
        for ep_id, indices in zip(unique_eps, grouped_indices):
            if len(indices) > 0:
                self.available_episodes.append(ep_id)
                # Keep indices sorted within episode for sequential reading (Time-Order)
                # This is CRITICAL for Cache Hits.
                indices.sort()
                self.episode_map[ep_id] = indices
                total_frames_mapped += len(indices)

        # --- 4. DDP Strict Balancing Calculations ---
        # We must align with DistributedSampler logic:
        # Each rank MUST yield exactly num_samples frames.
        
        total_frames_global = len(dataset)
        
        # Calculate samples per rank
        if self.drop_last and self.num_replicas > 1:
            self.num_samples = math.floor(total_frames_global / self.num_replicas)
        else:
            self.num_samples = math.ceil(total_frames_global / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas
        
        if self.rank == 0:
            logger.info(
                f"[Sampler] Strategy: {self.num_samples} frames per rank "
                f"(Total covered: {self.total_size}/{total_frames_global})"
            )

    def __iter__(self) -> Iterator[int]:
        # 1. Deterministic Shuffling (Seeded per epoch)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            # Shuffle the order of EPISODES (not frames)
            # This preserves the cache locality while randomizing training order
            rand_ep_indices = torch.randperm(len(self.available_episodes), generator=g).tolist()
        else:
            rand_ep_indices = list(range(len(self.available_episodes)))
            
        # 2. DDP Partitioning (Round Robin Episodes)
        # We assign episodes to ranks in a strided fashion.
        # Rank 0 gets Ep[0], Ep[4], ...
        # Rank 1 gets Ep[1], Ep[5], ...
        # Note: We do NOT pad episodes here. We pad the final frame stream later.
        my_ep_indices = rand_ep_indices[self.rank :: self.num_replicas]
        
        # 3. Stream Construction
        # Build the sequential stream of frames for this rank
        local_stream = []
        for list_idx in my_ep_indices:
            real_ep_id = self.available_episodes[list_idx]
            frames = self.episode_map[real_ep_id]
            local_stream.extend(frames)
            
        # 4. Strict DDP Balancing (Padding/Truncation)
        # DDP hangs if ranks yield different lengths.
        # We enforce len(yield) == self.num_samples
        
        current_len = len(local_stream)
        
        if current_len < self.num_samples:
            # Underflow: We need more data.
            # Pad by repeating the start of our own stream (safest)
            padding_needed = self.num_samples - current_len
            
            # Repeat stream as needed to cover padding
            # Logic: local_stream + local_stream + ...
            # We use a loop in case padding > current_len (rare but possible with massive scaling)
            extended_stream = local_stream
            while len(extended_stream) < self.num_samples:
                extended_stream.extend(local_stream)
            
            local_stream = extended_stream[:self.num_samples]
            
        elif current_len > self.num_samples:
            # Overflow: We have too much data.
            # Truncate. This might cut the last episode in half, but it's necessary for DDP.
            # Since we shuffle episodes every epoch, the cut point moves, so unbiased over time.
            local_stream = local_stream[:self.num_samples]
            
        # Final Verification
        assert len(local_stream) == self.num_samples, \
            f"Sampler Logic Error: Rank {self.rank} generated {len(local_stream)}, expected {self.num_samples}"
            
        return iter(local_stream)
    
    def __len__(self) -> int:
        """Required by DataLoader to determine batch counts."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Standard DDP hook to reshuffle data differently each epoch."""
        self.epoch = epoch