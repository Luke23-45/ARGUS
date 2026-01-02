"""
icu/utils/samplers.py
--------------------------------------------------------------------------------
SOTA Episode-Aware Sampler for Time-Series.
Author: APEX Research Team
Version: 3.5 (Balanced / Cache-Optimized / DDP-Strict)

Problem:
    Standard RandomSampler jumps between episodes randomly, causing cache thrashing.
    Additionally, imbalanced datasets (e.g., 1% sepsis) lead to generative collapse.

Solution:
    1. Episode-Awareness: Shuffles episodes but read frames sequentially.
    2. Weighted Sampling: Boosts the probability of selecting clinical event episodes.
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
        if isinstance(dataset, Subset):
            subset_indices = np.array(dataset.indices)
            root_ds = dataset.dataset
        else:
            subset_indices = np.arange(len(dataset))
            root_ds = dataset

        if not hasattr(root_ds, "cumulative_chunks"):
             raise ValueError("EpisodeAwareSampler requires a dataset with 'cumulative_chunks'.")

        # --- 3. Vectorized Index Mapping ---
        try:
            episode_ids = np.searchsorted(root_ds.cumulative_chunks, subset_indices, side='right')
        except Exception as e:
            logger.error(f"[Sampler] CRITICAL: Failed to map indices to episodes: {e}")
            raise e
        
        sort_order = np.argsort(episode_ids)
        sorted_indices = subset_indices[sort_order]
        sorted_ep_ids = episode_ids[sort_order]
        
        unique_eps, split_pos = np.unique(sorted_ep_ids, return_index=True)
        grouped_indices = np.split(sorted_indices, split_pos[1:])
        
        self.available_episodes: List[int] = []
        self.episode_map: Dict[int, np.ndarray] = {}
        
        for ep_id, indices in zip(unique_eps, grouped_indices):
            if len(indices) > 0:
                self.available_episodes.append(ep_id)
                indices.sort()
                self.episode_map[ep_id] = indices

        # --- 4. DDP Strict Balancing ---
        total_frames_global = len(dataset)
        if self.drop_last and self.num_replicas > 1:
            self.num_samples = math.floor(total_frames_global / self.num_replicas)
        else:
            self.num_samples = math.ceil(total_frames_global / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            rand_ep_indices = torch.randperm(len(self.available_episodes), generator=g).tolist()
        else:
            rand_ep_indices = list(range(len(self.available_episodes)))
            
        my_ep_indices = rand_ep_indices[self.rank :: self.num_replicas]
        
        local_stream = []
        for list_idx in my_ep_indices:
            real_ep_id = self.available_episodes[list_idx]
            frames = self.episode_map[real_ep_id]
            local_stream.extend(frames)
            
        current_len = len(local_stream)
        if current_len < self.num_samples:
            local_stream = (local_stream * (self.num_samples // max(1, current_len) + 1))[:self.num_samples]
        else:
            local_stream = local_stream[:self.num_samples]
            
        return iter(local_stream)
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class WeightedEpisodeSampler(EpisodeAwareSampler):
    """
    SOTA Balanced Sampler for Clinical Time-Series (v4.1).
    Ensures a target prevalence of under-represented classes (Sepsis) per batch.
    
    Logic:
    1. Scan all available episodes for Sepsis labels.
    2. Calculate weights to achieve `target_prevalence` (default 15%).
    3. Sample episodes proportionally using Multinomial sampling.
    """
    def __init__(self, 
                 dataset: Union[Sized, Subset], 
                 target_prevalence: float = 0.15,
                 shuffle: bool = True, 
                 seed: int = 42, 
                 drop_last: bool = False):
        super().__init__(dataset, shuffle, seed, drop_last)
        self.target_prevalence = target_prevalence
        
        # --- 5. Prevalence Analysis ---
        if self.rank == 0:
            logger.info(f"[Sampler] Analyzing episode labels for balanced sampling (Target: {target_prevalence*100:.1f}%)...")
        
        root_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        self.episode_weights = torch.ones(len(self.available_episodes))
        sepsis_flags = []
        
        for ep_id in self.available_episodes:
            try:
                # Key format: ep_000000_labels
                label_key = f"{root_ds.episode_metadata[ep_id]['episode_id']}_labels"
                raw_labels = root_ds._read_bytes(label_key)
                labels = np.frombuffer(raw_labels, dtype=np.float32)
                has_sepsis = np.any(labels > 0)
                sepsis_flags.append(has_sepsis)
            except Exception:
                sepsis_flags.append(False)
                
        sepsis_flags = torch.tensor(sepsis_flags)
        n_sepsis = sepsis_flags.sum().item()
        n_stable = len(sepsis_flags) - n_sepsis
        
        if n_sepsis > 0 and n_stable > 0:
            # W_s = (target * N_h) / (N_s * (1 - target))
            weight_sepsis = (target_prevalence * n_stable) / (n_sepsis * (1.0 - target_prevalence))
            self.episode_weights[sepsis_flags] = weight_sepsis
            
            if self.rank == 0:
                logger.info(f"[Sampler] Sepsis Detection: {n_sepsis} episodes found ({n_sepsis/len(sepsis_flags)*100:.2f}%)")
                logger.info(f"[Sampler] Prevalence Boost: {weight_sepsis:.2f}x weight for Sepsis episodes.")
        else:
            if self.rank == 0:
                logger.warning(f"[Sampler] Class imbalance extreme or data missing. Falling back to uniform.")

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            # Weighted Multinomial Episode Sampling
            rand_ep_indices = torch.multinomial(
                self.episode_weights, 
                num_samples=len(self.available_episodes), 
                replacement=True, 
                generator=g
            ).tolist()
        else:
            rand_ep_indices = list(range(len(self.available_episodes)))
            
        my_ep_indices = rand_ep_indices[self.rank :: self.num_replicas]
        
        local_stream = []
        for list_idx in my_ep_indices:
            real_ep_id = self.available_episodes[list_idx]
            frames = self.episode_map[real_ep_id]
            local_stream.extend(frames)
            
        if len(local_stream) < self.num_samples:
            local_stream = (local_stream * (self.num_samples // max(1, len(local_stream)) + 1))[:self.num_samples]
        else:
            local_stream = local_stream[:self.num_samples]
            
        return iter(local_stream)