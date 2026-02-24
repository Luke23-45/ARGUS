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
import os
import time
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import Sampler, Subset
from pathlib import Path
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
        # [FIX] PyTorch 2.x Sampler is abstract base - don't pass dataset to super().__init__()
        super().__init__()
        
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.consumed = 0
        
        # [SOTA FIX 2] Absolute DDP Seed Consensus
        # Rationale: This sampler globally shuffles then shards. If ranks have different seeds,
        # the global shuffles diverge, causing overlapping data and broken epochs.
        self.seed = seed
        if dist.is_available() and dist.is_initialized():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            seed_t = torch.tensor([self.seed], dtype=torch.long, device=device)
            dist.broadcast(seed_t, src=0)
            self.seed = int(seed_t.item())
            
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
            
        # [v4.2 SOTA FIX] Intra-Epoch Resumption
        # Resume from the exact 'consumed' offset to prevent duplicate batches.
        for i in range(self.consumed, len(local_stream)):
            self.consumed += 1
            yield int(local_stream[i])
            
        # Reset at end of full iteration
        self.consumed = 0
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        # [SOTA FIX] Only reset consumed if we are truly moving to a NEW epoch.
        # This allows re-runs of 'set_epoch' (e.g. in validation/sanity) without wiping the training offset.
        if epoch != self.epoch:
            self.consumed = 0
        self.epoch = epoch

    def state_dict(self) -> Dict[str, int]:
        """v4.2: Persist state for exact resumption."""
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "consumed": self.consumed
        }

    def load_state_dict(self, state_dict: Dict[str, int]):
        """v4.2: Restore state."""
        self.epoch = state_dict.get("epoch", 0)
        self.seed = state_dict.get("seed", self.seed)
        self.consumed = state_dict.get("consumed", 0)
        logger.info(f"[Sampler] State Restored: Epoch={self.epoch}, Consumed={self.consumed}")


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
        # [v2026 SOTA] Atomic Sampler Sync & Optimization
        # Rationale: Prevents I/O thumping by caching prevalence results and 
        # using a single LMDB transaction for the entire scan.
        cache_name = f"{root_ds.split}_prevalence_v1.npy"
        root_path = getattr(root_ds, 'root_path', Path('.'))
        cache_path = root_path / cache_name
        
        sepsis_flags = None
        
        # 1. Rank-Aware Sync Barrier (Wait-to-Load)
        if not cache_path.exists() and self.num_replicas > 1 and self.rank != 0:
            logger.info(f"[Sampler] Rank {self.rank} waiting for Rank 0 to finish prevalence scan...")
            for _ in range(60): # 5 min timeout
                if cache_path.exists(): break
                time.sleep(5)
                
        if cache_path.exists():
            try:
                # Use mmap=True for zero-copy read if large
                sepsis_flags = torch.from_numpy(np.load(str(cache_path)))
                if self.rank == 0:
                    logger.info(f"[Sampler] Prevalence cache loaded: {cache_path}")
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"[Sampler] Cache corrupted, falling back to scan: {e}")
                sepsis_flags = None

        if sepsis_flags is None:
            # [SOTA FIX] Check for Metadata-Only Fast-Path
            # Rationale: dataset_quality.py now embeds 'has_sepsis' in the index.
            # If present, we can build the flags in O(N) memory-speed without touching LMDB.
            try:
                if all('has_sepsis' in root_ds.episode_metadata.get(ep_id, {}) for ep_id in self.available_episodes):
                    if self.rank == 0:
                        logger.info(f"[Sampler] Using Metadata Fast-Path for Prevalence Scan (Zero-Wait)")
                    sepsis_flags = torch.tensor([
                        bool(root_ds.episode_metadata[ep_id]['has_sepsis']) 
                        for ep_id in self.available_episodes
                    ])
            except Exception as e:
                logger.debug(f"[Sampler] Fast-path skipped: {e}")

        if sepsis_flags is None:
            if self.rank == 0:
                logger.info(f"[Sampler] Performing Prevalence Scan (LMDB Fallback)...")
            
            sepsis_flags_list = []
            env_was_none = (getattr(root_ds, '_lmdb_env', None) is None)
            
            if env_was_none:
                if hasattr(root_ds, '_init_lmdb'): root_ds._init_lmdb()
                elif hasattr(root_ds, '_open_lmdb'): root_ds._open_lmdb()

            try:
                env = getattr(root_ds, '_lmdb_env', None)
                # [SOTA FIX] Single-Transaction Lifecycle
                if env is not None:
                    with env.begin(write=False) as txn:
                        for ep_id in self.available_episodes:
                            try:
                                meta = root_ds.episode_metadata[ep_id]
                                if 'has_sepsis' in meta:
                                    has_sepsis = bool(meta['has_sepsis'])
                                elif 'label' in meta:
                                    has_sepsis = (meta['label'] > 0)
                                else:
                                    label_key = f"{meta['episode_id']}_labels"
                                    raw_labels = txn.get(label_key.encode())
                                    if raw_labels is not None:
                                        labels = np.frombuffer(raw_labels, dtype=np.float32)
                                        has_sepsis = np.any(labels > 0)
                                    else:
                                        has_sepsis = False
                                sepsis_flags_list.append(has_sepsis)
                            except Exception:
                                sepsis_flags_list.append(False)
                else:
                    # Fallback for empty ranks or missing env
                    for ep_id in self.available_episodes:
                        try:
                            meta = root_ds.episode_metadata[ep_id]
                            label_key = f"{meta['episode_id']}_labels"
                            raw_labels = root_ds._read_bytes(label_key)
                            labels = np.frombuffer(raw_labels, dtype=np.float32)
                            sepsis_flags_list.append(np.any(labels > 0))
                        except Exception:
                            sepsis_flags_list.append(False)
                
                sepsis_flags = torch.tensor(sepsis_flags_list)
                
                # Rank 0 persists the cache atomically
                if self.rank == 0:
                    try:
                        # [SOTA FIX 1] Prevent np.save from secretly appending ".npy" to our ".tmp" file
                        temp_path = str(cache_path).replace('.npy', '.tmp.npy')
                        np.save(temp_path, sepsis_flags.numpy())
                        os.replace(temp_path, str(cache_path))
                    except Exception as e:
                        logger.warning(f"[Sampler] Failed to save prevalence cache: {e}")
            finally:
                if env_was_none and hasattr(root_ds, 'close'):
                    root_ds.close()

        if sepsis_flags is None:
            sepsis_flags = torch.zeros(len(self.available_episodes), dtype=torch.bool)
        
        sepsis_flags = sepsis_flags.bool()
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
            
        # [v4.2 SOTA FIX] Intra-Epoch Resumption
        for i in range(self.consumed, len(local_stream)):
            self.consumed += 1
            yield int(local_stream[i])
            
        self.consumed = 0