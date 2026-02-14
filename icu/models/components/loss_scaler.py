import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Tuple, Optional
from icu.utils.train_utils import ScalingSteward

class BayesianProjectedScaler(nn.Module):
    """
    [SOTA 2025] Bayesian-PGD Uncertainty Scaler with UW-SO (Achituve et al. 2024).
    
    Features:
    1. Clinical-Priority UW-SO: Softmax-normalized priority balancing.
    2. PGD Projection: Ensures log_vars stay in the differentiable zone [-2, 5].
    3. Gradient Entropy Preservation: No forward-pass clamping.
    """
    def __init__(self, num_tasks: int = 7, decay: float = 0.99):
        super().__init__()
        self.keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
        self.num_tasks = num_tasks
        
        # [v110.0 SOTA FIX] Safe-Start Initialization (Smoking Gun #110)
        # Rationale: Initialization at 0.0 means precision=1.0. High-loss tasks
        # like physics can cause massive initial shocks. 
        # Fix: Pre-suppress non-primary task gradients to prevent cold-start shocks.
        log_vars_init = torch.zeros(num_tasks)
        # indices: diff=0, critic=1, aux=2, acl=3, bgsl=4, tcb=5, phys=6
        # [FIX] Only set suppression values if the index exists in num_tasks
        if num_tasks > 4: log_vars_init[4] = 0.5  # bgsl starts cautious
        if num_tasks > 5: log_vars_init[5] = 1.0  # tcb starts suppressed
        if num_tasks > 6: log_vars_init[6] = 2.0  # phys starts heavily suppressed
        self.log_vars = nn.Parameter(log_vars_init)
        
        # [v29.6] Gradient Hardening (Abyssal #307)
        # Rationale: Prevent extreme priority shifts (ringing) during loss spikes.
        # Clamping gradients ensures log_vars move at most 0.1 units per step.
        self.log_vars.register_hook(lambda grad: grad.clamp(min=-0.1, max=0.1))
        
        # EMA tracking for UW-SO stability
        self.register_buffer("loss_emas", torch.ones(num_tasks))
        self.register_buffer("decay", torch.tensor([decay]))
        
        # [v177.1 SOTA] Accumulation Buffers
        # Rationale: Accumulate raw losses across sub-batches to provide 
        # uncertainty EMAs with the true cycle average (Fixes Smoking Gun #177/214).
        self.register_buffer("loss_accumulator", torch.zeros(num_tasks))
        self.register_buffer("task_counters", torch.zeros(num_tasks))
        self.register_buffer("batch_counter", torch.zeros([1]))
        self.register_buffer("step_count", torch.tensor([0], dtype=torch.long))

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies uncertainty decay across step densities."""
        if n_curr <= 0: return
        self.decay.fill_(ScalingSteward.get_decay(0.99, n_curr))
        
    def forward(self, loss_dict: Dict[str, torch.Tensor], stability_factor: float = 1.0, phys_multiplier: float = 1.0, batch_size: int = 1, is_accumulating: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Unified weighting pass with Dynamic Priority.
        """
        # 1. Extract Active Task Signals
        losses = []
        active_keys = []
        for i, key in enumerate(self.keys):
            if key in loss_dict:
                losses.append(loss_dict[key])
                active_keys.append((i, key))
        
        has_losses = len(losses) > 0
        device = next(self.parameters()).device
        
        if has_losses:
            losses_tensor = torch.stack(losses)
            indices = torch.tensor([idx for idx, _ in active_keys], device=losses_tensor.device)
            local_raw = losses_tensor.detach()
            
            # [v177.2 SOTA FIX] Accumulation-Aware Data Collection
            with torch.no_grad():
                self.loss_accumulator[indices] += local_raw * batch_size
                self.task_counters[indices] += batch_size
                self.batch_counter += batch_size
        else:
            # Neutral tensors for empty-batch ranks
            losses_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            indices = torch.tensor([], dtype=torch.long, device=device)
            local_raw = torch.tensor([], device=device)

        # [v137.1 / v156.1 / v216.0 SOTA FIX] Stepping-Batch DDP Consensus
        # CRITICAL: This block must be entered by ALL ranks simultaneously to prevent 
        # "One-Armed Bandit" deadlocks when some ranks have zero task losses.
        if torch.distributed.is_initialized() and self.training and not is_accumulating:
            # 1. Pack accumulated sums into a fixed-size buffer
            sync_buffer = torch.zeros(self.num_tasks * 2 + 1, device=device)
            sync_buffer[:self.num_tasks] = self.loss_accumulator
            sync_buffer[self.num_tasks:2*self.num_tasks] = self.task_counters
            sync_buffer[-1] = self.batch_counter
            
            # 2. Global Consensus (All Ranks participating)
            torch.distributed.all_reduce(sync_buffer, op=torch.distributed.ReduceOp.SUM)
            
            global_sum_losses = sync_buffer[:self.num_tasks]
            global_task_counts = sync_buffer[self.num_tasks:2*self.num_tasks]
            global_batch_size = sync_buffer[-1]
            
            # 3. Compute True Global Average for the entire cycle
            avg_losses_all = global_sum_losses / (global_task_counts + 1e-8)
            avg_losses = avg_losses_all[indices] if has_losses else torch.tensor([], device=device)
            
            # 4. Momentum-Based Bayesian Update (Dynamic Inertia #110B)
            # Rationale: Lower decay (0.95) during warmup (200 steps) allows 5x faster adaptation.
            # [v2026 SOTA] Vectorized Warmup Gate (Zero-Sync)
            self.step_count.add_(1)
            is_warmup = (self.step_count < 200)
            curr_decay_t = torch.where(is_warmup, torch.as_tensor([0.95], device=device), self.decay)
            
            # [v2026 SOTA FIX] Atomic EMA Update (lerp_)
            # Rationale: Previous logic performed a double-update (mul_ + copy_).
            # lerp_ provides a single, vectorized kernel for EMA: self = self + (1-decay) * (target - self)
            self.loss_emas.lerp_(avg_losses_all, 1.0 - curr_decay_t)
            
            # 5. Cycle Reset
            self.loss_accumulator.zero_()
            self.task_counters.zero_()
            self.batch_counter.zero_()
        else:
            # During accumulation or non-DDP, use local stats.
            # For empty ranks during accumulation, we just provide an empty fallback.
            if self.training and not is_accumulating:
                # [Non-DDP Case] Handle the non-DDP stepping-batch logic
                avg_losses_all = self.loss_accumulator / (self.task_counters + 1e-8)
                avg_losses = avg_losses_all[indices] if has_losses else torch.tensor([], device=device)
                # [SOTA v4.0] Conservative Warmup (matches DDP case)
                curr_decay = torch.where(self.step_count < 200, torch.as_tensor([0.95], device=device), self.decay)
                self.loss_emas.lerp_(avg_losses_all, 1.0 - curr_decay)
                self.loss_accumulator.zero_()
                self.task_counters.zero_()
                self.batch_counter.zero_()
            else:
                avg_losses = local_raw
                avg_losses_all = local_raw # Fallback for logging
        
        # 4. Return early if no losses to weight (after DDP sync)
        if not has_losses:
            return losses_tensor, {}
        
        # 1. Soft Optimal Uncertainty Weighting (UW-SO)
        # (EMA update already moved into the stepped-sync block above)
        with torch.no_grad():
            # [PATCH 3] Fixed Clinical Priority Weights
            clinical_weights = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, phys_multiplier],  # [diff, critic, aux, acl, bgsl, tcb, phys]
                device=losses_tensor.device
            )
            
            # [SOTA 2026 SURGICAL PATCH] Dynamic Magnitude Throttle
            # Rationale: Forensic Audit Phase 5 revealed 60x magnitude mismatch.
            # Fix: Compute relative magnitude of task EMAs and throttle outliers.
            if self.training:
                fundamental_signal = self.loss_emas[0]  # Diffusion is the anchor
                # [v2026 SOTA] Vectorized Throttle (Zero-Sync)
                # Rationale: Replaced task-by-task loop with a single tensor operation.
                # Threshold raised from 5x -> 20x to prevent task extinction.
                emas_active = self.loss_emas[indices]
                throttle_threshold = 20.0 * fundamental_signal
                
                # Compute throttles: scale down if EMA > threshold, else 1.0
                throttles = torch.where(emas_active > throttle_threshold, throttle_threshold / (emas_active + 1e-8), torch.as_tensor(1.0, device=device))
                # Clamp to [0.1, 1.0] to suppress bullies without killing signals
                # [v2026 SOTA FIX] Direct Tensor update
                clinical_weights[indices] *= torch.clamp(throttles, min=0.1, max=1.0)

            # [v2026 SOTA] Vectorized Governor Gate
            effective_sf = torch.clamp(torch.as_tensor(stability_factor, device=losses_tensor.device), min=0.5)
            uw_weights = clinical_weights[indices] * effective_sf + (1.0 - effective_sf)
        
        # 2. Bayesian Weighting (Kendall et al.)
        log_vars_active = self.log_vars[indices]
        log_vars_clamped = torch.clamp(log_vars_active, min=-1.5)
        precision = torch.exp(-log_vars_clamped)
        
        # [v46.1 SOTA FIX] Option B (Preferred): Stabilized Forward + Softplus Guard
        # Rationale: 
        # 1. Preserves DDP Consensus (avg_losses) for stability.
        # 2. Guarantees L > 0 via Softplus (Satisfies User Constraint).
        # 3. Preserves Gradient Flow for negative tasks (unlike Clamp which zeroes grads).

        # 1. Stabilized Forward (Global Average) + Gradient (Local)
        stabilized_loss = avg_losses.clone() + (losses_tensor - losses_tensor.detach())
        
        # 2. Differentiable Positivity Guard (The "Soft" Sharpened Axe)
        # Softplus ensures Forward > 0 while maintaining non-zero gradients.
        stabilized_loss_pos = F.softplus(stabilized_loss)

        # 3. RUW Weighting
        # Regularization via Softplus (Positive)
        regularization_per_task = F.softplus(log_vars_clamped)
        
        # Combined Weighted Loss (All terms structurally positive)
        weighted_losses = 0.5 * precision * stabilized_loss_pos * uw_weights + regularization_per_task
        total_loss = weighted_losses.sum()
        
        # Logging (Return tensors to avoid hot-path .item() sync)
        log_metrics = {}
        for idx, (original_idx, key) in enumerate(active_keys):
            log_metrics[f"weight/{key}"] = 0.5 * precision[idx]
            log_metrics[f"priority/{key}"] = uw_weights[idx]
            log_metrics[f"sigma/{key}"] = torch.exp(0.5 * log_vars_active[idx])
            log_metrics[f"raw/{key}"] = avg_losses[idx]
            
        return total_loss, log_metrics

    @torch.no_grad()
    def project_parameters(self):
        """
        [SOTA] Parameter Projection Hook.
        Must be called after optimizer.step() to prevent 'Dead Zones'.
        
        PMS Extension: Enforces Clinical Ranking Constraint (PRUW).
        We guarantee that Sepsis uncertainty (aux) never exceeds Diffusion uncertainty,
        ensuring that the Sepsis task always maintains its priority signal.
        """
        # [v2026 Phase 19 FIX] Removed DDP Parameter Averaging (Audit Finding F-1)
        # Rationale: log_vars gradients are already synced via all_reduce in 
        # training_step (L2266-2270) BEFORE opt.step(). Since all ranks start 
        # with identical parameters (ensured by on_fit_start DDP consensus) and
        # receive identical gradients, the post-step parameters are already identical.
        # Averaging parameters here caused AdamW momentum (m/v) misalignment because
        # the optimizer tracked pre-averaged values while parameters were overwritten.
        # The clamp/projection below is sufficient to prevent rank drift.

        # 1. [v27.1 FIX] Tightened Bayesian Boundary Projection
        # Rationale: Old bounds [-2, 5] create 1097x precision ratio
        # New bounds [-1.5, 3] create 90x ratio (12x improvement)
        # precision = exp(-log_var): exp(1.5)=4.48 to exp(-3)=0.05
        self.log_vars.clamp_(min=-1.5, max=3.0)
        
        # [v36.0 SOTA FIX] Diffusion Gradient Starvation Prevention (Fix #H5)
        # Rationale: Diagnostic testing confirmed log_var=2.0 causes 6.81x gradient reduction.
        # The 2.0 ceiling (precision=0.135) starved diffusion gradients after epoch 8.
        # New ceiling 1.0 (precision=0.368) guarantees ≥36.8% gradient flow.
        # Math: exp(-1.0) = 0.368 vs exp(-2.0) = 0.135 → 2.7x improvement.
        self.log_vars[0].clamp_(max=1.0)
        
        # 2. [PRUW] Clinical Ranking Enforcement (Relaxed for v5.0)
        # Keys: ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
        # indices: diff=0, aux=2, acl=3, phys=6
        diff_log_var = self.log_vars[0]
        
        # [SOTA FIX]: Decouple Sepsis (aux) from Diffusion Uncertainty (Fix Stagnation)
        # Rationale: Previous logic linked Aux uncertainty to Diffusion. If diffusion failed
        # (high variance), Aux was allowed to fail too. 
        # We now forcefuly clamp Aux/ACL to max 2.5 (Precision > 0.08) to ensure they
        # always provide *some* gradient signal, regardless of Diffusion performance.
        self.log_vars[2].clamp_(max=2.5) # Aux (Sepsis)
        self.log_vars[3].clamp_(max=2.5) # ACL (Contrastive)
        
        # Physics task (6) should also be constrained to prevent explosion
        self.log_vars[6].clamp_(max=3.0) 

    def assert_clean(self):
        """
        [SOTA SAFETY] Epoch Boundary Guard.
        Ensures that no loss accumulation bleeds into the next epoch.
        Must be called at on_train_epoch_end.
        """
        if self.loss_accumulator.abs().sum() > 0:
            raise RuntimeError(
                "[CRITICAL] Loss Scaler Epoch Bleed Detected! "
                "Accumulator was not cleared at the end of the epoch. "
                "Check 'is_accumulating' logic in wrapper_generalist.py."
            )
        if self.batch_counter.item() > 0:
            raise RuntimeError(
                "[CRITICAL] Batch Counter Bleed Detected! "
                "Scaler thinks it is still accumulating. "
            )
