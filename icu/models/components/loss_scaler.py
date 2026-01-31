import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
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
        self.decay = decay
        
        # [v177.1 SOTA] Accumulation Buffers
        # Rationale: Accumulate raw losses across sub-batches to provide 
        # uncertainty EMAs with the true cycle average (Fixes Smoking Gun #177/214).
        self.register_buffer("loss_accumulator", torch.zeros(num_tasks))
        self.register_buffer("task_counters", torch.zeros(num_tasks))
        self.register_buffer("batch_counter", torch.zeros(1))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def scale_dynamics(self, n_curr: int):
        """[SOTA v2026] Unifies uncertainty decay across step densities."""
        if n_curr <= 0: return
        self.decay = ScalingSteward.get_decay(0.99, n_curr)
        
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
            global_batch_size = sync_buffer[-1].item()
            
            # 3. Compute True Global Average for the entire cycle
            avg_losses_all = global_sum_losses / (global_task_counts + 1e-8)
            avg_losses = avg_losses_all[indices] if has_losses else torch.tensor([], device=device)
            
            # 4. Momentum-Based Bayesian Update (Dynamic Inertia #110B)
            # EMA now sees the clean, aggregated manifold state of the ENTIRE cycle.
            # Rationale: Lower decay (0.9) during warmup (200 steps) allows 10x faster adaptation.
            self.step_count += 1
            curr_decay = 0.90 if self.step_count < 200 else self.decay
            self.loss_emas.mul_(curr_decay).add_(avg_losses_all, alpha=1 - curr_decay)
            
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
                self.step_count += 1
                curr_decay = 0.90 if self.step_count < 200 else self.decay
                self.loss_emas.mul_(curr_decay).add_(avg_losses_all, alpha=1 - curr_decay)
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
                [0.5, 0.5, 1.5, 1.5, 1.0, 1.0, phys_multiplier],  # [diff, critic, aux, acl, bgsl, tcb, phys]
                device=losses_tensor.device
            )
            # [v27.1 FIX] Apply Adaptive Governor with Floor
            effective_sf = max(0.5, stability_factor)
            uw_weights = clinical_weights[indices] * effective_sf + (1.0 - effective_sf)
        
        # 2. Bayesian Weighting (Kendall et al.)
        log_vars_active = self.log_vars[indices]
        log_vars_clamped = torch.clamp(log_vars_active, min=-1.5)
        precision = torch.exp(-log_vars_clamped)
        
        # 2. Kendall et al. Regularization: L_reg = sum(0.5 * log_var)
        regularization = 0.5 * log_vars_clamped.sum()
        
        # [PATCH 1] Restrained Uncertainty Weighting (RUW)
        weighted_losses = 0.5 * (precision * (losses_tensor - losses_tensor.detach() + avg_losses) * uw_weights) + regularization
        total_loss = weighted_losses.sum()
        
        # Logging
        log_metrics = {}
        for idx, (original_idx, key) in enumerate(active_keys):
            log_metrics[f"weight/{key}"] = 0.5 * precision[idx].item()
            log_metrics[f"priority/{key}"] = uw_weights[idx].item()
            log_metrics[f"sigma/{key}"] = torch.exp(0.5 * log_vars_active[idx]).item()
            # Log the raw loss used for the current sub-batch calculation
            log_metrics[f"raw/{key}"] = avg_losses[idx].item()
            
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
        # 1. [v27.1 FIX] Tightened Bayesian Boundary Projection
        # Rationale: Old bounds [-2, 5] create 1097x precision ratio
        # New bounds [-1.5, 3] create 90x ratio (12x improvement)
        # precision = exp(-log_var): exp(1.5)=4.48 to exp(-3)=0.05
        self.log_vars.clamp_(min=-1.5, max=3.0)
        
        # 2. [PRUW] Clinical Ranking Enforcement
        # Keys: ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
        # indices: diff=0, aux=2, acl=3, phys=6
        diff_log_var = self.log_vars[0]
        
        # Sepsis tasks (aux, acl) must be at least as certain as the foundation
        # log_var_aux <= log_var_diff
        self.log_vars[2].clamp_(max=diff_log_var.item())
        self.log_vars[3].clamp_(max=diff_log_var.item())
        
        # Physics task (6) should also be constrained to prevent explosion
        self.log_vars[6].clamp_(max=5.0) 
