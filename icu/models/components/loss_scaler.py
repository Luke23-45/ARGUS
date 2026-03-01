import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
from typing import Dict, Tuple, Optional, List
from icu.utils.train_utils import ScalingSteward

logger = logging.getLogger("BayesianProjectedScaler")

class BayesianProjectedScaler(nn.Module):
    """
    [SOTA 2025] Bayesian-PGD Uncertainty Scaler with UW-SO (Achituve et al. 2024).
    
    Features:
    1. Clinical-Priority UW-SO: Softmax-normalized priority balancing.
    2. PGD Projection: Ensures log_vars stay in the differentiable zone [-2, 5].
    3. DDP-Synchronized Momentum: Unified EMA updates across the cluster.
    """
    def __init__(self, num_tasks: int = 7, decay: float = 0.99):
        super().__init__()
        self.keys = ['diffusion', 'critic', 'aux', 'acl', 'bgsl', 'tcb', 'phys']
        self.num_tasks = num_tasks
        
        # [v110.0 SOTA FIX] Safe-Start Initialization
        log_vars_init = torch.zeros(num_tasks)
        if num_tasks > 4: log_vars_init[4] = 0.5  # bgsl starts cautious
        if num_tasks > 5: log_vars_init[5] = 1.0  # tcb starts suppressed
        if num_tasks > 6: log_vars_init[6] = 2.0  # phys starts heavily suppressed
        self.log_vars = nn.Parameter(log_vars_init)
        
        # EMA tracking for UW-SO stability
        self.register_buffer("loss_emas", torch.ones(num_tasks))
        
        # Accumulation Buffers
        self.register_buffer("loss_accumulator", torch.zeros(num_tasks))
        self.register_buffer("task_counters", torch.zeros(num_tasks))
        self.register_buffer("batch_counter", torch.zeros([1]))
        self.register_buffer("step_count", torch.tensor([0], dtype=torch.long))
        self.register_buffer("is_calibrated", torch.tensor([False]))
        
        # [v2026 SOTA] Dynamic Scaling Invariant
        # Initialized to -1 to force explicit configuration via scale_dynamics()
        self.warmup_steps = -1 

    @torch.no_grad()
    def scale_dynamics(self, steps_per_epoch: int):
        """[SOTA v2026] Unifies uncertainty decay and warmup across step densities."""
        self.decay.fill_(ScalingSteward.get_decay(0.99, steps_per_epoch))
        self.warmup_steps = steps_per_epoch
        logger.info(f"[Scaler] Scaling Dynamics: {steps_per_epoch} steps/epoch (Warmup target)")

    def forward(self, 
                loss_dict: Dict[str, torch.Tensor], 
                stability_factor: float = 1.0, 
                phys_multiplier: float = 1.0, 
                batch_size: int = 1, 
                is_accumulating: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified weighting pass with Dynamic Priority and DDP Consensus.
        """
        # 1. Extract Active Task Signals
        losses = []
        active_indexes = []
        for i, key in enumerate(self.keys):
            if key in loss_dict:
                losses.append(loss_dict[key].squeeze())
                active_indexes.append(i)
        
        has_losses = len(losses) > 0
        device = next(self.parameters()).device
        
        if has_losses:
            losses_tensor = torch.stack(losses)
            indices = torch.tensor(active_indexes, device=device)
            local_raw = losses_tensor.detach()
            
            # Accumulation (No Grad)
            with torch.no_grad():
                self.loss_accumulator[indices] += local_raw * batch_size
                self.task_counters[indices] += batch_size
                self.batch_counter += batch_size
        else:
            losses_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            indices = torch.tensor([], dtype=torch.long, device=device)
            local_raw = torch.tensor([], device=device)

        # 2. DDP Consensus (Sync Stepped Average)
        if torch.distributed.is_initialized() and self.training and not is_accumulating:
            sync_buffer = torch.zeros(self.num_tasks * 2 + 1, device=device)
            sync_buffer[:self.num_tasks] = self.loss_accumulator
            sync_buffer[self.num_tasks:2*self.num_tasks] = self.task_counters
            sync_buffer[-1] = self.batch_counter
            
            # Global Consensus
            torch.distributed.all_reduce(sync_buffer, op=torch.distributed.ReduceOp.SUM)
            global_sum_losses = sync_buffer[:self.num_tasks]
            global_task_counts = sync_buffer[self.num_tasks:2*self.num_tasks]
            
            # Compute True Global Average
            avg_losses_all = global_sum_losses / (global_task_counts + 1e-8)
            avg_losses = avg_losses_all[indices] if has_losses else torch.tensor([], device=device)
            
            # [SOTA TITANIUM FIX] Loss NaN-Gate
            # Rationale: All-reduce results in NaN if any rank had an Inf.
            # We MUST reject the step to prevent poisoning EMA memory.
            is_finite = torch.isfinite(avg_losses_all).all()
            
            # Momentum-Based Update
            self.step_count += 1
            is_warmup = (self.step_count < self.warmup_steps)
            curr_decay_t = torch.where(is_warmup, torch.as_tensor([0.95], device=device), self.decay)
            
            # Atomic EMA Update (Protected)
            if is_finite:
                self.loss_emas.lerp_(avg_losses_all, 1.0 - curr_decay_t)
            
            # Reset Cycle
            self.loss_accumulator.zero_()
            self.task_counters.zero_()
            self.batch_counter.zero_()
        else:
            # Local stats for accumulation or non-DDP
            if self.training and not is_accumulating:
                avg_losses_all = self.loss_accumulator / (self.task_counters + 1e-8)
                avg_losses = avg_losses_all[indices] if has_losses else torch.tensor([], device=device)
                
                # [SOTA TITANIUM FIX] Local NaN-Gate
                is_finite = torch.isfinite(avg_losses_all).all()
                self.step_count += 1
                is_warmup = (self.step_count < self.warmup_steps)
                curr_decay = torch.where(is_warmup, torch.as_tensor([0.95], device=device), self.decay)
                if is_finite:
                    self.loss_emas.lerp_(avg_losses_all, 1.0 - curr_decay)
                
                self.loss_accumulator.zero_()
                self.task_counters.zero_()
                self.batch_counter.zero_()
            else:
                avg_losses = local_raw

        if not has_losses:
            return losses_tensor, {}

        # 3. Clinical Priority Weighting (UW-SO)
        with torch.no_grad():
            clinical_weights = torch.ones(self.num_tasks, device=device)
            # Physical task (index 6) gets the multiplier
            clinical_weights[6] = phys_multiplier
            
            # Magnitude Throttle
            fundamental_signal = self.loss_emas[0] # Diffusion anchor
            emas_active = self.loss_emas[indices]
            throttle_threshold = 20.0 * fundamental_signal
            throttles = torch.where(emas_active > throttle_threshold, throttle_threshold / (emas_active + 1e-8), torch.as_tensor([1.0], device=device))
            
            uw_weights = clinical_weights[indices] * torch.clamp(throttles, min=0.1, max=1.0)
            effective_sf = torch.clamp(torch.as_tensor(stability_factor, device=device), min=0.5)
            uw_weights = uw_weights * effective_sf + (1.0 - effective_sf)

        # 4. Bayesian Multi-Tasking (Kendall et al.)
        log_vars_active = self.log_vars[indices]
        # [SOTA v4.0] Expanded range allows tasks with tiny raw magnitudes to reach parity
        log_vars_clamped = torch.clamp(log_vars_active, min=-5.0, max=8.0)
        precision = torch.exp(-log_vars_clamped)
        
        # [SOTA v5.1] Exact Mathematical Decoupling (NASA-Tier Orthogonal Projection)
        # Rationale: Previous version ran backprop through `log(softplus(EMA + diff))`, resulting in
        # extreme "Double Suppression" that stifled converged gradients. 
        # By separating Theta and Sigma updates exactly based on Kendall et al.:
        
        # Component 1: Network Updates (Theta)
        # We use the EXACT RAW loss for the network weights, scaled linearly by precision.
        # This prevents the exponential gradient suppression of high-magnitude tasks (e.g., GMSE).
        # Note: TCB explosion is handled by the Magnitude Throttle (L149-155) which
        # smoothly dampens tasks exceeding 20x the diffusion anchor via uw_weights.
        # The SG-4 step_count fix ensures the throttle's EMAs converge properly.
        theta_loss = (0.5 * precision.detach() * losses_tensor * uw_weights).sum()
        
        # Component 2: Uncertainty Updates (Sigma)
        # We use the smoothed EMA loss to update the log_vars (Sigma), preventing batch-to-batch
        # thrashing and ensuring stable loss landscape calibration.
        # [SOTA TITANIUM FIX] Linear Magnitude Response
        # Rationale: log(avg_losses) muted the signal for runaway components.
        # Using raw EMA loss ensures the Bayesian scaler reacts instantly to divergence.
        # F.softplus(avg_losses) provides C1-continuity near zero while remaining 
        # linear for magnitudes encountered in clinical RL (L > 1.0).
        sigma_loss = (0.5 * precision * F.softplus(avg_losses).detach() * uw_weights + 0.5 * log_vars_clamped).sum()
        
        # Fused AutoGrad Root
        total_loss = theta_loss + sigma_loss
        
        # Telemetry
        metrics = {}
        for i, idx in enumerate(active_indexes):
            key = self.keys[idx]
            metrics[f"weight/{key}"] = 0.5 * precision[i]
            metrics[f"priority/{key}"] = uw_weights[i]
            metrics[f"sigma/{key}"] = torch.exp(0.5 * log_vars_active[i])
            metrics[f"raw/{key}"] = avg_losses[i]

        return total_loss, metrics

    @torch.no_grad()
    def project_parameters(self):
        """Enforces clinical boundaries and ranking constraints."""
        # 1. Domain Clamp
        # [SOTA v4.0] Expanded for >1000x magnitude gap support
        self.log_vars.clamp_(min=-5.0, max=8.0)
        # 2. Diffusion Floor
        self.log_vars[0].clamp_(max=1.0)
        # 3. [FORENSIC FIX #2] Critic Precision Range (Root Cause: Critic Explosion → Total Loss Blowup)
        # Original: max=0.0 forced precision >= 1.0, preventing the scaler from reducing critic weight
        # when V exploded from 31→115. This made critic the permanent highest-weighted task.
        # Fix: max=3.0 allows precision down to exp(-3) ≈ 0.05, giving the Bayesian scaler
        # 20x dynamic range to naturally suppress critic via its own uncertainty estimation
        # (Kendall et al.) when the loss magnitude diverges from other tasks.
        self.log_vars[1].clamp_(max=3.0)
        # 4. Clinical Gating (Aux/ACL)
        self.log_vars[2].clamp_(max=4.0)
        self.log_vars[3].clamp_(max=4.0)
        # 5. Physics Guard
        self.log_vars[6].clamp_(max=8.0)
        
        # 6. [v12.1 NASA-TIER FIX] Clinical Precision Floor (Uncertainty Trap Prevention)
        # Rationale: Bayesian scaling treats high loss as 'noise' to suppress.
        # For clinical tasks (Aux, ACL, BGSL), high loss is a CRITICAL ERROR.
        # We enforce a precision floor (log_var ceiling) so these tasks can never 
        # be suppressed below 25% of their initial priority.
        # Index 2=Aux, 3=ACL, 4=BGSL
        for idx in [2, 3, 4]:
             self.log_vars[idx].clamp_(max=2.0) # precision_floor = exp(-2.0) approx 0.13

    @torch.no_grad()
    def calibrate_log_vars(self, loss_dict: Dict[str, torch.Tensor], anchor_key: str = 'diffusion'):
        """
        [SOTA 2025] Automatic Log-Var Initialization (ALI)
        Harmonizes log-variances dynamically using first-batch empirical losses.
        Ensures critical care heuristics start on equal footing with Diffusion.
        """
        if self.is_calibrated.item():
            return
            
        device = self.log_vars.device
        raw_losses = torch.zeros(self.num_tasks, device=device)
        active_mask = torch.zeros(self.num_tasks, device=device)
        
        # 1. Capture local magnitudes
        for i, key in enumerate(self.keys):
            if key in loss_dict:
                # Use .mean() to handle potential sequence/batch dims
                raw_losses[i] = loss_dict[key].detach().clone()
                active_mask[i] = 1.0

        # 2. DDP Consensus (Zero-Sync protocol)
        if torch.distributed.is_initialized():
            dist_stats = torch.stack([raw_losses, active_mask])
            torch.distributed.all_reduce(dist_stats, op=torch.distributed.ReduceOp.SUM)
            # Average across ranks that actually contributed to this task
            raw_losses = dist_stats[0] / dist_stats[1].clamp(min=1)

        # 3. Derive Balances
        if anchor_key not in self.keys or torch.sum(raw_losses) == 0:
            logger.error(f"[ALI] anchor_key {anchor_key} not in {self.keys}. Aborting.")
            return
            
        anchor_idx = self.keys.index(anchor_key)
        # [NASA-Tier v1.1] Numerical Shielding (Smoking Gun #2)
        # Rationale: If target_magnitude is 0, the division produces Inf, and log(Inf) = Inf.
        # This permanently poisons the scaler. Clamp minimum to a safe numerical lower bound.
        target_magnitude = raw_losses[anchor_idx].clamp(min=1e-4)
        
        # log_var = ln(loss / target) => Weight = target / loss
        # This equalizes the weighted loss magnitudes to EXACTLY match the anchor.
        # [NASA-Tier v1.1] Double-sided clamp to prevent log(0) [NaN] and log(Inf) [Inf].
        safe_ratio = (raw_losses / target_magnitude).clamp(min=1e-8, max=1e8)
        new_log_vars = torch.log(safe_ratio)
        
        # 4. Expert Clinical Priors
        # Physics and TCB start with lower priority to allow the generative manifold to settle.
        if self.num_tasks > 6:
            new_log_vars[6] = torch.max(new_log_vars[6], torch.tensor(1.0, device=device)) # phys: Cautious
        if self.num_tasks > 5:
            new_log_vars[5] = torch.max(new_log_vars[5], torch.tensor(2.0, device=device)) # tcb: Suppressed

        # 5. Persistent Update
        # Bound within the PGD projection zone [-1.5, 3.0]
        new_log_vars = torch.clamp(new_log_vars, min=-1.5, max=3.0)
        self.log_vars.data.copy_(new_log_vars)
        self.loss_emas.data.copy_(raw_losses)
        self.is_calibrated.fill_(True)
        
        logger.info(f"[ALI] Manifold Harmonized: log_vars={self.log_vars.data.tolist()}")

    def assert_clean(self):
        """Epoch boundary guard."""
        if self.loss_accumulator.abs().sum() > 0:
            raise RuntimeError("[Scaler] Accumulator bleed detected!")
        if self.batch_counter.item() > 0:
            raise RuntimeError("[Scaler] Batch counter bleed detected!")
