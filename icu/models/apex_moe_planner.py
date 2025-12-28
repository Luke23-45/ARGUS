"""
icu/models/apex_moe_planner.py
--------------------------------------------------------------------------------
APEX-MoE: Adaptive Phase-Locked Mixture-of-Experts Planner (Ultimate v12.0).

Status: SAFETY-CRITICAL / PRODUCTION-READY
Architecture: Tri-Phase Specialist (Stable -> Pre-Shock -> Shock).

"Like a team of specialized physicians, each expert focuses on what
they do best, with seamless handoffs between them."

This model transforms a Generalist DiT (Diffusion Transformer) into a team of 
specialized experts. It uses "Phase-Locking" to ensure that the definition of 
clinical states remains consistent while the experts specialize in their 
respective dynamics.

Upgrades (Ultimate v12.0 - Guardian Supreme):
1.  **Physics-Guided MoE Sampling**: Active gradient steering during inference
    to enforce biological constraints in real-time.
2.  **Self-Conditioning Integration**: Two-pass "Analog Bits" mechanism during
    both training and inference for improved prediction refinement.
3.  **Physiological Consistency Loss**: Experts are trained with soft-penalties
    for violating biological "Hard Decks" (e.g., MAP < 65, HR > 180).
4.  **GT-Masked Routing (v4.1)**: Sub-phase specialization where each clinical
    phase (Stable/Pre-Shock/Shock) can have multiple sub-experts.
5.  **Expert Diversity Loss**: Prevents routing collapse by encouraging
    different experts to produce distinct predictions.
6.  **Loss-Free Load Balancing Option**: Dynamically adjusts expert biases
    without auxiliary loss gradients (SOTA 2024).
7.  **Temporal Smoothness Regularization**: Penalizes sudden jumps in predicted
    vitals to ensure realistic clinical trajectories.
8.  **Acuity-Aware Router Training**: Boosts learning signal for rare
    sepsis/shock cases (only ~3% of dataset).
9.  **Expert Capacity Monitoring**: Tracks token distribution across experts
    for debugging and optimization.
10. **Comprehensive Telemetry**: Full diagnostic output for training analysis.

Key Mechanisms:
1.  **Bootstrapping**: Clones experts from a converged Generalist to skip
    early training instability.
2.  **Phase-Locked Perception**: Freezes the History Encoder to maintain
    consistent clinical state definitions.
3.  **Gradient Surgery**: Hard-gates training data so Expert K *only* sees
    Phase K patients, enabling true specialization.
4.  **Chain Tethering**: Regularizes adjacent experts (0<->1, 1<->2) to
    ensure smooth transitions between clinical states.
5.  **Soft-Gated Inference**: Blends expert predictions based on Router
    probability masses using Top-K sparse routing.

References:
    - Fedus et al., "Switch Transformers" (Load Balancing, Capacity Factor)
    - DeepSeek-AI, "DeepSeekMoE" (Fine-grained experts, auxiliary losses)
    - Chen et al., "Analog Bits" (Self-Conditioning)
    - Sepsis-3 Consensus (2016) (Clinical phase definitions)
    - Peng et al., "AWR" (Advantage-weighted regression)

Dependencies:
    - icu.models.diffusion.ICUUnifiedPlanner (The Donor/Generalist)
"""

from __future__ import annotations

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional, Tuple, List, Union
from omegaconf import DictConfig

# Base Model Import
from icu.models.diffusion import ICUUnifiedPlanner, PhysiologicalConsistencyLoss

# Setup Logger
logger = logging.getLogger("APEX_MoE_Ultimate")

# =============================================================================
# APEX-MoE CLASS DEFINITION
# =============================================================================

class APEX_MoE_Planner(nn.Module):
    """
    The Tri-Phase Specialist Model (Ultimate Edition v12.0).
    
    Training (Forward):
        - Uses 'Gradient Surgery': Batch is physically split by Ground Truth Phase.
        - Expert 0 sees Stable, Expert 1 sees Pre-Shock, Expert 2 sees Shock.
        - With 6 experts (2 per phase), GT-Masked Routing selects sub-experts.
        
    Inference (Sample):
        - Uses 'Soft Gating': Router predicts probabilities P(Phase).
        - Top-K experts are activated and their outputs are blended.
        - Physics-Guided Sampling steers trajectories toward biological plausibility.
        - Self-Conditioning refines predictions across diffusion steps.
    
    Attributes:
        encoder: Frozen history encoder from the Generalist.
        router: Trainable classification head for expert routing.
        experts: ModuleList of N specialized diffusion backbones.
        value_head: Trainable critic for AWR.
        scheduler: Diffusion noise scheduler.
        normalizer: Clinical value normalizer.
        phys_loss: Physiological consistency loss function.
    """
    
    def __init__(
        self, 
        generalist_model: ICUUnifiedPlanner, 
        phase_weights: Optional[List[float]] = None,
        lambda_reg: float = 0.01,
        lambda_lb: float = 0.01,           # Load-balancing coefficient
        lambda_diversity: float = 0.001,   # Expert diversity coefficient
        use_loss_free_balancing: bool = False, # SOTA 2024: Bias-based balancing
        aux_loss_scale: float = 0.1        # Router classification loss scale
    ):
        super().__init__()
        logger.info(f"Initializing APEX-MoE Specialist (Ultimate v12.0: {generalist_model.cfg.num_phases} Experts)...")
        
        # =====================================================================
        # 1. BOOTSTRAPPING (Organ Harvesting from Generalist)
        # =====================================================================
        # We steal the pre-trained components from the converged generalist.
        # This skips early training instability and provides a strong prior.
        
        # Configuration & Hyperparameters
        self.cfg = generalist_model.cfg
        if self.cfg.num_phases < 2:
            raise ValueError(f"APEX-MoE requires at least 2 phases, found {self.cfg.num_phases}")
        
        logger.info(f" - Configuration: {self.cfg.num_phases} Experts | d_model={self.cfg.d_model}")
        
        # Core Components
        self.encoder = generalist_model.encoder       # History understanding
        self.scheduler = generalist_model.scheduler   # Noise scheduling
        self.normalizer = generalist_model.normalizer # Value normalization
        
        # Router (Auxiliary Head) - Critical for phase prediction
        if not hasattr(generalist_model, 'aux_head') or generalist_model.aux_head is None:
            raise RuntimeError("Generalist model has no Auxiliary Head (Router). Cannot build MoE.")
        self.router = generalist_model.aux_head
        
        # Value Head (Critic) - Critical for AWR advantage calculation
        self.value_head = generalist_model.value_head
        
        # Physics Loss (Fresh instance for clean state)
        self.phys_loss = PhysiologicalConsistencyLoss()
        
        # RoPE (If shared across layers)
        self.history_rope = getattr(generalist_model, 'history_rope', None)
        
        # =====================================================================
        # 2. PHASE-LOCKING (Freeze Perception Components)
        # =====================================================================
        # The "Eyes" (Encoder) must remain fixed so the "Hands" (Experts) train
        # against a stable clinical reality. This ensures consistent phase
        # definitions across all experts.
        
        # Freeze Encoder - Clinical state perception is fixed
        self.encoder.requires_grad_(False)
        
        # Trainable Components:
        # - Router: Learns sub-phase specialization (v4.1 feature)
        # - Value Head: Adapts critic to specialist dynamics
        self.router.requires_grad_(True)
        self.value_head.requires_grad_(True)
        
        if self.history_rope:
            self.history_rope.requires_grad_(False)
            
        # Put frozen parts in eval mode (disables dropout)
        self.encoder.eval()
        self.router.train()
        self.value_head.train()
        
        logger.info(" - Encoder FROZEN. Router/Value Head TRAINABLE (Sub-Phase Mode).")
        
        # =====================================================================
        # 3. CLONING (Expert Forking from Generalist)
        # =====================================================================
        # Create N independent copies of the Diffusion Backbone.
        # Each expert will specialize in a specific clinical phase.
        
        logger.info(f" - Cloning Generalist Backbone into {self.cfg.num_phases} Experts...")
        self.experts = nn.ModuleList([
            copy.deepcopy(generalist_model.backbone) 
            for _ in range(self.cfg.num_phases)
        ])
        
        # =====================================================================
        # 4. REGULARIZATION HYPERPARAMETERS
        # =====================================================================
        self.lambda_reg = lambda_reg           # Chain tethering strength
        self.lambda_lb = lambda_lb             # Load-balancing strength
        self.lambda_diversity = lambda_diversity  # Expert diversity strength
        self.use_loss_free_balancing = use_loss_free_balancing
        self.aux_loss_scale = aux_loss_scale
        
        # Loss-Free Balancing: Expert-wise bias buffer (SOTA 2024)
        # These biases are dynamically updated to guide routing
        if use_loss_free_balancing:
            self.register_buffer("expert_bias", torch.zeros(self.cfg.num_phases))
            self.bias_update_rate = 0.01  # Smooth update rate
        
        # Phase Weights for Loss Scaling
        # High-acuity phases (Pre-Shock, Shock) get higher weight to balance
        # the class imbalance (Stable ~85%, Pre-Shock ~12%, Shock ~3%)
        if phase_weights is None:
            # Auto-balance: Phase 0 is 1.0, others scale up
            self.phase_weights = [1.0] + [5.0] * (self.cfg.num_phases - 1)
        else:
            if len(phase_weights) != self.cfg.num_phases:
                raise ValueError(f"Phase weights mismatch. Expected {self.cfg.num_phases}, Got {len(phase_weights)}")
            self.phase_weights = phase_weights
            
        self.register_buffer("loss_weights", torch.tensor(self.phase_weights))
        
        logger.info(f" - Phase Weights: {self.phase_weights}")
        logger.info(f" - Regularization: lambda_reg={lambda_reg}, lambda_lb={lambda_lb}, lambda_div={lambda_diversity}")
        logger.info(f" - Loss-Free Balancing: {use_loss_free_balancing}")

    def train(self, mode: bool = True):
        """
        Robust Train Mode.
        Ensures frozen components stay in EVAL mode even when model.train() is called.
        This prevents dropout from activating in the encoder.
        """
        super().train(mode)
        # Force frozen parts to eval
        self.encoder.eval()
        # Trainable parts follow the mode
        if mode:
            self.router.train()
            self.value_head.train()
            for expert in self.experts:
                expert.train()
        return self

    # =========================================================================
    # NORMALIZATION WRAPPERS
    # =========================================================================

    def normalize(self, x_ts: torch.Tensor, x_static: Optional[torch.Tensor] = None):
        """Wrapper for the safety-patched ClinicalNormalizer."""
        return self.normalizer.normalize(x_ts, x_static)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization for clinical interpretability."""
        return self.normalizer.denormalize(x)

    # =========================================================================
    # TRAINING FORWARD PASS (Gradient Surgery)
    # =========================================================================

    def forward(
        self, 
        batch: Dict[str, torch.Tensor],
        awr_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        The "Gradient Surgery" Forward Pass with GT-Masked Routing.
        
        This is the core training logic for the MoE specialist. It ensures that:
        1. Each expert only sees patients from its specialized phase.
        2. The router learns to predict sub-phase assignments within GT constraints.
        3. Load balancing prevents expert collapse.
        4. Chain tethering ensures smooth transitions between phases.
        
        Args:
            batch: Dictionary containing observed_data, future_data, static_context,
                   phase_label/outcome_label, and optional src_mask.
            awr_weights: Optional per-sample AWR weights for advantage weighting.
        
        Returns:
            Dictionary containing loss, component losses, and telemetry.
        """
        if not self.training:
            return self.forward_inference(batch)
            
        # --- 1. Unpack & Normalize ---
        past = batch["observed_data"]
        fut = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # Pull AWR weights from batch if not passed explicitly
        awr_weights = awr_weights if awr_weights is not None else batch.get("awr_weights", None)
        
        # Robust Phase Label Resolution
        # Priority: phase_label > outcome_label (mapped)
        if "phase_label" in batch:
            gt_phases = batch["phase_label"].long()  # Values in {0, 1, 2}
        elif "outcome_label" in batch:
            gt = batch["outcome_label"].long()
            gt_phases = torch.zeros_like(gt)
            gt_phases[gt >= 1] = 2  # Map positive outcome to Shock (Phase 2)
        else:
            raise ValueError("Batch missing both 'phase_label' and 'outcome_label'. Cannot route experts.")

        past_norm, static_norm = self.normalize(past, static)
        fut_norm, _ = self.normalize(fut, None)
        
        B = past.shape[0]
        device = past.device
        N_GT_PHASES = 3  # Dataset provides 3 clinical phases
        N_SUB_EXPERTS = max(1, self.cfg.num_phases // N_GT_PHASES)  # Sub-experts per phase
        
        # Defensive validation
        assert gt_phases.max() < N_GT_PHASES, \
            f"Invalid phase label detected: max={gt_phases.max()}, expected < {N_GT_PHASES}"
        
        # --- 2. Shared Perception (Encoder is frozen) ---
        # [v16.0] FIX: Disable unsafe padding inference. Fixed window dataset has no padding.
        padding_mask = None
        
        with torch.no_grad():
            ctx_seq, global_ctx, ctx_mask = self.encoder(
                past_norm, static_norm, 
                imputation_mask=src_mask,
                padding_mask=padding_mask
            )

        # --- 3. Router Decision (Trainable) ---
        router_logits = self.router(global_ctx)  # [B, num_experts]
        
        # Apply Loss-Free Balancing bias if enabled
        if self.use_loss_free_balancing:
            # Add learnable bias to logits to guide routing
            router_logits = router_logits + self.expert_bias.unsqueeze(0)
        
        router_probs = F.softmax(router_logits, dim=-1)
        
        # --- 4. GT-Masked Routing ---
        # For each sample, select an expert from the allowed set for its GT phase.
        # Phase L (GT) -> allowed experts {L, L+3, L+6, ...}
        # Router picks the highest-probability one within this set.
        
        selected_experts = torch.zeros(B, dtype=torch.long, device=device)
        
        for gt_phase in range(N_GT_PHASES):
            mask = (gt_phases == gt_phase)
            if not mask.any():
                continue
            
            # Compute allowed experts for this GT phase
            # E.g., Phase 0 -> Experts {0, 3}, Phase 1 -> Experts {1, 4}, etc.
            allowed_experts = [gt_phase + i * N_GT_PHASES for i in range(N_SUB_EXPERTS)]
            allowed_experts = [exp for exp in allowed_experts if exp < self.cfg.num_phases]
            
            # Get router probabilities for allowed sub-experts
            sub_probs = router_probs[mask][:, allowed_experts]  # [N_mask, N_sub]
            sub_winner_idx = torch.argmax(sub_probs, dim=1)     # [N_mask]
            
            # Map back to absolute expert indices
            selected_experts[mask] = torch.tensor(allowed_experts, device=device)[sub_winner_idx]
        
        # --- 5. Expert Execution Loop (Sparse) ---
        total_loss = torch.tensor(0.0, device=device)
        
        # Phase-wise losses for monitoring
        loss_stable = torch.tensor(0.0, device=device)
        loss_preshock = torch.tensor(0.0, device=device)
        loss_crash = torch.tensor(0.0, device=device)
        
        # Telemetry
        batch_counts = {}
        expert_load = torch.zeros(self.cfg.num_phases, device=device)
        
        # Common Diffusion Params
        # [v15.2.1] SOTA: Support external t/noise for synchronized two-pass training
        t = batch.get("t", torch.randint(0, self.cfg.timesteps, (B,), device=device))
        noise_eps = batch.get("noise_eps", None)
        
        if noise_eps is None:
            noisy_fut, noise_eps = self.scheduler.add_noise(fut_norm, t)
        else:
            # Reconstruct noisy input using provided noise
            alpha_bar = self.scheduler.alphas_cumprod[t][:, None, None]
            noisy_fut = torch.sqrt(alpha_bar) * fut_norm + torch.sqrt(1 - alpha_bar) * noise_eps
        
        # Self-Conditioning: Check if wrapper provided a first-pass prediction
        # [v15.1] SOTA Parity Fix: Pass first_pass_pred into experts
        self_cond = batch.get("first_pass_pred", None)
        if self_cond is None:
            self_cond = torch.zeros_like(noisy_fut)
        else:
            self_cond = self_cond.detach()

        # Buffers for reconstruction
        full_pred_noise = torch.zeros_like(noisy_fut)
        full_pred_x0 = torch.zeros_like(noisy_fut)

        # Loop over experts that have at least one sample
        for expert_idx in range(self.cfg.num_phases):
            idx_mask = (selected_experts == expert_idx)
            indices = torch.nonzero(idx_mask).squeeze(-1)
            
            count = indices.numel() if indices.dim() > 0 else (1 if len(indices) > 0 else 0)
            batch_counts[f"count_expert_{expert_idx}"] = float(count)
            expert_load[expert_idx] = float(count)
            
            if count == 0:
                continue
                
            # Slicing batch for this expert
            sub_noisy = noisy_fut[indices]
            sub_t = t[indices]
            sub_ctx_seq = ctx_seq[indices]
            sub_global_ctx = global_ctx[indices]
            sub_ctx_mask = ctx_mask[indices] if ctx_mask is not None else None
            sub_self_cond = self_cond[indices]
            
            # Expert Forward Pass
            expert = self.experts[expert_idx]
            pred_noise = expert(
                sub_noisy, sub_t, 
                sub_ctx_seq, sub_global_ctx, 
                sub_ctx_mask,
                self_cond=sub_self_cond
            )

            # [v15.0] Differentiable Routing (The "Switch Transformer" Trick)
            # Scale output by prob/prob.detach() to allow gradients to flow to router
            # The expert loss now provides feedback on whether this expert was a "good choice"
            if self.training:
                # Get router probability for this specific expert on these samples
                # indices refers to batch indices routed here
                prob_selected = router_probs[indices, expert_idx]  # [sub_batch]
                
                # Scale: Forward=1.0, Backward=1.0/prob
                # We want: dL/dProb = dL/dOut * dOut/dProb
                # out = out_raw * prob
                # But we don't want to scale the magnitude of the prediction heavily during training
                # So we use the "pass-through" scaling: out = out_raw * (prob / prob.detach())
                # [v15.1] CLAMPED SCALE for router stability
                scale = prob_selected / (prob_selected.detach().clamp(min=1e-3))
                pred_noise = pred_noise * scale.view(-1, 1, 1)
            
            # Reconstruct x0 estimate (for self-conditioning and physiological monitoring)
            alpha_bar = self.scheduler.alphas_cumprod[sub_t][:, None, None]
            sub_x0 = (sub_noisy - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            
            # [v15.2.2] SOTA: Keep x0 differentiable to allow physiological pressure on experts/router
            full_pred_x0[indices] = sub_x0
            
            # [v15.2.2] Store noise for reconstruction
            full_pred_noise[indices] = pred_noise.detach()

            # Loss Computation (Per-sample for AWR weighting)
            loss_raw = F.mse_loss(pred_noise, noise_eps[indices], reduction='none').mean(dim=[1, 2])
            
            # Apply AWR weights if provided
            if awr_weights is not None:
                w_sub = awr_weights[indices]
                loss_mse = (loss_raw * w_sub).mean()
            else:
                loss_mse = loss_raw.mean()
            
            # Phase-based weighting
            # Map expert to base phase: expert % 3 -> {0, 1, 2}
            base_phase = expert_idx % N_GT_PHASES
            p_weight = self.loss_weights[base_phase]
            weighted_loss = loss_mse * p_weight
            
            # Accumulate phase-specific losses
            if base_phase == 0:
                loss_stable = loss_stable + loss_mse
            elif base_phase == 2:
                loss_crash = loss_crash + loss_mse
            else:
                loss_preshock = loss_preshock + loss_mse
            
            # Add to total (scaled by batch proportion to maintain magnitude)
            if B > 0:
                total_loss = total_loss + weighted_loss * (count / B)
            
            # Telemetry
            batch_counts[f"loss_expert_{expert_idx}"] = float(loss_mse.item())

        # --- 6. Chain Tethering Regularization (Manifold Smoothing) ---
        # Ensures smooth transitions between adjacent experts
        reg_loss = torch.tensor(0.0, device=device)
        smoothness_loss = torch.tensor(0.0, device=device)
        
        if self.lambda_reg > 0:
            subset_sz = min(B, 4)  # Small subset to minimize overhead
            sub_idx = torch.randperm(B, device=device)[:subset_sz]
            
            # Use the SAME 't' values that generated the noise
            t_reg = t[sub_idx]
            reg_noisy = noisy_fut[sub_idx]
            reg_ctx = ctx_seq[sub_idx]
            reg_glob = global_ctx[sub_idx]
            reg_mask = ctx_mask[sub_idx] if ctx_mask is not None else None
            # [v15.1.1] Use the same self-cond for regularization pass
            reg_self_cond = self_cond[sub_idx]
            
            # Run all experts on this subset
            outputs = []
            for expert in self.experts:
                outputs.append(expert(
                    reg_noisy, t_reg, reg_ctx, reg_glob, reg_mask,
                    self_cond=reg_self_cond
                ))
            
            # Chain Loss: Link experts by severity and within phases
            # Group 1: 0 -> 1 -> 2 (Severity continuum within first sub-ensemble)
            # Group 2: 3 -> 4 -> 5 (Severity continuum within second sub-ensemble)
            # Cross: 0-3, 1-4, 2-5 (Sub-experts for same phase)
            chain_loss = torch.tensor(0.0, device=device)
            N = self.cfg.num_phases
            G = N_GT_PHASES
            
            # Severity Continuum (Horizontal)
            # Expert L -> Expert L+1 (within same sub-ensemble)
            for i in [0, 1, 3, 4]:
                if i + 1 < N:
                    chain_loss = chain_loss + F.mse_loss(outputs[i], outputs[i + 1])
            
            # Expert Pairing (Vertical)
            # Expert i -> Expert i+G (Sub-experts for the same clinical state)
            for i in range(G):
                if i + G < N:
                    chain_loss = chain_loss + F.mse_loss(outputs[i], outputs[i + G])
            
            # Temporal Smoothness (Self-Regularization)
            # Penalize sudden jumps in predicted vitals over time
            smoothness = torch.tensor(0.0, device=device)
            for out in outputs:
                # out: [subset, T, D]
                # diff: [subset, T-1, D]
                diff = out[:, 1:, :] - out[:, :-1, :]
                smoothness = smoothness + (diff ** 2).mean()
            
            smoothness_loss = smoothness
            reg_loss = (chain_loss + 0.5 * smoothness_loss) * self.lambda_reg
            total_loss = total_loss + reg_loss

        # --- 7. Expert Diversity Loss ---
        # Encourages different experts to produce distinct predictions
        # Prevents routing collapse where all experts converge
        diversity_loss = torch.tensor(0.0, device=device)
        
        if self.lambda_diversity > 0 and len(outputs) >= 2:
            # Compare each pair of experts
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    # Negative correlation: We WANT differences
                    similarity = F.cosine_similarity(
                        outputs[i].flatten(1),
                        outputs[j].flatten(1),
                        dim=-1
                    ).mean()
                    # Add penalty if experts are too similar
                    diversity_loss = diversity_loss + similarity
            
            # Diversity Loss: Averaged across all expert pairs
            n_pairs = len(outputs) * (len(outputs) - 1) / 2
            diversity_loss = self.lambda_diversity * (diversity_loss / (n_pairs + 1e-8))
            total_loss = total_loss + diversity_loss

        # --- 8. Load-Balancing Loss (Switch Transformer Style) ---
        load_balance_loss = torch.tensor(0.0, device=device)
        
        if self.lambda_lb > 0 and B > 0 and not self.use_loss_free_balancing:
            # f_i: Fraction of samples ROUTED TO expert i
            # P_i: Mean probability assigned to expert i
            N = self.cfg.num_phases
            f = expert_load / B  # [N] - Load fraction per expert
            P = router_probs.mean(dim=0)  # [N] - Mean prob per expert
            
            # Load balance loss: N * Σ(f_i * P_i)
            # Minimized when load is uniform
            load_balance_loss = self.lambda_lb * N * (f * P).sum()
            total_loss = total_loss + load_balance_loss
        
        # Update expert biases for Loss-Free Balancing
        if self.use_loss_free_balancing and self.training:
            with torch.no_grad():
                target_load = 1.0 / self.cfg.num_phases
                # [v15.1] DDP SYNC for Load Balancing
                # We must all-reduce counts so all ranks agree on expert load
                if dist.is_initialized():
                    dist.all_reduce(expert_load, op=dist.ReduceOp.SUM)
                    total_B = torch.tensor(B, device=device, dtype=torch.float32)
                    dist.all_reduce(total_B, op=dist.ReduceOp.SUM)
                    current_load = expert_load / (total_B + 1e-8)
                else:
                    current_load = expert_load / max(B, 1)
                
                load_error = current_load - target_load
                # Decrease bias for overloaded experts, increase for underloaded
                self.expert_bias = self.expert_bias - self.bias_update_rate * load_error
        
        # --- 9. Router Cross-Entropy Loss (Sub-Expert Supervision) ---
        # [v15.0] Hierarchical Phase Supervision
        # Instead of just ensuring Expert 0 is Phase 0, we ensure 
        # Sum(Prob(Experts for Phase 0)) matches Phase 0.
        # This allows the router to freely allocate between sub-experts (e.g. 0 and 3)
        # while still adhering to the clinical ground truth.
        
        # Aggregate probs by phase group (stride = N_GT_PHASES)
        # Assuming sequential allocation: Ep 0 (P0), Ep 1 (P1), Ep 2 (P2), Ep 3 (P0)...
        phase_probs_list = []
        for p in range(N_GT_PHASES):
            # Indices for this phase: p, p+3, p+6...
            phase_indices = [p + i * N_GT_PHASES for i in range(N_SUB_EXPERTS)]
            phase_indices = [idx for idx in phase_indices if idx < self.cfg.num_phases]
            
            # Sum probs for all experts in this phase group
            # [B, N_sub] -> sum -> [B]
            if phase_indices:
                p_sum = router_probs[:, phase_indices].sum(dim=1)
            else:
                p_sum = torch.zeros(B, device=device)
            phase_probs_list.append(p_sum)
            
        # Stack to [B, N_GT_PHASES] -> e.g., [B, 3]
        phase_probs_agg = torch.stack(phase_probs_list, dim=1)
        
        # Normalize to ensure sum=1 (sanity check, usually already close)
        phase_probs_agg = phase_probs_agg / (phase_probs_agg.sum(dim=1, keepdim=True) + 1e-8)
        
        # Acuity-Aware Weighting
        # [v16.0] Use configured 'loss_weights' (derived from crash_weight)
        # instead of hardcoding '5.0'.
        # gt_phases is [B], loss_weights is [N_experts]. Result is [B].
        acuity_weights = self.loss_weights[gt_phases]
        
        # NLL Loss on aggregated probabilities
        # NLL(log(probs), target) is equivalent to CrossEntropy, but we constructed probs manually
        router_ce_loss_raw = F.nll_loss(
            torch.log(phase_probs_agg + 1e-8), 
            gt_phases, 
            reduction='none'
        )
        
        router_ce_loss = (router_ce_loss_raw * acuity_weights).mean()
        total_loss = total_loss + self.aux_loss_scale * router_ce_loss  # Weighted by config

        # --- 10. Physiological Consistency Regularization ---
        # [v15.2.2] Protect Hard Decks across the whole batch
        phys_loss = self.phys_loss(full_pred_x0)

        # --- 11. Value Head Output (For AWR in wrapper) ---
        pred_value = self.value_head(global_ctx)

        return {
            # Primary Losses
            "loss": total_loss,
            "diffusion_loss": loss_stable + loss_preshock + loss_crash,
            "phys_loss": phys_loss, # Return separately for wrapper curriculum
            
            # Component Losses
            "reg_loss": reg_loss,
            "smoothness_loss": smoothness_loss,
            "diversity_loss": diversity_loss,
            "router_ce_loss": router_ce_loss,
            "load_balance_loss": load_balance_loss,
            
            # Phase-wise Losses
            "stable_loss": loss_stable,
            "crash_loss": loss_crash,
            "preshock_loss": loss_preshock,
            
            # AWR Support
            "pred_value": pred_value.squeeze(-1),
            "pred_x0": full_pred_x0,
            "pred_noise": full_pred_noise,
            
            # Telemetry
            **batch_counts
        }

    # =========================================================================
    # HELPER: SPARSE EXPERT EXECUTION
    # =========================================================================

    def _run_sparse_experts(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        ctx_seq: torch.Tensor, 
        global_ctx: torch.Tensor, 
        ctx_mask: Optional[torch.Tensor],
        top_k_indices: torch.Tensor,
        top_k_probs: torch.Tensor,
        self_cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Runs selected experts and blends their outputs using Top-K routing.
        
        This is the core inference computation that:
        1. Identifies which experts are needed for the current batch
        2. Runs only those experts (sparse computation)
        3. Blends outputs using router probabilities
        
        Args:
            x_t: Current noisy prediction [B, T, D]
            t: Diffusion timesteps [B]
            ctx_seq: Context sequence from encoder [B, T_ctx, d_model]
            global_ctx: Global context vector [B, d_model]
            ctx_mask: Attention mask for context [B, T_ctx]
            top_k_indices: Selected expert indices [B, K]
            top_k_probs: Routing probabilities [B, K]
            self_cond: Self-conditioning tensor [B, T, D] or None
        
        Returns:
            Blended expert prediction [B, T, D]
        """
        accum_pred = torch.zeros_like(x_t)
        unique_experts = torch.unique(top_k_indices)
        
        for expert_idx_tensor in unique_experts:
            expert_idx = expert_idx_tensor.item()
            
            # Create mask: which samples have this expert in their Top-K?
            mask = (top_k_indices == expert_idx).any(dim=-1)  # [B]
            
            if not mask.any():
                continue
            
            # Slice self-conditioning if it exists
            sub_self_cond = self_cond[mask] if self_cond is not None else None
            
            # Run expert on relevant samples
            expert_out = self.experts[expert_idx](
                x_t[mask], t[mask],
                ctx_seq[mask], global_ctx[mask],
                ctx_mask[mask] if ctx_mask is not None else None,
                self_cond=sub_self_cond
            )
            
            # Get the weight for this expert for each relevant sample
            position_mask = (top_k_indices[mask] == expert_idx)  # [n_selected, K]
            weights = (top_k_probs[mask] * position_mask.float()).sum(dim=-1)  # [n_selected]
            weights = weights.view(-1, 1, 1)  # [n_selected, 1, 1]
            
            # Weighted accumulation
            accum_pred[mask] = accum_pred[mask] + weights * expert_out
            
        return accum_pred

    # =========================================================================
    # INFERENCE SAMPLING
    # =========================================================================

    @torch.no_grad()
    def sample(
        self, 
        batch: Dict[str, torch.Tensor], 
        num_steps: Optional[int] = None, 
        hard_gating: bool = False,
        top_k: int = 2,
        use_physics_guidance: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Inference Sampling with SOTA Features:
        - Top-K Sparse Routing
        - Physics-Guided Sampling (PGS)
        - Self-Conditioning ("Analog Bits")
        
        Args:
            batch: Input batch with observed_data and static_context.
            num_steps: Number of diffusion steps (default: cfg.timesteps).
            hard_gating: If True, uses argmax routing (top_k=1).
            top_k: Number of experts to activate per sample (default=2).
            use_physics_guidance: Enable active gradient steering.
        
        Returns:
            Dictionary with 'vitals', 'logits', 'probs', 'top_k_indices'.
        """
        past = batch["observed_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        B = past.shape[0]
        device = past.device
        
        # 1. Perception (Frozen Encoder)
        past_norm, static_norm = self.normalize(past, static)
        # [v16.0] FIX: Disable unsafe padding inference.
        padding_mask = None
        
        ctx_seq, global_ctx, ctx_mask = self.encoder(
            past_norm, static_norm, 
            imputation_mask=src_mask,
            padding_mask=padding_mask
        )
        
        # 2. Router Decision
        logits = self.router(global_ctx)  # [B, N_experts]
        
        # Apply Loss-Free Balancing bias if enabled
        if self.use_loss_free_balancing:
            logits = logits + self.expert_bias.unsqueeze(0)
        
        probs = F.softmax(logits, dim=-1)  # [B, N]
        
        # 3. Select Top-K Experts (Sparse Routing)
        if hard_gating:
            k = 1  # Single winner
        else:
            k = min(top_k, self.cfg.num_phases)  # Ensure K <= N_experts
        
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)  # [B, K], [B, K]
        
        # Renormalize K probabilities to sum to 1.0
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Diffusion Loop
        x_t = torch.randn(B, self.cfg.pred_len, self.cfg.input_dim, device=device)
        steps = num_steps or self.cfg.timesteps
        
        # Initialize Self-Conditioning Buffer
        x_self_cond = torch.zeros_like(x_t)
        
        # Check if Physics Guidance is enabled and available
        pgs_enabled = (
            use_physics_guidance and 
            hasattr(self.cfg, 'physics_guidance_scale') and 
            self.cfg.physics_guidance_scale > 0
        )
        
        for i in reversed(range(steps)):
            t_step = torch.full((B,), i, dtype=torch.long, device=device)
            
            # --- Physics-Guided Sampling (Active Gradient Steering) ---
            if pgs_enabled:
                with torch.enable_grad():
                    x_t_in = x_t.detach().requires_grad_(True)
                    
                    # Run experts to get epsilon prediction
                    out_eps = self._run_sparse_experts(
                        x_t_in, t_step, ctx_seq, global_ctx, ctx_mask,
                        top_k_indices, top_k_probs, x_self_cond
                    )
                    
                    # Reconstruct x0 approximation (DDIM equation)
                    alpha_bar = self.scheduler.alphas_cumprod[t_step][:, None, None]
                    x0_approx = (x_t_in - torch.sqrt(1 - alpha_bar) * out_eps) / torch.sqrt(alpha_bar)
                    
                    # Calculate Physiological Violation
                    loss = self.phys_loss(x0_approx)
                    grad = torch.autograd.grad(loss, x_t_in)[0]
                    
                # [v15.2] PER-SAMPLE GRADIENT NORMALIZATION (Safety Guard)
                # Compute norm across vitals dimensions [T, D] for each sample in batch
                # Effectively: grad = grad / ||grad||_2 per patient
                grad_norm = grad.norm(dim=(1, 2), keepdim=True) # [B, 1, 1]
                
                # Apply normalization where norm > epsilon
                safe_grad = grad / (grad_norm + 1e-8)
                
                # Steer away from violations
                x_t = x_t - self.cfg.physics_guidance_scale * safe_grad.detach()

            # --- Standard Denoising Step ---
            out = self._run_sparse_experts(
                x_t, t_step, ctx_seq, global_ctx, ctx_mask,
                top_k_indices, top_k_probs, x_self_cond
            )
            
            # Update Self-Conditioning (for next step)
            alpha_bar = self.scheduler.alphas_cumprod[t_step][:, None, None]
            x_self_cond = (x_t - torch.sqrt(1 - alpha_bar) * out) / torch.sqrt(alpha_bar)
            x_self_cond = x_self_cond.detach()
            
            # Apply Scheduler Step
            x_t = self.scheduler.step(out, t_step, x_t, use_ddim=self.cfg.use_ddim_sampling)
            
        return {
            "vitals": self.unnormalize(x_t),
            "logits": logits,
            "probs": probs,
            "top_k_indices": top_k_indices
        }

    def forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Helper for simple forward pass during validation.
        Returns the full sampling payload (vitals + routing info).
        """
        return self.sample(batch)
