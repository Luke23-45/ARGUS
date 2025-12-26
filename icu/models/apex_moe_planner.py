"""
icu/models/apex_moe_planner.py
--------------------------------------------------------------------------------
APEX-MoE: Adaptive Phase-Locked Mixture-of-Experts Planner (SOTA v2.0).

Status: SAFETY-CRITICAL / PRODUCTION-READY
Architecture: Tri-Phase Specialist (Stable -> Pre-Shock -> Shock).

This model transforms a Generalist DiT into a team of specialized experts.
It uses "Phase-Locking" to ensure that the definition of clinical states 
remains consistent while the experts specialize in their respective dynamics.

Key Mechanisms:
1.  **Bootstrapping**: Clones experts from a converged Generalist to skip early training instability.
2.  **Phase-Locked Perception**: Freezes the History Encoder and Router.
3.  **Gradient Surgery**: Hard-gates training data so Expert K *only* sees Phase K patients.
4.  **Chain Tethering**: Regularizes adjacent experts (0<->1, 1<->2) to ensure smooth transitions.
5.  **Soft-Gated Inference**: Blends expert predictions based on Router probability masses.

Dependencies:
    - icu.models.diffusion.ICUUnifiedPlanner (The Donor/Generalist)
"""

from __future__ import annotations

import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from omegaconf import DictConfig

# Base Model Import
from icu.models.diffusion import ICUUnifiedPlanner

# Setup Logger
logger = logging.getLogger("APEX_MoE_Specialist")

# =============================================================================
# APEX-MoE CLASS DEFINITION
# =============================================================================

class APEX_MoE_Planner(nn.Module):
    """
    The Tri-Phase Specialist Model.
    
    Training (Forward):
        - Uses 'Gradient Surgery': Batch is physically split by Ground Truth Phase.
        - Expert 0 sees Stable, Expert 1 sees Pre-Shock, Expert 2 sees Shock.
        
    Inference (Sample):
        - Uses 'Soft Gating': Router predicts probabilities P(Phase).
        - Final Output = sum(P_k * Expert_k(x)).
    """
    def __init__(
        self, 
        generalist_model: ICUUnifiedPlanner, 
        phase_weights: Optional[List[float]] = None,
        lambda_reg: float = 0.01,
        lambda_lb: float = 0.01  # [SOTA-Clinical] Load-balancing coefficient (Switch Transformer)
    ):
        super().__init__()
        logger.info(f"Initializing APEX-MoE Specialist (SOTA-Clinical: {generalist_model.cfg.num_phases} Experts)...")
        
        # --- 1. BOOTSTRAPPING (Organ Harvesting) ---
        # We steal the initialized components from the pre-trained generalist.
        
        # Config & Hyperparameters
        self.cfg = generalist_model.cfg
        if self.cfg.num_phases < 2:
            raise ValueError(f"APEX-MoE requires at least 2 phases, found {self.cfg.num_phases}")
        
        logger.info(f" - Configuration: {self.cfg.num_phases} Experts | d_model={self.cfg.d_model}")
        
        # Components
        self.encoder = generalist_model.encoder
        self.scheduler = generalist_model.scheduler
        self.normalizer = generalist_model.normalizer
        
        # Router (Aux Head) - Critical
        if not hasattr(generalist_model, 'aux_head') or generalist_model.aux_head is None:
            raise RuntimeError("Generalist model has no Auxiliary Head (Router). Cannot build MoE.")
        self.router = generalist_model.aux_head
        
        # Value Head (Critic) - Critical for AWR
        self.value_head = generalist_model.value_head
        
        # RoPE (If shared)
        self.history_rope = getattr(generalist_model, 'history_rope', None)
        
        # --- 2. PHASE-LOCKING (Freeze Perception) ---
        # The "Eyes" (Encoder) must remain fixed so the "Hands" (Experts) train
        # against a stable reality.
        # [v4.1 FIX] UNFREEZE Router to allow sub-phase specialization.
        # With 6 experts but only 3 GT phases, the router must learn which
        # sub-expert (e.g., Expert 0 vs Expert 3) should handle each sample.
        
        self.encoder.requires_grad_(False)
        # [v4.1] Router is now TRAINABLE for sub-phase discovery
        self.router.requires_grad_(True)
        # [PATCH] Unfreeze Value Head to allow Critic adaptation during specialization
        self.value_head.requires_grad_(True) 
        
        if self.history_rope:
            self.history_rope.requires_grad_(False)
            
        # Put frozen parts in eval mode
        self.encoder.eval()
        # [v4.1] Router stays in TRAIN mode
        self.router.train()
        # [PATCH] Keep Value Head in train mode
        self.value_head.train()
        
        logger.info(" - Encoder FROZEN. Router/Value Head TRAINABLE (v4.1 Sub-Phase Mode).")
        
        # --- 3. CLONING (Expert Forking) ---
        # Create N independent copies of the Diffusion Backbone
        logger.info(f" - Cloning Generalist Backbone into {self.cfg.num_phases} Experts...")
        self.experts = nn.ModuleList([
            copy.deepcopy(generalist_model.backbone) 
            for _ in range(self.cfg.num_phases)
        ])
        
        # --- 4. CONFIGURATION ---
        self.lambda_reg = lambda_reg
        self.lambda_lb = lambda_lb  # [SOTA-Clinical] Load-balancing coefficient
        
        # Default weighting if none provided: [1.0, 5.0, 5.0, ...] style
        if phase_weights is None:
            # Auto-balance: Phase 0 is 1.0, others scale up
            self.phase_weights = [1.0] + [5.0] * (self.cfg.num_phases - 1)
        else:
            if len(phase_weights) != self.cfg.num_phases:
                raise ValueError(f"Phase weights mismatch. Exp {self.cfg.num_phases}, Got {len(phase_weights)}")
            self.phase_weights = phase_weights
            
        self.register_buffer("loss_weights", torch.tensor(self.phase_weights))
        logger.info(f" - Phase Weights: {self.phase_weights}")
        logger.info(f" - Regularization: lambda_reg={lambda_reg}, lambda_lb={lambda_lb}")


    def train(self, mode: bool = True):
        """
        Robust Train Mode.
        Ensures frozen components stay in EVAL mode even when model.train() is called.
        [v4.1] Router is now trainable and stays in train mode.
        """
        super().train(mode)
        # Force frozen parts to eval
        self.encoder.eval()
        # [v4.1] Router is trainable - keep in train mode
        self.router.train()
        # [PATCH] Keep Value Head in train mode
        self.value_head.train()
        return self

    # --- Robust Normalization Handling ---
    def normalize(self, x_ts: torch.Tensor, x_static: Optional[torch.Tensor] = None):
        """Wrapper for the safety-patched ClinicalNormalizer."""
        return self.normalizer.normalize(x_ts, x_static)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(x)

    def forward(
        self, 
        batch: Dict[str, torch.Tensor],
        awr_weights: Optional[torch.Tensor] = None # [PATCH] Re-add AWR support
    ) -> Dict[str, torch.Tensor]:
        """
        The "Gradient Surgery" Forward Pass with GT-Masked Routing (v4.1).
        
        [v4.1] GT-Masked Routing:
        - GT phases from dataset: {0: Stable, 1: Pre-Shock, 2: Shock}
        - Model experts: {0, 1, 2, 3, 4, 5} (6 total, 2 per phase)
        - Mapping: Phase L -> Experts {L, L+3}
        - Router picks Top-1 within the allowed set for each sample.
        """
        if not self.training:
            return self.forward_inference(batch)
            
        # --- 1. Unpack & Normalize ---
        past = batch["observed_data"]
        fut = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # [CRITICAL FIX] Pull weights from batch if not passed explicitly as argument
        # This occurs when calling self.model(batch) from Lightning's training_step
        awr_weights = awr_weights if awr_weights is not None else batch.get("awr_weights", None)
        
        # [PATCH] Robust Phase Label Resolution
        if "phase_label" in batch:
            gt_phases = batch["phase_label"]  # Values in {0, 1, 2}
        elif "outcome_label" in batch:
            gt = batch["outcome_label"].long()
            gt_phases = torch.zeros_like(gt)
            gt_phases[gt == 1] = 2  # Map positive outcome to Shock (Phase 2)
        else:
            raise ValueError("Batch missing both 'phase_label' and 'outcome_label'. Cannot route experts.")

        past_norm, static_norm = self.normalize(past, static)
        fut_norm, _ = self.normalize(fut, None)
        
        B = past.shape[0]
        device = past.device
        N_GT_PHASES = 3  # Dataset provides 3 phases
        N_SUB_EXPERTS = self.cfg.num_phases // N_GT_PHASES  # 2 sub-experts per phase
        
        # [DEFENSIVE] Validate GT phases are in expected range
        assert gt_phases.max() < N_GT_PHASES, \
            f"Invalid phase label detected: max={gt_phases.max()}, expected < {N_GT_PHASES}"
        
        # --- 2. Shared Perception (Encoder frozen, Router trainable) ---
        with torch.no_grad():
            ctx_seq, global_ctx, ctx_mask = self.encoder(past_norm, static_norm, src_mask)
            
            if self.history_rope:
                context_cos, context_sin = self.history_rope(ctx_seq.shape[1], device)
            else:
                context_cos, context_sin = None, None

        # --- 3. Router Decision (TRAINABLE in v4.1) ---
        router_logits = self.router(global_ctx)  # [B, 6] - Router predicts over all 6 experts
        router_probs = F.softmax(router_logits, dim=-1)
        
        # --- 4. GT-Masked Routing ---
        # For each sample, compute which expert to route to:
        # Phase L (GT) -> allowed experts {L, L+3}
        # Router picks the higher-prob one within this set.
        
        selected_experts = torch.zeros(B, dtype=torch.long, device=device)
        
        for gt_phase in range(N_GT_PHASES):
            mask = (gt_phases == gt_phase)
            if not mask.any():
                continue
            
            # Allowed experts for this GT phase: {gt_phase, gt_phase + 3, ...}
            # [SOTA-Clinical] v4.1 supports arbitrary sub-experts per phase
            allowed_experts = [gt_phase + i * N_GT_PHASES for i in range(N_SUB_EXPERTS)]
            allowed_experts = [exp for exp in allowed_experts if exp < self.cfg.num_phases]
            
            # Get router probs for the allowed experts/sub-experts
            # If N_SUB_EXPERTS = 2, we pick the stronger of the two
            sub_probs = router_probs[mask][:, allowed_experts] # [N_mask, N_sub]
            sub_winner_idx = torch.argmax(sub_probs, dim=1)    # [N_mask] indices in {0, ..., N_sub-1}
            
            # Map back to absolute indices
            selected_experts[mask] = torch.tensor(allowed_experts, device=device)[sub_winner_idx]
        
        # --- 5. Expert Execution Loop (Sparse) ---
        total_loss = torch.tensor(0.0, device=device)
        
        loss_stable = torch.tensor(0.0, device=device)
        loss_preshock = torch.tensor(0.0, device=device)
        loss_crash = torch.tensor(0.0, device=device)
        
        batch_counts = {}
        
        # Common Diffusion Params
        t = torch.randint(0, self.cfg.timesteps, (B,), device=device)
        noisy_fut, noise_eps = self.scheduler.add_noise(fut_norm, t)

        # Loop over experts that have at least one sample
        for expert_idx in range(self.cfg.num_phases):
            idx_mask = (selected_experts == expert_idx)
            indices = torch.nonzero(idx_mask).squeeze(-1)
            
            count = indices.numel() if indices.dim() > 0 else (1 if len(indices) > 0 else 0)
            batch_counts[f"count_expert_{expert_idx}"] = float(count)
            
            if count == 0:
                continue
                
            # B. Slicing
            sub_noisy = noisy_fut[indices]
            sub_t = t[indices]
            sub_ctx_seq = ctx_seq[indices]
            sub_global_ctx = global_ctx[indices]
            sub_ctx_mask = ctx_mask[indices] if ctx_mask is not None else None
            
            # C. Expert Forward
            expert = self.experts[expert_idx]
            pred_noise = expert(
                sub_noisy, sub_t, 
                sub_ctx_seq, sub_global_ctx, 
                sub_ctx_mask
            )
            
            # D. Loss Computation (With AWR Support)
            # [PATCH] Bug #2: Use raw MSE per-sample to allow AWR weighting
            loss_raw = F.mse_loss(pred_noise, noise_eps[indices], reduction='none').mean(dim=[1, 2])
            
            if awr_weights is not None:
                # Extract and normalize weights for this sub-batch
                w_sub = awr_weights[indices]
                loss_mse = (loss_raw * w_sub).mean()
            else:
                loss_mse = loss_raw.mean()
            
            # E. Weighting & Accumulation
            # [v4.1] Use expert_idx for loss weighting, not phase_idx
            # Map expert to its base phase for legacy loss weights: expert % 3
            base_phase = expert_idx % 3  # Maps {0,3}->0, {1,4}->1, {2,5}->2
            p_weight = self.loss_weights[base_phase]
            weighted_loss = loss_mse * p_weight
            
            # [v4.1] Mapping to Standardized Output Keys (by base phase)
            if base_phase == 0:
                loss_stable += loss_mse
            elif base_phase == 2:
                loss_crash += loss_mse
            else:
                loss_preshock += loss_mse
            
            # Add to total (scaled by batch proportion to maintain magnitude)
            if B > 0:
                total_loss += weighted_loss * (count / B)
            
            # [Telemetry] Log individual expert losses
            batch_counts[f"loss_expert_{expert_idx}"] = float(loss_mse.item())

        # --- 4. Tethering Regularization (Manifold Smoothing) ---
        # We sample a small random subset to minimize overhead
        reg_loss = torch.tensor(0.0, device=device)
        if self.lambda_reg > 0:
            subset_sz = min(B, 4) # Keep it cheap
            sub_idx = torch.randperm(B)[:subset_sz]
            
            # [SAFETY FIX] Use the SAME 't' values that generated the noise
            # Original bug: t_reg was random, but noisy_fut was generated with different 't'
            t_reg = t[sub_idx]  # Reuse timestamps that match the noise level
            reg_noisy = noisy_fut[sub_idx]
            reg_ctx = ctx_seq[sub_idx]
            reg_glob = global_ctx[sub_idx]
            reg_mask = ctx_mask[sub_idx] if ctx_mask is not None else None
            
            # Run all experts on this subset
            outputs = []
            for expert in self.experts:
                outputs.append(expert(reg_noisy, t_reg, reg_ctx, reg_glob, reg_mask))
            
            # Chain Loss: Link experts by severity and within phases
            # Group 1: 0 -> 1 -> 2 | Group 2: 3 -> 4 -> 5 | Cross: 0-3, 1-4, 2-5
            chain_loss = 0.0
            N = self.cfg.num_phases
            G = 3 # Clinical Phases
            
            # 1. Severity Continuum (Horizontal)
            # Expert L -> Expert L+1 (within same sub-ensemble)
            for i in [0, 1, 3, 4]:
                if i+1 < N:
                    chain_loss += F.mse_loss(outputs[i], outputs[i+1])
            
            # 2. Expert Pairing (Vertical)
            # Expert i -> Expert i+G (Sub-experts specializing in the same clinical state)
            for i in range(G):
                if i+G < N:
                    chain_loss += F.mse_loss(outputs[i], outputs[i+G])
            
            # 3. Temporal Smoothness (Self-Regularization)
            # Penalize sudden jumps in the PREDICTED vitals over time
            smoothness_loss = 0.0
            for out in outputs:
                # out: [B, T, D]
                # diff: [B, T-1, D]
                diff = out[:, 1:] - out[:, :-1]
                smoothness_loss += (diff**2).mean()
            
            reg_loss = (chain_loss + 0.5 * smoothness_loss) * self.lambda_reg
            total_loss += reg_loss

        # --- 6. Load-Balancing Auxiliary Loss (Switch Transformer Style) ---
        # [v4.1] Updated to use SELECTED_EXPERTS (Router-driven) not GT phases
        load_balance_loss = torch.tensor(0.0, device=device)
        
        if self.lambda_lb > 0 and B > 0:
            # f_i: Fraction of samples ROUTED TO expert i (by Router)
            # P_i: Mean probability assigned to expert i
            N = self.cfg.num_phases
            f = torch.zeros(N, device=device)
            P = router_probs.mean(dim=0)  # [N] - Mean prob per expert
            
            for exp_idx in range(N):
                f[exp_idx] = (selected_experts == exp_idx).float().mean()
            
            # Load balance loss: N * Σ(f_i * P_i)
            load_balance_loss = self.lambda_lb * N * (f * P).sum()
            total_loss += load_balance_loss
        
        # --- 7. Router Cross-Entropy Loss (Sub-Expert Supervision) ---
        # [v4.1 CRITICAL FIX] Train router to predict BASE GT phase, not its own decisions
        # Original bug: router_ce_loss = F.cross_entropy(router_logits, selected_experts)
        # This created a self-supervision loop where the router learned to predict
        # its own Top-1 choices, not the ground truth.
        # 
        # Fix: Map selected_experts back to GT phases for supervision
        # {0,3}→0, {1,4}→1, {2,5}→2
        gt_phase_targets = selected_experts % N_GT_PHASES
        
        # [SOTA 2025] Acuity-Aware Router Weighting
        # Sepsis events are rare (3.1%). We boost the loss for Pre-Shock (1) and Shock (2)
        # to ensure the router doesn't bias towards the 'Healthy' majority.
        router_ce_loss_raw = F.cross_entropy(router_logits[:, :N_GT_PHASES], gt_phase_targets, reduction='none')
        acuity_weights = torch.ones_like(gt_phase_targets, dtype=torch.float32)
        acuity_weights[gt_phase_targets >= 1] = 5.0 # 5x focus on sepsis phases
        
        router_ce_loss = (router_ce_loss_raw * acuity_weights).mean()
        total_loss += 0.1 * router_ce_loss  # Weighted at 0.1

        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
            "load_balance_loss": load_balance_loss,  # [SOTA-Clinical] MoE health metric
            "stable_loss": loss_stable,
            "crash_loss": loss_crash,
            "preshock_loss": loss_preshock,
            "pred_value": self.value_head(global_ctx).squeeze(-1), # [SOTA] Explicit for AWR logging
            **batch_counts
        }


    @torch.no_grad()
    def sample(
        self, 
        batch: Dict[str, torch.Tensor], 
        num_steps: Optional[int] = None, 
        hard_gating: bool = False,
        top_k: int = 2  # [SOTA-Clinical] Top-K sparse routing (default=2)
    ) -> Dict[str, torch.Tensor]:
        """
        Inference Sampling with SOTA Top-K Sparse Routing.
        
        Args:
            hard_gating: If True, uses argmax(Router) -> Single Expert (top_k=1).
            top_k: Number of experts to activate per sample (default=2).
                   Set to cfg.num_phases for full soft-gating (legacy behavior).
        """
        past = batch["observed_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        B = past.shape[0]
        device = past.device
        
        # 1. Perception (Frozen)
        past_norm, static_norm = self.normalize(past, static)
        ctx_seq, global_ctx, ctx_mask = self.encoder(past_norm, static_norm, src_mask)
        
        # 2. Router Decision
        logits = self.router(global_ctx)  # [B, N_experts]
        probs = F.softmax(logits, dim=-1)  # [B, N]
        
        # 3. Select Top-K Experts (SOTA Sparse Routing)
        if hard_gating:
            k = 1  # Single winner
        else:
            k = min(top_k, self.cfg.num_phases)  # Ensure K <= N_experts
        
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)  # [B, K], [B, K]
        # Renormalize the K probabilities to sum to 1.0
        # [DEFENSIVE] Add epsilon to prevent division by zero
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Diffusion Loop with Sparse Routing
        x_t = torch.randn(B, self.cfg.pred_len, self.cfg.input_dim, device=device)
        steps = num_steps or self.cfg.timesteps
        
        for i in reversed(range(steps)):
            t = torch.full((B,), i, dtype=torch.long, device=device)
            
            # Sparse Expert Execution: Only run experts in someone's Top-K
            unique_experts = torch.unique(top_k_indices)
            
            # Accumulator for blended predictions
            accum_pred = torch.zeros_like(x_t)
            
            for expert_idx_tensor in unique_experts:
                expert_idx = expert_idx_tensor.item()
                
                # Create mask: which samples have this expert in their Top-K?
                mask = (top_k_indices == expert_idx).any(dim=-1)  # [B]
                
                if not mask.any():
                    continue
                
                # Run expert on relevant samples
                expert_out = self.experts[expert_idx](
                    x_t[mask], t[mask],
                    ctx_seq[mask], global_ctx[mask],
                    ctx_mask[mask] if ctx_mask is not None else None
                )
                
                # Get the weight for this expert for each relevant sample
                position_mask = (top_k_indices[mask] == expert_idx)  # [n_selected, K]
                weights = (top_k_probs[mask] * position_mask.float()).sum(dim=-1)  # [n_selected]
                weights = weights.view(-1, 1, 1)  # [n_selected, 1, 1]
                
                # Weighted accumulation
                accum_pred[mask] += weights * expert_out
            
            # Scheduler Step
            x_t = self.scheduler.step(accum_pred, t, x_t, use_ddim=self.cfg.use_ddim_sampling)
            
        return {
            "vitals": self.unnormalize(x_t),
            "logits": logits,
            "probs": probs,
            "top_k_indices": top_k_indices  # [SOTA] Expose routing for analysis
        }


    def forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Helper for simple forward pass during validation.
        Now returns the full payload (vitals + routing info).
        """
        return self.sample(batch)