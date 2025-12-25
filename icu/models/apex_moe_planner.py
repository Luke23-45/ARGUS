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
        lambda_reg: float = 0.01
    ):
        super().__init__()
        logger.info("Initializing APEX-MoE Specialist (Tri-Phase)...")
        
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
        # The "Eyes" (Encoder) and "Brain" (Router) must remain fixed 
        # so the "Hands" (Experts) train against a stable reality.
        
        self.encoder.requires_grad_(False)
        self.router.requires_grad_(False)
        # [PATCH] Unfreeze Value Head to allow Critic adaptation during specialization
        self.value_head.requires_grad_(True) 
        
        if self.history_rope:
            self.history_rope.requires_grad_(False)
            
        # Put frozen parts in eval mode
        self.encoder.eval()
        self.router.eval()
        # [PATCH] Keep Value Head in train mode
        self.value_head.train()
        
        logger.info(" - Perception Layer (Encoder/Router) FROZEN.")
        
        # --- 3. CLONING (Expert Forking) ---
        # Create N independent copies of the Diffusion Backbone
        logger.info(f" - Cloning Generalist Backbone into {self.cfg.num_phases} Experts...")
        self.experts = nn.ModuleList([
            copy.deepcopy(generalist_model.backbone) 
            for _ in range(self.cfg.num_phases)
        ])
        
        # --- 4. CONFIGURATION ---
        self.lambda_reg = lambda_reg
        
        # Default weighting if none provided: [1.0, 5.0, 10.0] style
        if phase_weights is None:
            # Auto-balance: Phase 0 is 1.0, others scale up
            self.phase_weights = [1.0] + [5.0] * (self.cfg.num_phases - 1)
        else:
            if len(phase_weights) != self.cfg.num_phases:
                raise ValueError(f"Phase weights mismatch. Exp {self.cfg.num_phases}, Got {len(phase_weights)}")
            self.phase_weights = phase_weights
            
        self.register_buffer("loss_weights", torch.tensor(self.phase_weights))
        logger.info(f" - Phase Weights: {self.phase_weights}")
        logger.info(f" - Tethering Regularization: lambda={lambda_reg}")

    def train(self, mode: bool = True):
        """
        Robust Train Mode.
        Ensures frozen components stay in EVAL mode even when model.train() is called.
        """
        super().train(mode)
        # Force frozen parts to eval
        self.encoder.eval()
        self.router.eval()
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
        The "Gradient Surgery" Forward Pass (Hard Gating).
        """
        if not self.training:
            return self.forward_inference(batch)
            
        # --- 1. Unpack & Normalize ---
        past = batch["observed_data"]
        fut = batch["future_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        # [PATCH] Robust Phase Label Resolution (Fixing Bug #3)
        # If dataset provides 'phase_label', use it. 
        # If not, fallback to 'outcome_label' mapping (Binary Compatibility).
        if "phase_label" in batch:
            phase_labels = batch["phase_label"]
        elif "outcome_label" in batch:
            # Fallback: Map 0->Stable(0), 1->Shock(Last Phase)
            # This allows training to proceed even if Pre-Shock (Phase 1) is missing
            gt = batch["outcome_label"].long()
            phase_labels = torch.zeros_like(gt)
            # Map Class 1 to the highest Phase Index (e.g., 2)
            phase_labels[gt == 1] = self.cfg.num_phases - 1
        else:
            raise ValueError("Batch missing both 'phase_label' and 'outcome_label'. Cannot route experts.")

        past_norm, static_norm = self.normalize(past, static)
        fut_norm, _ = self.normalize(fut, None)
        
        B = past.shape[0]
        device = past.device
        
        # --- 2. Shared Perception (No Grad) ---
        with torch.no_grad():
            # [SOTA Fix] Handle triplet return from fixed TemporalFusionEncoder
            # ctx_seq: [B, T_hist+1, D]
            # global_ctx: [B, D]
            # ctx_mask: [B, T_hist+1] (or similar)
            ctx_seq, global_ctx, ctx_mask = self.encoder(past_norm, static_norm, src_mask)
            
            # RoPE Prep (Shared)
            # The scheduler noise loop generates 't', so we prepare 'context_cos/sin'
            if self.history_rope:
                # History length is T_hist+1 (due to static token)
                context_cos, context_sin = self.history_rope(ctx_seq.shape[1], device)
            else:
                context_cos, context_sin = None, None

        # --- 3. Expert Execution Loop (With AWR Support) ---
        total_loss = torch.tensor(0.0, device=device)
        
        # [PATCH] Explicit Compatibility Accumulators (Fixing Bug #1)
        loss_stable = torch.tensor(0.0, device=device)
        loss_preshock = torch.tensor(0.0, device=device)
        loss_crash = torch.tensor(0.0, device=device)
        
        batch_counts = {}
        
        # Common Diffusion Params
        t = torch.randint(0, self.cfg.timesteps, (B,), device=device)
        noisy_fut, noise_eps = self.scheduler.add_noise(fut_norm, t)

        for phase_idx in range(self.cfg.num_phases):
            # A. Identification
            idx_mask = (phase_labels == phase_idx)
            indices = torch.nonzero(idx_mask).squeeze(-1)
            
            count = len(indices)
            batch_counts[f"count_phase_{phase_idx}"] = float(count)
            
            if count == 0:
                continue
                
            # B. Slicing
            sub_noisy = noisy_fut[indices]
            sub_t = t[indices]
            sub_ctx_seq = ctx_seq[indices]
            sub_global_ctx = global_ctx[indices]
            sub_ctx_mask = ctx_mask[indices] if ctx_mask is not None else None
            
            # C. Expert Forward
            expert = self.experts[phase_idx]
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
            # Apply configured phase weight (e.g. 10.0 for Shock)
            p_weight = self.loss_weights[phase_idx]
            weighted_loss = loss_mse * p_weight
            
            # [PATCH] Mapping to Standardized Output Keys
            if phase_idx == 0:
                loss_stable = loss_mse
            elif phase_idx == self.cfg.num_phases - 1:
                loss_crash = loss_mse
            else:
                # Sum intermediate phases for preshock slot
                loss_preshock += loss_mse
            
            # Add to total (scaled by batch proportion to maintain magnitude)
            if B > 0:
                total_loss += weighted_loss * (count / B)

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
            
            # Chain Loss: MSE(Exp0, Exp1) + MSE(Exp1, Exp2) ...
            chain_loss = 0.0
            for i in range(len(outputs) - 1):
                chain_loss += F.mse_loss(outputs[i], outputs[i+1])
                
            reg_loss = chain_loss * self.lambda_reg
            total_loss += reg_loss

        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
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
        hard_gating: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Inference Sampling (Soft or Hard Gated).
        
        Args:
            hard_gating: If True, uses argmax(Router) -> Single Expert.
                         If False, blends sum(Prob_i * Expert_i).
        """
        past = batch["observed_data"]
        static = batch["static_context"]
        src_mask = batch.get("src_mask", None)
        
        B = past.shape[0]
        
        # 1. Perception
        past_norm, static_norm = self.normalize(past, static)
        ctx_seq, global_ctx, ctx_mask = self.encoder(past_norm, static_norm, src_mask)
        
        # 2. Router Decision
        logits = self.router(global_ctx) # [B, N_phases]
        probs = F.softmax(logits, dim=-1) # [B, N]
        
        # 3. Diffusion Loop
        x_t = torch.randn(B, self.cfg.pred_len, self.cfg.input_dim, device=past.device)
        steps = num_steps or self.cfg.timesteps
        
        for i in reversed(range(steps)):
            t = torch.full((B,), i, dtype=torch.long, device=past.device)
            
            if hard_gating:
                # Efficient: Run only the winner expert per sample
                # Note: This is complex to vectorize efficiently in PyTorch without
                # running all experts or creating N sub-batches.
                # For simplicity/speed balance, we run soft gating usually.
                # Implementing naive hard gating via masking for correctness demo:
                
                preds = torch.zeros_like(x_t)
                winners = torch.argmax(probs, dim=-1) # [B]
                
                for k in range(self.cfg.num_phases):
                    idx = (winners == k)
                    if idx.any():
                        out_k = self.experts[k](
                            x_t[idx], t[idx], 
                            ctx_seq[idx], global_ctx[idx], 
                            ctx_mask[idx] if ctx_mask is not None else None
                        )
                        preds[idx] = out_k
                
                final_pred = preds
                
            else:
                # Soft Gating: Run ALL experts, weighted average
                # This captures uncertainty ("Is it Pre-Shock or Stable?")
                accum_pred = 0.0
                
                for k in range(self.cfg.num_phases):
                    # Local Expert Prediction
                    out_k = self.experts[k](x_t, t, ctx_seq, global_ctx, ctx_mask)
                    
                    # Weighting: P_k [B, 1] * Out_k [B, T, D]
                    w_k = probs[:, k].view(B, 1, 1)
                    accum_pred += w_k * out_k
                
                final_pred = accum_pred
            
            # Scheduler Step
            x_t = self.scheduler.step(final_pred, t, x_t, use_ddim=self.cfg.use_ddim_sampling)
            
        return {
            "vitals": self.unnormalize(x_t),
            "logits": logits,
            "probs": probs
        }

    def forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Helper for simple forward pass during validation.
        Now returns the full payload (vitals + routing info).
        """
        return self.sample(batch)