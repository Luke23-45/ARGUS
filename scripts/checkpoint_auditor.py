"""
scripts/checkpoint_auditor.py
--------------------------------------------------------------------------------
FORENSIC Checkpoint Auditor for ICU Research.

This script performs an **exhaustive** inspection of a PyTorch Lightning checkpoint
to verify serialization integrity for resumption capabilities.

CRITICAL STATE COMPONENTS TRACKED:
1. TemporalContrastiveBuffer (queue, queue_ptr)
2. SepsisGhostBank (10 buffers: raw_vitals, raw_masks, raw_labels, latent_anchors, uncertainties, prototype_ema, ptr, size, is_full)
3. BayesianProjectedScaler (log_vars [Parameter], loss_emas [Buffer])
4. GradNormBalancer (weights [Parameter], initial_losses [Buffer])
5. ICUAdvantageCalculator (7 buffers: beta, max_weight, ess_buffer, ess_momentum_buffer, clip_rate_buffer, adv_mean, adv_std, stats_count, stats_initialized)
6. DynamicClassBalancer (counts, initialized)
7. EMA State (ema_state_dict from TieredEMACallback)
8. Normalizer State (normalizer_state - manually saved)
9. Foundation Gradient EMA (fnd_grad_ema - manually saved, non-buffer)
10. Optimizer & Scheduler States

Usage:
    python scripts/checkpoint_auditor.py <path_to_checkpoint.ckpt>

Author: APEX Forensic Team
Version: 2.0 - Exhaustive Edition
"""

import torch
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict
import json

# ============================================================================
# EXPECTED STATE COMPONENTS
# ============================================================================

# Maps: component_name -> list of (expected_key_suffix, is_critical, description)
EXPECTED_COMPONENTS = {
    "tcb_buffer": [
        ("queue", True, "InfoNCE Memory Bank [capacity, d_model]"),
        ("queue_ptr", True, "Pointer for circular queue [scalar]"),
    ],
    "ghost_bank": [
        ("raw_vitals", True, "Stored sepsis trajectories [capacity, history_len, feature_dim]"),
        ("raw_masks", True, "Imputation masks for ghosts [capacity, history_len, feature_dim]"),
        ("raw_labels", True, "Phase labels for ghosts [capacity]"),
        ("latent_anchors", True, "Latent representations for CGA alignment [capacity, latent_dim]"),
        ("uncertainties", True, "Uncertainty scores for prioritized sampling [capacity, 1]"),
        ("prototype_ema", False, "Global sepsis manifold centroid [1, latent_dim]"),
        ("ptr", True, "Current write pointer [scalar]"),
        ("size", True, "Number of active ghosts [scalar]"),
        ("is_full", True, "Bank capacity flag [bool]"),
    ],
    "awr_calculator": [
        ("beta", True, "AWR Temperature (adaptive) [scalar]"),
        ("max_weight", True, "Maximum AWR weight clamp [scalar]"),
        ("ess_buffer", False, "Effective Sample Size [1]"),
        ("ess_momentum_buffer", False, "ESS target [1]"),
        ("clip_rate_buffer", False, "Clipping rate [1]"),
        ("adv_mean", True, "Global advantage mean for whitening [scalar]"),
        ("adv_std", True, "Global advantage std for whitening [scalar]"),
        ("stats_count", False, "Sample count for Welford's [scalar]"),
        ("stats_initialized", True, "Stats lock flag [bool]"),
    ],
    "loss_scaler": [
        ("log_vars", True, "Learnable uncertainty per task [num_tasks]"),
        ("loss_emas", True, "EMA of losses for UW-SO [num_tasks]"),
    ],
    "class_balancer": [
        ("counts", True, "Running class counts [num_classes]"),
        ("initialized", True, "Initialization flag [bool]"),
    ],
    "gradnorm": [
        ("weights", True, "Task weights [num_tasks] - nn.Parameter"),
        ("initial_losses", True, "Initial loss values for normalization [num_tasks]"),
    ],
    "model.normalizer": [
        ("ts_min", True, "Per-channel minimum values [feature_dim]"),
        ("ts_max", True, "Per-channel maximum values [feature_dim]"),
        ("is_calibrated", True, "Calibration flag [bool]"),
    ],
}

# Manual/External keys saved in on_save_checkpoint
MANUAL_EXTERNAL_KEYS = [
    ("normalizer_state", True, "Full normalizer state_dict (manual save)"),
    ("fnd_grad_ema", False, "PMS Foundation Gradient EMA (manual save)"),
    ("ema_state_dict", True, "TieredEMA shadow weights (from EMACallback)"),
    ("optimizer_states", True, "Optimizer states for resumption"),
    ("lr_schedulers", True, "LR scheduler states"),
    ("epoch", True, "Epoch counter"),
    ("global_step", True, "Global training step"),
    ("pytorch-lightning_version", False, "PL version for compat check"),
    ("state_dict", True, "Full module state_dict"),
    ("hyper_parameters", False, "Config snapshot"),
    ("callbacks", False, "Callback states (EMA, Saver)"),
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_tensor_summary(t: torch.Tensor) -> str:
    """Returns a summary string for a tensor."""
    if t.numel() == 0:
        return "[EMPTY]"
    if t.numel() == 1:
        # Scalar
        val = t.item() if t.is_floating_point() else t.item()
        return f"Shape: [] | Value: {val}"
    if t.is_floating_point():
        return f"Shape: {list(t.shape)} | Mean: {t.float().mean().item():.6f} | Std: {t.float().std().item():.6f} | Min/Max: [{t.min().item():.4f}, {t.max().item():.4f}]"
    else:
        return f"Shape: {list(t.shape)} | Min/Max: [{t.min().item()}, {t.max().item()}]"

def check_tensor_health(t: torch.Tensor, name: str) -> List[str]:
    """Returns a list of warnings for the tensor."""
    warnings = []
    if t.numel() == 0:
        warnings.append(f"EMPTY: {name}")
    elif t.is_floating_point():
        if torch.isnan(t).any():
            warnings.append(f"NaN DETECTED: {name}")
        if torch.isinf(t).any():
            warnings.append(f"Inf DETECTED: {name}")
        if t.abs().sum().item() == 0 and t.numel() > 1:
            warnings.append(f"ALL ZEROS: {name}")
    return warnings

# ============================================================================
# MAIN AUDITOR
# ============================================================================

def audit_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """
    Performs exhaustive inspection of a checkpoint file.
    
    Returns:
        A dictionary containing all findings.
    """
    report = {
        "path": ckpt_path,
        "top_level_keys": [],
        "components": {},
        "external_states": {},
        "warnings": [],
        "critical_missing": [],
        "summary": {}
    }
    
    if not os.path.exists(ckpt_path):
        report["warnings"].append(f"FATAL: Checkpoint file not found: {ckpt_path}")
        return report

    print("=" * 100)
    print(f"🔬 FORENSIC CHECKPOINT AUDIT: {ckpt_path}")
    print("=" * 100)

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        report["warnings"].append(f"FATAL: Failed to load checkpoint: {e}")
        print(f"❌ CRITICAL ERROR: Failed to load checkpoint: {e}")
        return report

    # =========================================================================
    # 1. TOP-LEVEL KEY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 50)
    print("📦 [1] TOP-LEVEL CHECKPOINT KEYS")
    print("=" * 50)
    for k in sorted(checkpoint.keys()):
        val = checkpoint[k]
        type_str = type(val).__name__
        if isinstance(val, torch.Tensor):
            size_str = str(list(val.shape))
        elif isinstance(val, (list, dict)):
            size_str = str(len(val))
        elif isinstance(val, (int, float, bool)):
            size_str = str(val)
        else:
            size_str = "-"
        print(f"  ✓ {k:30} | Type: {type_str:20} | Size/Value: {size_str}")
        report["top_level_keys"].append({"key": k, "type": type_str, "size": size_str})

    # =========================================================================
    # 2. MODULE STATE_DICT COMPONENT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 50)
    print("🧠 [2] MODULE state_dict COMPONENT ANALYSIS")
    print("=" * 50)
    
    if "state_dict" not in checkpoint:
        report["critical_missing"].append("state_dict")
        print("  ❌ CRITICAL: 'state_dict' NOT FOUND in checkpoint!")
    else:
        sd = checkpoint["state_dict"]
        
        for comp_name, expected_keys in EXPECTED_COMPONENTS.items():
            print(f"\n  --- Component: {comp_name} ---")
            comp_report = {"found": [], "missing": [], "warnings": []}
            
            for key_suffix, is_critical, description in expected_keys:
                full_key = f"{comp_name}.{key_suffix}"
                
                # Try exact match first
                if full_key in sd:
                    tensor = sd[full_key]
                    summary = format_tensor_summary(tensor)
                    health_warnings = check_tensor_health(tensor, full_key)
                    
                    status = "✅ [OK]" if not health_warnings else "⚠️ [WARN]"
                    print(f"    {status} {key_suffix:20} | {summary}")
                    
                    comp_report["found"].append({
                        "key": full_key, 
                        "summary": summary,
                        "warnings": health_warnings
                    })
                    report["warnings"].extend(health_warnings)
                else:
                    # Fuzzy search
                    matches = [sk for sk in sd.keys() if comp_name in sk and key_suffix in sk]
                    if matches:
                        print(f"    🔍 [FUZZY] {key_suffix} found as: {matches[0]}")
                        tensor = sd[matches[0]]
                        summary = format_tensor_summary(tensor)
                        comp_report["found"].append({"key": matches[0], "summary": summary})
                    else:
                        status = "❌ [MISSING]" if is_critical else "⚠️ [MISSING]"
                        print(f"    {status} {key_suffix:20} | {description}")
                        comp_report["missing"].append({"key": full_key, "critical": is_critical})
                        if is_critical:
                            report["critical_missing"].append(full_key)
            
            report["components"][comp_name] = comp_report

    # =========================================================================
    # 3. EXTERNAL/MANUAL STATE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 50)
    print("🔗 [3] EXTERNAL/MANUAL STATE ANALYSIS")
    print("=" * 50)
    
    for key, is_critical, description in MANUAL_EXTERNAL_KEYS:
        if key in checkpoint:
            val = checkpoint[key]
            if isinstance(val, dict):
                detail = f"Dict with {len(val)} keys"
            elif isinstance(val, list):
                detail = f"List with {len(val)} items"
            elif isinstance(val, torch.Tensor):
                detail = format_tensor_summary(val)
            else:
                detail = str(val)[:50]
            print(f"  ✅ [PRESENT] {key:25} | {detail}")
            report["external_states"][key] = {"present": True, "detail": detail}
        else:
            status = "❌ [MISSING]" if is_critical else "⚠️ [MISSING]"
            print(f"  {status} {key:25} | {description}")
            report["external_states"][key] = {"present": False, "critical": is_critical}
            if is_critical:
                report["critical_missing"].append(f"(external) {key}")

    # =========================================================================
    # 4. CALLBACKS STATE DEEP DIVE
    # =========================================================================
    print("\n" + "=" * 50)
    print("📞 [4] CALLBACKS STATE DEEP DIVE")
    print("=" * 50)
    
    if "callbacks" in checkpoint:
        callbacks = checkpoint["callbacks"]
        if isinstance(callbacks, dict):
            for cb_name, cb_state in callbacks.items():
                print(f"  📌 Callback: {cb_name}")
                if isinstance(cb_state, dict):
                    for k, v in cb_state.items():
                        if isinstance(v, torch.Tensor):
                            print(f"      {k}: {format_tensor_summary(v)}")
                        elif isinstance(v, dict):
                            print(f"      {k}: Dict with {len(v)} keys")
                        else:
                            print(f"      {k}: {str(v)[:60]}")
                else:
                    print(f"      (Non-dict state: {type(cb_state).__name__})")
    else:
        print("  ⚠️ No 'callbacks' key found in checkpoint.")

    # =========================================================================
    # 5. EMA STATE DEEP DIVE
    # =========================================================================
    print("\n" + "=" * 50)
    print("👤 [5] EMA (TEACHER) STATE DEEP DIVE")
    print("=" * 50)
    
    ema_found = False
    # Check direct key
    if "ema_state_dict" in checkpoint:
        ema_found = True
        ema_sd = checkpoint["ema_state_dict"]
        if isinstance(ema_sd, dict):
            if "shadow" in ema_sd:
                print(f"  ✅ EMA shadow weights: {len(ema_sd['shadow'])} parameters")
                # Sample key
                if ema_sd['shadow']:
                    first_key = list(ema_sd['shadow'].keys())[0]
                    print(f"     Sample Key: {first_key}")
            else:
                print(f"  ✅ EMA state_dict: {len(ema_sd)} keys (flat format)")
    
    # Check callbacks
    if "callbacks" in checkpoint and isinstance(checkpoint["callbacks"], dict):
        for cb_name, cb_state in checkpoint["callbacks"].items():
            if "ema" in cb_name.lower() or "EMA" in cb_name:
                ema_found = True
                print(f"  ✅ Found EMA in callback: {cb_name}")
                if isinstance(cb_state, dict) and "shadow" in cb_state:
                    print(f"     Shadow weights: {len(cb_state['shadow'])} parameters")
    
    if not ema_found:
        print("  ❌ NO EMA STATE FOUND IN CHECKPOINT!")
        report["critical_missing"].append("EMA state (ema_state_dict or callback)")

    # =========================================================================
    # 6. SPECIFIC BUFFER VALUE CHECKS
    # =========================================================================
    print("\n" + "=" * 50)
    print("🔍 [6] CRITICAL BUFFER VALUE CHECKS")
    print("=" * 50)
    
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        
        # TCB Queue Health
        if "tcb_buffer.queue" in sd:
            tcb_queue = sd["tcb_buffer.queue"]
            queue_sum = tcb_queue.abs().sum().item()
            queue_std = tcb_queue.std().item()
            if queue_sum == 0:
                print("  🚨 TCB Queue is ALL ZEROS! Contrastive memory is EMPTY!")
                report["warnings"].append("TCB Queue is empty (all zeros)")
            elif queue_std < 0.01:
                print(f"  ⚠️ TCB Queue has LOW variance (std={queue_std:.6f}). May be collapsed.")
            else:
                print(f"  ✅ TCB Queue: Sum={queue_sum:.2f}, Std={queue_std:.4f}")
        
        # Ghost Bank Size
        if "ghost_bank.size" in sd:
            bank_size = sd["ghost_bank.size"].item()
            print(f"  ℹ️ Ghost Bank Size: {bank_size}")
            if bank_size == 0:
                print("  🚨 Ghost Bank is EMPTY! No sepsis ghosts available for summoning!")
                report["warnings"].append("Ghost Bank is empty (size=0)")
        
        # AWR Stats Initialized
        if "awr_calculator.stats_initialized" in sd:
            initialized = sd["awr_calculator.stats_initialized"].item()
            if not initialized:
                print("  🚨 AWR stats NOT initialized! Whitening will use batch stats (unstable)!")
                report["warnings"].append("AWR stats not initialized")
            else:
                adv_mean = sd.get("awr_calculator.adv_mean", torch.tensor(0)).item()
                adv_std = sd.get("awr_calculator.adv_std", torch.tensor(1)).item()
                print(f"  ✅ AWR Stats: Mean={adv_mean:.4f}, Std={adv_std:.4f}, Initialized=True")
        
        # Loss Scaler log_vars
        if "loss_scaler.log_vars" in sd:
            log_vars = sd["loss_scaler.log_vars"]
            print(f"  ℹ️ Loss Scaler log_vars: {log_vars.tolist()}")
            # Check for extreme values
            if (log_vars.abs() > 4).any():
                print("  ⚠️ Some log_vars are extreme (|val| > 4). Task weighting may be unbalanced.")
        
        # GradNorm initial_losses
        if "gradnorm.initial_losses" in sd:
            init_losses = sd["gradnorm.initial_losses"]
            if init_losses.sum().item() == 0:
                print("  ⚠️ GradNorm initial_losses are ALL ZEROS. Will be re-initialized on resume.")
            else:
                print(f"  ✅ GradNorm initial_losses: {init_losses.tolist()}")

    # =========================================================================
    # 7. FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("📊 FINAL AUDIT SUMMARY")
    print("=" * 100)
    
    total_warnings = len(report["warnings"])
    total_critical = len(report["critical_missing"])
    
    if total_critical > 0:
        print(f"\n  🚨 CRITICAL MISSING STATES ({total_critical}):")
        for item in report["critical_missing"]:
            print(f"      • {item}")
    
    if total_warnings > 0:
        print(f"\n  ⚠️ WARNINGS ({total_warnings}):")
        for item in report["warnings"]:
            print(f"      • {item}")
    
    if total_critical == 0 and total_warnings == 0:
        print("\n  ✅ CHECKPOINT APPEARS HEALTHY - All critical states present and non-empty.")
    elif total_critical > 0:
        print("\n  ❌ CHECKPOINT HAS CRITICAL ISSUES - Resumption will likely cause trauma!")
    else:
        print("\n  ⚠️ CHECKPOINT HAS MINOR ISSUES - Resumption may be unstable.")
    
    report["summary"] = {
        "total_warnings": total_warnings,
        "total_critical": total_critical,
        "verdict": "HEALTHY" if total_critical == 0 and total_warnings == 0 else ("CRITICAL" if total_critical > 0 else "WARNING")
    }
    
    print("\n" + "=" * 100)
    print("Audit Complete.")
    print("=" * 100)
    
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_auditor.py <path_to_checkpoint.ckpt>")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    report = audit_checkpoint(ckpt_path)
    
    # Optionally save report as JSON
    # report_path = ckpt_path.replace(".ckpt", "_audit.json")
    # with open(report_path, "w") as f:
    #     json.dump(report, f, indent=2, default=str)
    # print(f"\nReport saved to: {report_path}")
