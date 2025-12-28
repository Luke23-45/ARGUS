"""
optimize_sota.py
--------------------------------------------------------------------------------
Physiologically-Constrained Bayesian Evolution (PCBE) Engine.
Author: APEX Research Team
Version: 4.0 (Final "Noble" Edition)

Objective:
    To mathematically discover the hyperparameter configuration that maximizes 
    Clinical Utility for Sepsis-3 prediction while strictly enforcing 
    physiological safety constraints.

Methodology ("The Noble Method"):
    1.  Core Engine: Multivariate Tree-Structured Parzen Estimator (TPE).
        - Models P(Score | Hyperparameters) to find non-linear interactions.
        - Uses multivariate sampling to capture dependencies (e.g., pos_weight vs beta).
        
    2.  Safety Interlock: "Soft-Barrier" Penalties + ASHA Pruning.
        - Instead of blindly killing unsafe trials (which gives 0 info to the optimizer),
          we apply a heavy "Safety Tax" to the objective score.
        - This teaches the Bayesian model *why* a configuration failed.
        
    3.  Persistence: SQLite-backed storage.
        - immune to crashes, timeouts, or OOM errors. 
        - Fully resumable "Grand Search".

    4.  Objective Function: Sepsis-3 Clinical Utility.
        - Score = 0.6*AUPRC + 0.3*AUROC - 0.1*ECE - SafetyTax
        - Prioritizes Precision (Positive Predictive Value) over simple AUC.

Usage:
    python icu/optimize_sota.py
"""

import os
import sys
import logging
import math
import json
import torch
import hydra
import optuna
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from torchmetrics import AveragePrecision
ROOT_DIR = Path(__file__).resolve().parents[1] # icu_research/
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# Project Imports
from icu.models.wrapper_generalist import ICUGeneralistWrapper
from icu.train.train_generalist import ICUGeneralistDataModule
from icu.utils.train_utils import get_hardware_context, set_seed
from icu.utils.callbacks import APEXTQDMProgressBar

# Configure SOTA Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("optimization_log.txt")
    ]
)
logger = logging.getLogger("APEX_PCBE_Optimizer")

# ==============================================================================
# 0. TELEMETRY: SCIENTIFIC PRECISION PROGRESS BAR
# ==============================================================================
class PreciseTQDMProgressBar(APEXTQDMProgressBar):
    """
    Custom progress bar that prevents 'Zero Clipping' of small learning rates.
    Shows LR in scientific notation (e.g., 2.0e-05).
    """
    def get_metrics(self, trainer, pl_module) -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        # Override LR display to scientific notation
        # The 'lr' key is injected by LearningRateMonitor
        for k in list(items.keys()):
            if k.lower() == "lr":
                try:
                    # Recalculate precisely from the actual optimizer
                    actual_lr = trainer.optimizers[0].param_groups[0]["lr"]
                    items[k] = f"{actual_lr:.1e}"
                except Exception:
                    pass
        return items

# ==============================================================================
# 1. METRIC INJECTION CALLBACK (The Missing Link)
# ==============================================================================
class ClinicalMetricInjector(pl.Callback):
    """
    Surgically injects AUPRC calculation into the validation loop without 
    modifying the original wrapper code.
    
    Why: The base wrapper calculates AUROC/ECE but misses AUPRC, which is 
    critical for the Sepsis-3 Utility Score (Weight = 0.6).
    """
    def __init__(self):
        super().__init__()
        # Binary Average Precision (Precision-Recall Curve Area)
        self.auprc = AveragePrecision(task="binary")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.auprc.reset()
        self.auprc.to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Extract predictions injected by validation_step return
        # The wrapper returns a dict with 'preds' and 'target'
        if outputs and isinstance(outputs, dict) and "preds" in outputs and "target" in outputs:
            preds = outputs["preds"]
            target = outputs["target"]
            
            # Convert logits to probability for sepsis class (assumed class 1+)
            if preds.shape[-1] > 1:
                # Multiclass: Sum probabilities of all sepsis stages (Pre-Shock + Shock)
                probs = torch.softmax(preds, dim=-1)[:, 1:].sum(dim=-1)
            else:
                # Binary
                probs = torch.sigmoid(preds.squeeze())
            
            # Robust Target: Any stage > 0 is "Sepsis"
            binary_target = (target > 0).long()
            
            self.auprc.update(probs, binary_target)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute and Log
        try:
            score = self.auprc.compute()
            pl_module.log("val/sepsis_auprc", score, prog_bar=True)
        except Exception:
            # Fallback for empty batches or singular class
            pl_module.log("val/sepsis_auprc", 0.0, prog_bar=True)


# ==============================================================================
# 2. SAFETY INTERLOCK & PRUNING CALLBACK
# ==============================================================================
class SafetyPruningCallback(pl.Callback):
    """
    Implements the "Kill Switch" and "Safety Tax" logic for PCBE.
    
    Logic:
    1. Pruning (Speed): If a model is simply performing poorly (ASHA), stop it.
    2. Safety Gates (Trust):
       - If Explained Variance < 0.0 (Critic Failure), immediate PRUNE.
       - If OOD Rate > 95% (Hallucination), immediate PRUNE.
       
    The actual "Safety Tax" is calculated in the Objective function using
    the metrics logged here.
    """
    def __init__(self, trial: optuna.trial.Trial):
        self.trial = trial

    def on_validation_epoch_end(self, trainer, pl_module):
        """Checks for pathological collapses early to save compute."""
        # Pruning is only reliable after the warmup and a bit of learning
        # We give it until Epoch 1 to show ANY sign of life
        if trainer.current_epoch < 1:
            return
            
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        # --- COMPUTE UTILITY SCORE ---
        # Get components (defaulting to safe values to prevent crash)
        # [REVERTED] aligned back to Local Injector keys
        auprc = metrics.get("val/sepsis_auprc", torch.tensor(0.0)).item()
        auroc = metrics.get("val/sepsis_auroc", torch.tensor(0.5)).item()
        ece   = metrics.get("val/clinical_ece", torch.tensor(0.5)).item() 
        ood_rate = metrics.get("val/ood_rate_avg", torch.tensor(0.0)).item()
        ev = metrics.get("train/explained_var", torch.tensor(0.0)).item()

        # KOMOROWSKI UTILITY FUNCTION (Base Score)
        # Weighting Precision (AUPRC) higher because False Negatives kill in Sepsis.
        utility_score = (0.6 * auprc) + (0.3 * auroc) - (0.1 * ece)
        
        # --- HARD SAFETY GATES (Immediate Termination) ---
        # Gate 1: Critic Collapse (Epoch 1)
        if epoch == 1 and ev < 0.01:
            message = f"GATE 1 KILLED: Critic Collapse (EV={ev:.4f})"
            logger.warning(message)
            self.trial.set_user_attr("fail_reason", "critic_collapse")
            raise optuna.TrialPruned(message)
            
        # Gate 2: Hallucination Mode (Epoch 3)
        if epoch == 3 and ood_rate > 0.95:
            message = f"GATE 2 KILLED: Hallucination (OOD={ood_rate:.2f})"
            logger.warning(message)
            self.trial.set_user_attr("fail_reason", "hallucination")
            raise optuna.TrialPruned(message)

        # --- ASHA PRUNING REPORT ---
        # We report the raw utility score. The "Safety Tax" is applied at the very end
        # to ensure the pruner works on the performance metric primarily.
        self.trial.report(utility_score, step=epoch)
        
        if self.trial.should_prune():
            message = f"ASHA PRUNED: Score {utility_score:.4f} below median at epoch {epoch}"
            logger.info(message)
            raise optuna.TrialPruned(message)


# ==============================================================================
# 3. THE OPTIMIZATION OBJECTIVE
# ==============================================================================
class PCBEOptimizer:
    def __init__(self, base_cfg: DictConfig):
        self.base_cfg = base_cfg
        self.hw_ctx = get_hardware_context()
        
        # Create output directory for study artifacts
        self.study_dir = Path("optimization_results")
        self.study_dir.mkdir(exist_ok=True)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        The Search Loop. Instantiates model with trial params and trains.
        """
        # --- A. HYPERPARAMETER SAMPLING (The "Search Space") ---
        # Centered on data stats but allowing architectural drift.
        
        # 1. Loss Balance (Toxicity vs Awareness)
        pos_weight = trial.suggest_float("pos_weight", 5.0, 30.0)
        
        # 2. RL Dynamics (Exploration vs Exploitation)
        awr_beta = trial.suggest_float("awr_beta", 0.05, 5.0, log=True)
        awr_max_weight = trial.suggest_int("awr_max_weight", 10, 100)
        
        # 3. Multi-Task Tradeoff (Diffusion Quality vs Sepsis Detection)
        aux_loss_scale = trial.suggest_float("aux_loss_scale", 0.05, 1.0)
        
        # 4. Optimization Physics & Throughput
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        # Physical batch size (Now targeting High-RAM utilization)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048]) 
        # Gradient Accumulation (unlimited effective batch size)
        accum_batches = trial.suggest_int("accumulate_grad_batches", 1, 8)
        
        # 5. Hardware Throttling (I/O Optimization)
        num_workers = trial.suggest_int("num_workers", 2, 8 if self.hw_ctx["accelerator"] == "gpu" else 2)
        prefetch_factor = trial.suggest_int("prefetch_factor", 2, 4)
        
        # --- B. CONFIGURATION INJECTION ---
        # Clone base config and inject trial parameters
        cfg = self.base_cfg.copy()
        
        # [v16.0] Disable struct safety to allow flexible HPO injection
        OmegaConf.set_struct(cfg, False)
        
        # Inject into Model/Train Config
        cfg.train.aux_loss_scale = aux_loss_scale
        cfg.train.awr_beta = awr_beta
        cfg.train.awr_max_weight = awr_max_weight
        cfg.train.pos_weight = pos_weight 
        cfg.train.batch_size = batch_size
        cfg.train.accumulate_grad_batches = accum_batches
        cfg.train.num_workers = num_workers
        cfg.train.prefetch_factor = prefetch_factor
        
        # [CRITICAL] Fix "Warmup Trap": Scale warmup down for short HPO trials
        # 500 steps is too long for a 15% data sweep (~66 steps/epoch).
        cfg.train.warmup_steps = 50 
        
        # [SOTA PERFORMANCE] Linear LR Scaling Rule
        # Scale LR by Effective Batch Size ratio relative to base (1024)
        effective_batch = batch_size * accum_batches
        base_batch = 1024
        scaled_lr = lr * (effective_batch / base_batch)
        cfg.train.lr = scaled_lr
        
        # Log effective batch size for analysis
        trial.set_user_attr("effective_batch_size", effective_batch)
        trial.set_user_attr("physical_batch_size", batch_size)
        
        # --- C. SYSTEM SETUP ---
        # Determinism is crucial for comparative optimization
        set_seed(42) 
        
        datamodule = ICUGeneralistDataModule(cfg, pin_memory=self.hw_ctx["pin_memory"])
        model = ICUGeneralistWrapper(cfg)
        
        # --- D. CALLBACKS ---
        safety_cb = SafetyPruningCallback(trial)
        metric_cb = ClinicalMetricInjector()
        
        # --- E. TRAINER ---
        # Lightweight trainer configuration for speed
        log_freq = 50
        trainer = pl.Trainer(
            default_root_dir=str(self.study_dir / f"trial_{trial.number}"),
            max_epochs=5, # Short horizon for searching (ASHA will prune early)
            accelerator=self.hw_ctx["accelerator"],
            devices=1, 
            precision=self.hw_ctx["precision"],
            callbacks=[safety_cb, metric_cb, PreciseTQDMProgressBar(refresh_rate=log_freq)],
            enable_checkpointing=False, # Save space
            logger=False, # Disable massive logging for sweep
            enable_progress_bar=True,
            num_sanity_val_steps=0, # Critical for Safety Gates to run at Epoch 0/1 correctly
            limit_train_batches=0.15, # [Balanced Speed] 15% is the SOTA sweet spot
            limit_val_batches=0.10,   # [Balanced Speed] 10% is enough for HPO ranking
            log_every_n_steps=log_freq
        )
        
        # --- F. EXECUTION ---
        try:
            trainer.fit(model, datamodule=datamodule)
        except optuna.TrialPruned as e:
            # Re-raise strictly for Optuna to handle state
            raise e
        except Exception as e:
            # Handle NaN loss or generic crashes as pruned (score = -1.0)
            logger.error(f"Trial {trial.number} CRASHED: {e}")
            return -1.0
            
        # --- G. SCORE CALCULATION & SAFETY TAX ---
        metrics = trainer.callback_metrics
        
        # [REVERTED] Re-aligned to local metric keys
        auprc = metrics.get("val/sepsis_auprc", torch.tensor(0.0)).item()
        auroc = metrics.get("val/sepsis_auroc", torch.tensor(0.5)).item()
        ece   = metrics.get("val/clinical_ece", torch.tensor(0.5)).item()
        ood_rate = metrics.get("val/ood_rate_avg", torch.tensor(0.0)).item()
        
        # Base Utility
        utility_score = (0.6 * auprc) + (0.3 * auroc) - (0.1 * ece)
        
        # The "Safety Tax" (Soft Barrier)
        # If OOD Rate > 10%, we start penalizing.
        # If OOD Rate is 50%, penalty is 2.0 * (0.5 - 0.1) = 0.8 (Massive hit to score)
        # This teaches TPE that high OOD is "expensive"
        safety_tax = 0.0
        if ood_rate > 0.10:
            safety_tax = 2.0 * (ood_rate - 0.10)
            
        final_score = utility_score - safety_tax
        
        # Log attributes for analysis
        trial.set_user_attr("final_utility", utility_score)
        trial.set_user_attr("safety_tax", safety_tax)
        trial.set_user_attr("ood_rate", ood_rate)
            
        return final_score


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================
@hydra.main(version_base=None, config_path="../conf", config_name="generalist")
def main(cfg: DictConfig):
    
    logger.info("="*80)
    logger.info("  🩺 PCBE OPTIMIZER STARTING (Noble Method v4.0)")
    logger.info("  Physiologically-Constrained Bayesian Evolution")
    logger.info("="*80)
    
    # 1. Setup Persistent Storage (SQLite)
    # This ensures that if the script crashes after 12 hours, we lose nothing.
    storage_url = "sqlite:///apex_optimization.db"
    
    # 2. Setup Sampler (Multivariate TPE)
    # Multivariate=True captures correlations (e.g., Low Beta needs High LR)
    sampler = optuna.samplers.TPESampler(
        seed=2025, 
        multivariate=True, 
        constant_liar=True, # Helps if we ever run parallel workers
        n_startup_trials=10
    )
    
    # 3. Setup Pruner (Hyperband)
    # Aggressively kills bottom 70% of trials at each rung
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, 
        max_resource=10, 
        reduction_factor=3
    )
    
    # 4. Create/Load Study
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name="apex_sepsis_sota_v4",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    logger.info(f"Study loaded. Existing trials: {len(study.trials)}")
    
    # 4.5 baseline Warmstart (The "Noble" Bootstrap)
    # If the study is new, enqueue a known good baseline configuration.
    if len(study.trials) == 0:
        logger.info("Enqueuing Baseline Warmstart Trial...")
        study.enqueue_trial({
            "pos_weight": 25.0,
            "awr_beta": 0.40,
            "awr_max_weight": 30,
            "aux_loss_scale": 0.20,
            "lr": 2.0e-4,
            "batch_size": 1024,
            "accumulate_grad_batches": 1,
            "num_workers": 4,
            "prefetch_factor": 2
        })
    
    # 5. Run Optimization
    optimizer_engine = PCBEOptimizer(cfg)
    
    # We aim for 50-60 completed trials for the "Noble" sweet spot
    TOTAL_TRIALS = 60
    remaining_trials = max(0, TOTAL_TRIALS - len(study.trials))
    
    if remaining_trials > 0:
        logger.info(f"Starting Search for {remaining_trials} more trials...")
        try:
            study.optimize(
                optimizer_engine.objective, 
                n_trials=remaining_trials, 
                timeout=None, # Run until completion
                gc_after_trial=True # Cleanup memory
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user. Saving current best.")
    else:
        logger.info("Study already complete!")

    # 6. Report & Serialize Results
    logger.info("="*80)
    logger.info("  🎉 OPTIMIZATION COMPLETE")
    logger.info("="*80)
    
    if len(study.trials) > 0:
        best_trial = study.best_trial
        logger.info(f"Best Score (Utility - Tax): {best_trial.value:.4f}")
        logger.info(f"Raw Utility: {best_trial.user_attrs.get('final_utility', 'N/A')}")
        logger.info(f"Safety Tax: {best_trial.user_attrs.get('safety_tax', 'N/A')}")
        
        logger.info("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
            
        # Serialize Best Config
        best_config_path = Path("best_sota_config.yaml")
        with open(best_config_path, "w") as f:
            f.write("# SOTA Sepsis Configuration (PCBE Discovered v4.0)\n")
            f.write(f"# Best Score: {best_trial.value:.4f}\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        
        # Serialize Optimization Report
        report_path = Path("optimization_report.json")
        with open(report_path, "w") as f:
            report = {
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "user_attrs": best_trial.user_attrs,
                "trial_id": best_trial.number
            }
            json.dump(report, f, indent=4)
                
        logger.info(f"Saved best configuration to {best_config_path}")
        logger.info("You may now run 'train_generalist.py' with these exact values.")
    else:
        logger.warning("No successful trials found.")

if __name__ == "__main__":
    # Ensure project root is in path
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    
    main()