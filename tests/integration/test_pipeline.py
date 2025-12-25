import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from icu.models.wrapper_generalist import ICUGeneralistWrapper
from icu.datasets import ICUSotaDataset, robust_collate_fn

def test_full_generalist_fit_step(dummy_lmdb, tmp_path):
    """
    Integration Test: Verifies that the full Generalist pipeline can run 
    one training and one validation step without crashing.
    """
    # 1. Create a "Mini-Config" matching the expected schema
    cfg = OmegaConf.create({
        "seed": 42,
        "run_name": "test_pipeline",
        "output_dir": str(tmp_path),
        "num_workers": 0,
        "dataset": {
            "dataset_dir": str(dummy_lmdb),
            "augment_noise": 0.01
        },
        "model": {
            "input_dim": 5,
            "static_dim": 2,
            "d_model": 64,
            "n_heads": 2,
            "n_layers": 2,
            "history_len": 24,
            "pred_len": 6,
            "use_auxiliary_head": True
        },
        "train": {
            "batch_size": 2,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "weight_decay": 0.01,
            "epochs": 1,
            "ema_decay": 0.99
        },
        "logging": {
            "use_wandb": False
        }
    })

    # 2. Initialize Model
    wrapper = ICUGeneralistWrapper(cfg)

    # 3. Setup DataLoaders
    # Train
    train_ds = ICUSotaDataset(
        dataset_dir=str(dummy_lmdb),
        split="train",
        history_len=cfg.model.history_len,
        pred_len=cfg.model.pred_len
    )
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=robust_collate_fn)

    # Val
    val_ds = ICUSotaDataset(
        dataset_dir=str(dummy_lmdb),
        split="val",
        history_len=cfg.model.history_len,
        pred_len=cfg.model.pred_len
    )
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=robust_collate_fn)

    # 4. Initialize Trainer (Dry Run)
    trainer = pl.Trainer(
        max_steps=1,             # Only 1 step to keep it fast
        num_sanity_val_steps=1,
        accelerator="cpu",       # Force CPU for stability in tests
        logger=False,            # Disable logging
        enable_checkpointing=False,
        fast_dev_run=True,       # Run a single batch of train, val, test
    )

    # 5. EXECUTE
    # This triggers on_fit_start (normalizer fitting), training_step, and validation_step
    trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 6. Verification
    assert trainer.state.finished, "Trainer did not complete successfully."
    assert "val/clinical_mse" in trainer.callback_metrics, "Logging failed to record clinical MSE."
    
    # Check that EMA model was created
    assert wrapper.ema is not None
    assert isinstance(wrapper.ema.get_model().backbone, torch.nn.Module)
