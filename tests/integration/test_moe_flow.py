import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from icu.models.wrapper_apex import ICUSpecialistWrapper
from icu.models.diffusion import ICUUnifiedPlanner, ICUConfig
from icu.datasets import ICUSotaDataset, robust_collate_fn

def test_moe_specialist_fit_step(dummy_lmdb, tmp_path):
    """
    Integration Test: Verifies that Phase 2 (APEX-MoE) can bootstrap 
    from a Phase 1 checkpoint and run a training step.
    """
    # 1. Create a dummy Phase 1 checkpoint
    p1_cfg = ICUConfig(
        input_dim=5, 
        static_dim=2, 
        d_model=64, 
        n_heads=2, 
        n_layers=2, 
        use_auxiliary_head=True
    )
    generalist = ICUUnifiedPlanner(p1_cfg)
    ckpt_path = tmp_path / "phase1_generalist.pt"
    # wrapper_apex expects model state dict at "model" or raw
    torch.save(generalist.state_dict(), ckpt_path)

    # 2. Create Phase 2 Config
    cfg = OmegaConf.create({
        "seed": 42,
        "run_name": "test_moe_pipeline",
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
            "pretrained_path": str(ckpt_path),
            "batch_size": 2,
            "lr": 1e-4,
            "min_lr": 1e-6,
            "weight_decay": 0.01,
            "epochs": 1,
            "ema_decay": 0.99,
            "crash_weight": 5.0
        },
        "logging": {
            "use_wandb": False
        }
    })

    # 3. Initialize Model (This triggers Bootstrapping)
    wrapper = ICUSpecialistWrapper(cfg)

    # 4. Setup DataLoaders
    train_ds = ICUSotaDataset(
        dataset_dir=str(dummy_lmdb),
        split="train",
        history_len=cfg.model.history_len,
        pred_len=cfg.model.pred_len
    )
    # We need at least one 'Crash' (1.0) and one 'Stable' (0.0) sample to test hard gating
    # Our dummy fixture generates them randomly, should be fine for a smoke test.
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=robust_collate_fn)

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_steps=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        fast_dev_run=True,
    )

    # 6. EXECUTE
    trainer.fit(wrapper, train_dataloaders=train_loader)

    # 7. Verification
    assert trainer.state.finished, "Trainer did not complete successfully."
    
    # Check that gradients flow to experts but NOT to encoder
    for name, param in wrapper.model.named_parameters():
        if "encoder" in name:
            assert param.requires_grad is False
        if "router" in name:
            assert param.requires_grad is False
        if "expert_stable" in name or "expert_crash" in name:
            assert param.requires_grad is True
