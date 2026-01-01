1 07:33:18,713][APEX_Generalist_v12][INFO] - [Rank 0] Calibrating Normalizer...
[2025-12-31 07:33:18,714][APEX_Normalizer_Ultimate][INFO] - [NORMALIZER] Calibrating from /content/ARGUS/sepsis_clinical_28/train_index.json...
[2025-12-31 07:33:19,363][APEX_Normalizer_Ultimate][INFO] - [NORMALIZER] Calibration Complete!
  Mode: robust_quantile
  Channels: 28 (7 log-transformed)
  Safety Margin: 5.0%
  Status: ONLINE
[2025-12-31 07:33:19,392][APEX_Generalist_v12][INFO] - [Rank 0] Sampling Trajectories for AWR Whitening...
[2025-12-31 07:33:19,393][APEX_Generalist_v12][INFO] - [Rank 0] AWR Calibration: Sampling trajectories (N=5000 of 451305)...
AWR Calibration: 100% 5000/5000 [00:08<00:00, 573.02it/s]
[2025-12-31 07:33:28,139][APEX_Generalist_v12][INFO] - AWR Stats: Mean=0.3237, Std=0.1849
[2025-12-31 07:33:28,139][APEX_Advantage_Ultimate][INFO] - [ADVANTAGE] Stats Locked: mu=0.3237, sigma=0.1849
Epoch 0:   0% 0/1763 [00:00<?, ?it/s]              [2025-12-31 07:33:45,938][APEX_Phase1_Engine][CRITICAL] - [CRITICAL] Training crashed: CAGrad.step() got an unexpected keyword argument 'closure'
[2025-12-31 07:33:45,945][APEX_Phase1_Engine][CRITICAL] - Traceback (most recent call last):
  File "/content/ARGUS/icu/train/train_generalist.py", line 487, in main
    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 49, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
    self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 1079, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 1123, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py", line 217, in run
    self.advance()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py", line 465, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 153, in run
    self.advance(data_fetcher)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 354, in advance
    batch_output = self.manual_optimization.run(kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/manual.py", line 95, in run
    self.advance(kwargs)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/manual.py", line 115, in advance
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 329, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/strategies/strategy.py", line 391, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/ARGUS/icu/models/wrapper_generalist.py", line 372, in training_step
    opt.step()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/core/optimizer.py", line 154, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/strategies/strategy.py", line 239, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/amp.py", line 76, in optimizer_step
    return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/precision.py", line 123, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/lr_scheduler.py", line 133, in wrapper
    return func.__get__(opt, opt.__class__)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/optimizer.py", line 517, in wrapper
    out = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
TypeError: CAGrad.step() got an unexpected keyword argument 'closure'

[2025-12-31 07:33:45,946][APEX_Phase1_Engine][INFO] - [DONE] Phase 1 Execution Finished.
Error executing job with overrides: []
Traceback (most recent call last):
  File "/content/ARGUS/icu/train/train_generalist.py", line 516, in main
    raise e
  File "/content/ARGUS/icu/train/train_generalist.py", line 487, in main
    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 49, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
    self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 1079, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 1123, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py", line 217, in run
    self.advance()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py", line 465, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 153, in run
    self.advance(data_fetcher)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 354, in advance
    batch_output = self.manual_optimization.run(kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/manual.py", line 95, in run
    self.advance(kwargs)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/manual.py", line 115, in advance
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 329, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/strategies/strategy.py", line 391, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/ARGUS/icu/models/wrapper_generalist.py", line 372, in training_step
    opt.step()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/core/optimizer.py", line 154, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/strategies/strategy.py", line 239, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/amp.py", line 76, in optimizer_step
    return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/precision.py", line 123, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/lr_scheduler.py", line 133, in wrapper
    return func.__get__(opt, opt.__class__)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/optimizer.py", line 517, in wrapper
    out = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
TypeError: CAGrad.step() got an unexpected keyword argument 'closure'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /content/drive/MyDrive/icu/logs/icu/wandb/offline-run-20251231_073316-i6uy2h0f
wandb: Find logs at: ../drive/MyDrive/icu/logs/icu/wandb/offline-run-20251231_073316-i6uy2h0f/logs