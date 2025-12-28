2025-12-28 02:53:02.794898: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1766890382.815422    5079 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1766890382.821483    5079 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1766890382.836375    5079 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766890382.836400    5079 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766890382.836404    5079 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766890382.836409    5079 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-12-28 02:53:02.841277: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

================================================================================
  🩺 APEX-MoE: Advanced Physiological Expert (Mixture-of-Experts)
  State-of-the-Art ICU Strategy v8.0 | Clinical Grade Intelligence
  Engine: PyTorch | Precision: Mixed | Guardrails: Active
================================================================================

[2025-12-28 02:53:09,382][APEX_Phase1_Engine][INFO] - Hardware Context: {'accelerator': 'gpu', 'devices': 1, 'precision': 'bf16-mixed', 'pin_memory': True, 'strategy': 'auto'}
/usr/local/lib/python3.12/dist-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
[2025-12-28 02:53:09,387][APEX_Generalist_v12][INFO] - Initializing ICUGeneralistWrapper (Ultimate Edition v12.0)...
[2025-12-28 02:53:10,004][APEX_Normalizer_Ultimate][INFO] - [NORMALIZER] Initialized: ts=28, static=6, mode=global_quantile
[2025-12-28 02:53:10,379][APEX_Advantage_Ultimate][INFO] - [ADVANTAGE] Initialized: beta=0.4, gamma=0.99, lambda=0.95, max_weight=20.0
[2025-12-28 02:53:10,388][APEX_Phase1_Engine][INFO] - [MODEL] Parameters: Trainable=89.31M, Total=89.31M
Using bfloat16 Automatic Mixed Precision (AMP)
💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
Trainer already configured with model summary callbacks: [<class 'pytorch_lightning.callbacks.model_summary.ModelSummary'>]. Skipping setting a default `ModelSummary` callback.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
`Trainer(limit_val_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
[2025-12-28 02:53:10,429][APEX_Phase1_Engine][INFO] - ================================================================================
[2025-12-28 02:53:10,429][APEX_Phase1_Engine][INFO] - [LAUNCH] Starting Phase 1 Generalist Training
[2025-12-28 02:53:10,429][APEX_Phase1_Engine][INFO] - [CONFIG] Output Dir: outputs/phase1
[2025-12-28 02:53:10,430][APEX_Phase1_Engine][INFO] - [CONFIG] Epochs: 100
[2025-12-28 02:53:10,430][APEX_Phase1_Engine][INFO] - [CONFIG] Batch Size: 1024
[2025-12-28 02:53:10,430][APEX_Phase1_Engine][INFO] - [CONFIG] Learning Rate: 0.0002
[2025-12-28 02:53:10,430][APEX_Phase1_Engine][INFO] - ================================================================================
[2025-12-28 02:53:10,517][APEX_Data_Frontier][INFO] - [Tier 0] Valid Local Data Found at '/content/ARGUS/sepsis_clinical_28'. System Ready.
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id hz32rj3p.
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in outputs/phase1/wandb/offline-run-20251228_025310-hz32rj3p
[2025-12-28 02:53:11,453][APEX_Data_Frontier][INFO] - [TRAIN] Loading Index: /content/ARGUS/sepsis_clinical_28/train_index.json
[2025-12-28 02:53:11,795][APEX_Data_Frontier][INFO] - [TRAIN] Initialized. Windows: 451,305 | Episodes: 23,952
[2025-12-28 02:53:11,796][APEX_Data_Frontier][INFO] - Augmentation Active: Noise=0.01, MaskDrop=0.0
[2025-12-28 02:53:11,797][APEX_Data_Frontier][INFO] - [VAL] Loading Index: /content/ARGUS/sepsis_clinical_28/val_index.json
[2025-12-28 02:53:11,846][APEX_Data_Frontier][INFO] - [VAL] Initialized. Windows: 51,685 | Episodes: 2,668
[2025-12-28 02:53:11,846][APEX_Phase1_Engine][INFO] - [Rank 0] Datasets Ready: Train=451305, Val=51685
[2025-12-28 02:53:11,848][icu.callbacks][INFO] - ClinicalMetricCallback: Metrics ready on cuda:0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[2025-12-28 02:53:11,980][OptimizerFactory][INFO] - AdamW Configured: Fused=True. Decayed: 57, Raw: 93
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/model_summary/model_summary.py:242: Precision bf16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.

   | Name                      | Type                         | Params | Mode  | FLOPs
--------------------------------------------------------------------------------------------
0  | model                     | ICUUnifiedPlanner            | 89.3 M | train | 0    
1  | model.encoder             | TemporalFusionEncoder        | 21.3 M | train | 0    
2  | model.encoder.vitals_proj | Linear                       | 34.6 K | train | 0    
3  | model.encoder.static_proj | Linear                       | 5.4 K  | train | 0    
4  | model.encoder.rope        | RotaryEmbedding              | 0      | train | 0    
5  | model.encoder.layers      | ModuleList                   | 18.9 M | train | 0    
6  | model.encoder.pool        | TimeAttentionPooling         | 2.4 M  | train | 0    
7  | model.encoder.out_norm    | RMSNorm                      | 768    | train | 0    
8  | model.backbone            | DiffusionActionHead          | 67.4 M | train | 0    
9  | model.backbone.in_proj    | Linear                       | 43.8 K | train | 0    
10 | model.backbone.rope       | RotaryEmbedding              | 0      | train | 0    
11 | model.backbone.time_mlp   | Sequential                   | 1.2 M  | train | 0    
12 | model.backbone.blocks     | ModuleList                   | 66.2 M | train | 0    
13 | model.backbone.final_norm | RMSNorm                      | 768    | train | 0    
14 | model.backbone.out_head   | Linear                       | 21.5 K | train | 0    
15 | model.scheduler           | NoiseScheduler               | 0      | train | 0    
16 | model.aux_head            | Sequential                   | 297 K  | train | 0    
17 | model.aux_head.0          | Linear                       | 295 K  | train | 0    
18 | model.aux_head.1          | SiLU                         | 0      | train | 0    
19 | model.aux_head.2          | Dropout                      | 0      | train | 0    
20 | model.aux_head.3          | Linear                       | 2.3 K  | train | 0    
21 | model.value_head          | Sequential                   | 297 K  | train | 0    
22 | model.value_head.0        | Linear                       | 295 K  | train | 0    
23 | model.value_head.1        | SiLU                         | 0      | train | 0    
24 | model.value_head.2        | Linear                       | 2.3 K  | train | 0    
25 | model.phys_loss           | PhysiologicalConsistencyLoss | 0      | train | 0    
26 | model.normalizer          | ClinicalNormalizer           | 0      | train | 0    
27 | phys_loss                 | PhysiologicalConsistencyLoss | 0      | train | 0    
28 | train_loss_total          | MeanMetric                   | 0      | train | 0    
29 | train_loss_diff           | MeanMetric                   | 0      | train | 0    
30 | train_loss_critic         | MeanMetric                   | 0      | train | 0    
31 | train_loss_phys           | MeanMetric                   | 0      | train | 0    
32 | train_loss_aux            | MeanMetric                   | 0      | train | 0    
33 | train_awr_ess             | MeanMetric                   | 0      | train | 0    
34 | train_explained_var       | MeanMetric                   | 0      | train | 0    
35 | val_mse_global            | MeanSquaredError             | 0      | train | 0    
36 | val_mse_hemo              | MeanSquaredError             | 0      | train | 0    
37 | val_mse_labs              | MeanSquaredError             | 0      | train | 0    
38 | val_mse_electrolytes      | MeanSquaredError             | 0      | train | 0    
39 | val_acc_sepsis            | MulticlassAccuracy           | 0      | train | 0    
40 | val_auroc_sepsis          | BinaryAUROC                  | 0      | train | 0    
41 | val_ood_rate              | MeanMetric                   | 0      | train | 0    
42 | val_safe_traj_count       | MeanMetric                   | 0      | train | 0    
43 | val_phys_violation_rate   | MeanMetric                   | 0      | train | 0    
44 | val_ece                   | MeanMetric                   | 0      | train | 0    
45 | val_oe                    | MeanMetric                   | 0      | train | 0    
--------------------------------------------------------------------------------------------
89.3 M    Trainable params
0         Non-trainable params
89.3 M    Total params
357.228   Total estimated model params size (MB)
204       Modules in train mode
0         Modules in eval mode
0         Total Flops
[2025-12-28 02:53:12,614][APEX_Generalist_v12][INFO] - [Rank 0] Calibrating Normalizer...
[2025-12-28 02:53:12,619][APEX_Normalizer_Ultimate][INFO] - [NORMALIZER] Calibrating from /content/ARGUS/sepsis_clinical_28/train_index.json...
[2025-12-28 02:53:14,276][APEX_Normalizer_Ultimate][INFO] - [NORMALIZER] Calibration Complete!
  Mode: robust_quantile
  Channels: 28 (7 log-transformed)
  Safety Margin: 5.0%
  Status: ONLINE
[2025-12-28 02:53:14,407][APEX_Generalist_v12][INFO] - [Rank 0] Syncing EMA shadow with calibrated normalizer...
[2025-12-28 02:53:14,410][APEX_Generalist_v12][INFO] - [Rank 0] EMA shadow sync complete.
[2025-12-28 02:53:14,413][APEX_Generalist_v12][INFO] - [Rank 0] Sampling Trajectories for AWR Whitening...
[2025-12-28 02:53:14,415][APEX_Generalist_v12][INFO] - [Rank 0] Starting Population Scan (N=451305)... This may take a few minutes.
AWR Calibration: 100% 451305/451305 [11:52<00:00, 633.53it/s]
[2025-12-28 03:05:06,967][APEX_Generalist_v12][INFO] - AWR Stats: Mean=0.3226, Std=0.1816
[2025-12-28 03:05:06,968][APEX_Advantage_Ultimate][INFO] - [ADVANTAGE] Stats Locked: mu=0.3226, sigma=0.1816
Epoch 0:  14% 60/440 [04:19<27:24,  4.33s/it, L_step=2.369, D=0.997, critic_L=1.816, awr_ess=0.977, awr_max_w=20.000, explained_var=-0.063, E=1.703, reward_mean=0.327, curr_phys_weight=0.010, lr=0.000, GN=8.625]
