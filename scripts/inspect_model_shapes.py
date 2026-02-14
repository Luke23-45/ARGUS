
import torch
import torch.nn as nn
from icu.models.diffusion import ICUConfig, TemporalFusionEncoder
from icu.models.components.loss_scaler import BayesianProjectedScaler
from icu.core.gradnorm import GradNormBalancer
from icu.models.components.temporal_sampler import TemporalSampler
from icu.models.components.bypass_context import LateralBypass
from icu.models.components.distributional_critic import DistributionalValueHead
from icu.models.components.sequence_aux_head import SequenceAuxHead
from icu.utils.stabilization import StableContrastiveLoss

def check_module(name, module):
    print(f"Checking {name}...")
    for n, p in module.named_parameters():
        if p.ndim == 0:
            print(f"  [SCALAR DETECTED] {name}.{n} has shape {p.shape}")
        elif p.ndim == 1 and p.shape[0] == 1:
            print(f"  [Vector-1] {name}.{n} has shape {p.shape}")

print("--- Starting Shape Inspection ---")

# 1. Loss Scalar
scaler = BayesianProjectedScaler(num_tasks=7)
check_module("LossScaler", scaler)

# 2. GradNorm
gn = GradNormBalancer(num_tasks=7, shared_params=[torch.tensor([1.0], requires_grad=True)])
check_module("GradNorm", gn)

# 3. Temporal Sampler
ts = TemporalSampler(d_model=512)
check_module("TemporalSampler", ts)

# 4. Lateral Bypass
lb = LateralBypass(input_dim=28, d_model=512)
check_module("LateralBypass", lb)

# 5. Contrastive Loss (though it tracks buffers)
cl = StableContrastiveLoss(d_model=512)
check_module("StableContrastiveLoss", cl)

# 6. Check common patterns in custom modules via dummy config
cfg = ICUConfig()
try:
    enc = TemporalFusionEncoder(cfg)
    check_module("Encoder", enc)
except Exception as e:
    print(f"Skipping Encoder init due to: {e}")

print("--- Inspection Complete ---")
