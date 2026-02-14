
import torch
import torch.nn as nn
from icu.models.diffusion import ICUConfig
# Import ALL components
from icu.models.components.geometric_projector import GeometricProjector
from icu.models.components.temporal_sampler import TemporalSampler
from icu.models.components.nth_encoder import NTHEncoderBlock
from icu.models.components.sequence_aux_head import SequenceAuxHead
from icu.models.components.alb_encoder import AsymmetricLatentBottleneck
from icu.models.components.risk_scorer import PhysiologicalRiskScorer
from icu.models.components.adaptive_sampler import StateAwareSampler
from icu.models.components.clinical_governor import ConfidenceAwareGovernor
from icu.models.components.distributional_critic import DistributionalValueHead

def check_module(name, module):
    print(f"Checking {name}...")
    try:
        for n, p in module.named_parameters():
            if p.ndim == 0:
                print(f"  [SCALAR DETECTED] {name}.{n} has shape {p.shape}")
            elif p.ndim == 1 and p.shape[0] == 1:
                print(f"  [Vector-1] {name}.{n} has shape {p.shape}")
    except Exception as e:
        print(f"  Error checking {name}: {e}")

print("--- Starting Full Inspection ---")
cfg = ICUConfig()

# 1. Risk Scorer
try:
    rs = PhysiologicalRiskScorer(map_threshold=65.0)
    check_module("PhysiologicalRiskScorer", rs)
except Exception as e: print(f"Skip RiskScorer: {e}")

# 2. Adaptive Sampler
try:
    asamp = StateAwareSampler(min_steps=50)
    check_module("StateAwareSampler", asamp)
except Exception as e: print(f"Skip AdaptiveSampler: {e}")

# 3. Clinical Governor
try:
    gov = ConfidenceAwareGovernor(base_p=0.99)
    check_module("ConfidenceAwareGovernor", gov)
except Exception as e: print(f"Skip Governor: {e}")

# 4. Geometric Projector
try:
    gp = GeometricProjector(d_model=512)
    check_module("GeometricProjector", gp)
except Exception as e: print(f"Skip GeometricProjector: {e}")

# 5. NTH Encoder
try:
    nth = NTHEncoderBlock(cfg)
    check_module("NTHEncoderBlock", nth)
except Exception as e: 
    print(f"Skip NTHEncoder: {e}")

# 6. ALB Encoder
# It seems AsymmetricLatentBottleneck takes (encoder, cfg).
# I'll create a dummy encoder.
class DummyEncoder(nn.Module):
    def forward(self, *args, **kwargs): return None
try:
    alb = AsymmetricLatentBottleneck(DummyEncoder(), cfg)
    check_module("ALBEncoder", alb)
except Exception as e: print(f"Skip ALBEncoder: {e}")

# 7. Distributional Value Head
try:
    dvh = DistributionalValueHead(d_model=512, pred_len=6, num_quantiles=25)
    check_module("DistValueHead", dvh)
except Exception as e: print(f"Skip DistValueHead: {e}")

print("--- Inspection Complete ---")
