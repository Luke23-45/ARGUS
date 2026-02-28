import sys
import os
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from icu.models.diffusion import ICUUnifiedPlanner
import hydra
from omegaconf import OmegaConf

def test_shapes():
    # Set up config search path directly
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(config_name="generalist.yaml")
        
    cfg = cfg.model
    
    model = ICUUnifiedPlanner(cfg)
    
    B = 2
    T_obs = 24
    T_pred = 6
    
    past = torch.randn(B, T_obs, 28)
    fut = torch.randn(B, T_pred, 28)
    static = torch.randn(B, 6)
    
    src_mask = torch.ones(B, T_obs, 28)
    padding_mask = torch.zeros(B, T_obs).bool()
    
    batch = {
        "observed_data": past,
        "future_data": fut,
        "static_context": static,
        "src_mask": src_mask,
        "padding_mask": padding_mask,
        "phase_label": torch.randint(0, 6, (B,))
    }
    
    try:
        out = model(batch)
        print("Test passed without crashing!")
        print(f"diffusion_loss shape: {out['diffusion_loss'].shape}")
        if 'aux_logits' in out and out['aux_logits'] is not None:
            print(f"aux_logits shape: {out['aux_logits'].shape}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shapes()
