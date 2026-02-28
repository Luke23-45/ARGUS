import torch
from icu.models.diffusion import ICUUnifiedPlanner
from icu.config.config_schema import ICUConfig

def test_shapes():
    cfg = ICUConfig(
        input_dim=28,
        static_dim=6,
        history_len=24,
        pred_len=6,
        d_model=768,
        n_heads=12,
        n_layers=8,
        encoder_layers=4,
        use_auxiliary_head=True,
        num_phases=6,
        timesteps=500,
        use_self_conditioning=True
    )
    
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
    
    out = model(batch)
    print("Test passed without crashing!")

if __name__ == "__main__":
    test_shapes()
