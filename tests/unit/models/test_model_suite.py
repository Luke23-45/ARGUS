import pytest
import torch
import torch.nn as nn
from icu.models.diffusion import (
    ICUConfig, 
    RotaryEmbedding, 
    apply_rotary_pos_emb, 
    DiTBlock1D,
    ICUUnifiedPlanner
)
from icu.models.apex_moe_planner import APEX_MoE_Planner

# ==============================================================================
# 1. CORE PHYSICS TESTS (RoPE)
# ==============================================================================

def test_rope_rotation_invariance():
    """Verify RoPE rotations preserve L2 norm (magnitude)."""
    dim = 64
    seq_len = 10
    rope = RotaryEmbedding(dim=dim // 8) # head_dim
    cos, sin = rope(seq_len, device="cpu")
    
    # Random query [B, H, L, D]
    q = torch.randn(1, 8, seq_len, dim // 8)
    q_norm_before = torch.norm(q, dim=-1)
    
    q_rotated = apply_rotary_pos_emb(q, cos, sin)
    q_norm_after = torch.norm(q_rotated, dim=-1)
    
    # Rotation should be norm-preserving
    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5)

# ==============================================================================
# 2. ARCHITECTURE COMPONENT TESTS (DiT)
# ==============================================================================

def test_dit_block_shapes():
    """Verify DiTBlock1D forward pass and shape consistency."""
    cfg = ICUConfig(d_model=128, n_heads=4)
    block = DiTBlock1D(cfg)
    
    B, L_fut, L_hist = 2, 6, 25 # (24 history + 1 static)
    x = torch.randn(B, L_fut, 128)
    context = torch.randn(B, L_hist, 128)
    condition = torch.randn(B, 128)
    
    # Mock RoPE tensors
    cos = torch.randn(1, 1, L_fut, 32)
    sin = torch.randn(1, 1, L_fut, 32)
    ctx_cos = torch.randn(1, 1, L_hist, 32)
    ctx_sin = torch.randn(1, 1, L_hist, 32)
    
    out = block(x, context, condition, cos, sin, ctx_cos, ctx_sin)
    assert out.shape == (B, L_fut, 128)

# ==============================================================================
# 3. APEX-MOE GATING & CLONING TESTS
# ==============================================================================

def test_moe_expert_separation():
    """Verify APEX-MoE correctly clones backbone and freezes shared weights."""
    base_cfg = ICUConfig(d_model=64, n_layers=2)
    generalist = ICUUnifiedPlanner(base_cfg)
    
    moe = APEX_MoE_Planner(generalist)
    
    # 1. Verify backbone cloning
    # Parameters in experts should be distinct objects after cloning
    for p_stb, p_crsh in zip(moe.expert_stable.parameters(), moe.expert_crash.parameters()):
        assert p_stb is not p_crsh
        assert torch.equal(p_stb, p_crsh) # Initialized the same
        
    # 2. Verify frozen perception layer
    # The History Encoder should be frozen
    for p in moe.encoder.parameters():
        assert p.requires_grad is False

def test_hard_gating_logic():
    """Verify Hard Gating correctly routes trajectories based on labels."""
    cfg = ICUConfig(d_model=64)
    generalist = ICUUnifiedPlanner(cfg)
    moe = APEX_MoE_Planner(generalist)
    
    # Create a mixed batch
    B = 4
    batch = {
        "observed_data": torch.randn(B, 24, 5),
        "future_data": torch.randn(B, 6, 5),
        "static_context": torch.randn(B, 2),
        "outcome_label": torch.tensor([0.0, 1.0, 0.0, 1.0]) # Stable, Crash, Stable, Crash
    }
    
    output = moe(batch)
    
    # In APEX forward, we expect a breakdown of batch sizes
    assert output["stable_batch_size"] == 2
    assert output["crash_batch_size"] == 2
    assert "loss" in output

def test_moe_inference_blending():
    """Verify Soft Gating blends experts during sampling."""
    cfg = ICUConfig(d_model=64, use_auxiliary_head=True)
    generalist = ICUUnifiedPlanner(cfg)
    moe = APEX_MoE_Planner(generalist)
    moe.eval()
    
    B = 2
    batch = {
        "observed_data": torch.randn(B, 24, 5),
        "static_context": torch.randn(B, 2)
    }
    
    with torch.no_grad():
        # sampling should return raw clinical values (unnormalized)
        # We just want to ensure it runs without shape/logic crashes
        sample = moe.sample(batch, num_steps=5)
        
    assert sample.shape == (B, cfg.pred_len, cfg.input_dim)
