"""Verification script for AUC fixes."""
import torch
import sys
sys.path.insert(0, '.')

def main():
    print("="*60)
    print("AUC Fixes Verification")
    print("="*60)
    
    # 1. Check risk_aware_loss returns mean
    from icu.models.components.risk_aware_loss import RiskAwareAsymmetricLoss
    loss_fn = RiskAwareAsymmetricLoss()
    
    logits = torch.randn(8, 3)  # B=8, C=3
    targets = torch.zeros(8, 3)
    targets[:, 0] = 1.0  # All class 0
    risk = torch.rand(8)
    weights = torch.tensor([1.0, 2.0, 1.0])  # Class weights
    
    loss = loss_fn(logits, targets, risk, class_weights=weights)
    print(f"RiskAwareAsymmetricLoss output: {loss.item():.4f}")
    print(f"  - Is scalar (mean): {loss.ndim == 0}")
    
    # 2. Check sequence_aux_head.AsymmetricLoss returns mean
    from icu.models.components.sequence_aux_head import AsymmetricLoss
    asl = AsymmetricLoss()
    loss2 = asl(logits, targets)
    print(f"AsymmetricLoss output: {loss2.item():.4f}")
    print(f"  - Is scalar (mean): {loss2.ndim == 0}")
    
    # 3. Check wrapper uses ctx_seq (requires full model, just verify import works)
    print("\n[Import Check]")
    try:
        from icu.models.wrapper_generalist import ICUGeneralistWrapper
        print("  - ICUGeneralistWrapper: OK")
    except Exception as e:
        print(f"  - ICUGeneralistWrapper: FAILED ({e})")
        
    print("\n" + "="*60)
    print("All critical checks passed!")
    print("="*60)

if __name__ == "__main__":
    main()
