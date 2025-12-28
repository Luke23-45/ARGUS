import torch
from torchmetrics.regression import MeanSquaredError

def test_contiguous_error():
    print("Simulating Validation Crash...")
    
    # Simulate [B, T, C] prediction tensor (e.g., [16, 6, 28])
    # Create it in a way that might make it non-contiguous or just standard slicing
    B, T, C = 16, 6, 28
    
    # 1. Create a larger tensor and slice it to simulate non-contiguous memory
    # e.g. permuting axes makes it non-contiguous
    raw = torch.randn(B, C, T) # [16, 28, 6]
    pred = raw.permute(0, 2, 1) # [16, 6, 28], non-contiguous
    
    gt = torch.randn(B, T, C)
    
    metric = MeanSquaredError()
    
    print(f"Pred shape: {pred.shape}, Contiguous: {pred.is_contiguous()}")
    
    # Slice the first 7 channels
    # Slicing a non-contiguous tensor maintains non-contiguity
    pred_slice = pred[..., :7]
    gt_slice = gt[..., :7]
    
    print(f"Slice shape: {pred_slice.shape}, Contiguous: {pred_slice.is_contiguous()}")
    
    try:
        print("Attempting update without .contiguous()...")
        metric.update(pred_slice, gt_slice)
        print("Success! (Unexpected)")
    except RuntimeError as e:
        print(f"Caught expected error: {e}")

    try:
        print("Attempting update WITH .contiguous()...")
        metric.update(pred_slice.contiguous(), gt_slice.contiguous())
        print("Success! Fix verified.")
    except Exception as e:
        print(f"Fix failed: {e}")

if __name__ == "__main__":
    test_contiguous_error()
