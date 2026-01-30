import torch
import torch.nn as nn
import logging
from icu.models.components.temporal_buffer import TemporalContrastiveBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AQ")

def test_tcb_homogenization():
    logger.info("Simulating TCB Diversity Collapse (Smoking Gun #21)...")
    
    d_model = 128
    tcb = TemporalContrastiveBuffer(d_model=d_model, capacity=1024)
    
    # 1. Homogenization Phase: Fill the buffer with very similar "Easy Negatives"
    # Imagine a patient with stable hemodynamics for 24 hours
    base_neg = torch.randn(1, d_model)
    easy_negatives = base_neg + 0.01 * torch.randn(1024, d_model)
    
    logger.info("Filling buffer with 1024 similar easy-negatives...")
    tcb._dequeue_and_enqueue(easy_negatives)
    
    # 2. Measure "Blinded" NCE Loss on regular data
    q = torch.randn(16, d_model)
    k = torch.randn(16, d_model)
    
    out_blind = tcb(q, k)
    logger.info(f"Blinded NCE Loss: {out_blind['nce_loss']:.4f}")
    
    # 3. Novelty Shock Phase: A genuinely different "Hard Negative" appears
    # This represents a different patient phenotype (e.g. Cardiogenic vs Sepsis)
    novel_neg = torch.randn(1, d_model) * 10.0 # Extreme difference
    
    # Re-calculate loss with the novel negative as the positive target (k)
    # This simulates the model suddenly being forced to differentiate from something it hasn't seen
    out_shock = tcb(q, novel_neg.expand(16, -1))
    
    logger.info(f"Novelty Shock NCE Loss: {out_shock['nce_loss']:.4f}")
    
    ratio = out_shock['nce_loss'] / out_blind['nce_loss']
    logger.info(f"Shock Intensity Ratio: {ratio:.2f}x")
    
    if ratio > 5.0:
        logger.error("❌ SMOKING GUN #21: TCB exhibits high Novelty Shock sensitivity due to pool homogenization.")
        logger.warning("Recovery: We need 'Shuffled Buffer Ingestion' or 'Global Cross-Rank Shuffling' to maintain entropy.")
    else:
        logger.info("✅ TCB handled the novelty shock within 5x magnitude.")

if __name__ == "__main__":
    test_tcb_homogenization()
