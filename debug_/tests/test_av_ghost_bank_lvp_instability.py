import torch
from icu.models.components.ghost_bank import SepsisGhostBank
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test_AV")

def test_ghost_bank_lvp():
    logger.info("Simulating Ghost Bank LVP Instability (Smoking Gun #23)...")
    
    capacity = 10
    bank = SepsisGhostBank(capacity=capacity)
    
    # 1. Fill the bank with 'Redundant' samples but with 0 uncertainty
    vitals = torch.randn(capacity, 24, 28)
    masks = torch.ones(capacity, 24, 28)
    labels = torch.zeros(capacity, dtype=torch.long)
    # Identical latents = High Redundancy
    latents = torch.ones(capacity, 512) 
    uncertainties = torch.zeros(capacity, 1) # ZERO uncertainty
    
    bank.update(vitals, masks, labels, latents, uncertainties)
    
    # 2. Trigger the LVP search
    # This happens when we try to update a full bank with a new diverse sample
    new_vit = torch.randn(1, 24, 28)
    new_mask = torch.ones(1, 24, 28)
    new_label = torch.ones(1, dtype=torch.long)
    new_lat = torch.randn(1, 512) # Diverse
    new_unc = torch.ones(1, 1)
    
    try:
        bank.update(new_vit, new_mask, new_label, new_lat, new_unc)
        logger.info("✅ Ghost Bank updated without crashing despite zero uncertainty.")
        
        # Check if the bank is still healthy (no NaNs)
        if torch.isnan(bank.latent_anchors).any():
             logger.error("❌ Bank contains NaNs after update!")
        else:
             logger.info("✅ Bank remains numerically healthy.")
             
    except Exception as e:
        logger.error(f"❌ Ghost Bank crashed during LVP search: {e}")

if __name__ == "__main__":
    test_ghost_bank_lvp()
