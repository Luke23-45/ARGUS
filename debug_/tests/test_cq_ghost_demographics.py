import torch
import unittest
from icu.models.components.ghost_bank import SepsisGhostBank

class TestCQGhostDemographics(unittest.TestCase):
    def test_demographic_preservation(self):
        print("\n" + "="*60)
        print("AUDIT: Verifying Ghost Bank Demographic Preservation")
        print("="*60)

        # Config
        history_len = 24
        feature_dim = 28
        latent_dim = 512
        num_ghosts = 4
        
        bank = SepsisGhostBank(capacity=10, history_len=history_len, feature_dim=feature_dim, latent_dim=latent_dim)
        
        # 1. Fill bank with dummy data containing non-zero demographics
        # Column 22 is "Age"
        vitals = torch.zeros(5, history_len, feature_dim)
        vitals[:, :, 22] = 65.0 # Set Age to 65
        vitals[:, :, 23] = 1.0  # Set Gender to 1
        
        masks = torch.ones(5, history_len, feature_dim)
        labels = torch.ones(5, dtype=torch.long)
        latents = torch.randn(5, latent_dim)
        uncertainties = torch.randn(5, 1)
        
        print("Filling bank with dummy sepsis cases (Age=65, Gender=1)...")
        bank.update(vitals=vitals, masks=masks, labels=labels, latents=latents, uncertainties=uncertainties)
        
        # 2. Sample from bank
        print("Sampling ghosts...")
        ghost_batch = bank.sample(num_ghosts=num_ghosts, seed=42)
        
        sampled_static = ghost_batch["static"]
        
        # 3. Verify
        print(f"Sampled Static Context Shape: {sampled_static.shape}")
        
        age_sum = sampled_static[:, 0].sum().item()
        gender_sum = sampled_static[:, 1].sum().item()
        
        if age_sum == 65.0 * num_ghosts:
            print(f"✅ PASS: Age preserved ({age_sum/num_ghosts:.1f}).")
        else:
            print(f"❌ FAIL: Age lost! Expected {65.0}, got {age_sum/num_ghosts:.1f}")
            
        if gender_sum == 1.0 * num_ghosts:
            print(f"✅ PASS: Gender preserved ({gender_sum/num_ghosts:.1f}).")
        else:
            print(f"❌ FAIL: Gender lost! Expected {1.0}, got {gender_sum/num_ghosts:.1f}")

if __name__ == "__main__":
    unittest.main()
