
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os

# Mock the Component (GhostBank)
class MockGhostBank(nn.Module):
    def __init__(self):
        super().__init__()
        # Register a buffer that tracks state (like 'size' or 'ptr' in the real bank)
        self.register_buffer("counter", torch.tensor(0))
        self.register_buffer("data_store", torch.zeros(10))

    def update(self):
        self.counter += 1
        self.data_store[self.counter % 10] = self.counter

# Mock the Wrapper with BOTH automatic and manual saving
class MockWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.ghost_bank = MockGhostBank()
        
    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        self.ghost_bank.update()
        loss = self(batch).sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    # REPLICATING THE GENERALIST LOGIC
    def on_save_checkpoint(self, checkpoint):
        checkpoint["ghost_bank_state"] = self.ghost_bank.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ghost_bank_state" in checkpoint:
            print("(!) Loading from MANUAL key 'ghost_bank_state'")
            self.ghost_bank.load_state_dict(checkpoint["ghost_bank_state"])

def verify_redundancy():
    print("--- STARTING VERIFICATION ---")
    
    # 1. Train and Save
    model = MockWrapper()
    trainer = pl.Trainer(max_epochs=1, default_root_dir="temp_test_ckpt", enable_checkpointing=True)
    # Fake data
    data = torch.randn(10, 1)
    train_loader = torch.utils.data.DataLoader(data, batch_size=2)
    
    print("\n1. Training for 1 epoch (Mocking update)...")
    trainer.fit(model, train_loader)
    
    # State after training
    print(f"Model Counter after training: {model.ghost_bank.counter.item()}")
    
    # Save checkpoint manually to control path
    ckpt_path = "temp_test_ckpt/redundancy_test.ckpt"
    trainer.save_checkpoint(ckpt_path)
    
    # 2. Inspect Checkpoint
    print(f"\n2. Inspecting Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    
    keys = checkpoint.keys()
    print(f"Top-level Keys: {list(keys)}")
    
    # Check Automatic State Dict
    state_dict = checkpoint['state_dict']
    has_auto = "ghost_bank.counter" in state_dict
    print(f"\n[Automatic] 'ghost_bank.counter' in state_dict? {has_auto}")
    if has_auto:
        print(f"Value in state_dict: {state_dict['ghost_bank.counter'].item()}")
        
    # Check Manual Key
    has_manual = "ghost_bank_state" in checkpoint
    print(f"\n[Manual] 'ghost_bank_state' in checkpoint? {has_manual}")
    if has_manual:
        manual_state = checkpoint['ghost_bank_state']
        print(f"Value in manual dict: {manual_state['counter'].item()}")

    # 3. verify duplication size
    # In a real scenario, data_store would be huge. Here it's small, but we can verify presence.
    if has_auto and has_manual:
        print("\n[CONCLUSION] REDUNDANCY CONFIRMED.")
        print("The state is present TWICE: once in 'state_dict' (automatic) and once in 'ghost_bank_state' (manual).")
        print("Removing 'on_save_checkpoint' manual save will reduce checkpoint size without losing data.")
    else:
        print("\n[CONCLUSION] NO REDUNDANCY FOUND (Unexpected).")

    # Clean up
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("\nCleanup complete.")

if __name__ == "__main__":
    verify_redundancy()
