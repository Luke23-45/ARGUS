import sys
import os
import torch
print("Step 0: Start", flush=True)

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(ROOT_DIR)
print(f"Step 1: Path Added ({ROOT_DIR})", flush=True)

try:
    from icu.models.components.ghost_bank import SepsisGhostBank
    print("Step 2: GhostBank Imported", flush=True)
except Exception as e:
    print(f"Step 2 Error: {e}", flush=True)

try:
    from icu.train.train_generalist import ICUGeneralistDataModule
    print("Step 3: DataModule Imported", flush=True)
except Exception as e:
    print(f"Step 3 Error: {e}", flush=True)

try:
    from icu.models.wrapper_generalist import ICUGeneralistWrapper
    print("Step 4: Wrapper Imported", flush=True)
except Exception as e:
    print(f"Step 4 Error: {e}", flush=True)

print("Step 5: All Imports Done", flush=True)
