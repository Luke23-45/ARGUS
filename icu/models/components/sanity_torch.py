import sys
import os
print("Torch Sanity Start", flush=True)
import torch
print(f"Torch Version: {torch.__version__}", flush=True)
print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)
print("Torch Sanity End", flush=True)
