import sys
import os
print("Sanity Check Start", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}", flush=True)
print("Sanity Check End", flush=True)
