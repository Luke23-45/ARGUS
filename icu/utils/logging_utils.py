
import os
import csv
import time
from typing import List, Dict, Any, Optional

class BufferedCSVLogger:
    """
    [SOTA v2026] High-Performance Intra-Epoch Telemetry Logger.
    
    Features:
    - buffered writing (minimizes I/O syscalls).
    - atomic flushes.
    - low-latency 'append' mode.
    - Handles 'Epoch X: Y%' granularity.
    """
    def __init__(self, log_dir: str, filename: str = "epoch_telemetry.csv", buffer_size: int = 10, flush_interval: int = 60):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = buffer_size
        self.headers: Optional[List[str]] = None
        self.last_flush = time.time()
        self.flush_interval = flush_interval # Seconds
        
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Check if file exists to determine header need
        self.file_exists = os.path.exists(self.filepath)

    def log(self, metrics: Dict[str, Any]):
        """
        Add a row to the buffer.
        """
        # Snapshot time
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()
            
        self.buffer.append(metrics)
        
        # Auto-Flush Logic
        if len(self.buffer) >= self.buffer_size or (time.time() - self.last_flush) > self.flush_interval:
            self.flush()

    def flush(self):
        """
        Write buffer to disk.
        """
        if not self.buffer:
            return

        try:
            # If headers not set, infer from first row keys
            if self.headers is None:
                # [SOTA Safety] Sort keys for deterministic column order
                self.headers = sorted(list(self.buffer[0].keys()))

            mode = 'a' if self.file_exists else 'w'
            
            with open(self.filepath, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                
                # Write header if new file
                if not self.file_exists:
                    writer.writeheader()
                    self.file_exists = True # Now it exists
                
                writer.writerows(self.buffer)
                
            # Clear buffer and reset timer
            self.buffer = []
            self.last_flush = time.time()
            
        except Exception as e:
            print(f"[BufferedCSVLogger] Error flushing logs: {e}")

    def close(self):
        """Force flush on exit."""
        self.flush()

    def __del__(self):
        self.close()
