"""
scripts/upload_to_hf_v3.py
--------------------------------------------------------------------------------
APEX-MoE Data Persistence Suite (v3.0 - Stable Release)

Description:
    Standardized ingestion engine for Clinical Datasets.
    
    CORRECTIONS FROM v2:
    - Reverts to standard `upload_folder` API (Most Stable).
    - Removes deprecated `multi_commits` flag (Fixes Error #1).
    - Removes internal `upload_large_folder` call (Fixes Error #2).
    - Retains auto-LFS injection (Critical for LMDB).

Usage:
    python scripts/upload_to_hf_v3.py --repo_id hellxhell/sepsis-clinical-28 --token <YOUR_HF_TOKEN>
"""

import os
import argparse
import logging
import sys
from pathlib import Path
from huggingface_hub import HfApi, login

# --- CONFIGURATION ---
DEFAULT_REPO = "hellxhell/sepsis-clinical-28"
DATASET_DIR = "sepsis_clinical_28"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("APEX_Persistence")

def ensure_lfs_attributes(folder_path: Path):
    """
    Auto-injects .gitattributes to ensure LMDB/Binary files are treated as LFS.
    """
    lfs_path = folder_path / ".gitattributes"
    
    # Patterns for Clinical/ML Data
    lfs_patterns = [
        "*.mdb filter=lfs diff=lfs merge=lfs -text",
        "*.pt filter=lfs diff=lfs merge=lfs -text",
        "*.safetensors filter=lfs diff=lfs merge=lfs -text",
        "*.parquet filter=lfs diff=lfs merge=lfs -text"
    ]
    
    existing_content = ""
    if lfs_path.exists():
        existing_content = lfs_path.read_text()
    
    missing_patterns = [p for p in lfs_patterns if p not in existing_content]
    
    if missing_patterns:
        logger.info("⚙️  Injecting LFS configurations...")
        with open(lfs_path, "a") as f:
            if existing_content and not existing_content.endswith("\n"):
                f.write("\n")
            f.write("\n".join(missing_patterns) + "\n")
        logger.info("   -> .gitattributes updated.")
    else:
        logger.info("✅ LFS Configuration verified.")

def upload_dataset(repo_id: str, token: str = None, private: bool = False):
    """
    Stable Ingestion Logic using standard HfApi.upload_folder
    """
    if token:
        login(token=token)
        logger.info("🔐 Authenticated with Hugging Face Hub.")

    local_path = Path(DATASET_DIR)
    
    # 1. Path Guard
    if not local_path.exists():
        logger.error(f"FATAL: Source directory '{DATASET_DIR}' not found.")
        return

    # 2. LFS Injection
    ensure_lfs_attributes(local_path)

    api = HfApi()

    # 3. Create Repo (Idempotent)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        logger.info(f"📡 Target Repository: {repo_id}")
    except Exception as e:
        logger.error(f"❌ Repo access warning (might already exist): {e}")

    # 4. Uploading
    logger.info("🚀 Starting Upload (Standard Protocol)...")
    logger.info("   This may take time. The library handles retries/hashing automatically.")

    try:
        # PURE STANDARD CALL - No experimental flags
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="APEX-MoE: SOTA Clinical 28 Dataset (Sync)",
            ignore_patterns=[".git*", "*.tmp"]
        )
        
        logger.info("\n" + "="*80)
        logger.info(" 🎉 SUCCESS: Dataset Persistence Complete.")
        logger.info(f" 🔗 URL: https://huggingface.co/datasets/{repo_id}")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        logger.info("💡 TIP: If 'ConnectionError', simply run this script again. It will resume.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX-MoE Stable Uploader")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO, help="HF Repository ID")
    parser.add_argument("--token", type=str, help="Hugging Face API Token")
    parser.add_argument("--private", action="store_true", default=False,help="Make repository private")
    
    args = parser.parse_args()
    
    print("\n" + "#"*80)
    print("  🚀 APEX-MoE DATA PERSISTENCE ENGINE (v3.0)")
    print("#"*80 + "\n")
    
    upload_dataset(repo_id=args.repo_id, token=args.token, private=args.private)