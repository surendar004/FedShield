"""Isolation utilities for quarantining suspicious files.
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
QUARANTINE_DIR = os.path.join(ROOT, 'data', 'quarantined')
os.makedirs(QUARANTINE_DIR, exist_ok=True)


def isolate_file(path: str) -> str:
    """Move a file into the quarantine directory. Returns new path."""
    if not os.path.exists(path):
        # for demo, create a small placeholder file if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('placeholder')

    dest = os.path.join(QUARANTINE_DIR, os.path.basename(path))
    try:
        shutil.move(path, dest)
    except Exception:
        # fallback: copy
        shutil.copy2(path, dest)
    return dest
