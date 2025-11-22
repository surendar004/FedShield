"""Cleanup script: null out `threat_type` for events where `is_threat` is False.
Creates a backup `global_logs.json.bak` before modifying.
Run with the project's venv Python: `\.venv\Scripts\python.exe scripts\clean_threat_labels.py`
"""
import json
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / 'server' / 'global_logs.json'
BACKUP = DATA_FILE.with_suffix('.json.bak')

if not DATA_FILE.exists():
    print('No global_logs.json file found at', DATA_FILE)
    exit(1)

shutil.copy2(DATA_FILE, BACKUP)
print('Backup created at', BACKUP)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    try:
        logs = json.load(f)
    except Exception as e:
        print('Failed to parse logs file:', e)
        exit(1)

changed = False
for entry in logs:
    # Treat both `is_threat` and `is_threat`-like keys conservatively
    if not entry.get('is_threat') and entry.get('threat_type') is not None:
        entry['threat_type'] = None
        changed = True

if changed:
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print('Updated logs written to', DATA_FILE)
else:
    print('No changes needed; all non-threat entries already have no threat_type')

print('Done.')
