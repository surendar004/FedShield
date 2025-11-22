#!/usr/bin/env python3
"""Sanitize `server/global_logs.json` by extracting JSON objects and saving a clean array.
Creates a backup `global_logs.json.bak.<timestamp>` before overwriting.
"""
import json
import os
import time
from json import JSONDecoder

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(ROOT, 'server', 'global_logs.json')
if not os.path.exists(DATA_FILE):
    print('No global_logs.json found at', DATA_FILE)
    raise SystemExit(1)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
bak = DATA_FILE + '.bak.' + time.strftime('%Y%m%d-%H%M%S')
with open(bak, 'w', encoding='utf-8') as f:
    f.write(content)
print('Backed up to', bak)

decoder = JSONDecoder()
objs = []
idx = 0
length = len(content)
# Try to parse whole file first
try:
    data = json.loads(content)
    if isinstance(data, list):
        objs = [o for o in data if isinstance(o, dict)]
    elif isinstance(data, dict):
        objs = [data]
    else:
        print('Top-level JSON parsed but is not list/dict; cleaning by scanning')
        raise ValueError('invalid top-level')
except Exception:
    # Fallback: scan for objects and arrays
    while idx < length:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, offset = decoder.raw_decode(content, idx)
            idx += offset
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        objs.append(x)
            elif isinstance(obj, dict):
                objs.append(obj)
            else:
                # skip non-dict scalars
                pass
        except Exception:
            # If parsing at this idx fails, try to skip one character (tolerant)
            idx += 1

print(f'Collected {len(objs)} JSON object entries')

# Write cleaned file
with open(DATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(objs, f, indent=2, ensure_ascii=False)

print('Wrote cleaned', DATA_FILE)
