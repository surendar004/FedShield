"""Post several randomized events to the local API once (helper for dashboard demo)."""
from __future__ import annotations
import random
import uuid
from datetime import datetime
import requests

API = "http://127.0.0.1:5000/api/report_threat"

CLIENTS = ["client1", "client2", "client3", "client4", "client5"]
FILES = [
    "data/sample_file_1.txt",
    "data/sample_file_2.txt",
    "data/malware.exe",
    "data/email_phish.html",
    "data/archive.zip",
]

def random_event():
    client_id = random.choice(CLIENTS)
    file_path = random.choice(FILES)
    cpu_pct = round(random.choice([12.5, 14.0, 13.2, 11.1, 90.0, 23.5, 34.2, 47.1]), 1)
    net_bytes = random.choice([256, 512, 1024, 2048, 4096, 1048576, 2097152])
    file_access_count = random.choice([0, 1, 2, 3, 5, 8, 13, 21, 34, 55])
    is_threat = file_path.endswith('.exe') or cpu_pct > 80 or net_bytes >= 1048576 or file_access_count > 25
    return {
        
        "client_id": client_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu_pct": cpu_pct,
        "net_bytes": net_bytes,
        "file_access_count": file_access_count,
        "file_path": file_path,
        "is_threat": bool(is_threat),
        "action": "none" if not is_threat else "quarantine",
    }

def main(n=5):
    print(f"Posting {n} randomized events to {API}")
    for i in range(n):
        payload = random_event()
        try:
            r = requests.post(API, json=payload, timeout=5)
            print(f"POST {r.status_code}: {payload['client_id']} {payload['file_path']} threat={payload['is_threat']}")
        except Exception as e:
            print("Failed to POST:", e)

if __name__ == '__main__':
    main()
