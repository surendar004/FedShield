"""
Lightweight simulator that POSTs randomized threat events to the local API every minute.
Does not modify server logic; uses the existing /api/report_threat endpoint.
"""
from __future__ import annotations
import random
import time
import uuid
from datetime import datetime
import requests

API = "http://localhost:5000/api/report_threat"

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
    # Mark threat with some probability
    is_threat = file_path.endswith(".exe") or cpu_pct > 80 or net_bytes >= 1048576 or file_access_count > 25
    return {
        "id": str(uuid.uuid4()),
        "client_id": client_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu_pct": cpu_pct,
        "net_bytes": net_bytes,
        "file_access_count": file_access_count,
        "file_path": file_path,
        "is_threat": bool(is_threat),
        "action": "none" if not is_threat else "quarantine",
    }

def main():
    print("Starting threat simulator. Posting an event every 60 seconds. Press Ctrl+C to stop.")
    while True:
        payload = random_event()
        try:
            r = requests.post(API, json=payload, timeout=5)
            print(f"[{datetime.utcnow().isoformat()}Z] POST {r.status_code} -> {payload['client_id']} {payload['file_path']} is_threat={payload['is_threat']}")
        except Exception as e:
            print("Simulator error:", e)
        time.sleep(60)

if __name__ == "__main__":
    main()

