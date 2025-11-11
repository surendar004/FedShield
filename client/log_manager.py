"""Logging and reporting utilities for client nodes.
"""
import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOCAL_LOG = os.path.join(LOG_DIR, 'threat_logs.json')


def write_local_log(entry: dict):
    logs = []
    if os.path.exists(LOCAL_LOG):
        try:
            with open(LOCAL_LOG, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOCAL_LOG, 'w') as f:
        json.dump(logs, f, indent=2)


def post_to_server(entry: dict, server_url='http://localhost:5000/api/report_threat', retries=3, backoff=1):
    for attempt in range(retries):
        try:
            r = requests.post(server_url, json=entry, timeout=5)
            if r.status_code in (200, 201):
                return True
            else:
                time.sleep(backoff)
        except Exception:
            time.sleep(backoff)
    return False
