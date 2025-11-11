"""Client node: loads local model, streams sample logs, detects anomalies, isolates files, and reports to Flask API.
Optionally connects to Flower for federated updates.
"""
import argparse
import time
import os
import sys
import json
import logging

# Ensure repository root is on sys.path so imports like `from client.local_model` work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from client.logging_config import setup_logging
from client.local_model import load_local_model, train_and_save

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)
from client.log_manager import write_local_log, post_to_server
from client.isolation import isolate_file

DATA_CSV = os.path.join(ROOT, 'data', 'sample_logs.csv')


def run_client(client_id='client1', simulate=True):
    print(f'[{client_id}] Starting client node')
    # ensure model exists
    model = load_local_model()

    # load sample logs
    import pandas as pd
    df = pd.read_csv(DATA_CSV)

    for i, row in df.iterrows():
        # Extract values as native Python types using .at accessor
        cpu_pct = float(row.at['cpu_pct'])
        net_bytes = float(row.at['net_bytes'])
        file_access_count = int(row.at['file_access_count'])
        
        # Create feature array in same order as model was trained
        features = [[cpu_pct, net_bytes, file_access_count]]
        pred = model.predict(features)[0]
        is_threat = pred == -1
        event = {
            'client_id': client_id,
            'timestamp': row.at['timestamp'] if 'timestamp' in df.columns else None,
            'cpu_pct': float(cpu_pct),
            'net_bytes': float(net_bytes),
            'file_access_count': int(file_access_count),
            'file_path': row.get('file_path'),
            'is_threat': bool(is_threat),
            'action': 'none'
        }

        if is_threat:
            # isolate the file and update event
            file_path = row.at['file_path'] if 'file_path' in df.columns else None
            if file_path:
                try:
                    newpath = isolate_file(str(file_path))
                    event['action'] = 'quarantine'
                    event['quarantined_path'] = newpath
                except Exception as e:
                    event['action'] = 'quarantine_failed'
                    event['error'] = str(e)
            else:
                event['action'] = 'quarantine_failed'
                event['error'] = 'No file path provided'

        write_local_log(event)
        posted = post_to_server(event)
        print(f'[{client_id}] Row {i} threat={is_threat} posted={posted}')
        time.sleep(1)

    print(f'[{client_id}] Client finished streaming logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='client1')
    args = parser.parse_args()
    run_client(client_id=args.id)
