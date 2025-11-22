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
import importlib

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)
from client.log_manager import write_local_log, post_to_server
from client.isolation import isolate_file

DATA_CSV = os.path.join(ROOT, 'data', 'sample_logs.csv')


def run_client(client_id='client1', simulate=True):
    print(f'[{client_id}] Starting client node')
    # Optional metrics: try to import psutil; if unavailable, metrics will be limited
    try:
        import psutil
        _HAS_PSUTIL = True
        _PS_PROC = psutil.Process()
    except Exception:
        _HAS_PSUTIL = False
        _PS_PROC = None
    # simple per-process counter for sent events
    _access_count = 0
    # Try to import the local_model module lazily to avoid importing heavy
    # sklearn C-extensions at module import time for every client process.
    model = None
    # Only attempt to load the heavy sklearn-backed model if explicitly allowed
    # via environment variable `CLIENT_USE_MODEL=true`. This prevents many
    # concurrent client processes from importing sklearn C-extensions and
    # exhausting the paging file / memory.
    use_model = os.environ.get('CLIENT_USE_MODEL', 'false').lower() in ('1', 'true', 'yes')
    if use_model:
        try:
            local_model_mod = importlib.import_module('client.local_model')
            try:
                model = local_model_mod.load_local_model()
            except Exception as e:
                logger.warning('Could not load trained model, will use heuristic: %s', e)
                model = None
        except Exception as e:
            logger.warning('Failed to import client.local_model (skipping model load): %s', e)
            model = None
    else:
        model = None

    # load sample logs
    import pandas as pd
    df = pd.read_csv(DATA_CSV)

    # Read environment overrides
    try:
        env = os.environ
        env_threat_rate = float(env.get('CLIENT_THREAT_RATE')) if env.get('CLIENT_THREAT_RATE') is not None else None
    except Exception:
        env_threat_rate = None
    env_threat_type = os.environ.get('CLIENT_THREAT_TYPE')
    # normalize threat type (lowercase, strip spaces) to match API schema
    if env_threat_type:
        env_threat_type = env_threat_type.strip().lower().replace(' ', '_')
        # validate against allowed types to avoid server-side validation warnings
        allowed_types = {'malware', 'unauthorized_access', 'data_leak', 'phishing', 'system_anomaly'}
        if env_threat_type not in allowed_types:
            logger.warning('CLIENT_THREAT_TYPE "%s" is not a known type; ignoring it', env_threat_type)
            env_threat_type = None
    sleep_override = float(os.environ.get('CLIENT_SLEEP')) if os.environ.get('CLIENT_SLEEP') else 1.0

    for i, row in df.iterrows():
        # Extract values as native Python types using .at accessor
        cpu_pct = float(row.at['cpu_pct'])
        net_bytes = float(row.at['net_bytes'])
        file_access_count = int(row.at['file_access_count'])
        
        # Create feature array in same order as model was trained
        features = [[cpu_pct, net_bytes, file_access_count]]
        if model is not None:
            pred = model.predict(features)[0]
            model_detected_threat = pred == -1
        else:
            model_detected_threat = False

        # If environment threat rate provided, use probabilistic override; otherwise use model or heuristic
        import random
        if env_threat_rate is not None:
            is_threat = random.random() < float(env_threat_rate)
        elif model is not None:
            is_threat = model_detected_threat
        else:
            # Lightweight heuristic: high cpu or many file accesses indicate threat, else low random chance
            is_threat = (cpu_pct > 80.0) or (file_access_count > 40) or (random.random() < 0.02)
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

        # Attach threat_type only when the event is actual threat or when forced via env.
        # This avoids sending misleading labels for non-detected events.
        force_label = os.environ.get('CLIENT_FORCE_LABEL', 'false').lower() in ('1', 'true', 'yes')
        if (is_threat and env_threat_type) or force_label:
            event['threat_type'] = env_threat_type
        else:
            event['threat_type'] = None

        # Attach lightweight client metrics to payload before sending
        try:
            _access_count += 1
            # base measurements
            base_cpu = float(psutil.cpu_percent(interval=0.05)) if _HAS_PSUTIL else float(random.uniform(5.0, 12.0))
            base_mem = float(_PS_PROC.memory_info().rss / 1024**2) if _HAS_PSUTIL and _PS_PROC else float(random.uniform(10.0, 40.0))
            base_net = float(net_bytes)
            base_faccess = int(file_access_count)

            # Amplify metrics for detected threats so they stand out in dashboard
            if is_threat:
                cpu_metric = round(base_cpu + random.uniform(8.0, 20.0), 2)
                access_metric = base_faccess + random.randint(5, 20)
                net_metric = int(base_net) + random.randint(50_000, 500_000)
            else:
                cpu_metric = round(base_cpu + random.uniform(0.0, 3.0), 2)
                access_metric = base_faccess + random.randint(0, 2)
                net_metric = int(base_net) + random.randint(0, 20_000)

            metrics = {
                'access_count': int(access_metric),
                'cpu_percent': float(cpu_metric),
                'memory_mb': float(round(base_mem, 2)),
                'net_bytes': int(net_metric),
                'last_sent_ts': time.time()
            }
            event['client_metrics'] = metrics
        except Exception:
            # Do not block sending if metrics fail
            event['client_metrics'] = None

        write_local_log(event)
        posted = post_to_server(event)
        print(f'[{client_id}] Row {i} threat={is_threat} posted={posted}')
        time.sleep(sleep_override)

    print(f'[{client_id}] Client finished streaming logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='client1')
    args = parser.parse_args()
    run_client(client_id=args.id)
