"""FedSIG+ client node that simulates signature generation every 2 minutes.

The behaviour is unchanged from the original FedShield prototype; only the
terminology and inline documentation were updated to describe the FedSIG+
workflow stages (Event Monitoring → Rule Evaluation → Signature Generation
→ Trust Filtering → Federated Aggregation → Global Update).
"""
import argparse
import time
import os
import sys
import json
import logging
from datetime import datetime
import random
import uuid

# Ensure repository root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from client.logging_config import setup_logging
from client.local_model import load_local_model
from client.log_manager import write_local_log, post_to_server
from client.isolation import isolate_file

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Signature profile definitions (FedSIG+ "Rule Evaluation" stage)
SIGNATURE_PROFILES = [
    {
        'type': 'malware',
        'name': 'Malware Detection',
        'cpu_pct': (85.0, 95.0),
        'net_bytes': (500000, 2000000),
        'file_access_count': (30, 60),
        'file_extensions': ['.exe', '.dll', '.bat', '.scr'],
        'file_names': ['malware.exe', 'trojan.dll', 'virus.bat', 'suspicious.scr']
    },
    {
        'type': 'unauthorized_access',
        'name': 'Unauthorized Access',
        'cpu_pct': (60.0, 80.0),
        'net_bytes': (200000, 1000000),
        'file_access_count': (20, 40),
        'file_extensions': ['.log', '.conf', '.key', '.pem'],
        'file_names': ['access.log', 'config.conf', 'private.key', 'secret.pem']
    },
    {
        'type': 'data_leak',
        'name': 'Data Leak',
        'cpu_pct': (40.0, 70.0),
        'net_bytes': (1000000, 5000000),
        'file_access_count': (10, 30),
        'file_extensions': ['.db', '.sql', '.csv', '.xlsx'],
        'file_names': ['database.db', 'backup.sql', 'users.csv', 'data.xlsx']
    },
    {
        'type': 'phishing',
        'name': 'Phishing Alert',
        'cpu_pct': (20.0, 50.0),
        'net_bytes': (10000, 500000),
        'file_access_count': (5, 20),
        'file_extensions': ['.html', '.htm', '.php', '.js'],
        'file_names': ['phish.html', 'fake_login.htm', 'malicious.php', 'tracker.js']
    },
    {
        'type': 'system_anomaly',
        'name': 'System Anomaly',
        'cpu_pct': (70.0, 90.0),
        'net_bytes': (100000, 800000),
        'file_access_count': (15, 35),
        'file_extensions': ['.tmp', '.sys', '.dmp'],
        'file_names': ['temp_file.tmp', 'system.sys', 'crash.dmp']
    }
]


def generate_threat_event(client_id, threat_config, threat_index):
    """Generate a signature contribution based on a profile."""
    threat_type = threat_config['type']
    
    # Generate values within threat type ranges
    cpu_pct = round(random.uniform(*threat_config['cpu_pct']), 1)
    net_bytes = random.randint(int(threat_config['net_bytes'][0]), int(threat_config['net_bytes'][1]))
    file_access_count = random.randint(int(threat_config['file_access_count'][0]), int(threat_config['file_access_count'][1]))
    
    # Select file based on threat type
    if random.random() > 0.3:  # 70% chance to use threat-specific file
        file_name = random.choice(threat_config['file_names'])
    else:
        file_name = f"data/suspicious_{threat_type}_{threat_index}.{threat_config['file_extensions'][0][1:]}"
    
    file_path = f"data/{file_name}"
    
    event = {
        'id': str(uuid.uuid4()),
        'client_id': client_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'cpu_pct': cpu_pct,
        'net_bytes': net_bytes,
        'file_access_count': file_access_count,
        'file_path': file_path,
        'is_threat': True,
        'threat_type': threat_type,
        'threat_name': threat_config['name'],
        'action': 'none'
    }
    
    # Try to quarantine
    try:
        newpath = isolate_file(file_path)
        event['action'] = 'quarantine'
        event['quarantined_path'] = newpath
    except Exception as e:
        event['action'] = 'quarantine_failed'
        event['error'] = str(e)
    
    # NOTE: Trust scoring lives outside this prototype. In FedSIG+, this is
    # where the client would update its trust context before forwarding the
    # signature. We log the event unchanged to preserve workflow compatibility.
    return event


def generate_normal_event(client_id):
    """Generate a normal (non-threat) event."""
    cpu_pct = round(random.uniform(10.0, 30.0), 1)
    net_bytes = random.choice([256, 512, 1024, 2048, 4096])
    file_access_count = random.randint(0, 5)
    file_path = f"data/sample_file_{random.randint(1, 10)}.txt"
    
    return {
        'id': str(uuid.uuid4()),
        'client_id': client_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'cpu_pct': cpu_pct,
        'net_bytes': net_bytes,
        'file_access_count': file_access_count,
        'file_path': file_path,
        'is_threat': False,
        'action': 'none'
    }


def run_enhanced_client(client_id='client1', interval_seconds=120):
    """
    Run enhanced client that generates different threat types every 2 minutes.
    
    Args:
        client_id: Unique identifier for this client
        interval_seconds: Time interval between threat detections (default 120 = 2 minutes)
    """
    print(f'[{client_id}] Starting enhanced client node')
    print(f'[{client_id}] Threat detection interval: {interval_seconds} seconds (2 minutes)')
    
    # Load model (for compatibility, though we're generating synthetic threats)
    try:
        model = load_local_model()
        print(f'[{client_id}] Model loaded')
    except Exception as e:
        print(f'[{client_id}] Warning: Could not load model: {e}')
        model = None
    
    threat_index = 0
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Generate threat every interval_seconds
            if elapsed >= interval_seconds:
                # Cycle through signature profiles (Poster Step 2: Rule Evaluation)
                threat_config = SIGNATURE_PROFILES[threat_index % len(SIGNATURE_PROFILES)]
                
                print(f'[{client_id}] [{datetime.now().strftime("%H:%M:%S")}] Detecting: {threat_config["name"]}')
                
                # Generate threat event
                event = generate_threat_event(client_id, threat_config, threat_index)
                
                # Log and post
                write_local_log(event)
                posted = post_to_server(event)
                
                if posted:
                    print(f'[{client_id}] ✓ Threat reported: {threat_config["name"]} (Type: {threat_config["type"]})')
                else:
                    print(f'[{client_id}] ✗ Failed to post threat: {threat_config["name"]}')
                
                threat_index += 1
                start_time = current_time  # Reset timer
            else:
                # Generate normal events occasionally
                if random.random() < 0.1:  # 10% chance every loop
                    event = generate_normal_event(client_id)
                    write_local_log(event)
                    post_to_server(event)
            
            # Sleep for a short interval
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print(f'\n[{client_id}] Shutting down...')
    except Exception as e:
        print(f'[{client_id}] Error: {e}')
        logger.exception(f"Error in enhanced client: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced client node with different threat types')
    parser.add_argument('--id', default='client1', help='Client ID')
    parser.add_argument('--interval', type=int, default=120, help='Threat detection interval in seconds (default: 120 = 2 minutes)')
    args = parser.parse_args()
    
    run_enhanced_client(client_id=args.id, interval_seconds=args.interval)

